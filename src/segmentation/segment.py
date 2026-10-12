"""
Step 2 of 2: segment only.
============================================================================

Reads --classifier-csv (produced by src/classification/classify.py), takes every row where
included == True, and runs nnU-Net on its harmonized .nii.gz path (the
nifti_path column). No TensorFlow dependency at all - only nnunetv2/torch.

Safety features carried over from the combined pipeline:
    - Each batch of cases runs in its own spawned subprocess. If it
      segfaults, only that subprocess dies - the run continues with the
      next batch instead of losing everything.
    - Before running, each case's spacing is compared against the model's
      trained target spacing to estimate the resampled voxel count; cases
      far enough over a safety ceiling are skipped rather than attempted,
      since that mismatch is a plausible cause of native OOM segfaults
      (e.g. a PNG assumed to be 1mm spacing vs. a model trained on much
      finer real spacing).

IMPORTANT: if your classifier CSV includes cases converted from PNG (no real
spacing info - see src/classification/classify.py's fake 1mm assumption) and your nnU-Net
model was trained on a genuinely different modality/spacing (e.g. MRI vs.
ultrasound), technical success here does not imply meaningful segmentations.
Inspect actual output volumes, don't just trust that the run completed.

Requirements
------------
    pip install nnunetv2 torch pandas nibabel
"""

import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("segment_only")


class NNUNetSegmenter:
    # ADJUST ME: see module docstring - rough safety ceiling on estimated
    # resampled voxel count before a case is skipped rather than attempted.
    MAX_ESTIMATED_VOXELS = 200_000_000

    def __init__(self, model_folder: Path, folds, checkpoint: str, device: str):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=(device == "cuda"),
            device=torch.device(device),
            verbose=False,
            allow_tqdm=True,
        )
        self.predictor.initialize_from_trained_model_folder(str(model_folder), folds, checkpoint)

    def _estimate_resampled_voxels(self, nifti_path: Path) -> Optional[float]:
        if not str(nifti_path).lower().endswith((".nii", ".nii.gz")):
            return None  # not a NIfTI file - spacing-based resample estimate doesn't apply
        import nibabel as nib
        try:
            img = nib.load(str(nifti_path))
            shape = img.shape
            original_spacing = img.header.get_zooms()[:3]
            target_spacing = self.predictor.configuration_manager.spacing
            if len(target_spacing) != len(original_spacing):
                return None
            scale = [o / t for o, t in zip(original_spacing, target_spacing)]
            resampled_shape = [s * sc for s, sc in zip(shape, scale)]
            return float(np.prod(resampled_shape))
        except Exception as e:
            log.warning("Could not estimate resampled size for %s (%s) - proceeding anyway", nifti_path, e)
            return None

    def filter_oversized(self, nifti_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        safe, skipped = [], []
        for p in nifti_paths:
            est = self._estimate_resampled_voxels(p)
            if est is not None and est > self.MAX_ESTIMATED_VOXELS:
                log.warning("%s: estimated resampled size ~%.0f voxels exceeds safety limit "
                            "(%.0f) - likely spacing mismatch with training data - skipping",
                            p.name, est, self.MAX_ESTIMATED_VOXELS)
                skipped.append(p)
            else:
                safe.append(p)
        return safe, skipped

    def predict_batch(self, nifti_paths: List[Path], output_dir: Path) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        input_lists = [[str(p)] for p in nifti_paths]
        output_truncated = [str(output_dir / p.stem.replace(".nii", "")) for p in nifti_paths]
        self.predictor.predict_from_files(
            input_lists, output_truncated, save_probabilities=False, overwrite=True,
        )
        file_ending = self.predictor.dataset_json["file_ending"]
        return [Path(o + file_ending) for o in output_truncated]


def _segment_batch_subprocess_worker(model_folder: str, folds, checkpoint: str, device: str,
                                     batch_paths: List[str], output_dir: str, result_queue) -> None:
    """Runs in its own spawned OS process - if it segfaults, only it dies."""
    try:
        segmenter = NNUNetSegmenter(Path(model_folder), folds, checkpoint, device)
        paths = [Path(p) for p in batch_paths]
        safe_paths, skipped_paths = segmenter.filter_oversized(paths)
        seg_paths = segmenter.predict_batch(safe_paths, Path(output_dir)) if safe_paths else []
        result_queue.put({
            "ok": True,
            "segmented": dict(zip([p.stem.replace(".nii", "") for p in safe_paths], [str(s) for s in seg_paths])),
            "skipped_oversized": [p.stem.replace(".nii", "") for p in skipped_paths],
        })
    except Exception as e:
        result_queue.put({"ok": False, "error": str(e)})


def _peek_model_file_ending(model_folder: Path) -> str:
    """Reads file_ending straight out of the trained model's dataset.json,
    without initializing the full predictor - this is what determines which
    reader/writer class nnU-Net will use for every input file at inference
    time, regardless of the extension on the file we actually hand it."""
    import json
    dataset_json_path = model_folder / "dataset.json"
    if not dataset_json_path.exists():
        raise SystemExit(f"No dataset.json found at {dataset_json_path} - is --nnunet-model-folder correct?")
    with open(dataset_json_path) as f:
        return json.load(f)["file_ending"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier-csv", type=Path, required=True, help="Output CSV from src/classification/classify.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nnunet-model-folder", type=Path, required=True)
    parser.add_argument("--nnunet-folds", nargs="+", default=["all"])
    parser.add_argument("--nnunet-checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--batch-size", type=int, default=20,
                         help="Cases per isolated subprocess call. Smaller = a crash loses less "
                         "progress and is easier to localize to a specific case.")
    parser.add_argument("--manifest-out", type=Path, default=Path("segmentation_manifest.csv"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.classifier_csv)
    included = df[df["included"] == True].copy()  # noqa: E712
    if included.empty:
        raise SystemExit(f"No included==True rows found in {args.classifier_csv}")
    log.info("%d/%d case(s) from the classifier CSV are marked included", len(included), len(df))

    # Peek at the target model's expected file format BEFORE building input
    # paths - nnU-Net reads every input using whatever reader/writer class was
    # recorded in the model's own dataset.json at training time, regardless
    # of what extension the file we hand it actually has. Feeding it the
    # wrong format (e.g. .nii.gz to a model trained on .png) fails with an
    # opaque "no backend to open" error rather than a clear mismatch message.
    model_file_ending = _peek_model_file_ending(args.nnunet_model_folder)
    log.info("Target model expects file_ending=%s", model_file_ending)

    case_ids, nifti_paths = [], []
    skipped_format_mismatch = []
    for _, row in included.iterrows():
        case_id = row["case_id"]
        if model_file_ending in (".nii", ".nii.gz"):
            path = row["nifti_path"]
        else:
            # Model expects its native (e.g. png) format - use the original
            # file directly if we have one; dicom cases and load-failed rows
            # have no usable original_path, so they can't be fed to a
            # non-nifti model and are skipped.
            original = row.get("original_path")
            if pd.isna(original):
                skipped_format_mismatch.append(case_id)
                continue
            path = original
        case_ids.append(case_id)
        nifti_paths.append(Path(path))

    if skipped_format_mismatch:
        log.warning("%d case(s) skipped - target model expects %s but no matching file is available "
                    "for these (e.g. DICOM cases when the model wants a native 2D format): %s",
                    len(skipped_format_mismatch), model_file_ending,
                    skipped_format_mismatch[:10] + (["..."] if len(skipped_format_mismatch) > 10 else []))

    rows = [{"case_id": cid, "nifti_path": str(p), "status": "pending"} for cid, p in zip(case_ids, nifti_paths)]
    seg_paths_by_case = {}

    mp_ctx = multiprocessing.get_context("spawn")  # required for CUDA safety, not fork

    for start in range(0, len(nifti_paths), args.batch_size):
        batch_ids = case_ids[start:start + args.batch_size]
        batch_paths = [str(p) for p in nifti_paths[start:start + args.batch_size]]
        log.info("Batch %d-%d / %d", start, start + len(batch_paths), len(nifti_paths))

        result_queue = mp_ctx.Queue()
        proc = mp_ctx.Process(
            target=_segment_batch_subprocess_worker,
            args=(str(args.nnunet_model_folder), args.nnunet_folds, args.nnunet_checkpoint,
                  args.device, batch_paths, str(args.output_dir), result_queue),
        )
        proc.start()
        proc.join()

        if proc.exitcode != 0:
            log.error("Batch %d-%d CRASHED (exit code %s) - cases in this batch left unsegmented; "
                      "continuing with the next batch.", start, start + len(batch_paths), proc.exitcode)
        else:
            result = result_queue.get() if not result_queue.empty() else {"ok": False, "error": "no result returned"}
            if result["ok"]:
                seg_paths_by_case.update(result["segmented"])
                for case_id in result["skipped_oversized"]:
                    for row in rows:
                        if row["case_id"] == case_id:
                            row["status"] = "skipped_oversized_resample"
            else:
                log.error("Batch %d-%d FAILED - %s", start, start + len(batch_paths), result["error"])

        for row in rows:
            if row["case_id"] in seg_paths_by_case:
                row["status"] = "segmented"
                row["segmentation_path"] = str(seg_paths_by_case[row["case_id"]])
            elif "segmentation_path" not in row:
                row["segmentation_path"] = None
        pd.DataFrame(rows).to_csv(args.manifest_out, index=False)

    df_out = pd.DataFrame(rows)
    print(f"\n{len(df_out)} case(s) processed -> manifest at {args.manifest_out}")
    print(df_out["status"].value_counts().to_string())


if __name__ == "__main__":
    main()