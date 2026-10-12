"""
Step 1 of 2: classify only.
============================================================================

Give it one input path (any mix of DICOM series, NIfTI, or PNG/JPG, any
folder structure). For each case it:

    1. Harmonizes it to a persistent .nii.gz (DICOM series and PNGs are
       converted; NIfTI passes through) under --harmonized-dir. Unlike a
       tmp dir, this is NOT cleaned up - the next script needs these files.
    2. Gets a representative 2D slice and runs it through your TF
       include/exclude classifier.
    3. Writes one row per case to --output-csv, immediately after each case
       is classified (not batched at the end) - so a crash partway through
       still leaves you with every result up to that point.

This script has NO nnU-Net/torch dependency - only tensorflow, opencv,
nibabel, pandas. Run step 2 (src/segmentation/segment.py) separately, in whatever
environment has nnunetv2/torch, pointing it at --output-csv from this run.

Requirements
------------
    pip install tensorflow opencv-python nibabel pandas
    pip install SimpleITK   # for DICOM series reading
"""

import argparse
import logging
import os

# Must be set before TensorFlow is imported/initialized.
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# Classifier is a single small forward pass per case - doesn't need a GPU,
# and some layers (e.g. EfficientNet's normalization) explicitly request XLA
# JIT compilation regardless of TF_XLA_FLAGS, which fails on clusters without
# a full CUDA toolkit (missing libdevice.10.bc). CPU sidesteps that entirely.
tf.config.set_visible_devices([], "GPU")
assert not tf.config.get_visible_devices("GPU"), "Expected TensorFlow to have no visible GPUs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("classify_only")

TARGET_H = 288  # ADJUST ME: match whatever your classifier was actually trained with
TARGET_W = 256
CLASS_NAMES = ["exclude", "include"]  # default; override with --class-names

NIFTI_EXTS = (".nii", ".nii.gz")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
KNOWN_NON_DATA_EXTS = (".csv", ".txt", ".json", ".md", ".xlsx", ".xls", ".log",
                       ".ini", ".yaml", ".yml", ".db", ".ds_store")


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def _has_nifti_ext(name: str) -> bool:
    return any(name.lower().endswith(ext) for ext in NIFTI_EXTS)


def _is_dicom_series_dir(d: Path) -> bool:
    import SimpleITK as sitk
    try:
        return len(sitk.ImageSeriesReader().GetGDCMSeriesIDs(str(d))) > 0
    except Exception:
        return False


def discover_cases(root: Path) -> List[Dict]:
    cases = []
    candidate_dirs = set()
    skipped_non_data = 0

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        if _has_nifti_ext(name):
            cases.append({"case_id": p.stem.replace(".nii", ""), "kind": "nifti", "files": [p]})
        elif name.lower().endswith(IMAGE_EXTS):
            cases.append({"case_id": p.stem, "kind": "png", "files": [p]})
        elif name.lower().endswith(KNOWN_NON_DATA_EXTS) or name.startswith("."):
            skipped_non_data += 1
        else:
            candidate_dirs.add(p.parent)

    if skipped_non_data:
        log.info("Ignored %d non-data sidecar file(s) (csv/txt/json/etc.)", skipped_non_data)

    for d in sorted(candidate_dirs):
        if _is_dicom_series_dir(d):
            cases.append({"case_id": d.name, "kind": "dicom", "files": sorted(d.glob("*"))})
        else:
            log.warning("%s: contains non-image files but is not a valid DICOM series - skipping", d)

    log.info("Discovered %d case(s): %d nifti, %d png, %d dicom",
              len(cases),
              sum(c["kind"] == "nifti" for c in cases),
              sum(c["kind"] == "png" for c in cases),
              sum(c["kind"] == "dicom" for c in cases))
    return cases


# ---------------------------------------------------------------------------
# Harmonization to persistent .nii.gz + representative slice extraction
# ---------------------------------------------------------------------------

def load_volume_and_representative_slice(
    case: Dict, harmonized_dir: Path, slice_selection: str
) -> Tuple[Path, np.ndarray]:
    import nibabel as nib

    kind = case["kind"]
    case_id = case["case_id"]

    if kind == "nifti":
        nifti_path = case["files"][0]
        arr = np.asanyarray(nib.load(str(nifti_path)).dataobj)

    elif kind == "dicom":
        import SimpleITK as sitk
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(case["files"][0].parent))
        if not series_ids:
            raise ValueError(f"{case_id}: no DICOM series found")
        file_names = reader.GetGDCMSeriesFileNames(str(case["files"][0].parent), series_ids[0])
        reader.SetFileNames(file_names)
        image = reader.Execute()
        nifti_path = harmonized_dir / f"{case_id}.nii.gz"
        sitk.WriteImage(image, str(nifti_path))
        arr = np.asanyarray(nib.load(str(nifti_path)).dataobj)

    elif kind == "png":
        gray = cv2.imread(str(case["files"][0]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"{case_id}: could not read image")
        arr = gray[np.newaxis, :, :].astype(np.float32)  # pseudo-volume (1, H, W)
        nifti_path = harmonized_dir / f"{case_id}.nii.gz"
        nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(nifti_path))  # ADJUST ME: fake 1mm spacing

    else:
        raise ValueError(f"Unknown case kind: {kind}")

    slice_2d = pick_representative_slice(arr, slice_selection)
    rgb = to_rgb_uint8(slice_2d)
    return nifti_path, rgb


def pick_representative_slice(arr: np.ndarray, slice_selection: str) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    n_slices = arr.shape[-1]
    if slice_selection == "middle" or n_slices == 1:
        return arr[..., n_slices // 2]
    best_idx = max(range(n_slices), key=lambda i: np.count_nonzero(arr[..., i]))
    return arr[..., best_idx]


def to_rgb_uint8(slice_2d: np.ndarray) -> np.ndarray:
    slice_2d = slice_2d.astype(np.float32)
    rng = slice_2d.max() - slice_2d.min()
    norm = (slice_2d - slice_2d.min()) / rng * 255.0 if rng > 0 else slice_2d
    gray_uint8 = norm.astype(np.uint8)
    return np.stack([gray_uint8] * 3, axis=-1)


# ---------------------------------------------------------------------------
# TF include/exclude classifier
# ---------------------------------------------------------------------------

class InclExclClassifier:
    def __init__(self, model_path: Path, class_names: List[str]):
        log.info("Loading TF classifier from %s", model_path)
        self.model = tf.keras.models.load_model(str(model_path))
        self.class_names = class_names

    def predict(self, rgb_uint8: np.ndarray) -> Tuple[str, float]:
        resized = tf.image.resize(
            tf.convert_to_tensor(rgb_uint8, dtype=tf.float32), (TARGET_H, TARGET_W)
        )
        batch = np.expand_dims(resized.numpy(), axis=0)
        probs = self.model.predict(batch, verbose=0)
        if probs.max() > 1 or probs.min() < 0:  # logits, not already softmaxed
            probs = tf.nn.softmax(probs, axis=1).numpy()
        predicted_idx = int(np.argmax(probs, axis=1)[0])
        return self.class_names[predicted_idx], float(np.max(probs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tf-model", type=Path, required=True)
    parser.add_argument("--harmonized-dir", type=Path, required=True,
                         help="Persistent output dir for converted .nii.gz files - src/segmentation/segment.py reads from here")
    parser.add_argument("--output-csv", type=Path, default=Path("classifier_results.csv"))
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--class-names", nargs="+", default=CLASS_NAMES,
                         help="Class names in model output order (default: exclude include). "
                         "For the coronal/sagittal model use: Coronal Saggital")
    parser.add_argument("--positive-class", default="include",
                         help="Class that marks a case as included==True (default: include). "
                         "For the coronal/sagittal model use: Saggital")
    parser.add_argument("--slice-selection", choices=["middle", "max-content"], default="middle")
    parser.add_argument("--resume", action="store_true",
                         help="Append to an existing --output-csv instead of overwriting it. Only use this "
                         "if that file was written by THIS version of the script - appending onto a CSV "
                         "from an older schema silently corrupts it (columns won't line up).")
    args = parser.parse_args()

    args.harmonized_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(args.input)
    classifier = InclExclClassifier(args.tf_model, args.class_names)

    if args.resume and args.output_csv.exists():
        output_csv_exists = True
        log.info("--resume set: appending to existing %s", args.output_csv)
    else:
        if args.output_csv.exists():
            log.warning("%s already exists and --resume was not set - overwriting it fresh.", args.output_csv)
            args.output_csv.unlink()
        output_csv_exists = False
    n_written = 0

    for case in cases:
        case_id = case["case_id"]
        # Original path, when the case is a single file (png/nifti) - useful
        # if the target nnU-Net model turns out to be trained on that native
        # format rather than nii.gz. Dicom cases have no single "original
        # path" (a whole directory of files), so this stays empty for those.
        original_path = str(case["files"][0]) if case["kind"] in ("png", "nifti") and len(case["files"]) == 1 else None
        try:
            nifti_path, rep_slice = load_volume_and_representative_slice(case, args.harmonized_dir, args.slice_selection)
        except Exception as e:
            log.error("%s: failed to load/harmonize - %s", case_id, e)
            row = {"case_id": case_id, "kind": case["kind"], "status": "load_failed",
                   "error": str(e), "nifti_path": None, "original_path": original_path,
                   "classifier_label": None, "classifier_confidence": None, "included": False}
        else:
            try:
                label, confidence = classifier.predict(rep_slice)
                included = (label == args.positive_class) and (confidence >= args.min_confidence)
                row = {"case_id": case_id, "kind": case["kind"], "status": "classified",
                       "error": None, "nifti_path": str(nifti_path), "original_path": original_path,
                       "classifier_label": label,
                       "classifier_confidence": round(confidence, 4), "included": included}
                log.info("%s: %s (confidence=%.2f) -> included=%s", case_id, label, confidence, included)
            except Exception as e:
                log.error("%s: classifier failed - %s", case_id, e)
                row = {"case_id": case_id, "kind": case["kind"], "status": "classifier_failed",
                       "error": str(e), "nifti_path": str(nifti_path), "original_path": original_path,
                       "classifier_label": None, "classifier_confidence": None, "included": False}

        pd.DataFrame([row]).to_csv(args.output_csv, mode="a", header=not output_csv_exists, index=False)
        output_csv_exists = True
        n_written += 1

    print(f"\n{n_written} case(s) classified -> {args.output_csv}")
    print(f"Harmonized volumes written to {args.harmonized_dir}")
    print("Next: run src/segmentation/segment.py with --classifier-csv pointing at this CSV")


if __name__ == "__main__":
    main()