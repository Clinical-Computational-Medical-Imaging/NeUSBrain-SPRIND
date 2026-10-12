"""

export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_results=/path/to/nnUNet_results

python3 src/segmentation/nnunet_inference.py \
    --dataset 006 \
    --folds all



That's it - everything else is auto-detected from the dataset number:
  - nnUNet_raw/Dataset006_* and nnUNet_results/Dataset006_* are located automatically
    (searches by the zero-padded dataset id, so '6', '006', and the full folder name
    'Dataset006_NEUSBrainSkull' are all accepted for --dataset).
  - The trained model variant (trainer__plans__configuration folder) is auto-discovered
    under nnUNet_results/<dataset>/. If more than one exists (e.g. you trained both '2d'
    and '3d_fullres'), it's auto-picked by a configuration priority
    (3d_fullres > 3d_lowres > 2d > 3d_cascade_fullres); if that's still ambiguous
    (e.g. two different trainers for the same configuration), you'll get a clear error
    listing the options and asking you to pass --trainer/--plans/--configuration.
  - The file format (.png, .nii.gz, ...) is read from dataset.json's "file_ending" key -
    no hardcoded pattern, works for 2D natural-image datasets and 3D NIfTI datasets alike.
  - Foreground labels (including region-based labels mapped to a list of values, and an
    "ignore" label) are read from dataset.json rather than assumed to be plain ints.
  - --gt_dir defaults to nnUNet_raw/<dataset>/labelsTs.
  - If the resolved configuration is 3d_cascade_fullres, the 3d_lowres stage is run
    first automatically and fed in as the previous-stage segmentation - cascade models
    were previously unsupported by this script.

You can still override any auto-detected piece explicitly (--trainer, --plans,
--configuration, --input_dir, --output_dir, --gt_dir, --checkpoint_name) if you need to
point at something non-standard.

Requirements
------------
    pip install nnunetv2 matplotlib pillow numpy torch nibabel
    (SimpleITK is used as an optional fallback reader for formats other than
    the ones PIL/nibabel handle, e.g. .mha/.nrrd - only imported if needed)
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Preference order used only to auto-pick between multiple trained
# configurations for the same dataset, when --configuration isn't given.
CONFIGURATION_PRIORITY = ["3d_fullres", "3d_lowres", "2d", "3d_cascade_fullres"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / model-variant resolution - the "just give me the number" part
# ─────────────────────────────────────────────────────────────────────────────

def resolve_dataset_folder(base_dir: Path, dataset_arg: str) -> Path:
    """Find the Dataset<NNN>_<Name> folder under base_dir (nnUNet_raw or
    nnUNet_results) for a given dataset id. Accepts '6', '006', or the full
    'Dataset006_NEUSBrainSkull' folder name."""
    base_dir = Path(base_dir)
    if dataset_arg.lower().startswith("dataset"):
        candidate = base_dir / dataset_arg
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{candidate} does not exist")

    try:
        dataset_id = int(dataset_arg)
    except ValueError:
        raise ValueError(
            f"--dataset {dataset_arg!r} isn't a number or a 'DatasetNNN_...' folder name"
        )

    matches = sorted(base_dir.glob(f"Dataset{dataset_id:03d}_*"))
    if not matches:
        raise FileNotFoundError(f"No Dataset{dataset_id:03d}_* folder found under {base_dir}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple folders match Dataset{dataset_id:03d}_* under {base_dir}: "
            f"{[m.name for m in matches]} - pass the full dataset name via --dataset instead"
        )
    return matches[0]


def resolve_model_variant(
    results_dataset_dir: Path,
    trainer: Optional[str] = None,
    plans: Optional[str] = None,
    configuration: Optional[str] = None,
) -> Path:
    """Find the trained model variant folder (TRAINER__PLANS__CONFIGURATION)
    under a dataset's nnUNet_results folder, auto-picking one if unambiguous."""
    candidates = [d for d in results_dataset_dir.iterdir() if d.is_dir() and d.name.count("__") == 2]

    def matches(name: str) -> bool:
        t, p, c = name.split("__")
        if trainer and t != trainer:
            return False
        if plans and p != plans:
            return False
        if configuration and c != configuration:
            return False
        return True

    if trainer or plans or configuration:
        candidates = [d for d in candidates if matches(d.name)]

    if not candidates:
        raise FileNotFoundError(
            f"No trained model variant found under {results_dataset_dir} "
            f"matching trainer={trainer}, plans={plans}, configuration={configuration}"
        )

    if len(candidates) == 1:
        return candidates[0]

    # Multiple candidates remain - try to break the tie purely by configuration
    # priority (handles the common "trained 2d and 3d_fullres, didn't say which" case).
    by_config: Dict[str, List[Path]] = {}
    for d in candidates:
        cfg = d.name.split("__")[2]
        by_config.setdefault(cfg, []).append(d)

    for pref in CONFIGURATION_PRIORITY:
        if pref in by_config and len(by_config[pref]) == 1:
            print(f"Multiple trained model variants found; auto-picking configuration '{pref}': {by_config[pref][0].name}")
            return by_config[pref][0]

    names = [d.name for d in candidates]
    raise ValueError(
        f"Multiple trained model variants found under {results_dataset_dir} and none could "
        f"be auto-selected: {names}. Disambiguate with --trainer/--plans/--configuration."
    )


def load_dataset_json(raw_dataset_dir: Path, model_variant_dir: Path) -> dict:
    """Prefer nnUNet_raw's dataset.json (the canonical source); fall back to
    the copy nnU-Net stores inside the trained model folder, in case the raw
    dataset was pruned/moved after training."""
    for candidate in (raw_dataset_dir / "dataset.json", model_variant_dir / "dataset.json"):
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    raise FileNotFoundError(f"No dataset.json found in {raw_dataset_dir} or {model_variant_dir}")


def foreground_label_items(labels: Dict[str, Union[int, list]]) -> Dict[str, Union[int, list]]:
    """dataset.json's 'labels' includes 'background' (always) and sometimes an
    'ignore' label - neither should be scored as a foreground class. Values
    can be a plain int, or a list of ints for region-based training."""
    return {name: val for name, val in labels.items() if name.lower() not in ("background", "ignore")}


# ─────────────────────────────────────────────────────────────────────────────
# Generic array loading - works for 2D natural images and 3D NIfTI alike
# ─────────────────────────────────────────────────────────────────────────────

def load_array(path: Path) -> np.ndarray:
    suffix = "".join(path.suffixes).lower()
    if suffix in (".nii", ".nii.gz"):
        import nibabel as nib
        return np.asarray(nib.load(str(path)).dataobj)
    try:
        return np.array(Image.open(path))
    except Exception as pil_error:
        try:
            import SimpleITK as sitk
            return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        except Exception:
            raise ValueError(
                f"Don't know how to read {path} (tried PIL, then SimpleITK as fallback): {pil_error}"
            )


def middle_slice_for_display(arr: np.ndarray) -> np.ndarray:
    """For 3D volumes, visualization shows a single middle slice (Dice is still
    computed on the full volume elsewhere - this is display-only)."""
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        return arr[..., arr.shape[-1] // 2]
    return arr


def collect_data(directory: Path, pattern: str) -> List[Path]:
    return sorted(directory.rglob(pattern))


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    input_dir: Path,
    output_dir: Path,
    model_variant_dir: Path,
    results_dataset_dir: Path,
    folds: Tuple,
    checkpoint_name: str = "checkpoint_final.pth",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration = model_variant_dir.name.split("__")[2]
    prev_stage_dir = None
    if configuration == "3d_cascade_fullres":
        print("3d_cascade_fullres detected - running the 3d_lowres stage first "
              "to produce the previous-stage segmentations it needs.")
        lowres_variant_dir = resolve_model_variant(results_dataset_dir, configuration="3d_lowres")
        prev_stage_dir = output_dir / "_lowres_stage_segs"
        run_inference(input_dir, prev_stage_dir, lowres_variant_dir, results_dataset_dir,
                      folds, checkpoint_name=checkpoint_name)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        str(model_variant_dir),
        use_folds=folds,
        checkpoint_name=checkpoint_name,
    )

    predictor.predict_from_files(
        str(input_dir),
        str(output_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=str(prev_stage_dir) if prev_stage_dir else None,
        num_parts=1,
        part_id=0,
    )


def case_id_from_image_path(image_path: Path, file_ending: str) -> str:
    """nnU-Net input images are named '<case_id>_0000<file_ending>' (channel suffix); strip it."""
    suffix = f"_0000{file_ending}"
    name = image_path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return image_path.stem


# ─────────────────────────────────────────────────────────────────────────────
# Statistics against ground truth
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray, label_value: Union[int, list]) -> float:
    if isinstance(label_value, (list, tuple)):
        pred_mask = np.isin(pred, label_value)
        gt_mask = np.isin(gt, label_value)
    else:
        pred_mask = pred == label_value
        gt_mask = gt == label_value
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    return float("nan") if denom == 0 else 2.0 * intersection / denom


def compute_statistics(
    output_dir: Path, gt_dir: Path, labels: Dict[str, Union[int, list]], file_ending: str
) -> List[dict]:
    """Per-case Dice score for every foreground label, comparing each predicted
    segmentation in output_dir against the matching file in gt_dir (labelsTs)."""
    seg_files = collect_data(output_dir, pattern=f"*{file_ending}")
    if not seg_files:
        print(f"No prediction files found in {output_dir} matching '*{file_ending}' - no statistics computed")
        return []

    foreground_labels = foreground_label_items(labels)

    rows = []
    for seg_path in seg_files:
        case_id = seg_path.name[: -len(file_ending)]
        gt_path = gt_dir / f"{case_id}{file_ending}"
        if not gt_path.exists():
            print(f"{seg_path.name}: no matching ground truth at {gt_path} - skipping stats for this case")
            continue

        try:
            pred = load_array(seg_path)
            gt = load_array(gt_path)
        except Exception as e:
            print(f"{seg_path.name}: could not read prediction/GT ({e}) - skipping stats for this case")
            continue

        if pred.shape != gt.shape:
            print(f"{seg_path.name}: shape mismatch (pred={pred.shape}, gt={gt.shape}) - skipping stats for this case")
            continue

        row = {"case_id": case_id}
        for label_name, label_value in foreground_labels.items():
            row[f"dice_{label_name}"] = dice_score(pred, gt, label_value)
        rows.append(row)

    return rows


def save_statistics_csv(rows: List[dict], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-case statistics -> {csv_path}")


def print_summary_statistics(rows: List[dict]) -> None:
    if not rows:
        print("No statistics to summarize (no ground-truth matches found).")
        return

    dice_keys = [k for k in rows[0].keys() if k.startswith("dice_")]
    print(f"\nSummary statistics over {len(rows)} case(s):")
    for key in dice_keys:
        values = [r[key] for r in rows if not np.isnan(r[key])]
        n_missing = len(rows) - len(values)
        if values:
            print(
                f"  {key:20s} mean={np.mean(values):.4f}  std={np.std(values):.4f}  "
                f"min={np.min(values):.4f}  max={np.max(values):.4f}  n={len(values)}"
                + (f"  ({n_missing} case(s) had no foreground in either pred or GT)" if n_missing else "")
            )
        else:
            print(f"  {key:20s} no valid cases (label absent in both pred and GT for all cases)")


def statistics_by_case(rows: List[dict]) -> Dict[str, dict]:
    return {row["case_id"]: row for row in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    input_dir: Path,
    output_dir: Path,
    save_dir: Path,
    file_ending: str,
    gt_dir: Optional[Path] = None,
    stats_by_case: Optional[Dict[str, dict]] = None,
    max_label_value: int = 1,
) -> None:
    """For each predicted segmentation, plot the original image, ground truth (if available),
    predicted mask, and an overlay, and save the figure as a PNG per case. If per-case Dice
    stats are available, they're added to the figure title. 3D volumes are shown as a single
    middle slice (Dice itself is computed on the full volume in compute_statistics)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    seg_files = collect_data(output_dir, pattern=f"*{file_ending}")
    if not seg_files:
        print(f"No prediction files found in {output_dir} matching '*{file_ending}'")
        return

    stats_by_case = stats_by_case or {}

    for seg_path in seg_files:
        case_id = seg_path.name[: -len(file_ending)]
        image_path = input_dir / f"{case_id}_0000{file_ending}"
        if not image_path.exists():
            print(f"{seg_path.name}: no matching input image at {image_path} - skipping")
            continue

        try:
            image = load_array(image_path)
            seg = load_array(seg_path)
        except Exception as e:
            print(f"{seg_path.name}: could not read image/segmentation ({e}) - skipping visualization")
            continue

        is_3d = seg.ndim == 3 and seg.shape[-1] not in (3, 4)
        image = middle_slice_for_display(image)
        if image.ndim == 3 and image.shape[-1] in (3, 4):
            image = image.mean(axis=-1)
        seg = middle_slice_for_display(seg)

        gt = None
        if gt_dir is not None:
            gt_path = gt_dir / f"{case_id}{file_ending}"
            if gt_path.exists():
                try:
                    gt = middle_slice_for_display(load_array(gt_path))
                except Exception as e:
                    print(f"{seg_path.name}: could not read GT for visualization ({e})")

        n_panels = 4 if gt is not None else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

        vmax = max(int(seg.max()), int(gt.max()) if gt is not None else 1, max_label_value)

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Image")
        axes[0].axis("off")

        panel_idx = 1
        if gt is not None:
            axes[panel_idx].imshow(gt, cmap="jet", vmin=0, vmax=vmax)
            axes[panel_idx].set_title("Ground truth")
            axes[panel_idx].axis("off")
            panel_idx += 1

        axes[panel_idx].imshow(seg, cmap="jet", vmin=0, vmax=vmax)
        axes[panel_idx].set_title("Predicted segmentation")
        axes[panel_idx].axis("off")
        panel_idx += 1

        axes[panel_idx].imshow(image, cmap="gray")
        axes[panel_idx].imshow(np.ma.masked_where(seg == 0, seg), cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        axes[panel_idx].set_title("Overlay")
        axes[panel_idx].axis("off")

        title = case_id + ("  (middle slice of 3D volume)" if is_3d else "")
        case_stats = stats_by_case.get(case_id)
        if case_stats:
            dice_str = "  ".join(
                f"{k.replace('dice_', '')}={v:.3f}" for k, v in case_stats.items() if k.startswith("dice_") and not np.isnan(v)
            )
            if dice_str:
                title = f"{title}   Dice: {dice_str}"

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_dir / f"{case_id}.png", dpi=150)
        plt.close(fig)

    print(f"Saved {len(seg_files)} visualization(s) to {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True,
                         help="Dataset id: '6', '006', or the full 'Dataset006_NEUSBrainSkull' name. "
                              "Everything else is auto-detected from this.")
    parser.add_argument("--input_dir", type=Path, default=None,
                         help="Default: <nnUNet_raw>/<dataset>/imagesTs")
    parser.add_argument("--output_dir", type=Path, default=None,
                         help="Default: ./<dataset_name>_predictions")
    parser.add_argument("--gt_dir", type=Path, default=None,
                         help="Default: <nnUNet_raw>/<dataset>/labelsTs")
    parser.add_argument("--vis_dir", type=Path, default=None,
                         help="Default: <output_dir>/visualizations")
    parser.add_argument("--folds", type=str, nargs="+", default=["all"],
                         help="Folds to use, e.g. 0 1 2 3 4, or 'all'")
    parser.add_argument("--trainer", type=str, default=None,
                         help="Override auto-detected trainer (only needed if multiple trained variants exist)")
    parser.add_argument("--plans", type=str, default=None,
                         help="Override auto-detected plans identifier (e.g. nnUNetPlans)")
    parser.add_argument("--configuration", type=str, default=None,
                         help="Override auto-detected configuration (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres)")
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint_final.pth")
    parser.add_argument("--skip_stats", action="store_true",
                         help="Skip Dice statistics computation entirely (e.g. if you have no ground truth for this run).")
    args = parser.parse_args()

    raw_dataset_dir = resolve_dataset_folder(Path(nnUNet_raw), args.dataset)
    results_dataset_dir = resolve_dataset_folder(Path(nnUNet_results), args.dataset)
    dataset_name = raw_dataset_dir.name
    print(f"Resolved dataset: {dataset_name}")

    model_variant_dir = resolve_model_variant(results_dataset_dir, args.trainer, args.plans, args.configuration)
    print(f"Using trained model variant: {model_variant_dir.name}")

    dataset_json = load_dataset_json(raw_dataset_dir, model_variant_dir)
    file_ending = dataset_json.get("file_ending", ".png")
    labels = dataset_json["labels"]
    print(f"File format: {file_ending}   Labels: {labels}")

    input_dir = args.input_dir or (raw_dataset_dir / "imagesTs")
    output_dir = args.output_dir or Path(f"./{dataset_name}_predictions")
    gt_dir = args.gt_dir or (raw_dataset_dir / "labelsTs")
    vis_dir = args.vis_dir or (output_dir / "visualizations")

    folds = tuple(f if f == "all" else int(f) for f in args.folds)

    run_inference(
        input_dir,
        output_dir,
        model_variant_dir,
        results_dataset_dir,
        folds,
        checkpoint_name=args.checkpoint_name,
    )

    max_label_value = max(
        (v if isinstance(v, int) else max(v) for v in labels.values() if isinstance(v, (int, list, tuple))),
        default=1,
    )

    stats_rows: List[dict] = []
    stats_lookup: Dict[str, dict] = {}
    if not args.skip_stats:
        if not gt_dir.exists():
            print(f"Ground truth folder {gt_dir} does not exist - skipping statistics. "
                  f"Pass --gt_dir explicitly or --skip_stats to silence this.")
        else:
            stats_rows = compute_statistics(output_dir, gt_dir, labels, file_ending)
            save_statistics_csv(stats_rows, output_dir / "statistics.csv")
            print_summary_statistics(stats_rows)
            stats_lookup = statistics_by_case(stats_rows)

    plot_results(
        input_dir,
        output_dir,
        vis_dir,
        file_ending,
        gt_dir=gt_dir if gt_dir.exists() else None,
        stats_by_case=stats_lookup,
        max_label_value=max_label_value,
    )


if __name__ == "__main__":
    main()