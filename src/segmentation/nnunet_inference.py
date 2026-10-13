#!/usr/bin/env python3

"""
Batch nnU-Net v2 inference + evaluation for single-channel 2D models.

Required:
    --image   Folder with original images (searched recursively)
    --mask    Folder with ground-truth masks (searched recursively)
    --model   Trained nnU-Net *trainer* folder, i.e. the folder that contains
              dataset.json, plans.json and fold_X / fold_all subfolders
              (e.g. .../Dataset001_X/nnUNetTrainer__nnUNetPlans__2d)
    --output  Output folder

Folds (--fold):
    --fold 0            single fold (default)
    --fold 0 1 2        ensemble of the given folds
    --fold ensemble     ensemble of every fold_<number> folder
    --fold all          the model trained with `nnUNetv2_train ... all`
                        (folder fold_all). If fold_all does not exist, falls
                        back to ensembling all numbered folds (with a warning).

Supported formats (images and masks):
    DICOM, PNG, JPEG, BMP, TIFF, NIfTI (.nii / .nii.gz)

Image -> mask matching:
    By filename without extension (case-insensitive). A trailing nnU-Net
    channel suffix such as "_0000" is ignored if no exact match exists.
    Ambiguous matches are resolved by relative subfolder, otherwise reported.

DICOM processing:
    1. Read pixels, apply rescale slope/intercept, invert MONOCHROME1,
       convert colour to luminance.
    2. Crop to the ultrasound region (SequenceOfUltrasoundRegions),
       fall back to contour-based cropping, else use the full image.
    3. Paste the prediction back into the original image size.

Outputs:
    predictions/                 predicted label maps (.nii.gz)
    overlays/                    PNGs (green = ground truth, red = prediction)
    images/                      original images as .nii.gz
    prediction_statistics.csv    per-case metrics
    mean_statistics.csv          mean metrics (+ case counts)
    summary_statistics.csv       mean / std / median / min / max per metric
    processing_log.csv           one row per image with its final status

Important:
    Intensity handling (--intensity) and input spacing (--spacing) must match
    how the training data was created, otherwise results will be degraded.
"""

import argparse
import json
import re
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import torch
from PIL import Image

try:  # pydicom >= 3
    from pydicom.pixels import apply_modality_lut
except ImportError:  # pydicom 2.x
    from pydicom.pixel_data_handlers.util import apply_modality_lut


# ============================================================
# File formats
# ============================================================

RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NIFTI_EXTENSIONS = {".nii", ".nii.gz"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}

# Files that are never images; skipped without trying to open them.
NON_IMAGE_EXTENSIONS = {
    ".json", ".csv", ".txt", ".md", ".py", ".pkl", ".pth", ".log",
    ".xml", ".yaml", ".yml", ".zip", ".npz", ".npy",
}

METRIC_COLUMNS = ["dice", "iou", "sensitivity", "specificity", "precision"]


def get_extension(path):
    name = Path(path).name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(path).suffix.lower()


def image_stem(path):
    """Filename without its (possibly double) extension."""
    name = Path(path).name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def detect_format(path):
    """Return 'dicom', 'raster' or 'nifti'; raise ValueError otherwise."""
    path = Path(path)
    ext = get_extension(path)

    if ext in NIFTI_EXTENSIONS:
        return "nifti"
    if ext in RASTER_EXTENSIONS:
        return "raster"
    if ext in NON_IMAGE_EXTENSIONS:
        raise ValueError(f"Not an image: {path}")

    # Unknown or missing extension: sniff the content.
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True)
        return "dicom"
    except Exception:
        pass

    if ext in DICOM_EXTENSIONS:
        raise ValueError(f"Unreadable DICOM file: {path}")

    try:
        with Image.open(path) as img:
            img.verify()
        return "raster"
    except Exception:
        pass

    raise ValueError(f"Unsupported image format: {path}")


def list_supported_files(root, exclude=()):
    """Recursively collect (path, format) for supported files under root."""
    root = Path(root)
    exclude = [Path(e).resolve() for e in exclude]
    files = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        resolved = path.resolve()
        if any(ex == resolved or ex in resolved.parents for ex in exclude):
            continue
        try:
            files.append((path, detect_format(path)))
        except Exception:
            continue

    return files


# ============================================================
# Image <-> mask matching
# ============================================================

def case_key(path):
    """Stem without a trailing nnU-Net channel suffix, e.g. '_0000'."""
    return re.sub(r"_\d{4}$", "", image_stem(path)).casefold()


def build_mask_indices(mask_files):
    exact, stripped = {}, {}
    for path, _ in mask_files:
        exact.setdefault(image_stem(path).casefold(), []).append(path)
        stripped.setdefault(case_key(path), []).append(path)
    return exact, stripped


def match_mask(image_path, image_root, mask_root, exact_index, stripped_index):
    """
    Find the mask for an image. Exact stem match first, then match with the
    channel suffix removed. Returns None if nothing matches.
    """
    rel_parent = image_path.relative_to(image_root).parent

    candidates = (
        (exact_index, image_stem(image_path).casefold()),
        (stripped_index, case_key(image_path)),
    )

    for index, key in candidates:
        matches = index.get(key, [])
        if not matches:
            continue
        if len(matches) == 1:
            return matches[0]

        same_folder = [
            m for m in matches
            if m.relative_to(mask_root).parent == rel_parent
        ]
        if len(same_folder) == 1:
            return same_folder[0]

        raise ValueError(
            f"Ambiguous mask match for {image_path.name}: "
            + ", ".join(str(m) for m in matches)
        )

    return None


# ============================================================
# Model configuration and folds
# ============================================================

def get_model_config(model_path):
    dataset_json = model_path / "dataset.json"

    if not dataset_json.exists():
        hints = sorted(
            str(p.parent) for p in model_path.rglob("dataset.json")
        )[:5]
        msg = f"dataset.json not found in {model_path}."
        if hints:
            msg += (
                " --model must point to the trainer folder. Candidates:\n  "
                + "\n  ".join(hints)
            )
        raise FileNotFoundError(msg)

    with open(dataset_json, encoding="utf-8") as f:
        config = json.load(f)

    ending = config.get("file_ending")
    if not ending:
        raise ValueError("file_ending missing from dataset.json")

    channels = config.get("channel_names") or config.get("modality") or {}
    if len(channels) != 1:
        raise ValueError(
            f"This script supports single-channel models only; "
            f"dataset.json lists {len(channels)} channels: {channels}"
        )

    return config, ending.lower()


def get_training_spacing(model_path):
    """Best-effort read of the configuration's target spacing from plans.json."""
    try:
        configuration = model_path.name.split("__")[-1]
        with open(model_path / "plans.json", encoding="utf-8") as f:
            plans = json.load(f)
        return configuration, plans["configurations"][configuration]["spacing"]
    except Exception:
        return None, None


def resolve_folds(model_path, fold_args, checkpoint):
    """
    Translate --fold arguments into what nnU-Net expects.

    'all' in nnU-Net means the model trained on all data (folder 'fold_all'),
    which is passed to nnU-Net as the string 'all', not as a list of numbers.
    """
    numbered = sorted(
        int(m.group(1))
        for p in model_path.iterdir()
        if p.is_dir() and (m := re.fullmatch(r"fold_(\d+)", p.name))
    )
    has_fold_all = (model_path / "fold_all").is_dir()
    tokens = [str(f).strip().lower() for f in fold_args]

    if tokens == ["all"]:
        if has_fold_all:
            folds = ("all",)
        elif numbered:
            print(
                "[WARN] --fold all: no 'fold_all' folder found; "
                f"ensembling numbered folds {numbered} instead."
            )
            folds = tuple(numbered)
        else:
            raise RuntimeError(f"No fold folders found in {model_path}")

    elif tokens == ["ensemble"]:
        if not numbered:
            raise RuntimeError(f"No fold_<number> folders found in {model_path}")
        folds = tuple(numbered)

    else:
        try:
            folds = tuple(int(t) for t in tokens)
        except ValueError:
            raise ValueError(
                f"Invalid --fold {fold_args}. Use numbers, 'all' or 'ensemble'."
            ) from None

    # Every requested fold must have the checkpoint; nnU-Net's own error
    # for a missing checkpoint is easy to misread.
    problems = []
    for fold in folds:
        fold_dir = model_path / f"fold_{fold}"
        if not fold_dir.is_dir():
            problems.append(f"fold_{fold}: folder missing")
            continue
        if not (fold_dir / checkpoint).is_file():
            available = sorted(p.name for p in fold_dir.glob("*.pth"))
            problems.append(
                f"fold_{fold}: {checkpoint} missing "
                f"(available: {available or 'none - training unfinished?'})"
            )

    if problems:
        raise RuntimeError(
            "Cannot use the requested folds:\n  " + "\n  ".join(problems)
            + "\nUse --fold to pick finished folds, or --checkpoint "
              "checkpoint_best.pth if only that exists."
        )

    return folds


# ============================================================
# Image loading
# ============================================================

def dicom_to_gray(ds):
    """Read a single-frame DICOM as a float32 grayscale 2D array."""
    if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
        raise ValueError("Multi-frame DICOM is not supported")

    array = ds.pixel_array
    samples = int(getattr(ds, "SamplesPerPixel", 1))

    if samples > 1:
        if array.ndim != 3 or array.shape[-1] != samples:
            raise ValueError(f"Unexpected colour DICOM shape {array.shape}")
        # pydicom >= 3 converts YBR to RGB when decoding; use luminance.
        rgb = array[..., :3].astype(np.float32)
        array = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    else:
        array = apply_modality_lut(array, ds).astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            array = array.max() + array.min() - array

    if array.ndim != 2:
        raise ValueError(f"Only 2D DICOM is supported, got {array.shape}")

    return array.astype(np.float32)


def read_nifti_2d(path):
    """Read a NIfTI as a 2D array; returns (array, sitk_image, spacing_xy)."""
    img = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(img)

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a single 2D NIfTI image, got {array.shape}")

    spacing = tuple(float(s) for s in img.GetSpacing()[:2])
    return array, img, spacing


def load_image(path, fmt):
    """Return dict with the 2D float32 image and its metadata."""
    if fmt == "dicom":
        ds = pydicom.dcmread(str(path))
        return {"array": dicom_to_gray(ds), "ds": ds,
                "sitk": None, "spacing": None}

    if fmt == "raster":
        with Image.open(path) as img:
            array = np.array(img.convert("L"), dtype=np.float32)
        return {"array": array, "ds": None, "sitk": None, "spacing": None}

    if fmt == "nifti":
        array, img, spacing = read_nifti_2d(path)
        return {"array": array.astype(np.float32), "ds": None,
                "sitk": img, "spacing": spacing}

    raise ValueError(f"Unsupported format: {fmt}")


def to_label_array(array, path):
    """Validate and convert a label image to integers."""
    array = np.asarray(array)

    if array.ndim == 3:
        # Colour-stored masks are only acceptable if all channels agree.
        channels = array[..., :3] if array.shape[-1] >= 3 else array
        if not all(np.array_equal(channels[..., 0], channels[..., c])
                   for c in range(1, channels.shape[-1])):
            raise ValueError(
                f"Mask {path.name} is a true colour image; expected a "
                "single-channel label map"
            )
        array = channels[..., 0]

    if array.ndim != 2:
        raise ValueError(f"Mask {path.name} is not 2D: {array.shape}")

    if not np.isfinite(array).all():
        raise ValueError(f"Mask {path.name} contains NaN/inf")

    if not np.array_equal(array, np.round(array)):
        raise ValueError(f"Mask {path.name} contains non-integer values")

    return array.astype(np.int64)


def load_mask(path, fmt):
    """Load a label map WITHOUT any intensity conversion."""
    if fmt == "raster":
        with Image.open(path) as img:
            # Keep palette indices / raw values; do not convert to 'L'.
            if img.mode in ("RGBA", "LA"):
                img = img.convert(img.mode[:-1])
            array = np.array(img)
        return to_label_array(array, path), None

    if fmt == "nifti":
        array, img, _ = read_nifti_2d(path)
        return to_label_array(array, path), img

    if fmt == "dicom":
        ds = pydicom.dcmread(str(path))
        return to_label_array(ds.pixel_array, path), None

    raise ValueError(f"Unsupported mask format: {fmt}")


def read_prediction(path, ending):
    if ending in NIFTI_EXTENSIONS:
        array, _, _ = read_nifti_2d(path)
    else:
        with Image.open(path) as img:
            array = np.array(img)
    return to_label_array(array, Path(path))


def normalize_uint8(array):
    array = np.asarray(array, dtype=np.float32)

    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or infinite values")

    low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)

    return np.round((array - low) / (high - low) * 255).astype(np.uint8)


def check_geometry(image_sitk, mask_sitk):
    if image_sitk is None or mask_sitk is None:
        return

    if image_sitk.GetSize() != mask_sitk.GetSize():
        raise ValueError(
            f"Image size {image_sitk.GetSize()} != "
            f"mask size {mask_sitk.GetSize()}"
        )

    for name in ("GetSpacing", "GetOrigin", "GetDirection"):
        a = np.array(getattr(image_sitk, name)())
        b = np.array(getattr(mask_sitk, name)())
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-4):
            raise ValueError(
                f"Image and mask differ in {name[3:].lower()}: {a} vs {b}"
            )


# ============================================================
# Ultrasound cropping (DICOM only)
# ============================================================

def bbox_from_regions(ds, shape):
    """First valid 2D tissue region from SequenceOfUltrasoundRegions."""
    regions = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if not regions:
        return None

    height, width = shape

    for region in regions:
        if getattr(region, "RegionSpatialFormat", None) != 1:
            continue

        coords = [
            getattr(region, name, None)
            for name in ("RegionLocationMinX0", "RegionLocationMinY0",
                         "RegionLocationMaxX1", "RegionLocationMaxY1")
        ]
        if None in coords:
            continue

        x0, y0, x1, y1 = (int(c) for c in coords)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1 + 1), min(height, y1 + 1)

        if x1 > x0 and y1 > y0:
            return x0, y0, x1, y1

    return None


def bbox_from_contour(gray, min_area=10000):
    """Fallback: bounding box of the largest bright connected region."""
    _, threshold = cv2.threshold(
        normalize_uint8(gray), 10, 255, cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area:
            boxes.append((x, y, x + w, y + h))

    if not boxes:
        return None

    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def get_crop_bbox(ds, gray):
    """Returns (bbox, method) or (None, None)."""
    bbox = bbox_from_regions(ds, gray.shape)
    if bbox is not None:
        return bbox, "us_regions"

    bbox = bbox_from_contour(gray)
    if bbox is not None:
        return bbox, "contour"

    return None, None


# ============================================================
# Model input
# ============================================================

def write_model_input(array, base_path, ending, intensity, spacing):
    """Write one 2D image in the model's input format."""
    if intensity == "minmax":
        data = normalize_uint8(array)
    else:
        data = np.asarray(array, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("Image contains NaN or infinite values")

    target = str(base_path) + ending

    if ending in RASTER_EXTENSIONS:
        if intensity == "raw":
            if (data.min() < 0 or data.max() > 255
                    or not np.array_equal(data, np.round(data))):
                raise ValueError(
                    "Raw intensities do not fit uint8 for a "
                    f"{ending} model; use --intensity minmax"
                )
            data = data.astype(np.uint8)
        Image.fromarray(data).save(target)
        return

    if ending in NIFTI_EXTENSIONS:
        # nnU-Net's SimpleITKIO treats 2D NIfTI as a single slice and adds
        # the dummy 999 spacing itself.
        img = sitk.GetImageFromArray(data.astype(np.float32))
        img.SetSpacing(tuple(float(s) for s in spacing))
        sitk.WriteImage(img, target)
        return

    raise ValueError(f"Unsupported model input extension: {ending}")


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(prediction, ground_truth):
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(ground_truth, dtype=bool)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    def ratio(num, den):
        return num / den if den else np.nan

    both_empty = (tp + fp + fn) == 0

    return {
        # Correctly predicting "nothing" counts as a perfect overlap
        # instead of NaN (which pandas would silently drop from the mean).
        "dice": 1.0 if both_empty else 2 * tp / (2 * tp + fp + fn),
        "iou": 1.0 if both_empty else tp / (tp + fp + fn),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "gt_positive_pixels": int(gt.sum()),
        "pred_positive_pixels": int(pred.sum()),
        "empty_case": both_empty,
    }


# ============================================================
# Output utilities
# ============================================================

def save_nifti(array, path, reference=None, spacing=None):
    """Save a 2D array; copy geometry from a compatible reference image."""
    array = np.asarray(array)

    if reference is not None:
        size = reference.GetSize()
        if reference.GetDimension() == 2 and array.shape[::-1] == size:
            img = sitk.GetImageFromArray(array)
            img.CopyInformation(reference)
            sitk.WriteImage(img, str(path))
            return
        if (reference.GetDimension() == 3 and size[2] == 1
                and array.shape[::-1] == size[:2]):
            img = sitk.GetImageFromArray(array[np.newaxis])
            img.CopyInformation(reference)
            sitk.WriteImage(img, str(path))
            return

    img = sitk.GetImageFromArray(array)
    if spacing is not None:
        img.SetSpacing(tuple(float(s) for s in spacing))
    sitk.WriteImage(img, str(path))


def save_overlay(image, prediction, ground_truth, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(normalize_uint8(image), cmap="gray")

    if ground_truth.any():
        ax.contour(ground_truth.astype(np.uint8), levels=[0.5],
                   colors="lime", linewidths=1.5)
    if prediction.any():
        ax.contour(prediction.astype(np.uint8), levels=[0.5],
                   colors="red", linewidths=1.5)

    ax.legend(
        handles=[Line2D([0], [0], color="lime", label="Ground truth"),
                 Line2D([0], [0], color="red", label="Prediction")],
        loc="lower right", fontsize=8,
    )
    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def unique_output_stem(relative_path, used):
    parts = list(relative_path.parent.parts) + [image_stem(relative_path)]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", "__".join(parts)) or "image"

    candidate, n = stem, 2
    while candidate.casefold() in used:
        candidate = f"{stem}_{n}"
        n += 1

    used.add(candidate.casefold())
    return candidate


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch nnU-Net v2 inference with evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--image", required=True, help="Original image folder.")
    parser.add_argument("--mask", required=True, help="Ground-truth mask folder.")
    parser.add_argument("--model", required=True,
                        help="nnU-Net trainer folder (contains dataset.json).")
    parser.add_argument("--output", required=True, help="Output folder.")

    parser.add_argument("--fold", nargs="+", default=["0"],
                        help="Fold number(s), 'all' (fold_all) or 'ensemble'.")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")

    parser.add_argument("--gt-label", type=int, default=1,
                        help="Foreground label value in the ground-truth masks.")
    parser.add_argument("--pred-label", type=int, default=None,
                        help="Foreground label in the model output "
                             "(default: same as --gt-label).")

    parser.add_argument("--intensity", choices=["minmax", "raw"],
                        default="minmax",
                        help="minmax: rescale each image to 0-255 (old "
                             "behaviour). raw: pass intensities unchanged. "
                             "Must match how the training data was made.")
    parser.add_argument("--spacing", type=float, nargs=2, default=None,
                        metavar=("SX", "SY"),
                        help="Pixel spacing written to NIfTI model inputs. "
                             "Default: source spacing for NIfTI images, "
                             "1.0 1.0 otherwise.")

    parser.add_argument("--no-crop", action="store_true",
                        help="Disable ultrasound cropping for DICOM.")
    parser.add_argument("--metrics-region", choices=["full", "crop"],
                        default="full",
                        help="Compute metrics on the full image or only "
                             "inside the crop box (affects specificity).")

    parser.add_argument("--no-mirroring", action="store_true",
                        help="Disable test-time mirroring (faster).")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    pred_label = args.pred_label if args.pred_label is not None else args.gt_label

    image_root = Path(args.image).resolve()
    mask_root = Path(args.mask).resolve()
    model_path = Path(args.model).resolve()
    output_root = Path(args.output).resolve()

    for name, path in [("Image folder", image_root),
                       ("Mask folder", mask_root),
                       ("Model folder", model_path)]:
        if not path.is_dir():
            raise NotADirectoryError(f"{name} does not exist: {path}")

    # Fail fast on model problems before touching any data.
    config, ending = get_model_config(model_path)
    folds = resolve_folds(model_path, args.fold, args.checkpoint)
    configuration, train_spacing = get_training_spacing(model_path)

    prediction_dir = output_root / "predictions"
    overlay_dir = output_root / "overlays"
    image_output_dir = output_root / "images"
    for folder in (prediction_dir, overlay_dir, image_output_dir):
        folder.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("BATCH nnU-Net INFERENCE")
    print("=" * 70)
    print(f"Images:        {image_root}")
    print(f"Masks:         {mask_root}")
    print(f"Model:         {model_path}")
    print(f"Output:        {output_root}")
    print(f"Input type:    {ending}")
    print(f"Folds:         {folds}  ({args.checkpoint})")
    print(f"Intensity:     {args.intensity}")
    if train_spacing is not None:
        print(f"Train spacing: {train_spacing}  ({configuration})")
    print(f"Labels:        GT={args.gt_label}, prediction={pred_label}")
    print(f"Device:        {device}")
    print("=" * 70)

    # Don't treat masks or previous outputs inside the image folder as images.
    image_files = list_supported_files(
        image_root, exclude=[mask_root, output_root]
    )
    mask_files = list_supported_files(mask_root, exclude=[output_root])
    print(f"Found {len(image_files)} images, {len(mask_files)} masks")

    if not image_files:
        raise RuntimeError("No supported images found.")

    exact_index, stripped_index = build_mask_indices(mask_files)
    mask_formats = dict(mask_files)

    # One log entry per image, updated as it moves through the pipeline.
    log = {}

    def set_status(image_path, status, message="", mask_path=None):
        entry = log.setdefault(str(image_path), {
            "image": str(image_path), "mask": "", "status": "", "message": ""
        })
        if mask_path is not None:
            entry["mask"] = str(mask_path)
        entry["status"] = status
        entry["message"] = message

    cases = []
    for image_path, image_fmt in image_files:
        try:
            mask_path = match_mask(image_path, image_root, mask_root,
                                   exact_index, stripped_index)
        except Exception as exc:
            set_status(image_path, "ERROR", str(exc))
            print(f"[ERROR] {image_path.name}: {exc}")
            continue

        if mask_path is None:
            set_status(image_path, "SKIPPED", "No matching mask")
            print(f"[SKIP]  No mask for {image_path.name}")
            continue

        cases.append((image_path, image_fmt, mask_path))

    print(f"Matched cases: {len(cases)}")
    if not cases:
        raise RuntimeError("No matching image-mask pairs were found.")

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    results = []
    used_stems = set()

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        nnunet_input = temp / "input"
        nnunet_output = temp / "prediction"
        nnunet_input.mkdir()
        nnunet_output.mkdir()

        case_mapping = {}

        # ----------------------------------------------------
        # Prepare inputs
        # ----------------------------------------------------
        for i, (image_path, image_fmt, mask_path) in enumerate(cases):
            case_name = f"case_{i:05d}"

            try:
                image = load_image(image_path, image_fmt)
                mask, mask_sitk = load_mask(mask_path, mask_formats[mask_path])

                if image["array"].shape != mask.shape:
                    raise ValueError(
                        f"Image shape {image['array'].shape} != "
                        f"mask shape {mask.shape}"
                    )
                check_geometry(image["sitk"], mask_sitk)

                if args.gt_label not in mask:
                    print(f"[INFO]  {image_path.name}: mask has no label "
                          f"{args.gt_label} (values: {np.unique(mask)[:10]})")

                height, width = image["array"].shape
                bbox, crop_method = (0, 0, width, height), "none"

                if image_fmt == "dicom" and not args.no_crop:
                    detected, method = get_crop_bbox(image["ds"], image["array"])
                    if detected is not None:
                        bbox, crop_method = detected, method
                    else:
                        print(f"[INFO]  {image_path.name}: no ultrasound "
                              "region found; using full image.")

                x0, y0, x1, y1 = bbox

                spacing = (args.spacing or image["spacing"] or (1.0, 1.0))

                write_model_input(
                    image["array"][y0:y1, x0:x1],
                    nnunet_input / f"{case_name}_0000",
                    ending, args.intensity, spacing,
                )

                case_mapping[case_name] = {
                    "image_path": image_path,
                    "image_fmt": image_fmt,
                    "mask_path": mask_path,
                    "mask_fmt": mask_formats[mask_path],
                    "image": image["array"],
                    "image_sitk": image["sitk"],
                    "spacing": spacing,
                    "mask": mask,
                    "bbox": bbox,
                    "crop_method": crop_method,
                }

                set_status(image_path, "PREPARED", mask_path=mask_path)
                print(f"[PREP]  {image_path.name} ({image_fmt}, "
                      f"crop={crop_method} {bbox})")

            except Exception as exc:
                set_status(image_path, "ERROR", f"Preparation: {exc}",
                           mask_path=mask_path)
                print(f"[ERROR] Preparing {image_path.name}: {exc}")
                if args.verbose:
                    traceback.print_exc()

        if not case_mapping:
            raise RuntimeError("No images could be prepared for inference.")

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=not args.no_mirroring,
            perform_everything_on_device=(device.type == "cuda"),
            device=device,
            verbose=args.verbose,
            verbose_preprocessing=args.verbose,
            allow_tqdm=True,
        )

        predictor.initialize_from_trained_model_folder(
            str(model_path),
            use_folds=folds,
            checkpoint_name=args.checkpoint,
        )

        print(f"\nRunning inference on {len(case_mapping)} cases...")

        predictor.predict_from_files(
            str(nnunet_input),
            str(nnunet_output),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
        )

        prediction_index = {
            p.name[:-len(ending)]: p
            for p in nnunet_output.iterdir()
            if p.name.lower().endswith(ending)
        }

        # ----------------------------------------------------
        # Restore, evaluate, save
        # ----------------------------------------------------
        for case_name, info in case_mapping.items():
            image_path = info["image_path"]

            try:
                if case_name not in prediction_index:
                    raise FileNotFoundError(f"No prediction for {case_name}")

                pred_crop = read_prediction(prediction_index[case_name], ending)

                x0, y0, x1, y1 = info["bbox"]
                expected = (y1 - y0, x1 - x0)
                if pred_crop.shape != expected:
                    raise ValueError(f"Prediction shape {pred_crop.shape} != "
                                     f"crop shape {expected}")

                prediction = np.zeros(info["image"].shape, dtype=np.uint8)
                prediction[y0:y1, x0:x1] = pred_crop

                pred_binary = prediction == pred_label
                gt_binary = info["mask"] == args.gt_label

                gt_outside_crop = int(gt_binary.sum()
                                      - gt_binary[y0:y1, x0:x1].sum())

                if args.metrics_region == "crop":
                    metrics = calculate_metrics(pred_binary[y0:y1, x0:x1],
                                                gt_binary[y0:y1, x0:x1])
                else:
                    metrics = calculate_metrics(pred_binary, gt_binary)

                stem = unique_output_stem(
                    image_path.relative_to(image_root), used_stems
                )
                prediction_path = prediction_dir / f"{stem}.nii.gz"
                overlay_path = overlay_dir / f"{stem}.png"
                image_nifti_path = image_output_dir / f"{stem}.nii.gz"

                save_nifti(prediction, prediction_path,
                           info["image_sitk"], info["spacing"])

                if info["image_sitk"] is not None:
                    sitk.WriteImage(info["image_sitk"], str(image_nifti_path))
                else:
                    save_nifti(info["image"].astype(np.float32),
                               image_nifti_path, spacing=info["spacing"])

                if not args.no_overlays:
                    save_overlay(
                        info["image"], pred_binary, gt_binary, overlay_path,
                        f"{image_path.name}  Dice={metrics['dice']:.3f}",
                    )

                metrics.update({
                    "case": case_name,
                    "image_name": image_path.name,
                    "image": str(image_path),
                    "mask": str(info["mask_path"]),
                    "image_format": info["image_fmt"],
                    "mask_format": info["mask_fmt"],
                    "prediction": str(prediction_path),
                    "overlay": "" if args.no_overlays else str(overlay_path),
                    "crop_method": info["crop_method"],
                    "crop_x0": x0, "crop_y0": y0,
                    "crop_x1": x1, "crop_y1": y1,
                    "gt_pixels_outside_crop": gt_outside_crop,
                })
                results.append(metrics)

                set_status(image_path, "SUCCESS")

                warn = (f"  [GT outside crop: {gt_outside_crop}px]"
                        if gt_outside_crop else "")
                print(f"[DONE]  {image_path.name}: Dice={metrics['dice']:.4f}, "
                      f"IoU={metrics['iou']:.4f}{warn}")

            except Exception as exc:
                set_status(image_path, "ERROR", f"Evaluation: {exc}")
                print(f"[ERROR] Evaluating {image_path.name}: {exc}")
                if args.verbose:
                    traceback.print_exc()

    # ========================================================
    # Export
    # ========================================================
    log_df = pd.DataFrame(list(log.values()))
    log_path = output_root / "processing_log.csv"
    log_df.to_csv(log_path, index=False)

    status_counts = log_df["status"].value_counts().to_dict()

    if not results:
        print("\nNo valid predictions were evaluated.")
        print(f"See processing log: {log_path}")
        return

    results_df = pd.DataFrame(results)
    leading = ["case", "image_name"] + METRIC_COLUMNS
    results_df = results_df[leading + [c for c in results_df.columns
                                       if c not in leading]]

    statistics_path = output_root / "prediction_statistics.csv"
    results_df.to_csv(statistics_path, index=False)

    mean_df = results_df[METRIC_COLUMNS].mean().to_frame().T
    mean_df["n_cases"] = len(results_df)
    mean_df["n_empty_cases"] = int(results_df["empty_case"].sum())
    mean_path = output_root / "mean_statistics.csv"
    mean_df.to_csv(mean_path, index=False)

    summary_df = results_df[METRIC_COLUMNS].agg(
        ["mean", "std", "median", "min", "max", "count"]
    )
    summary_path = output_root / "summary_statistics.csv"
    summary_df.to_csv(summary_path)

    print("\n" + "=" * 70)
    print("BATCH INFERENCE FINISHED")
    print("=" * 70)
    print(f"Evaluated: {status_counts.get('SUCCESS', 0)}")
    print(f"Skipped:   {status_counts.get('SKIPPED', 0)}")
    print(f"Errors:    {status_counts.get('ERROR', 0)}")
    print(f"Empty GT + empty prediction (Dice=1): "
          f"{int(results_df['empty_case'].sum())}")

    print("\nSummary:")
    print(summary_df.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nOutput files:")
    print(f"Predictions: {prediction_dir}")
    print(f"Overlays:    {overlay_dir}")
    print(f"Images:      {image_output_dir}")
    print(f"Per case:    {statistics_path}")
    print(f"Mean:        {mean_path}")
    print(f"Summary:     {summary_path}")
    print(f"Log:         {log_path}")


if __name__ == "__main__":
    main()
