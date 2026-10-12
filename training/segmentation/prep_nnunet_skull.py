#!/usr/bin/env python3
"""
build_merged_dataset.py

Merge two datasets into one nnU-Net v2 raw dataset:

  1. An existing nnU-Net raw dataset (nnUNet_raw/DatasetXXX_name), which is
     just copied over as-is: its imagesTr/labelsTr and dataset.json define
     the file format (file_ending), channel names and label scheme that the
     merged dataset must follow.

  2. A second dataset laid out as:

         dcm_root/<patient_id>/...      (DICOM files, any sub-layout)
         seg_root/<patient_id>/...      (NIfTI segmentation files)

     For each patient, every DICOM is matched to a segmentation file by
     SeriesDescription (same matching rule as nnunet_inference.py:
     <SeriesDescription>.nii.gz / .nii, searched recursively under that
     patient's segmentation folder). Some segmentations contain label 1
     and label 2; only label 1 is kept as foreground, label 2 (and
     anything else) is treated as background.

Each DICOM is converted to grayscale (matching the include/exclude
classifier's own conversion: cv2.normalize for single-channel images,
cv2.cvtColor(..., COLOR_RGB2GRAY) / COLOR_RGBA2GRAY for color images), then
cropped to the ultrasound region: the first 2D tissue region from
SequenceOfUltrasoundRegions, or a contour-based fallback if that tag is
missing. The matching segmentation is cropped with the exact same box, so
image and label stay pixel-aligned. Images are converted to the file format
used by the existing dataset (--file_ending is read from its dataset.json,
so this always matches). Labels are always written with the same file_ending
as the images.

IMPORTANT: the existing dataset's own images (copied as-is, see below) were
NOT produced with this cropping (they still contain the full DICOM frame,
including background outside the US cone). Mixing full-frame cases with
newly cropped cases in one training set means the model sees two different
input distributions. If you want a fully consistent merged dataset, the
existing dataset would need to be rebuilt from its original DICOMs with the
same crop, rather than copied from its current imagesTr/labelsTr.

Usage:
    python build_merged_dataset.py \\
        --existing_dataset /data/.../nnUNet_raw/Dataset003_submission \\
        --dcm_root '/data/.../data (US only)_unique_ids' \\
        --seg_root '/data/.../segmentations' \\
        --output_dataset /data/.../nnUNet_raw/Dataset010_merged \\
        --dataset_name Dataset010_merged

Then run the usual nnU-Net planning/preprocessing on --output_dataset.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from PIL import Image


# ============================================================
# Existing dataset info
# ============================================================

def read_dataset_json(dataset_path):
    path = dataset_path / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset.json not found in {dataset_path}")
    with open(path) as f:
        return json.load(f)


def find_existing_cases(dataset_path, file_ending):
    """
    Return {case_name: {"image": path, "label": path}} for an existing
    nnU-Net raw dataset. Assumes a single input channel (_0000).
    """
    images_tr = dataset_path / "imagesTr"
    labels_tr = dataset_path / "labelsTr"

    cases = {}
    suffix = f"_0000{file_ending}"

    for img_path in sorted(images_tr.glob(f"*{suffix}")):
        case_name = img_path.name[: -len(suffix)]
        label_path = labels_tr / f"{case_name}{file_ending}"

        if not label_path.exists():
            print(f"[WARNING] No label for existing case {case_name}; skipping.")
            continue

        cases[case_name] = {"image": img_path, "label": label_path}

    return cases


def detect_case_naming(case_names):
    """
    Look at existing case names (e.g. 'case_00188') and figure out the
    prefix, the zero-padding width, and the next free index, so new cases
    can continue the same numbering instead of getting their own scheme.

    Falls back to prefix "case_", width 5, starting at 0, if no case name
    ends in digits.
    """
    pattern = re.compile(r"^(.*?)(\d+)$")

    best = None  # (prefix, width, max_index)

    for name in case_names:
        match = pattern.match(name)
        if not match:
            continue

        prefix, digits = match.group(1), match.group(2)
        index = int(digits)

        if best is None or prefix == best[0]:
            if best is None:
                best = (prefix, len(digits), index)
            else:
                best = (prefix, max(best[1], len(digits)), max(best[2], index))

    if best is None:
        return "case_", 5, 0

    prefix, width, max_index = best
    return prefix, width, max_index + 1


# ============================================================
# DICOM / segmentation matching (new dataset)
# ============================================================

def find_dicom_files(root):
    files = sorted(root.rglob("*.dcm"))
    if files:
        return files

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
            files.append(p)
        except Exception:
            pass

    return sorted(files)


def find_seg_file(patient_seg_dir, series_description):
    name = str(series_description).strip()

    for ext in (".nii.gz", ".nii"):
        direct = patient_seg_dir / f"{name}{ext}"
        if direct.exists():
            return direct

    for ext in (".nii.gz", ".nii"):
        matches = list(patient_seg_dir.rglob(f"{name}{ext}"))
        if matches:
            return matches[0]

    return None


def dicom_to_gray_uint8(ds):
    """
    Convert a DICOM's pixel array to a uint8 grayscale image, using the
    same conversion as the include/exclude classifier:

      (H, W)      -> min-max normalized to 0-255
      (H, W, 1)   -> squeezed, then as above
      (H, W, 3)   -> cv2.cvtColor(..., COLOR_RGB2GRAY)
      (H, W, 4)   -> cv2.cvtColor(..., COLOR_RGBA2GRAY)
    """
    arr = ds.pixel_array

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    if arr.ndim == 2:
        gray = cv2.normalize(arr.astype(np.float32), None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    else:
        raise ValueError(f"Unexpected pixel array shape {arr.shape}")

    return gray


# ============================================================
# Ultrasound region cropping (matches predict_incl_excl.py)
# ============================================================

def bbox_from_regions(ds, shape):
    """
    Bounding box (x0, y0, x1, y1) of the first 2D tissue region
    (RegionSpatialFormat == 1) in SequenceOfUltrasoundRegions, or None.
    """
    regions = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if not regions:
        return None

    h, w = shape

    for r in regions:
        if getattr(r, "RegionSpatialFormat", None) != 1:
            continue

        x0 = getattr(r, "RegionLocationMinX0", None)
        y0 = getattr(r, "RegionLocationMinY0", None)
        x1 = getattr(r, "RegionLocationMaxX1", None)
        y1 = getattr(r, "RegionLocationMaxY1", None)

        if None in (x0, y0, x1, y1):
            continue

        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(x1), w), min(int(y1), h)

        if x1 > x0 and y1 > y0:
            return (x0, y0, x1, y1)

    return None


def bbox_from_contour(gray_u8):
    """Fallback: bounding box of the single contour with area > 10000 px."""
    _, thresh = cv2.threshold(gray_u8, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 10000:
            boxes.append((x, y, x + w, y + h))

    return boxes[0] if len(boxes) == 1 else None


def get_crop_bbox(ds, gray_u8):
    """Region-based crop first, contour-based fallback. None if both fail."""
    return bbox_from_regions(ds, gray_u8.shape) or bbox_from_contour(gray_u8)


def load_label1(seg_path):
    """Load a NIfTI segmentation; only label 1 is kept as foreground."""
    arr = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path))))
    return (arr == 1).astype(np.uint8)


# ============================================================
# Writers (format follows the existing dataset's file_ending)
# ============================================================

def write_image(array_uint8, out_path, file_ending):
    if file_ending == ".png":
        Image.fromarray(array_uint8).save(out_path)
    else:
        img = sitk.GetImageFromArray(array_uint8[None].astype(np.float32))
        img.SetSpacing((1.0, 1.0, 999.0))
        sitk.WriteImage(img, str(out_path))


def write_label(array_uint8, out_path, file_ending):
    if file_ending == ".png":
        Image.fromarray(array_uint8).save(out_path)
    else:
        img = sitk.GetImageFromArray(array_uint8[None])
        img.SetSpacing((1.0, 1.0, 999.0))
        sitk.WriteImage(img, str(out_path))


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge an existing nnU-Net dataset with a new DICOM+segmentation "
                    "dataset, keeping only label 1."
    )
    parser.add_argument("--existing_dataset", required=True,
                        help="Path to the existing nnUNet_raw/DatasetXXX_name folder.")
    parser.add_argument("--dcm_root", required=True,
                        help="Root folder: dcm_root/<patient_id>/... (DICOMs).")
    parser.add_argument("--seg_root", required=True,
                        help="Root folder: seg_root/<patient_id>/... (segmentation NIfTIs).")
    parser.add_argument("--output_dataset", required=True,
                        help="Path to the new merged nnUNet_raw/DatasetYYY_name folder.")
    parser.add_argument("--dataset_name", required=True,
                        help="Name written into the merged dataset.json, e.g. Dataset010_merged.")
    parser.add_argument("--no_crop", action="store_true",
                        help="Disable cropping to the ultrasound region.")
    parser.add_argument("--crop_failure", choices=["skip", "full"], default="skip",
                        help="If no ultrasound region can be found: skip the "
                             "image or use the full image. Default: skip.")
    parser.add_argument("--foreground_label_name", default=None,
                        help="Override the foreground label name in dataset.json. "
                             "Default: taken from the existing dataset.json.")
    parser.add_argument("--test_fraction", type=float, default=0.2,
                        help="Fraction of the NEW (converted) cases to hold out into "
                             "imagesTs/labelsTs instead of imagesTr/labelsTr. Cases "
                             "copied from --existing_dataset always stay in the "
                             "training set. Default: 0.2. Set to 0 to disable.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the train/test split. Default: 42.")
    parser.add_argument("--no_test_labels", action="store_true",
                        help="Do not write labelsTs for the held-out test cases "
                             "(by default their GT is written too, for later evaluation).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print what would be done; do not write any files.")
    args = parser.parse_args()

    if not 0.0 <= args.test_fraction < 1.0:
        raise ValueError("--test_fraction must be in [0, 1).")

    existing_dataset = Path(args.existing_dataset)
    dcm_root = Path(args.dcm_root)
    seg_root = Path(args.seg_root)
    output_dataset = Path(args.output_dataset)

    for label, p in [("Existing dataset", existing_dataset),
                      ("DCM root", dcm_root),
                      ("Segmentation root", seg_root)]:
        if not p.exists():
            raise FileNotFoundError(f"{label} does not exist:\n{p}")

    # --------------------------------------------------------
    # Read existing dataset
    # --------------------------------------------------------
    existing_json = read_dataset_json(existing_dataset)
    file_ending = existing_json["file_ending"]
    channel_names = existing_json.get("channel_names", existing_json.get("modality"))
    labels = dict(existing_json["labels"])

    if len(channel_names) != 1:
        raise NotImplementedError(
            f"Existing dataset has {len(channel_names)} input channels; "
            f"this script only supports single-channel (grayscale) datasets."
        )

    # Foreground label value/name: everything except "background"
    fg_names = [k for k in labels if k != "background"]
    if len(fg_names) != 1:
        raise NotImplementedError(
            f"Expected exactly one foreground label in the existing dataset.json, "
            f"found: {fg_names}. This script assumes a binary segmentation task."
        )
    fg_name = args.foreground_label_name or fg_names[0]
    fg_value = labels[fg_names[0]]

    print("=" * 70)
    print("Existing dataset")
    print("=" * 70)
    print(f"Path          : {existing_dataset}")
    print(f"File ending   : {file_ending}")
    print(f"Channel names : {channel_names}")
    print(f"Labels        : {labels}")

    existing_cases = find_existing_cases(existing_dataset, file_ending)
    print(f"Existing cases: {len(existing_cases)}")

    # --------------------------------------------------------
    # Match new DICOMs to segmentations, patient by patient
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("New dataset (DICOM + segmentation)")
    print("=" * 70)

    patient_dirs = sorted(p for p in dcm_root.iterdir() if p.is_dir())
    print(f"Found {len(patient_dirs)} patient folders under {dcm_root}")

    new_cases = []  # list of dicts: case_name, dicom, seg, series_description, patient_id
    n_no_seg = 0
    n_read_error = 0
    n_no_label1 = 0

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        seg_patient_dir = seg_root / patient_id

        if not seg_patient_dir.exists():
            print(f"[WARNING] No segmentation folder for patient {patient_id}; skipping patient.")
            continue

        dicom_files = find_dicom_files(patient_dir)

        for dicom_path in dicom_files:
            try:
                ds = pydicom.dcmread(str(dicom_path))
                series_description = str(getattr(ds, "SeriesDescription", "")).strip()
            except Exception as e:
                print(f"[WARNING] Could not read {dicom_path}: {e}")
                n_read_error += 1
                continue

            if not series_description:
                print(f"[WARNING] No SeriesDescription in {dicom_path}")
                continue

            seg_path = find_seg_file(seg_patient_dir, series_description)

            if seg_path is None:
                n_no_seg += 1
                continue

            new_cases.append({
                "dicom": dicom_path,
                "seg": seg_path,
                "series_description": series_description,
                "patient_id": patient_id,
                "ds": ds,
            })

    print(f"\nMatched (image+seg) : {len(new_cases)}")
    print(f"Skipped (no seg)    : {n_no_seg}")
    print(f"Skipped (read error): {n_read_error}")

    if not existing_cases and not new_cases:
        print("\nNothing to write, exiting.")
        return

    # ----- convert and write new cases -----
    kept_cases = []

    for case in new_cases:
        seg_arr = load_label1(case["seg"])

        if not seg_arr.any():
            n_no_label1 += 1
            continue

        case["seg_arr"] = seg_arr
        kept_cases.append(case)

    print(f"Skipped (no label 1): {n_no_label1}")
    new_cases = kept_cases

    prefix, width, next_index = detect_case_naming(existing_cases.keys())
    print(f"Continuing numbering as {prefix}{'0' * width}... "
          f"(next free index: {next_index})")

    for case in new_cases:
        case["case_name"] = f"{prefix}{next_index:0{width}d}"
        next_index += 1

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(new_cases))
    n_test = int(round(len(new_cases) * args.test_fraction))
    test_indices = set(order[:n_test].tolist())

    for idx, case in enumerate(new_cases):
        case["split"] = "test" if idx in test_indices else "train"

    print(f"New cases split: {len(new_cases) - n_test} train / {n_test} test "
          f"(test_fraction={args.test_fraction}, seed={args.seed})")


    # --------------------------------------------------------
    # Report label composition of new segmentations
    # --------------------------------------------------------
    log_rows = []

    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN — no files will be written")
    else:
        print("Writing merged dataset")
    print("=" * 70)
    print(f"Output: {output_dataset}")

    images_out = output_dataset / "imagesTr"
    labels_out = output_dataset / "labelsTr"
    images_ts_out = output_dataset / "imagesTs"
    labels_ts_out = output_dataset / "labelsTs"

    if not args.dry_run:
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        if n_test > 0:
            images_ts_out.mkdir(parents=True, exist_ok=True)
            if not args.no_test_labels:
                labels_ts_out.mkdir(parents=True, exist_ok=True)

    # ----- copy existing cases as-is -----
    for case_name, info in existing_cases.items():
        log_rows.append({"case_name": case_name, "source": "existing", "split": "train",
                         "patient_id": "", "series_description": "",
                         "dicom": "", "seg": str(info["label"])})

        if args.dry_run:
            continue

        shutil.copy2(info["image"], images_out / f"{case_name}_0000{file_ending}")
        shutil.copy2(info["label"], labels_out / f"{case_name}{file_ending}")

    for case in new_cases:
        case_name = case["case_name"]
        seg_arr = case["seg_arr"]
        seg_binary = (seg_arr == 1).astype(np.uint8) * fg_value

        log_rows.append({
            "case_name": case_name, "source": "new",
            "patient_id": case["patient_id"],
            "series_description": case["series_description"],
            "dicom": str(case["dicom"]), "seg": str(case["seg"]),
            "gt_positive_pixels": int((seg_arr == 1).sum()),
        })

        if args.dry_run:
            continue

        try:
            gray_u8 = dicom_to_gray_uint8(case["ds"])
        except Exception as e:
            print(f"[WARNING] Could not load {case['dicom']}: {e}; skipping.")
            continue

        if gray_u8.shape != seg_binary.shape:
            print(f"[WARNING] Shape mismatch for {case_name}: "
                  f"image={gray_u8.shape}, seg={seg_binary.shape}; skipping.")
            continue

        if args.no_crop:
            bbox = (0, 0, gray_u8.shape[1], gray_u8.shape[0])
        else:
            bbox = get_crop_bbox(case["ds"], gray_u8)

            if bbox is None:
                if args.crop_failure == "full":
                    print(f"[WARNING] No ultrasound region found for {case_name}; "
                          f"using full image.")
                    bbox = (0, 0, gray_u8.shape[1], gray_u8.shape[0])
                else:
                    print(f"[WARNING] No ultrasound region found for {case_name}; "
                          f"skipping.")
                    continue

        x0, y0, x1, y1 = bbox
        image_crop = gray_u8[y0:y1, x0:x1]
        label_crop = seg_binary[y0:y1, x0:x1]

        if case["split"] == "test":
            write_image(image_crop, images_ts_out / f"{case_name}_0000{file_ending}", file_ending)
            if not args.no_test_labels:
                write_label(label_crop, labels_ts_out / f"{case_name}{file_ending}", file_ending)
        else:
            write_image(image_crop, images_out / f"{case_name}_0000{file_ending}", file_ending)
            write_label(label_crop, labels_out / f"{case_name}{file_ending}", file_ending)

        log_rows[-1]["split"] = case["split"]
        log_rows[-1]["crop_x0"] = x0
        log_rows[-1]["crop_y0"] = y0
        log_rows[-1]["crop_x1"] = x1
        log_rows[-1]["crop_y1"] = y1

        print(f"  {case_name}: {case['series_description']} "
              f"(crop x={x0}:{x1}, y={y0}:{y1}, "
              f"fg px={int((label_crop == fg_value).sum())}, split={case['split']})")

    n_new_train = len([r for r in log_rows if r["source"] == "new" and r["split"] == "train"])
    n_new_test = len([r for r in log_rows if r["source"] == "new" and r["split"] == "test"])
    total_train_cases = len(existing_cases) + n_new_train

    # --------------------------------------------------------
    # dataset.json
    # --------------------------------------------------------
    merged_json = dict(existing_json)
    merged_json["name"] = args.dataset_name
    merged_json["numTraining"] = total_train_cases
    if n_new_test > 0:
        merged_json["numTest"] = n_new_test
    merged_json["channel_names"] = channel_names
    merged_json["labels"] = {"background": 0, fg_name: fg_value}
    merged_json.pop("modality", None)

    if not args.dry_run:
        with open(output_dataset / "dataset.json", "w") as f:
            json.dump(merged_json, f, indent=2)

    # --------------------------------------------------------
    # Log / summary
    # --------------------------------------------------------
    log_df = pd.DataFrame(log_rows)

    if not args.dry_run:
        log_file = output_dataset / "merge_log.csv"
        log_df.to_csv(log_file, index=False)
    else:
        log_file = "(not written, dry run)"

    print("\n" + "=" * 70)
    print("DONE" if not args.dry_run else "DRY RUN FINISHED")
    print("=" * 70)
    print(f"Existing cases copied      : {len(existing_cases)}")
    print(f"New cases converted (train): {n_new_train}")
    print(f"New cases converted (test) : {n_new_test}")
    print(f"Total training cases       : {total_train_cases}")
    print(f"Labels                : {merged_json['labels']}")
    print(f"Log                   : {log_file}")

    if not args.dry_run:
        print(f"\nNext steps:")
        print(f"  nnUNetv2_plan_and_preprocess -d <DATASET_ID> --verify_dataset_integrity")


if __name__ == "__main__":
    main()