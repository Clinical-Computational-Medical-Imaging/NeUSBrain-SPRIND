#!/usr/bin/env python3
"""
prepare_midsagittal_dataset.py

Build an nnU-Net v2 raw dataset (binary segmentation of label 2) from:

  - A CSV file with columns "image_path" and "user_label". Only rows whose
    user_label matches --label_filter (default: "mid_saggital") are used.
    image_path points at one DICOM instance, e.g.:

        /data/.../data (US only)_unique_ids/40/1000374C/1000374D/1000374E/100037A3

    The patient id is the first path component under --dcm_root (here: "40").

  - A DICOM SeriesDescription of the form "<patient_id>_<unique5digitid>",
    which is used to find the matching segmentation:

        <seg_root>/<patient_id>/<SeriesDescription>.nii[.gz]

  Only images whose segmentation actually contains label 2 are kept
  (label 2 -> foreground; everything else, including label 1, -> background).

Each kept DICOM is converted to NIfTI. Rather than re-deriving orientation
from DICOM tags (error-prone: ImageOrientationPatient / ImagePositionPatient
are not always reliable or present for ultrasound), the image is written
onto the *exact grid of its own ground-truth NIfTI* (size, spacing, origin,
direction) via CopyInformation. That guarantees image and mask line up
pixel-for-pixel. As a sanity check, the DICOM's own PixelSpacing (or
ultrasound-region spacing) is compared against the GT's spacing and a
warning is printed if they disagree by more than 1%.

RGB/RGBA DICOMs are collapsed to a single grayscale channel (standard
luminance weights, kept in float — no artificial 0-255 rescale, since
nnU-Net normalizes intensities itself during preprocessing).

Usage:
    python prepare_midsagittal_dataset.py \\
        --dcm_root /path/to/dicom_root \\
        --seg_root /path/to/segmentations \\
        --csv_file /path/to/labels.csv \\
        --output_dataset /data/.../nnUNet_raw/Dataset011_midsagittal \\
        --dataset_name Dataset011_midsagittal \\
        --dry_run
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk


# ============================================================
# Case naming (continue an existing dataset's numbering, if given)
# ============================================================

def find_existing_case_names(dataset_path):
    if dataset_path is None:
        return []

    images_tr = Path(dataset_path) / "imagesTr"
    if not images_tr.exists():
        return []

    names = []
    for p in images_tr.iterdir():
        name = p.name
        for suffix in ("_0000.nii.gz", "_0000.nii", "_0000.png"):
            if name.endswith(suffix):
                names.append(name[: -len(suffix)])
                break

    return names


def detect_case_naming(case_names):
    """
    Figure out prefix / zero-padding width / next free index from existing
    case names (e.g. 'case_00188' -> next index 189). Falls back to
    prefix "case_", width 5, starting at 0.
    """
    pattern = re.compile(r"^(.*?)(\d+)$")
    best = None

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
# DICOM helpers
# ============================================================

def grayscale_channel_collapse(ds):
    """
    Collapse a DICOM's pixel array to single-channel float32, preserving
    the original intensity scale (no 0-255 rescale — nnU-Net normalizes
    itself). RGB/RGBA is collapsed with standard luminance weights.
    """
    arr = ds.pixel_array
    samples = int(getattr(ds, "SamplesPerPixel", 1))

    if arr.ndim == 3 and samples in (3, 4) and arr.shape[-1] == samples:
        arr = arr.astype(np.float32)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        return gray.astype(np.float32)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    raise ValueError(f"Unexpected pixel array shape {arr.shape}")


def get_dicom_in_plane_spacing(ds):
    """
    In-plane spacing (spacing_x, spacing_y) in mm from a DICOM dataset,
    matching the sitk (x=columns, y=rows) axis order, or None.
    """
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        value = getattr(ds, tag, None)
        if value is not None and len(value) == 2:
            row_sp, col_sp = float(value[0]), float(value[1])
            if row_sp > 0 and col_sp > 0:
                return (col_sp, row_sp)

    regions = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if regions:
        region = regions[0]
        dx = getattr(region, "PhysicalDeltaX", None)
        dy = getattr(region, "PhysicalDeltaY", None)
        unit_x = getattr(region, "PhysicalUnitsXDirection", None)
        unit_y = getattr(region, "PhysicalUnitsYDirection", None)

        if dx and dy and unit_x == 3 and unit_y == 3:  # 3 == cm
            return (abs(float(dx)) * 10.0, abs(float(dy)) * 10.0)

    return None


def check_spacing(dicom_spacing, gt_img, transposed, name):
    """Warn if DICOM in-plane spacing and GT in-plane spacing disagree."""
    if dicom_spacing is None:
        return

    sizes = gt_img.GetSize()
    spacing = gt_img.GetSpacing()
    in_plane = tuple(sp for sz, sp in zip(sizes, spacing) if sz > 1)

    if len(in_plane) != 2:
        return

    expected = tuple(dicom_spacing[::-1]) if transposed else tuple(dicom_spacing)

    if not np.allclose(expected, in_plane, rtol=0.01):
        print(f"[WARNING] {name}: DICOM spacing {expected} mm differs from "
              f"GT spacing {in_plane} mm. Using GT geometry anyway.")


def array_to_grid(array2d, ref_img):
    """
    Wrap a 2D array (already in ref_img's axis order) as a SimpleITK image
    with ref_img's exact geometry (size, spacing, origin, direction).
    """
    target_shape = ref_img.GetSize()[::-1]  # sitk (x,y,z) -> array (z,y,x)
    array = np.ascontiguousarray(array2d).reshape(target_shape)
    img = sitk.GetImageFromArray(array)
    img.CopyInformation(ref_img)
    return img


def find_seg_file(patient_seg_dir, series_description):
    name = str(series_description).strip()

    for ext in (".nii", ".nii.gz"):
        direct = patient_seg_dir / f"{name}{ext}"
        if direct.exists():
            return direct

    for ext in (".nii", ".nii.gz"):
        matches = list(patient_seg_dir.rglob(f"{name}{ext}"))
        if matches:
            return matches[0]

    return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build a mid-sagittal, label-2 nnU-Net dataset from a CSV-filtered "
                    "DICOM list and matching segmentations."
    )
    parser.add_argument("--dcm_root", required=True,
                        help="Root folder: dcm_root/<patient_id>/... — must be the "
                             "prefix of every image_path in the CSV.")
    parser.add_argument("--seg_root", required=True,
                        help="Root folder: seg_root/<patient_id>/<SeriesDescription>.nii[.gz]")
    parser.add_argument("--csv_file", required=True,
                        help="CSV with columns 'image_path' and 'user_label'.")
    parser.add_argument("--output_dataset", required=True,
                        help="Path to the new nnUNet_raw/DatasetYYY_name folder.")
    parser.add_argument("--dataset_name", required=True,
                        help="Name written into dataset.json, e.g. Dataset011_midsagittal.")
    parser.add_argument("--label_filter", default="mid_saggital",
                        help="user_label value to keep (case-insensitive, trimmed). "
                             "Default: mid_saggital.")
    parser.add_argument("--source_label", type=int, default=2,
                        help="Segmentation label value to treat as foreground. Default: 2.")
    parser.add_argument("--foreground_label_name", default="label2",
                        help="Foreground label name written into dataset.json.")
    parser.add_argument("--existing_dataset", default=None,
                        help="Optional existing nnU-Net dataset whose case numbering "
                             "this dataset should continue.")
    parser.add_argument("--test_fraction", type=float, default=0.2,
                        help="Fraction of the matched cases to hold out into "
                             "imagesTs/labelsTs instead of imagesTr/labelsTr. "
                             "Default: 0.2. Set to 0 to disable.")
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

    dcm_root = Path(args.dcm_root)
    seg_root = Path(args.seg_root)
    csv_file = Path(args.csv_file)
    output_dataset = Path(args.output_dataset)

    for label, p in [("DCM root", dcm_root), ("Segmentation root", seg_root),
                      ("CSV file", csv_file)]:
        if not p.exists():
            raise FileNotFoundError(f"{label} does not exist:\n{p}")

    # --------------------------------------------------------
    # Read and filter the CSV
    # --------------------------------------------------------
    df = pd.read_csv(csv_file)

    missing_cols = {"image_path", "user_label"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required column(s): {missing_cols}")

    target = args.label_filter.strip().lower()
    mask = df["user_label"].astype(str).str.strip().str.lower() == target
    filtered = df[mask]

    print("=" * 70)
    print("CSV filtering")
    print("=" * 70)
    print(f"CSV file        : {csv_file}")
    print(f"Total rows      : {len(df)}")
    print(f"user_label      : {df['user_label'].astype(str).str.strip().value_counts().to_dict()}")
    print(f"Matching '{args.label_filter}': {len(filtered)}")

    if filtered.empty:
        print("\nNo rows match --label_filter, nothing to do.")
        return

    # --------------------------------------------------------
    # Case naming
    # --------------------------------------------------------
    existing_names = find_existing_case_names(args.existing_dataset)
    prefix, width, next_index = detect_case_naming(existing_names)
    print(f"\nCase numbering starts at {prefix}{next_index:0{width}d}")

    # --------------------------------------------------------
    # Phase 1: match, filter by label-2 presence, load into memory
    # --------------------------------------------------------
    kept = []
    n_not_under_root = 0
    n_read_error = 0
    n_no_seg = 0
    n_no_label = 0
    n_shape_mismatch = 0

    for _, row in filtered.iterrows():
        dicom_path = Path(row["image_path"])

        try:
            rel = dicom_path.relative_to(dcm_root)
        except ValueError:
            print(f"[WARNING] {dicom_path} is not under --dcm_root; skipping.")
            n_not_under_root += 1
            continue

        patient_id = rel.parts[0]

        try:
            ds = pydicom.dcmread(str(dicom_path))
            series_description = str(getattr(ds, "SeriesDescription", "")).strip()
        except Exception as e:
            print(f"[WARNING] Could not read {dicom_path}: {e}")
            n_read_error += 1
            continue

        if not series_description:
            print(f"[WARNING] No SeriesDescription in {dicom_path}; skipping.")
            continue

        seg_path = find_seg_file(seg_root / patient_id, series_description)

        if seg_path is None:
            n_no_seg += 1
            continue

        gt_img = sitk.ReadImage(str(seg_path))
        gt_array = np.squeeze(sitk.GetArrayFromImage(gt_img))

        if not (gt_array == args.source_label).any():
            n_no_label += 1
            continue

        try:
            image_array = grayscale_channel_collapse(ds)
        except Exception as e:
            print(f"[WARNING] Could not load {dicom_path}: {e}; skipping.")
            continue

        label_binary = (gt_array == args.source_label).astype(np.uint8)

        transposed = False
        if image_array.shape != label_binary.shape:
            if image_array.T.shape == label_binary.shape:
                image_array = image_array.T
                transposed = True
            else:
                print(f"[WARNING] Shape mismatch for {dicom_path.name} "
                      f"({series_description}): image={image_array.shape}, "
                      f"GT={label_binary.shape}; skipping.")
                n_shape_mismatch += 1
                continue

        dicom_spacing = get_dicom_in_plane_spacing(ds)

        kept.append({
            "dicom": dicom_path,
            "gt": seg_path,
            "patient_id": patient_id,
            "series_description": series_description,
            "image_array": image_array,
            "label_binary": label_binary,
            "gt_img": gt_img,
            "transposed": transposed,
            "dicom_spacing": dicom_spacing,
        })

    print(f"\nNot under --dcm_root : {n_not_under_root}")
    print(f"Read errors          : {n_read_error}")
    print(f"No matching seg      : {n_no_seg}")
    print(f"No label {args.source_label} in seg   : {n_no_label}")
    print(f"Shape mismatch       : {n_shape_mismatch}")
    print(f"Matched + kept       : {len(kept)}")

    if not kept:
        print("\nNothing matched all criteria, exiting.")
        return

    # --------------------------------------------------------
    # Assign case names (continuing --existing_dataset's numbering)
    # --------------------------------------------------------
    for case in kept:
        case["case_name"] = f"{prefix}{next_index:0{width}d}"
        next_index += 1

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(kept))
    n_test = int(round(len(kept) * args.test_fraction))
    test_indices = set(order[:n_test].tolist())

    for idx, case in enumerate(kept):
        case["split"] = "test" if idx in test_indices else "train"

    print(f"\nSplit: {len(kept) - n_test} train / {n_test} test "
          f"(test_fraction={args.test_fraction}, seed={args.seed})")

    # --------------------------------------------------------
    # Phase 2: write
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("DRY RUN — no files will be written" if args.dry_run else "Writing dataset")
    print("=" * 70)

    images_tr = output_dataset / "imagesTr"
    labels_tr = output_dataset / "labelsTr"
    images_ts = output_dataset / "imagesTs"
    labels_ts = output_dataset / "labelsTs"

    if not args.dry_run:
        images_tr.mkdir(parents=True, exist_ok=True)
        labels_tr.mkdir(parents=True, exist_ok=True)
        if n_test > 0:
            images_ts.mkdir(parents=True, exist_ok=True)
            if not args.no_test_labels:
                labels_ts.mkdir(parents=True, exist_ok=True)

    log_rows = []

    for case in kept:
        case_name = case["case_name"]
        gt_img = case["gt_img"]

        check_spacing(case["dicom_spacing"], gt_img, case["transposed"],
                     f"{case_name} ({case['series_description']})")

        log_rows.append({
            "case_name": case_name,
            "split": case["split"],
            "patient_id": case["patient_id"],
            "series_description": case["series_description"],
            "dicom": str(case["dicom"]),
            "gt": str(case["gt"]),
            "fg_pixels": int(case["label_binary"].sum()),
            "transposed": case["transposed"],
        })

        print(f"  {case_name}: {case['series_description']} "
              f"(fg px={int(case['label_binary'].sum())}, split={case['split']})")

        if args.dry_run:
            continue

        is_test = case["split"] == "test"
        image_dir = images_ts if is_test else images_tr
        label_dir = labels_ts if is_test else labels_tr

        image_file = image_dir / f"{case_name}_0000.nii.gz"
        sitk.WriteImage(array_to_grid(case["image_array"], gt_img), str(image_file))

        if not is_test or not args.no_test_labels:
            label_file = label_dir / f"{case_name}.nii.gz"
            sitk.WriteImage(array_to_grid(case["label_binary"], gt_img), str(label_file))

    n_train = len(kept) - n_test

    # --------------------------------------------------------
    # dataset.json
    # --------------------------------------------------------
    dataset_json = {
        "channel_names": {"0": "USG"},
        "labels": {"background": 0, args.foreground_label_name: 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": args.dataset_name,
    }
    if n_test > 0:
        dataset_json["numTest"] = n_test

    if not args.dry_run:
        with open(output_dataset / "dataset.json", "w") as f:
            json.dump(dataset_json, f, indent=2)

        log_file = output_dataset / "prepare_log.csv"
        pd.DataFrame(log_rows).to_csv(log_file, index=False)
    else:
        log_file = "(not written, dry run)"

    print("\n" + "=" * 70)
    print("DONE" if not args.dry_run else "DRY RUN FINISHED")
    print("=" * 70)
    print(f"Train cases : {n_train}")
    print(f"Test cases  : {n_test}")
    print(f"Labels      : {dataset_json['labels']}")
    print(f"Log         : {log_file}")

    if not args.dry_run:
        print(f"\nNext steps:")
        print(f"  nnUNetv2_plan_and_preprocess -d <DATASET_ID> --verify_dataset_integrity")


if __name__ == "__main__":
    main()