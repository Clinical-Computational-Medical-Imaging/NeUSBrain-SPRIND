import os
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib

from pydantic import BaseModel, ConfigDict


# ============================================================
# Configuration
# ============================================================

TARGET_TAGS = {
    "relevant_coronal",
    "mid_saggital",
}

# DICOM PhysicalUnitsXDirection/YDirection code for centimeters.
# (see DICOM PS3.3 C.8.5.5.1.7 - Ultrasound Region Calibration)
DICOM_UNITS_CM = "3"


# ============================================================
# Convert CSV tag -> output class
# ============================================================

def get_class_from_tag(tag: str) -> Optional[str]:
    """
    Convert the tag used in the CSV to the output directory.

        relevant_coronal -> coronal
        mid_saggital     -> sagittal

    Returns None for unsupported tags.
    """

    tag = str(tag).strip().lower()

    if tag == "relevant_coronal":
        return "coronal"

    if tag == "mid_saggital":
        return "sagittal"

    return None


# ============================================================
# Ultrasound region metadata
# ============================================================

class UltrasoundRegionBase(BaseModel):

    ultrasound_region_uid: str

    spatial_format: Optional[int] = None
    data_type: Optional[int] = None
    flags: Optional[int] = None

    location_min_x0: Optional[int] = None
    location_min_y0: Optional[int] = None
    location_max_x1: Optional[int] = None
    location_max_y1: Optional[int] = None

    reference_pixel_x0: Optional[int] = None
    reference_pixel_y0: Optional[int] = None

    units_x: Optional[str] = None
    units_y: Optional[str] = None

    physical_value_x: Optional[float] = None
    physical_value_y: Optional[float] = None

    delta_x: Optional[float] = None
    delta_y: Optional[float] = None

    frequency: Optional[int] = None
    instance_id: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


def parse_ultrasound_regions(
    dcm: pydicom.FileDataset,
) -> Optional[List[UltrasoundRegionBase]]:

    try:

        if not hasattr(dcm, "SequenceOfUltrasoundRegions"):
            return None

        regions = []

        for i, region in enumerate(dcm.SequenceOfUltrasoundRegions):

            region_obj = UltrasoundRegionBase(
                ultrasound_region_uid=f"{dcm.SOPInstanceUID}_{i}",
                spatial_format=getattr(region, "RegionSpatialFormat", None),
                data_type=getattr(region, "RegionDataType", None),
                flags=getattr(region, "RegionFlags", None),
                location_min_x0=getattr(region, "RegionLocationMinX0", None),
                location_min_y0=getattr(region, "RegionLocationMinY0", None),
                location_max_x1=getattr(region, "RegionLocationMaxX1", None),
                location_max_y1=getattr(region, "RegionLocationMaxY1", None),
                reference_pixel_x0=getattr(region, "ReferencePixelX0", None),
                reference_pixel_y0=getattr(region, "ReferencePixelY0", None),
                units_x=str(getattr(region, "PhysicalUnitsXDirection", None)),
                units_y=str(getattr(region, "PhysicalUnitsYDirection", None)),
                physical_value_x=getattr(region, "ReferencePixelPhysicalValueX", None),
                physical_value_y=getattr(region, "ReferencePixelPhysicalValueY", None),
                delta_x=getattr(region, "PhysicalDeltaX", None),
                delta_y=getattr(region, "PhysicalDeltaY", None),
                frequency=getattr(region, "TransducerFrequency", None),
            )

            regions.append(region_obj)

        return regions

    except Exception as e:

        raise ValueError(f"Failed to parse DICOM UltrasoundRegion: {e}") from e


# ============================================================
# Pixel spacing (physical size per pixel, in mm)
# ============================================================

def get_pixel_spacing_mm(
    dcm: pydicom.FileDataset,
    regions: Optional[List[UltrasoundRegionBase]],
) -> Tuple[float, float]:
    """
    Returns (spacing_x, spacing_y) in mm/pixel for the ORIGINAL,
    uncropped image: spacing along columns (x) and rows (y).

    Falls back to 1.0mm isotropic (i.e. "unknown") if nothing usable
    is found, which at least keeps the image undistorted even though
    absolute measurements in the output NIfTI won't be meaningful.
    """

    if regions:

        for r in regions:

            if (
                r.spatial_format == 1
                and r.delta_x
                and r.delta_y
            ):

                factor_x = 10.0 if r.units_x == DICOM_UNITS_CM else 1.0
                factor_y = 10.0 if r.units_y == DICOM_UNITS_CM else 1.0

                return (
                    abs(r.delta_x) * factor_x,
                    abs(r.delta_y) * factor_y,
                )

    if hasattr(dcm, "PixelSpacing"):

        try:

            row_spacing, col_spacing = [
                float(v) for v in dcm.PixelSpacing
            ]

            return col_spacing, row_spacing

        except Exception:
            pass

    return 1.0, 1.0


# ============================================================
# Contour extraction
# ============================================================

def extract_ultrasound_image(
    whole_img: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:

    contours, _ = cv2.findContours(
        whole_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    img_contour = []

    for c in contours:

        x, y, w, h = cv2.boundingRect(c)

        area = w * h

        if area > 10000:
            img_contour.append((x, y, w, h))

    if len(img_contour) != 1:

        print(f"    Warning: expected 1 contour, found {len(img_contour)}")

        return None

    return img_contour[0]


# ============================================================
# Find ultrasound crop
# ============================================================

def find_ultrasound_crop(
    dcm: pydicom.FileDataset,
    gray: np.ndarray,
    regions: Optional[List[UltrasoundRegionBase]],
) -> Optional[Tuple[int, int, int, int]]:

    # --------------------------------------------------------
    # First try DICOM region metadata
    # --------------------------------------------------------

    if regions:

        for r in regions:

            if (
                r.spatial_format == 1
                and None not in (
                    r.location_min_x0,
                    r.location_min_y0,
                    r.location_max_x1,
                    r.location_max_y1,
                )
            ):

                x0 = int(r.location_min_x0)
                y0 = int(r.location_min_y0)
                x1 = int(r.location_max_x1)
                y1 = int(r.location_max_y1)

                x0 = max(0, min(x0, gray.shape[1]))
                x1 = max(0, min(x1, gray.shape[1]))
                y0 = max(0, min(y0, gray.shape[0]))
                y1 = max(0, min(y1, gray.shape[0]))

                if x1 > x0 and y1 > y0:
                    return (x0, y0, x1 - x0, y1 - y0)

    # --------------------------------------------------------
    # Fall back to contour
    # --------------------------------------------------------

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    return extract_ultrasound_image(thresh)


# ============================================================
# Frame selection (handles cine / multi-frame DICOMs)
# ============================================================

def select_single_frame(
    dcm: pydicom.FileDataset,
    pixel_array: np.ndarray,
) -> np.ndarray:
    """
    US DICOMs are sometimes cine loops (multiple frames). We only
    want a single still frame. Frame is always the leading axis
    per the DICOM standard, so peel it off explicitly using
    NumberOfFrames rather than guessing from array shape (guessing
    is what causes silent, hard-to-debug shape bugs downstream).
    """

    n_frames = int(getattr(dcm, "NumberOfFrames", 1) or 1)

    if n_frames > 1 and pixel_array.ndim >= 3 and pixel_array.shape[0] == n_frames:

        print(f"    Multi-frame DICOM ({n_frames} frames); using frame 0")

        return pixel_array[0]

    return pixel_array


# ============================================================
# Load and preprocess DICOM
# ============================================================

def load_and_preprocess(
    dcm_path: str,
    img_height: int,
    img_width: int,
):
    """
    Returns a dict with:

        image:
            (H, W) float32 grayscale (NOT tripled to RGB - these
            are single-channel US images, and saving them as fake
            3-channel volumes is what made viewers show them as a
            bogus "3D" stack of 3 identical slices).

        bbox:
            (x, y, w, h) crop box, in ORIGINAL pixel coordinates.

        original_shape:
            (H, W) of the original (uncropped) grayscale image.

        spacing_xy_mm:
            (spacing_x, spacing_y) mm/pixel of the ORIGINAL image.
    """

    try:
        dcm = pydicom.dcmread(dcm_path)
    except Exception as e:
        print(f"    Cannot read DICOM: {e}")
        return None

    try:
        pixel_array = dcm.pixel_array
    except Exception as e:
        print(f"    Cannot extract pixel array: {e}")
        return None

    arr = select_single_frame(dcm, pixel_array)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    if arr.ndim == 2:

        gray = cv2.normalize(
            arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX,
        ).astype(np.uint8)

    elif arr.ndim == 3 and arr.shape[2] == 3:

        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    elif arr.ndim == 3 and arr.shape[2] == 4:

        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    else:

        print(f"    Unexpected pixel shape {arr.shape}")
        return None

    original_shape = gray.shape

    # --------------------------------------------------------
    # Physical spacing (needed for a correct affine later)
    # --------------------------------------------------------

    regions = parse_ultrasound_regions(dcm)

    spacing_xy_mm = get_pixel_spacing_mm(dcm, regions)

    # --------------------------------------------------------
    # Find ultrasound region / crop
    # --------------------------------------------------------

    bbox = find_ultrasound_crop(dcm, gray, regions)

    if bbox is None:
        return None

    x, y, w, h = bbox

    cropped = gray[y:y + h, x:x + w]

    if cropped.size == 0:
        print("    Empty image crop")
        return None

    resized = cv2.resize(
        cropped, (img_width, img_height), interpolation=cv2.INTER_LINEAR,
    )

    return {
        "image": resized.astype(np.float32),
        "bbox": bbox,
        "original_shape": original_shape,
        "spacing_xy_mm": spacing_xy_mm,
    }


# ============================================================
# Load and preprocess labelmap
# ============================================================

def preprocess_labelmap(
    nii_path: str,
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    output_height: int,
    output_width: int,
) -> Optional[np.ndarray]:

    try:
        nii = nib.load(nii_path)
        label = nii.get_fdata()
    except Exception as e:
        print(f"    Cannot read labelmap: {e}")
        return None

    label = np.squeeze(label)

    # --------------------------------------------------------
    # Labels drawn on an RGB-converted US frame can come back as
    # (H, W, 3/4) even though the mask itself is single-channel
    # (identical values duplicated across channels). Collapse it.
    # --------------------------------------------------------

    if label.ndim == 3:

        channel_axis = None

        for ax, size in enumerate(label.shape):

            if size in (3, 4):
                channel_axis = ax
                break

        if channel_axis is None:
            print(f"    Unexpected labelmap shape {label.shape}")
            return None

        label = np.take(label, 0, axis=channel_axis)

    if label.ndim != 2:
        print(f"    Unexpected labelmap shape {label.shape}")
        return None

    # --------------------------------------------------------
    # The label may have been saved with rows/cols swapped
    # relative to the DICOM pixel array the bbox was computed
    # from. Detect and correct that before cropping.
    # --------------------------------------------------------

    if label.shape != image_shape and label.shape == image_shape[::-1]:

        print(
            f"    Labelmap shape {label.shape} is transposed relative "
            f"to image shape {image_shape}; correcting."
        )

        label = label.T

    x, y, w, h = bbox

    if y + h > label.shape[0] or x + w > label.shape[1]:

        print(
            f"    Image crop is outside labelmap dimensions "
            f"(bbox=({x},{y},{w},{h}), label_shape={label.shape}, "
            f"image_shape={image_shape})"
        )

        return None

    cropped = label[y:y + h, x:x + w]

    if cropped.size == 0:
        return None

    # nearest-neighbour preserves class IDs
    resized = cv2.resize(
        cropped.astype(np.int32),
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized


# ============================================================
# Orientation-correct affine
# ============================================================

def build_affine(spacing_x_mm: float, spacing_y_mm: float) -> np.ndarray:
    """
    numpy/OpenCV arrays are stored (row, col) with row 0 at the TOP
    of the image and increasing downward. NIfTI viewers instead
    index (i, j, k) and map i -> the affine's first column, j -> the
    second, both walked in a "world space" that (by convention)
    increases upward/rightward.

    If we hand a NIfTI viewer a plain identity affine with the raw
    (row, col) array, it ends up displaying the array's row axis
    (vertical) as if it were the world's first/horizontal axis -
    which is exactly what shows up as "rotated 90 degrees". Negating
    the row-axis spacing (see array_to_nifti_layout below, which
    also transposes the array) is the standard fix: it accounts for
    the row-major top-down storage vs. the bottom-up world axis.
    """

    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = spacing_x_mm
    affine[1, 1] = -spacing_y_mm
    affine[2, 2] = 1.0

    return affine


def array_to_nifti_layout(arr2d: np.ndarray) -> np.ndarray:
    """
    Transpose a (H, W) array to (W, H) and add a singleton Z axis,
    so the array's axis order matches the affine built by
    build_affine (axis 0 <-> x/columns, axis 1 <-> y/rows).
    Must be applied identically to both the image and its label so
    they stay in registration.
    """

    return arr2d.T[:, :, np.newaxis]


# ============================================================
# Save 2D array as NIfTI
# ============================================================

def save_nifti(
    array2d: np.ndarray,
    output_path: Path,
    affine: np.ndarray,
    is_label: bool = False,
):
    """
    Save a single 2D grayscale image or label as a proper,
    correctly-oriented, correctly-spaced 2D-as-3D NIfTI volume
    (shape (W, H, 1)).
    """

    volume = array_to_nifti_layout(array2d)

    if is_label:
        volume = volume.astype(np.int16)
    else:
        volume = volume.astype(np.float32)

    nii = nib.Nifti1Image(volume, affine)

    dtype = np.int16 if is_label else np.float32
    nii.set_data_dtype(dtype)

    # Set both qform and sform consistently so viewers that only
    # honor one or the other still display the image the same way,
    # and so nothing tries to re-derive/reorient axes on its own.
    nii.set_qform(affine, code=1)
    nii.set_sform(affine, code=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    nib.save(nii, str(output_path))


# ============================================================
# Process one image/label pair
# ============================================================

def process_single_pair(
    match: dict,
    output_dir: str,
    img_height: int,
    img_width: int,
):

    image_path = match["image_path"]
    label_path = match["label_path"]

    image_id = match["image_id"]
    image_class = match["class"]

    try:

        result = load_and_preprocess(image_path, img_height, img_width)

        if result is None:

            return {
                "success": False,
                **match,
                "reason": "image preprocessing failed",
            }

        image = result["image"]
        bbox = result["bbox"]
        original_shape = result["original_shape"]
        spacing_x_orig, spacing_y_orig = result["spacing_xy_mm"]

        label = preprocess_labelmap(
            label_path, bbox, original_shape, img_height, img_width,
        )

        if label is None:

            return {
                "success": False,
                **match,
                "reason": "label preprocessing failed",
            }

        # ----------------------------------------------------
        # Effective spacing after crop + resize
        # ----------------------------------------------------

        _, _, crop_w, crop_h = bbox

        spacing_x_out = spacing_x_orig * (crop_w / img_width)
        spacing_y_out = spacing_y_orig * (crop_h / img_height)

        affine = build_affine(spacing_x_out, spacing_y_out)

        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        output_dir = Path(output_dir)

        image_dir = output_dir / "images" / image_class
        label_dir = output_dir / "labels" / image_class

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        image_output = image_dir / f"{image_id}.nii.gz"
        label_output = label_dir / f"{image_id}.nii.gz"

        save_nifti(image, image_output, affine, is_label=False)
        save_nifti(label, label_output, affine, is_label=True)

        return {
            "success": True,
            **match,
            "output_image": str(image_output),
            "output_label": str(label_output),
            "crop_x": bbox[0],
            "crop_y": bbox[1],
            "crop_w": bbox[2],
            "crop_h": bbox[3],
            "original_height": original_shape[0],
            "original_width": original_shape[1],
            "spacing_x_mm": spacing_x_out,
            "spacing_y_mm": spacing_y_out,
        }

    except Exception as e:

        return {
            "success": False,
            **match,
            "reason": f"processing exception: {e}",
        }


# ============================================================
# Parallel preprocessing
# ============================================================

def process_dataset_parallel(
    matches: List[dict],
    output_dir: Path,
    img_height: int,
    img_width: int,
    num_workers: Optional[int] = None,
):

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 2)

    print(f"Preprocessing with {num_workers} workers")

    processed = []
    failed = []

    for image_class in ("sagittal", "coronal"):

        (output_dir / "images" / image_class).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / image_class).mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        futures = [
            executor.submit(
                process_single_pair, match, str(output_dir), img_height, img_width,
            )
            for match in matches
        ]

        total = len(futures)

        for i, future in enumerate(as_completed(futures), start=1):

            result = future.result()

            if result["success"]:
                processed.append(result)
            else:
                failed.append(result)

            if i % 50 == 0 or i == total:

                print(
                    f"\rProcessed: {i:6d}/{total} | "
                    f"Success: {len(processed):6d} | "
                    f"Failed: {len(failed):6d}",
                    end="",
                    flush=True,
                )

    print()

    pd.DataFrame(processed).to_csv(output_dir / "processed.csv", index=False)
    pd.DataFrame(failed).to_csv(output_dir / "processing_failures.csv", index=False)

    print()
    print("==========================================")
    print("Preprocessing complete")
    print("==========================================")
    print(f"Successfully processed: {len(processed)}")
    print(f"Failed:                 {len(failed)}")


# ============================================================
# Find matching DICOM + labelmap
# ============================================================

def find_matching_images_parallel(
    csv: pd.DataFrame,
    labels_dataset: Path,
    target_tags: set,
    num_workers: Optional[int] = None,
):

    print("Indexing labelmaps...")

    label_names = {}

    for path in labels_dataset.rglob("*.nii"):

        if path.name not in label_names:
            label_names[path.name] = str(path)

    for path in labels_dataset.rglob("*.nii.gz"):

        if path.name not in label_names:
            label_names[path.name] = str(path)

    print(f"Found {len(label_names)} labelmaps")

    jobs = []

    for idx, row in csv.iterrows():

        tag = str(row.get("user_label", "")).strip()

        if tag not in target_tags:
            continue

        image_class = get_class_from_tag(tag)

        if image_class is None:
            continue

        image_path = row.get("image_path")

        if pd.isna(image_path):
            continue

        image_path = str(image_path)

        if not os.path.exists(image_path):
            continue

        jobs.append(
            (idx, image_path, tag, image_class, str(labels_dataset), label_names)
        )

    print(f"Images selected: {len(jobs)}")

    def worker(job):

        (
            csv_index,
            image_path,
            tag,
            image_class,
            labels_dataset,
            label_names,
        ) = job

        try:

            ds = pydicom.dcmread(image_path, stop_before_pixels=True)

            image_id = str(getattr(ds, "SeriesDescription", "")).strip()

            if not image_id:

                return {
                    "success": False,
                    "reason": "missing SeriesDescription",
                    "csv_index": csv_index,
                    "image_path": image_path,
                }

            patient_id = image_id.split("_")[0]

            expected = Path(labels_dataset) / patient_id / f"{image_id}.nii"

            if expected.exists():

                label_path = str(expected)

            else:

                expected_gz = (
                    Path(labels_dataset) / patient_id / f"{image_id}.nii.gz"
                )

                if expected_gz.exists():

                    label_path = str(expected_gz)

                else:

                    label_path = (
                        label_names.get(f"{image_id}.nii")
                        or label_names.get(f"{image_id}.nii.gz")
                    )

            if label_path is None:

                return {
                    "success": False,
                    "reason": "labelmap not found",
                    "csv_index": csv_index,
                    "image_path": image_path,
                    "image_id": image_id,
                    "patient_id": patient_id,
                    "user_label": tag,
                }

            return {
                "success": True,
                "csv_index": csv_index,
                "user_label": tag,
                "class": image_class,
                "image_path": image_path,
                "image_id": image_id,
                "patient_id": patient_id,
                "label_path": label_path,
            }

        except Exception as e:

            return {
                "success": False,
                "reason": f"DICOM read failed: {e}",
                "csv_index": csv_index,
                "image_path": image_path,
                "user_label": tag,
            }

    from concurrent.futures import ThreadPoolExecutor

    if num_workers is None:
        num_workers = min(32, (os.cpu_count() or 4) * 2)

    matches = []
    failures = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:

        futures = [executor.submit(worker, job) for job in jobs]

        total = len(futures)

        for i, future in enumerate(as_completed(futures), start=1):

            result = future.result()

            if result["success"]:
                matches.append(result)
            else:
                failures.append(result)

            if i % 100 == 0 or i == total:

                print(
                    f"\rMatching: {i:6d}/{total} | "
                    f"Matched: {len(matches):6d} | "
                    f"Failed: {len(failures):6d}",
                    end="",
                    flush=True,
                )

    print()

    return matches, failures


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--images-dataset", required=True)
    ap.add_argument("--labels-dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--workers", type=int, default=None)

    args = ap.parse_args()

    labels_dataset = Path(args.labels_dataset)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(args.csv)

    if "image_path" not in df.columns:
        raise ValueError("CSV must contain 'image_path'")

    if "user_label" not in df.columns:
        raise ValueError("CSV must contain 'tag'")

    matches, matching_failures = find_matching_images_parallel(
        csv=df,
        labels_dataset=labels_dataset,
        target_tags=TARGET_TAGS,
        num_workers=args.workers,
    )

    print(f"\nMatched pairs: {len(matches)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(matches).to_csv(output_dir / "matched_pairs.csv", index=False)
    pd.DataFrame(matching_failures).to_csv(
        output_dir / "matching_failures.csv", index=False
    )

    process_dataset_parallel(
        matches=matches,
        output_dir=output_dir,
        img_height=args.height,
        img_width=args.width,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()