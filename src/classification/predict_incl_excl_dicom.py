"""
predict_incl_excl.py

Reads a folder of DICOM files, preprocesses each image using the
ultrasound region extraction pipeline, runs the include/exclude
classifier, and writes results to a CSV.

Usage:
    python predict_incl_excl.py \
        --data_dir /path/to/dcm/files \
        --model_path /path/to/model.keras \
        --output_csv /path/to/results.csv \
        [--img_height 180] [--img_width 180] [--batch_size 32]
"""

import argparse
import os
import traceback
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Pydantic models (from your codebase)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DICOM parsing helpers (from your codebase)
# ---------------------------------------------------------------------------

def parse_ultrasound_regions(dcm: pydicom.FileDataset) -> Optional[List[UltrasoundRegionBase]]:
    try:
        if not hasattr(dcm, "SequenceOfUltrasoundRegions"):
            return None

        regions = []
        for i, region in enumerate(dcm.SequenceOfUltrasoundRegions):
            region_obj = UltrasoundRegionBase(
                ultrasound_region_uid=str(dcm.SOPInstanceUID) + "_" + str(i),
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
        raise ValueError(f"Failed to parse DICOM UltrasoundRegion object: {e}") from e


def extract_ultrasound_image(whole_img: np.ndarray, visualize: bool = False):
    """
    Finds the bounding box of the largest ultrasound region in a
    binary/grayscale image using contour detection.

    Returns (x, y, w, h) or None if not exactly one large contour found.
    """
    contours, _ = cv2.findContours(whole_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contour = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > 10000:
            img_contour.append((x, y, w, h))

    if len(img_contour) != 1:
        print(f"  Warning: expected 1 contour, found {len(img_contour)} — skipping.")
        return None

    return img_contour[0]


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def load_and_preprocess(
    dcm_path: str,
    img_height: int,
    img_width: int,
) -> Optional[np.ndarray]:
    """
    Loads a DICOM file, extracts the ultrasound pixel region, and returns
    a float32 RGB array of shape (img_height, img_width, 3) ready for the
    model, or None on failure.
    """
    try:
        dcm = pydicom.dcmread(dcm_path)
    except Exception as e:
        print(f"  Cannot read DICOM: {e}")
        return None

    # --- pixel data ---
    try:
        pixel_array = dcm.pixel_array  # shape: (H, W) or (H, W, 3)
    except Exception as e:
        print(f"  Cannot extract pixel array: {e}")
        return None

    # Normalise to uint8 grayscale — handle all common DICOM shapes:
    #   (H, W)      → true grayscale
    #   (H, W, 1)   → grayscale with redundant channel axis
    #   (H, W, 3)   → RGB
    #   (H, W, 4)   → RGBA
    arr = pixel_array
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]          # squeeze single channel → (H, W)

    if arr.ndim == 2:
        # Already grayscale — just normalise to uint8
        gray = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    else:
        print(f"  Unexpected pixel array shape {arr.shape} — skipping.")
        return None

    # --- try region-based crop first (uses SequenceOfUltrasoundRegions) ---
    regions = parse_ultrasound_regions(dcm)
    cropped = None

    if regions:
        # Use the bounding box from the first 2D tissue region if available
        for r in regions:
            if (
                r.spatial_format == 1  # 2D tissue
                and None not in (r.location_min_x0, r.location_min_y0,
                                 r.location_max_x1, r.location_max_y1)
            ):
                x0, y0 = r.location_min_x0, r.location_min_y0
                x1, y1 = r.location_max_x1, r.location_max_y1
                cropped = gray[y0:y1, x0:x1]
                break

    # --- fall back to contour-based extraction ---
    if cropped is None:
        # Threshold to isolate the US cone from the black background
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        bbox = extract_ultrasound_image(thresh, visualize=False)
        if bbox is None:
            return None
        x, y, w, h = bbox
        cropped = gray[y:y + h, x:x + w]

    if cropped.size == 0:
        print("  Empty crop — skipping.")
        return None

    # --- resize and convert to 3-channel float32 for EfficientNet ---
    resized = cv2.resize(cropped, (img_width, img_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)          # (H, W, 3)
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def predict_folder(
    data_dir: str,
    model_path: str,
    output_csv: str,
    img_height: int = 180,
    img_width: int = 180,
    batch_size: int = 32,
) -> None:

    # Collect all DICOM files (recursively)
    dcm_paths = sorted([
        str(p) for p in Path(data_dir).rglob("*")
        if p.suffix.lower() in {".dcm", ".dicom", ""}
        and p.is_file()
    ])
    # Also accept files without extension that pydicom can still read
    # Filter: keep only readable DICOMs
    print(f"Found {len(dcm_paths)} candidate files in {data_dir}")

    # Load model
    print(f"Loading model from {model_path} …")
    model = tf.keras.models.load_model(model_path)

    # Determine class names from model output layer
    num_classes = model.output_shape[-1]
    # Default label map — adjust if your folder order differs
    class_labels = {0: "exclude", 1: "include"} if num_classes == 2 else {
        i: str(i) for i in range(num_classes)
    }

    records = []
    batch_imgs: List[np.ndarray] = []
    batch_files: List[str] = []

    def run_batch():
        if not batch_imgs:
            return
        arr = np.stack(batch_imgs, axis=0)               # (B, H, W, 3)
        logits = model.predict(arr, verbose=0)            # (B, num_classes)
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        for fpath, prob in zip(batch_files, probs):
            pred_idx = int(np.argmax(prob))
            records.append({
                "file": fpath,
                "prediction": class_labels[pred_idx],
                "confidence": float(prob[pred_idx]),
                **{f"prob_{class_labels[i]}": float(prob[i]) for i in range(num_classes)},
                "error": "",
            })

    for i, dcm_path in enumerate(dcm_paths):
        print(f"[{i+1}/{len(dcm_paths)}] Processing {os.path.basename(dcm_path)} …", end=" ")

        img = load_and_preprocess(dcm_path, img_height, img_width)

        if img is None:
            print("SKIPPED")
            records.append({
                "file": dcm_path,
                "prediction": "error",
                "confidence": None,
                **{f"prob_{class_labels[j]}": None for j in range(num_classes)},
                "error": "preprocessing failed",
            })
            continue

        print("OK")
        batch_imgs.append(img)
        batch_files.append(dcm_path)

        if len(batch_imgs) == batch_size:
            run_batch()
            batch_imgs.clear()
            batch_files.clear()

    # Final partial batch
    run_batch()

    # Write CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nDone. Results written to {output_csv}")
    print(df["prediction"].value_counts().to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Include/exclude classifier for neonatal brain US DICOMs.")
    parser.add_argument("--data_dir",    required=True,  help="Folder containing .dcm files (searched recursively).")
    parser.add_argument("--model_path",  required=True,  help="Path to the saved Keras model (.keras or SavedModel dir).")
    parser.add_argument("--output_csv",  required=True,  help="Path for the output CSV file.")
    parser.add_argument("--img_height",  type=int, default=180, help="Image height expected by the model (default: 180).")
    parser.add_argument("--img_width",   type=int, default=180, help="Image width expected by the model (default: 180).")
    parser.add_argument("--batch_size",  type=int, default=32,  help="Inference batch size (default: 32).")
    args = parser.parse_args()

    predict_folder(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_csv=args.output_csv,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
    )