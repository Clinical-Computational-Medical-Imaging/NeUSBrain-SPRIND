"""
Measure the height of the cerebellar vermis on 2D midsagittal
segmentations (e.g. fetal / neonatal ultrasound).

Input:  label maps (NIfTI or DICOM) in which the vermis has its own
        label value (default 2), optionally plus the original image
        DICOMs for calibration and QC overlays.
Output: one CSV row per label map and (optionally) one QC image each.

Method ("chord through the centroid"):

    1. Take the vermis label, keep the largest connected component and
       fill holes inside it.
    2. Map every pixel to world coordinates in mm, using the NIfTI
       affine or the DICOM pixel spacing. Transposed, flipped or
       anisotropic pixel grids therefore all measure the same.
    3. Compute the centroid of the vermis and its principal axis (PCA).
       The axis is limited to +-20 deg from the image vertical, because
       for an almost round vermis the free PCA axis can tip over.
    4. Height = length of the straight line through the centroid along
       that axis, from where it enters to where it leaves the vermis
       (like a caliper line drawn by hand through the middle).
       AP diameter = the same line perpendicular to it.

Usage:

    # label maps only (spacing taken from the label files themselves)
    python vermis_height.py --input labels/ --output results/

    # label maps + original DICOMs (spacing and QC background from the images)
    python vermis_height.py --input labels/ --images dicoms/ --output results/

See README.md for details on input layout, output columns and flags.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom


# label value of the vermis in the segmentation
DEFAULT_VERMIS_LABEL = 2

# maximum angle (deg) between the measurement axis and the image vertical
DEFAULT_MAX_AXIS_ANGLE = 20.0

# height / AP below this -> the principal axis is poorly defined
MIN_ELONGATION = 1.2

# DICOM PhysicalUnitsX/YDirection code for centimeters
DICOM_UNITS_CM = 3

NIFTI_SUFFIXES = (".nii.gz", ".nii")


# ============================================================
# Find input files
# ============================================================

def is_dicom(path: Path) -> bool:
    """True for *.dcm / *.dicom files and extension-less DICOM files."""

    if path.suffix.lower() in (".dcm", ".dicom"):
        return True

    if path.suffix:
        return False

    # DICOMs often have no extension; check the "DICM" marker after the preamble
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def is_nifti(path: Path) -> bool:

    return path.name.lower().endswith(NIFTI_SUFFIXES)


def find_label_files(input_path: Path) -> list:
    """A single label map, or all NIfTI / DICOM files below a folder."""

    if input_path.is_file():
        return [input_path]

    return sorted(p for p in input_path.rglob("*") if p.is_file() and (is_nifti(p) or is_dicom(p)))


def sample_id_from_path(path: Path) -> str:
    """
    File name without extension and without a trailing "_seg", so that
    "case01.nii.gz", "case01_seg.nii.gz" and "case01_seg.dcm" all
    become "case01" and can be matched to the image "case01.dcm".
    """

    name = path.name

    for ext in NIFTI_SUFFIXES + (".dcm", ".dicom"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break

    if name.endswith("_seg"):
        name = name[: -len("_seg")]

    return name


def index_images(images_dir: Path) -> Dict[str, Path]:
    """
    Maps a sample id to its image DICOM. Every image is listed under its
    file name (without extension) and under its SeriesDescription, so
    label maps may be named after either. Segmentation DICOMs
    (Modality SEG) in the folder are skipped.
    """

    index = {}

    for path in sorted(images_dir.rglob("*")):

        if not path.is_file() or not is_dicom(path):
            continue

        try:
            dcm = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue

        if str(getattr(dcm, "Modality", "")).upper() == "SEG":
            continue

        keys = [sample_id_from_path(path), str(getattr(dcm, "SeriesDescription", "")).strip()]

        # first file wins on duplicate keys
        for key in keys:
            if key:
                index.setdefault(key, path)

    return index


# ============================================================
# Load DICOMs
# ============================================================

def get_pixel_spacing_mm(dcm: pydicom.Dataset) -> Tuple[float, float]:
    """
    (spacing_x, spacing_y) in mm per pixel.

    Ultrasound DICOMs store their calibration in the
    SequenceOfUltrasoundRegions (often in cm); other DICOMs use
    PixelSpacing. Falls back to (1.0, 1.0), which is flagged as
    "spacing_uncalibrated" in the output.
    """

    for region in getattr(dcm, "SequenceOfUltrasoundRegions", []):

        delta_x = getattr(region, "PhysicalDeltaX", None)
        delta_y = getattr(region, "PhysicalDeltaY", None)

        # RegionSpatialFormat 1 = 2D tissue image (not a Doppler trace etc.)
        if getattr(region, "RegionSpatialFormat", None) == 1 and delta_x and delta_y:

            factor_x = 10.0 if getattr(region, "PhysicalUnitsXDirection", None) == DICOM_UNITS_CM else 1.0
            factor_y = 10.0 if getattr(region, "PhysicalUnitsYDirection", None) == DICOM_UNITS_CM else 1.0

            return abs(float(delta_x)) * factor_x, abs(float(delta_y)) * factor_y

    if hasattr(dcm, "PixelSpacing"):

        # PixelSpacing is (row spacing, column spacing) = (y, x)
        row_spacing, col_spacing = [float(v) for v in dcm.PixelSpacing]

        return col_spacing, row_spacing

    return 1.0, 1.0


def load_dicom_pixels(path: Path) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Returns (pixel array as (H, W) or (H, W, 3), (spacing_x, spacing_y) in mm).
    For multi-frame DICOMs (cine loops) only the first frame is used.
    """

    dcm = pydicom.dcmread(str(path))

    arr = dcm.pixel_array

    n_frames = int(getattr(dcm, "NumberOfFrames", 1) or 1)

    if n_frames > 1 and arr.ndim >= 3 and arr.shape[0] == n_frames:
        arr = arr[0]

    # single-channel or RGBA -> (H, W) or (H, W, 3)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    return arr, get_pixel_spacing_mm(dcm)


def spacing_to_affine(spacing_xy_mm: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pixel -> world mapping for a DICOM pixel grid:
    world_mm = A @ [row, col] + t = (col * spacing_x, row * spacing_y, 0).
    """

    spacing_x, spacing_y = spacing_xy_mm

    A = np.array([[0.0, spacing_x], [spacing_y, 0.0], [0.0, 0.0]])

    return A, np.zeros(3)


def to_gray(arr: np.ndarray) -> np.ndarray:
    """Image as 8-bit grayscale, used as QC background."""

    if arr.ndim == 3:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    return cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ============================================================
# Load label maps as 2D array + pixel -> world mapping
# ============================================================

def load_label_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (label (H, W), A (3, 2), t (3,)) with

        world_mm = A @ [row, col] + t

    taken from the NIfTI affine. Singleton axes are dropped. Labels
    drawn on RGB frames can come with an extra channel axis of size 3
    or 4 (identical values); only its first channel is used.
    """

    nii = nib.load(str(path))

    data = np.asanyarray(nii.dataobj)

    # dims beyond 3 (time / vector) are not covered by the affine
    while data.ndim > 3:
        data = data[..., 0]

    while data.ndim < 3:
        data = data[..., np.newaxis]

    spatial_axes = [ax for ax in range(3) if data.shape[ax] > 1]

    if len(spatial_axes) == 3:

        channel_axes = [ax for ax in spatial_axes if data.shape[ax] in (3, 4)]

        if not channel_axes:
            raise ValueError(f"label map is 3D {data.shape}, expected a single 2D slice")

        spatial_axes.remove(channel_axes[-1])

    if len(spatial_axes) != 2:
        raise ValueError(f"unexpected label map shape {data.shape}")

    index = tuple(slice(None) if ax in spatial_axes else 0 for ax in range(3))

    # nibabel returns Fortran-ordered arrays; OpenCV needs C order
    label = np.ascontiguousarray(np.rint(data[index]).astype(np.int32))

    A = nii.affine[:3, spatial_axes]
    t = nii.affine[:3, 3]

    return label, A, t


def load_label_dicom(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as load_label_nifti, for label maps stored as DICOM."""

    arr, spacing_xy_mm = load_dicom_pixels(path)

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    label = np.ascontiguousarray(arr.astype(np.int32))

    A, t = spacing_to_affine(spacing_xy_mm)

    return label, A, t


def load_label(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if is_nifti(path):
        return load_label_nifti(path)

    return load_label_dicom(path)


# ============================================================
# Clean the mask
# ============================================================

def largest_component_filled(mask: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """
    Keeps the largest connected component of the mask and fills holes
    inside it (by redrawing its outer contour filled).

    Returns (cleaned mask, number of components, fraction of the
    original foreground that the kept component makes up).
    """

    mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.zeros_like(mask, dtype=bool), 0, 0.0

    largest = max(contours, key=cv2.contourArea)

    filled = np.zeros_like(mask_u8)
    cv2.drawContours(filled, [largest], -1, 1, thickness=cv2.FILLED)

    kept = filled.astype(bool)

    fraction = float((kept & mask).sum()) / float(mask.sum())

    return kept, len(contours), fraction


# ============================================================
# Measure
# ============================================================

def clamp_axis(
    major_axis: np.ndarray,
    A: np.ndarray,
    vertical_axis: int,
    max_axis_angle: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (major axis, minor axis, signed angle of the unclamped
    major axis from the image vertical in degrees).

    The image vertical is the world direction of array axis
    `vertical_axis` (0 = rows, i.e. top-bottom in a normally displayed
    image). If max_axis_angle is set, the major axis is rotated within
    the image plane to at most that angle from the vertical.
    """

    # orthonormal basis of the image plane: e1 = vertical, e2 = horizontal
    e1 = A[:, vertical_axis] / np.linalg.norm(A[:, vertical_axis])

    other = A[:, 1 - vertical_axis]
    e2 = other - (other @ e1) * e1
    e2 /= np.linalg.norm(e2)

    angle = np.degrees(np.arctan2(major_axis @ e2, major_axis @ e1))

    # an axis has no direction: fold into [-90, 90]
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    used = angle if max_axis_angle is None else float(np.clip(angle, -max_axis_angle, max_axis_angle))

    th = np.radians(used)

    major = np.cos(th) * e1 + np.sin(th) * e2
    minor = -np.sin(th) * e1 + np.cos(th) * e2

    return major, minor, float(angle)


def chord_through(
    mask: np.ndarray,
    A: np.ndarray,
    t: np.ndarray,
    centroid: np.ndarray,
    direction: np.ndarray,
    reach_mm: float,
) -> Tuple[float, float]:
    """
    Walks along the line centroid + s * direction (s in mm, from
    -reach_mm to +reach_mm) and returns (s_min, s_max): the outermost
    points of the line that lie inside the mask, i.e. where the line
    enters and leaves the vermis contour.
    """

    # sample at a quarter of the smallest pixel size
    step = 0.25 * min(np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1]))

    s = np.arange(-reach_mm, reach_mm + step, step)

    points = centroid + s[:, None] * direction

    # world -> nearest pixel (row, col)
    pixels = np.rint((points - t) @ np.linalg.pinv(A).T).astype(int)

    rows, cols = pixels[:, 0], pixels[:, 1]

    valid = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])

    inside = np.zeros(len(s), dtype=bool)
    inside[valid] = mask[rows[valid], cols[valid]]

    if not inside.any():
        return 0.0, 0.0

    return float(s[inside].min()), float(s[inside].max())


def measure_vermis(
    mask: np.ndarray,
    A: np.ndarray,
    t: np.ndarray,
    max_axis_angle: Optional[float] = DEFAULT_MAX_AXIS_ANGLE,
    vertical_axis: int = 0,
) -> dict:
    """
    Height and AP diameter of a (cleaned) vermis mask, measured as
    chords through its centroid (see module docstring).
    """

    rows, cols = np.nonzero(mask)

    centers = np.stack([rows, cols], axis=1).astype(np.float64)

    world_centers = centers @ A.T + t

    centroid = world_centers.mean(axis=0)

    # principal axes of the (planar) point cloud; the third
    # component is ~0 since all points lie in the image plane
    _, _, vt = np.linalg.svd(world_centers - centroid, full_matrices=False)

    major_axis, minor_axis, pca_angle = clamp_axis(vt[0], A, vertical_axis, max_axis_angle)

    # far enough from the centroid to leave the vermis in every direction:
    # largest coordinate of any pixel corner relative to the centroid + 1 mm
    corner_offsets = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    corners = (centers[:, np.newaxis, :] + corner_offsets[np.newaxis]).reshape(-1, 2)
    reach = float(np.abs(corners @ A.T + t - centroid).max()) + 1.0

    span_major = chord_through(mask, A, t, centroid, major_axis, reach)
    span_minor = chord_through(mask, A, t, centroid, minor_axis, reach)

    height_mm = span_major[1] - span_major[0]
    ap_mm = span_minor[1] - span_minor[0]

    pixel_area_mm2 = float(np.linalg.norm(np.cross(A[:, 0], A[:, 1])))

    # end points of the measurement lines in (row, col), for the QC image
    A_pinv = np.linalg.pinv(A)

    def to_pixel(world_point):
        return A_pinv @ (world_point - t)

    return {
        "height_mm": height_mm,
        "ap_mm": ap_mm,
        "area_mm2": float(len(rows)) * pixel_area_mm2,
        "elongation": height_mm / ap_mm if ap_mm > 0 else np.nan,
        "pca_angle_deg": pca_angle,
        "axis_clamped": max_axis_angle is not None and abs(pca_angle) > max_axis_angle,
        "centroid_px": to_pixel(centroid),
        "height_line": [to_pixel(centroid + major_axis * s) for s in span_major],
        "ap_line": [to_pixel(centroid + minor_axis * s) for s in span_minor],
    }


# ============================================================
# QC image
# ============================================================

def save_qc_image(
    raw_mask: np.ndarray,
    kept_mask: np.ndarray,
    result: dict,
    output_path: Path,
    background: Optional[np.ndarray] = None,
):
    """
    Draws the mask and the measurement lines:

        red    = height (chord along the vermis axis)
        blue   = AP diameter
        yellow = centroid

    Without a background the kept vermis is light gray and discarded
    fragments dark gray; on top of an image they are drawn as green
    and orange contours. The image is drawn in array (row, col) layout,
    which can be transposed or flipped compared to a DICOM viewer; the
    measurement itself is not affected by that.
    """

    h, w = raw_mask.shape

    # upscale small images so the lines and text stay readable
    scale = max(1, 900 // max(h, w))
    size = (w * scale, h * scale)

    if background is None:

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[raw_mask] = (80, 80, 80)
        canvas[kept_mask] = (160, 160, 160)

        canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_NEAREST)

    else:

        canvas = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_NEAREST)

        for mask, color in ((raw_mask & ~kept_mask, (0, 165, 255)), (kept_mask, (0, 255, 0))):

            mask_up = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(canvas, contours, -1, color, 1)

    def to_xy(p):
        # (row, col) pixel center -> (x, y) on the upscaled canvas
        row, col = p
        return (int(round((col + 0.5) * scale)), int(round((row + 0.5) * scale)))

    a, b = result["height_line"]
    cv2.line(canvas, to_xy(a), to_xy(b), (0, 0, 255), 2)

    a, b = result["ap_line"]
    cv2.line(canvas, to_xy(a), to_xy(b), (255, 128, 0), 2)

    cv2.circle(canvas, to_xy(result["centroid_px"]), 4, (0, 255, 255), -1)

    text = f"height {result['height_mm']:.1f} mm   AP {result['ap_mm']:.1f} mm"
    cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), canvas)


# ============================================================
# Process one label map
# ============================================================

def measure_label(
    label: np.ndarray,
    A: np.ndarray,
    t: np.ndarray,
    label_id: int,
    qc_path: Optional[Path],
    background: Optional[np.ndarray],
    options: dict,
) -> dict:
    """Cleans the vermis mask, measures it and collects quality flags."""

    spacing_row_mm = float(np.linalg.norm(A[:, 0]))
    spacing_col_mm = float(np.linalg.norm(A[:, 1]))

    record = {"spacing_row_mm": spacing_row_mm, "spacing_col_mm": spacing_col_mm}

    raw_mask = label == label_id

    if not raw_mask.any():
        return {**record, "success": False, "reason": f"no pixels with label {label_id}"}

    kept_mask, n_components, kept_fraction = largest_component_filled(raw_mask)

    result = measure_vermis(kept_mask, A, t, **options)

    flags = []

    if n_components > 1:
        flags.append("multiple_components")

    if result["elongation"] < MIN_ELONGATION:
        flags.append("axis_unstable")

    if result["axis_clamped"]:
        flags.append("axis_clamped")

    if np.isclose(spacing_row_mm, 1.0) and np.isclose(spacing_col_mm, 1.0):
        flags.append("spacing_uncalibrated")

    if qc_path is not None:
        save_qc_image(raw_mask, kept_mask, result, qc_path, background)

    return {
        **record,
        "success": True,
        "reason": "",
        "height_mm": result["height_mm"],
        "ap_mm": result["ap_mm"],
        "area_mm2": result["area_mm2"],
        "elongation": result["elongation"],
        "pca_angle_deg": result["pca_angle_deg"],
        "n_components": n_components,
        "kept_fraction": kept_fraction,
        "flags": ",".join(flags),
    }


def process_label_file(
    label_path: Path,
    image_index: Optional[Dict[str, Path]],
    label_id: int,
    qc_dir: Optional[Path],
    options: dict,
) -> dict:
    """
    Measures one label map.

    Without images the pixel spacing comes from the label file itself.
    With images (image_index given) the label must belong to an image
    of the same size; spacing and QC background then come from that
    image DICOM, and the label's own spacing is ignored (segmentation
    tools often do not store it correctly).
    """

    sample_id = sample_id_from_path(label_path)

    record = {"sample_id": sample_id, "label_path": str(label_path), "image_path": ""}

    qc_path = qc_dir / f"{sample_id}.png" if qc_dir is not None else None

    label, A, t = load_label(label_path)

    if image_index is None:
        return {**record, **measure_label(label, A, t, label_id, qc_path, None, options)}

    image_path = image_index.get(sample_id)

    if image_path is None:
        return {**record, "success": False, "reason": "no matching image DICOM"}

    record["image_path"] = str(image_path)

    pixels, spacing_xy_mm = load_dicom_pixels(image_path)

    background = to_gray(pixels)

    # some tools save the label transposed relative to the image
    if label.shape != background.shape:

        if label.shape == background.shape[::-1]:
            label = np.ascontiguousarray(label.T)
        else:
            return {
                **record,
                "success": False,
                "reason": f"label shape {label.shape} does not match image shape {background.shape}",
            }

    A, t = spacing_to_affine(spacing_xy_mm)

    return {**record, **measure_label(label, A, t, label_id, qc_path, background, options)}


# ============================================================
# Command line
# ============================================================

def parse_args():

    ap = argparse.ArgumentParser(
        description="Measure vermis height (chord through the centroid) on 2D midsagittal segmentations.",
    )

    ap.add_argument(
        "--input", required=True,
        help="Label map (NIfTI .nii/.nii.gz or DICOM) or a folder of them (searched recursively)",
    )
    ap.add_argument(
        "--output", required=True,
        help="Output folder; receives measurements.csv and the QC images in qc/",
    )
    ap.add_argument(
        "--images", default=None,
        help="Optional folder with the original image DICOMs. Labels are matched by file name "
             "(case01_seg.nii.gz <-> case01.dcm) or SeriesDescription; spacing and QC background "
             "are then taken from the image",
    )
    ap.add_argument(
        "--label", type=int, default=DEFAULT_VERMIS_LABEL,
        help=f"Label value of the vermis (default {DEFAULT_VERMIS_LABEL})",
    )
    ap.add_argument(
        "--max-axis-angle", type=float, default=DEFAULT_MAX_AXIS_ANGLE,
        help=f"Limit the measurement axis to this many degrees from the image vertical "
             f"(default {DEFAULT_MAX_AXIS_ANGLE:g}); negative = no limit",
    )
    ap.add_argument(
        "--vertical-axis", type=int, default=0, choices=[0, 1],
        help="Array axis that runs top-bottom in the image (default 0 = rows)",
    )
    ap.add_argument("--no-qc", action="store_true", help="Do not write QC images")

    return ap.parse_args()


def main():

    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    options = {
        "max_axis_angle": None if args.max_axis_angle < 0 else args.max_axis_angle,
        "vertical_axis": args.vertical_axis,
    }

    qc_dir = None if args.no_qc else output_dir / "qc"

    label_files = find_label_files(input_path)

    print(f"Found {len(label_files)} label maps in {input_path}")

    image_index = None

    if args.images:
        image_index = index_images(Path(args.images))
        print(f"Found {len(image_index)} image keys in {args.images}")

    results = []

    for i, label_path in enumerate(label_files, start=1):

        print(f"[{i}/{len(label_files)}] {label_path.name}")

        # one broken file must not stop the whole run
        try:
            result = process_label_file(label_path, image_index, args.label, qc_dir, options)
        except Exception as e:
            result = {
                "sample_id": sample_id_from_path(label_path),
                "label_path": str(label_path),
                "success": False,
                "reason": f"{type(e).__name__}: {e}",
            }

        if result["success"]:
            print(f"    height {result['height_mm']:.2f} mm, AP {result['ap_mm']:.2f} mm"
                  + (f"  [{result['flags']}]" if result["flags"] else ""))
        else:
            print(f"    failed: {result['reason']}")

        results.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "measurements.csv"

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    n_ok = int(df["success"].sum()) if len(df) else 0

    print(f"\nDone: {n_ok}/{len(df)} measured. Results in {output_csv}")


if __name__ == "__main__":
    main()
