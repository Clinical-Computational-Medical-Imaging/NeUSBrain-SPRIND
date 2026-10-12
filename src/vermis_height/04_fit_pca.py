"""
Fit a plane-specific linear PCA shape model (from build_pca_shape_model.py)
to a segmentation - today a ground-truth labelmap for validation, in the
future a CNN prediction, using the exact same interface either way.

For each input segmentation:

    1. Extract its label==2 border (dense, for evaluation; a coarser
       arc-length resample for the initial correspondence guess).

    2. Get the model's mean shape roughly onto this segmentation's own
       border via the same GPA-mean + CPD correspondence procedure
       used during training (build_shape_correspondence.py), so we
       have point-corresponding "observed" landmarks y_obs.

    3. Iteratively fit the PCA model to y_obs (classic Active Shape
       Model fitting): alternately (a) solve for the best rigid+scale
       pose mapping the current model instance onto y_obs, and
       (b) update the shape parameters b by projecting the pose-
       inverted observation onto the PCA modes, clamping b to a
       plausible range (+/- k standard deviations per mode) so the
       fit stays a valid instance of the learned shape space rather
       than free-form registration.

    4. Output the final fitted landmarks (in the segmentation's own
       pixel space) plus HD95 (and supporting distance statistics)
       between the fitted shape and the segmentation's true border.

This is preparation for landmark-based measurements (e.g. a
consistent vermis height) in a later step: once a shape reliably
fits, specific landmark indices can be used to define measurements
directly, immune to noisy CNN border pixels.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import cv2
import nibabel as nib
import pandas as pd
from scipy.spatial import cKDTree

try:
    from pycpd import DeformableRegistration
except ImportError as e:
    raise ImportError(
        "This script requires pycpd. Install with:\n"
        "    pip install pycpd --break-system-packages"
    ) from e


TARGET_LABEL_VALUE = 2
MAX_EXPECTED_UNIQUE_LABELS = 3

TARGET_DENSITY_DEFAULT = 400   # dense border points used for CPD target + HD95
CLAMP_STD_DEFAULT = 3.0        # keep fitted shape parameters within +/- this many std devs
ASM_MAX_ITERS = 50
ASM_TOL = 1e-4

CPD_ALPHA_DEFAULT = 2.0
CPD_BETA_DEFAULT = 2.0
CPD_MAX_ITERATIONS = 200

OUTLIER_MAD_MULTIPLIER_DEFAULT = 3.0
MIN_SAMPLES_FOR_AUTO_THRESHOLD = 5  # below this, an auto batch threshold isn't statistically meaningful


# ============================================================
# Loading labelmaps / PCA models
# ============================================================

def get_sample_id(path: Path) -> str:
    """
    Path.stem only strips one suffix, so "case.nii.gz" -> "case.nii".
    Strip the trailing ".nii" too, to match the sample_id convention
    used by build_shape_correspondence.py.
    """

    return path.stem[:-4] if path.stem.endswith(".nii") else path.stem


def load_label_2d(nii_path: Path) -> Optional[np.ndarray]:
    """
    Load a labelmap. No transpose/flip is applied - matches the
    preprocessing script's (H, W, 1), no-reorientation convention.
    Also returns pixel spacing (mm) read from the NIfTI affine.
    """

    try:
        nii = nib.load(str(nii_path))
        data = nii.get_fdata()
    except Exception as e:
        print(f"    Cannot read {nii_path}: {e}")
        return None

    data = np.squeeze(data)

    if data.ndim != 2:
        print(f"    Unexpected label shape {data.shape} in {nii_path}")
        return None

    return data


def get_spacing_mm(nii_path: Path) -> Tuple[float, float]:
    """
    Returns (spacing_row_mm, spacing_col_mm) read from the affine
    saved by the preprocessing script (affine[0,0] = row spacing,
    affine[1,1] = col spacing - see preprocess_ultrasound.py's
    build_affine). Falls back to (1.0, 1.0) if unavailable.
    """

    try:
        nii = nib.load(str(nii_path))
        affine = nii.affine
        return abs(affine[0, 0]), abs(affine[1, 1])
    except Exception:
        return 1.0, 1.0


def validate_label_count(
    label_2d: np.ndarray,
    max_unique_labels: int = MAX_EXPECTED_UNIQUE_LABELS,
) -> Tuple[bool, np.ndarray]:

    unique_labels = np.unique(label_2d)
    return len(unique_labels) <= max_unique_labels, unique_labels


def load_pca_model(npz_path: Path) -> Dict[str, np.ndarray]:

    data = np.load(npz_path, allow_pickle=True)

    return {
        "mean": data["mean"],
        "components": data["components"],
        "eigenvalues": data["eigenvalues"],
        "n_landmarks": int(data["n_landmarks"]),
    }


# ============================================================
# Border extraction (same conventions as build_shape_correspondence.py)
# ============================================================

def resample_contour(
    contour: np.ndarray,
    n_points: int,
) -> Optional[np.ndarray]:

    closed = np.vstack([contour, contour[0]])

    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    if total_length == 0:
        return None

    sample_dist = np.linspace(0, total_length, n_points, endpoint=False)

    resampled = np.empty((n_points, 2), dtype=np.float64)

    for i, d in enumerate(sample_dist):

        idx = np.searchsorted(cum_length, d, side="right") - 1
        idx = int(np.clip(idx, 0, len(seg_lengths) - 1))

        seg_start = closed[idx]
        seg_end = closed[idx + 1]
        seg_len = seg_lengths[idx]

        t = 0.0 if seg_len == 0 else (d - cum_length[idx]) / seg_len

        resampled[i] = seg_start + t * (seg_end - seg_start)

    return resampled


def extract_border_points(
    label_2d: np.ndarray,
    n_points: int,
    target_label: int = TARGET_LABEL_VALUE,
) -> Optional[np.ndarray]:

    mask = (label_2d == target_label).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    contour = contour.reshape(-1, 2).astype(np.float64)

    if len(contour) < 3:
        return None

    return resample_contour(contour, n_points)


# ============================================================
# Procrustes alignment
# ============================================================

def procrustes_align(
    shape: np.ndarray,
    reference: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Similarity-align `shape` onto `reference` (no reflection).
    Returns (aligned_shape, transform); see build_pca_shape_model.py
    for the same convention:
        aligned = scale * (shape - shape_mean) @ R + reference_mean
    """

    shape_mean = shape.mean(axis=0)
    reference_mean = reference.mean(axis=0)

    shape_c = shape - shape_mean
    ref_c = reference - reference_mean

    norm_shape = np.linalg.norm(shape_c)
    norm_ref = np.linalg.norm(ref_c)

    if norm_shape == 0 or norm_ref == 0:
        transform = {
            "R": np.eye(2), "scale": 1.0,
            "shape_mean": shape_mean, "reference_mean": reference_mean,
        }
        return shape.copy(), transform

    shape_n = shape_c / norm_shape
    ref_n = ref_c / norm_ref

    U, S, Vt = np.linalg.svd(shape_n.T @ ref_n)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    scale = S.sum() * (norm_ref / norm_shape)

    aligned = (shape_n @ R) * scale * norm_shape + reference_mean

    transform = {
        "R": R, "scale": scale,
        "shape_mean": shape_mean, "reference_mean": reference_mean,
    }

    return aligned, transform


# ============================================================
# Step A: establish landmark correspondence on the new segmentation
# (mirrors build_shape_correspondence.py's per-sample CPD step)
# ============================================================

def register_mean_shape_to_target(
    mean_shape: np.ndarray,
    target_dense: np.ndarray,
    target_rough: np.ndarray,
    alpha: float,
    beta: float,
    max_iterations: int,
) -> np.ndarray:
    """
    mean_shape: (N, 2) GPA mean shape (zero-centered, unit scale;
                 "shape space").
    target_rough: (N, 2) arc-length resampled border of the new
                 segmentation - same point count as mean_shape, used
                 only to get a rigid initial guess.
    target_dense: (M, 2) dense border of the new segmentation, used
                 as the actual CPD fit target.

    Returns (N, 2) corresponding landmarks on the new segmentation.
    """

    initial_guess, _ = procrustes_align(mean_shape, target_rough)

    reg = DeformableRegistration(
        X=target_dense, Y=initial_guess,
        alpha=alpha, beta=beta, max_iterations=max_iterations,
    )

    deformed_points, _ = reg.register()

    return deformed_points


# ============================================================
# Step B: ASM-style PCA fitting (pose + shape parameters)
# ============================================================

def fit_pca_to_points(
    pca_model: Dict[str, np.ndarray],
    y_obs: np.ndarray,
    clamp_std: float = CLAMP_STD_DEFAULT,
    max_iters: int = ASM_MAX_ITERS,
    tol: float = ASM_TOL,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Iteratively solve for PCA shape parameters `b` and pose (R, scale,
    translation) that best explain the corresponding observed points
    y_obs, constraining b to stay within a plausible range of the
    learned shape space (classic Active Shape Model fitting).

    Returns (fitted_shape_in_target_space, b, pose_transform).
    """

    mean_flat = pca_model["mean"]
    components = pca_model["components"]
    eigenvalues = pca_model["eigenvalues"]
    n_landmarks = pca_model["n_landmarks"]

    std = np.sqrt(np.clip(eigenvalues, 0, None))
    b = np.zeros(len(eigenvalues))

    for iteration in range(max_iters):

        x_flat = mean_flat + b @ components
        x = x_flat.reshape(n_landmarks, 2)

        # Model shapes are zero-centroid by construction (GPA), so
        # this pose maps x -> y_obs directly.
        _, transform = procrustes_align(x, y_obs)

        R = transform["R"]
        scale = transform["scale"]
        y_obs_mean = transform["reference_mean"]

        # Invert the pose to bring y_obs into model ("shape") space.
        y_prime = (y_obs - y_obs_mean) @ R.T / scale

        dx = y_prime.reshape(-1) - mean_flat

        b_new = components @ dx

        # Keep the fit a plausible instance of the learned shape
        # space rather than letting it deform arbitrarily.
        b_new = np.clip(b_new, -clamp_std * std, clamp_std * std)

        delta = np.linalg.norm(b_new - b)
        b = b_new

        if delta < tol:
            break

    x_final_flat = mean_flat + b @ components
    x_final = x_final_flat.reshape(n_landmarks, 2)

    fitted_shape, final_transform = procrustes_align(x_final, y_obs)

    return fitted_shape, b, final_transform


# ============================================================
# Step C: HD95 and supporting distance statistics
# ============================================================

def compute_hd95(
    points_a: np.ndarray,
    points_b: np.ndarray,
    spacing: Tuple[float, float] = (1.0, 1.0),
) -> Dict[str, float]:
    """
    Symmetric HD95 (the standard segmentation-evaluation convention):
        hd95 = max( P95( d(a -> b) ), P95( d(b -> a) ) )

    Points are (x, y) pixel coordinates; scaling by `spacing`
    (spacing_x_mm, spacing_y_mm) converts distances to mm before
    computing Hausdorff statistics.
    """

    sx, sy = spacing

    a_scaled = points_a * np.array([sx, sy])
    b_scaled = points_b * np.array([sx, sy])

    tree_b = cKDTree(b_scaled)
    d_ab, _ = tree_b.query(a_scaled)

    tree_a = cKDTree(a_scaled)
    d_ba, _ = tree_a.query(b_scaled)

    all_d = np.concatenate([d_ab, d_ba])

    return {
        "hd95_mm": float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))),
        "hd100_max_mm": float(max(d_ab.max(), d_ba.max())),
        "mean_surface_distance_mm": float(all_d.mean()),
        "median_surface_distance_mm": float(np.median(all_d)),
    }


def compute_outlier_threshold(values: np.ndarray, mad_multiplier: float) -> float:
    """
    Robust outlier threshold: median + mad_multiplier * MAD (scaled to
    approximate std under normality). Robust to the outliers
    themselves inflating the threshold, unlike mean/std.
    """

    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826

    if mad == 0:
        return median * 1.5 + 1e-6

    return median + mad_multiplier * mad


# ============================================================
# Orientation correction (rigid re-orientation of the whole fit)
# ============================================================

def signed_angle_from_vertical_deg(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """
    Signed angle of the line through point_a/point_b relative to
    vertical, in (-90, 90]. A LINE has no inherent direction, so this
    is normalized to not depend on which point is "a" vs "b".
    """

    # Points are in the transposed NIfTI array space of
    # 01_preprocess_data.py: point[0] ("x") = original image row
    # (vertical), point[1] ("y") = original image column (horizontal).
    d_vertical = point_a[0] - point_b[0]
    d_horizontal = point_a[1] - point_b[1]

    angle = float(np.degrees(np.arctan2(d_horizontal, d_vertical)))

    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180

    return angle


def rotate_points(points: np.ndarray, degrees: float, center: np.ndarray) -> np.ndarray:

    theta = np.radians(degrees)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return (points - center) @ R.T + center


def correct_orientation(
    fitted_shape: np.ndarray,
    landmark_a: int,
    landmark_b: int,
    max_angle_deviation_deg: float,
) -> Dict:
    """
    The five measurement candidates around landmark_a/landmark_b are
    NOT independent - they're all built from nearby points deformed
    by the same smooth CPD field and placed by the same estimated
    rigid pose, so a pose error shows up as a consistent (if not
    perfectly parallel) tilt across all of them, not just one bad
    point. Since a rotation error doesn't distort the shape - it only
    misorients it - it can be corrected exactly and safely by
    rigidly rotating the WHOLE fitted shape about its own centroid,
    using the one thing known for certain: this measurement axis
    should be close to vertical.

    Only acts if the current angle exceeds max_angle_deviation_deg;
    otherwise returns the shape unchanged.
    """

    point_a = fitted_shape[landmark_a]
    point_b = fitted_shape[landmark_b]

    angle_before = signed_angle_from_vertical_deg(point_a, point_b)

    if abs(angle_before) <= max_angle_deviation_deg:

        return {
            "shape": fitted_shape,
            "corrected": False,
            "correction_deg": 0.0,
            "angle_before_deg": angle_before,
            "angle_after_deg": angle_before,
        }

    centroid = fitted_shape.mean(axis=0)
    # The angle is measured from the x axis towards y, so rotating
    # by -angle brings the axis back onto x (= original vertical).
    correction_deg = -angle_before

    corrected_shape = rotate_points(fitted_shape, correction_deg, centroid)

    angle_after = signed_angle_from_vertical_deg(
        corrected_shape[landmark_a], corrected_shape[landmark_b],
    )

    return {
        "shape": corrected_shape,
        "corrected": True,
        "correction_deg": correction_deg,
        "angle_before_deg": angle_before,
        "angle_after_deg": angle_after,
    }


# ============================================================
# Per-sample pipeline
# ============================================================

def fit_single_segmentation(
    label_path: Path,
    pca_model: Dict[str, np.ndarray],
    target_density: int,
    clamp_std: float,
    cpd_alpha: float,
    cpd_beta: float,
    cpd_max_iterations: int,
    landmark_a: Optional[int] = None,
    landmark_b: Optional[int] = None,
    orientation_max_angle_deg: float = 20.0,
) -> Optional[Dict]:

    label_2d = load_label_2d(label_path)

    if label_2d is None:
        return {"sample_id": get_sample_id(label_path), "success": False, "reason": "could not load label"}

    is_valid, unique_labels = validate_label_count(label_2d)

    if not is_valid:

        print(
            f"    ERROR: {get_sample_id(label_path)} has {len(unique_labels)} unique labels "
            f"{unique_labels.tolist()} (max expected: {MAX_EXPECTED_UNIQUE_LABELS}); "
            f"this indicates an error in the labelmap - skipping"
        )

        return {
            "sample_id": get_sample_id(label_path), "success": False,
            "reason": f"labelmap has {len(unique_labels)} unique labels {unique_labels.tolist()}",
        }

    n_landmarks = pca_model["n_landmarks"]

    target_rough = extract_border_points(label_2d, n_landmarks)
    target_dense = extract_border_points(label_2d, target_density)

    if target_rough is None or target_dense is None:
        return {"sample_id": get_sample_id(label_path), "success": False, "reason": "border extraction failed (label 2 not found)"}

    mean_shape = pca_model["mean"].reshape(n_landmarks, 2)

    try:
        y_obs = register_mean_shape_to_target(
            mean_shape, target_dense, target_rough,
            alpha=cpd_alpha, beta=cpd_beta, max_iterations=cpd_max_iterations,
        )
    except Exception as e:
        return {"sample_id": get_sample_id(label_path), "success": False, "reason": f"CPD correspondence failed: {e}"}

    fitted_shape, b, pose = fit_pca_to_points(
        pca_model, y_obs, clamp_std=clamp_std,
    )

    orientation_info = None

    if landmark_a is not None and landmark_b is not None:

        orientation_info = correct_orientation(
            fitted_shape, landmark_a, landmark_b, orientation_max_angle_deg,
        )
        fitted_shape = orientation_info["shape"]

        if orientation_info["corrected"]:
            print(
                f"    {get_sample_id(label_path)}: measurement axis was "
                f"{orientation_info['angle_before_deg']:+.1f} deg from vertical - "
                f"rigidly re-oriented by {orientation_info['correction_deg']:+.1f} deg "
                f"(now {orientation_info['angle_after_deg']:+.1f} deg)"
            )

    spacing = get_spacing_mm(label_path)  # (row_mm, col_mm) i.e. (y, x)
    spacing_xy = (spacing[1], spacing[0])  # reorder to (x, y) to match point ordering

    # Densify the fitted landmark polygon so HD95 compares two fairly
    # dense curves rather than sparse landmarks vs. a dense border.
    fitted_dense = resample_contour(fitted_shape, target_density)

    hd_stats = compute_hd95(fitted_dense, target_dense, spacing=spacing_xy)

    result = {
        "sample_id": get_sample_id(label_path),
        "success": True,
        "fitted_shape": fitted_shape,
        "b": b,
        "pose": pose,
        **hd_stats,
    }

    if orientation_info is not None:
        result["orientation_corrected"] = orientation_info["corrected"]
        result["orientation_correction_deg"] = orientation_info["correction_deg"]
        result["angle_before_deg"] = orientation_info["angle_before_deg"]
        result["angle_after_deg"] = orientation_info["angle_after_deg"]

    return result


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pca-model", required=True, help="Path to pca_shape_model.npz for this plane")
    ap.add_argument("--input", required=True, help="A single labelmap file, or a directory of them")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-density", type=int, default=TARGET_DENSITY_DEFAULT)
    ap.add_argument("--clamp-std", type=float, default=CLAMP_STD_DEFAULT)
    ap.add_argument("--cpd-alpha", type=float, default=CPD_ALPHA_DEFAULT)
    ap.add_argument("--cpd-beta", type=float, default=CPD_BETA_DEFAULT)
    ap.add_argument("--cpd-max-iterations", type=int, default=CPD_MAX_ITERATIONS)
    ap.add_argument(
        "--max-hd95-mm", type=float, default=None,
        help="Fixed HD95 (mm) cutoff for a good fit. Recommended once you have a baseline from validation data - required for meaningful filtering when fitting a single new segmentation. If omitted and >= 5 samples are processed in this run, an automatic median+MAD threshold is used instead.",
    )
    ap.add_argument(
        "--outlier-mad-multiplier", type=float, default=OUTLIER_MAD_MULTIPLIER_DEFAULT,
        help="Used only when --max-hd95-mm is omitted and this run has >= 5 samples: threshold = median + this * MAD of HD95 across the batch.",
    )
    ap.add_argument(
        "--landmark-a", type=int, default=None,
        help="Index of one vermis-height measurement landmark. If given together with --landmark-b, the fitted shape is rigidly re-oriented (about its own centroid) whenever this measurement axis deviates too far from vertical - correcting a pose/rotation bias in the fit without distorting the shape itself.",
    )
    ap.add_argument(
        "--landmark-b", type=int, default=None,
        help="Index of the other vermis-height measurement landmark; see --landmark-a.",
    )
    ap.add_argument(
        "--orientation-max-angle-deg", type=float, default=20.0,
        help="Max allowed deviation from vertical (degrees) for the landmark-a/landmark-b axis before the whole fitted shape is rigidly re-oriented (default 20). Only used if --landmark-a/--landmark-b are given.",
    )

    args = ap.parse_args()

    pca_model = load_pca_model(Path(args.pca_model))

    input_path = Path(args.input)

    if input_path.is_dir():
        label_files = sorted(input_path.glob("*.nii.gz")) + sorted(input_path.glob("*.nii"))
    else:
        label_files = [input_path]

    if not label_files:
        print(f"No labelmaps found at {input_path}")
        return

    output_dir = Path(args.output_dir)
    landmarks_dir = output_dir / "fitted_landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, label_path in enumerate(label_files):

        print(f"[{i + 1}/{len(label_files)}] Fitting {label_path.name}...")

        result = fit_single_segmentation(
            label_path, pca_model,
            target_density=args.target_density,
            clamp_std=args.clamp_std,
            cpd_alpha=args.cpd_alpha,
            cpd_beta=args.cpd_beta,
            cpd_max_iterations=args.cpd_max_iterations,
            landmark_a=args.landmark_a,
            landmark_b=args.landmark_b,
            orientation_max_angle_deg=args.orientation_max_angle_deg,
        )

        if result is None or not result.get("success"):
            reason = result.get("reason", "unknown") if result else "unknown"
            print(f"    FAILED: {reason}")
            results.append({"sample_id": get_sample_id(label_path), "success": False, "reason": reason})
            continue

        sample_id = result["sample_id"]

        # Save fitted landmarks (post-orientation-correction, if applied)
        df = pd.DataFrame(result["fitted_shape"], columns=["x", "y"])
        df.insert(0, "landmark_index", np.arange(len(df)))
        df.to_csv(landmarks_dir / f"{sample_id}.csv", index=False)

        print(
            f"    HD95: {result['hd95_mm']:.3f} mm | "
            f"mean surface dist: {result['mean_surface_distance_mm']:.3f} mm"
        )

        row = {
            "sample_id": sample_id,
            "success": True,
            "hd95_mm": result["hd95_mm"],
            "hd100_max_mm": result["hd100_max_mm"],
            "mean_surface_distance_mm": result["mean_surface_distance_mm"],
            "median_surface_distance_mm": result["median_surface_distance_mm"],
        }

        if "orientation_corrected" in result:
            row["orientation_corrected"] = result["orientation_corrected"]
            row["orientation_correction_deg"] = result["orientation_correction_deg"]
            row["angle_before_deg"] = result["angle_before_deg"]
            row["angle_after_deg"] = result["angle_after_deg"]

        results.append(row)

    results_df = pd.DataFrame(results)

    successful = results_df[results_df["success"] == True] if len(results_df) else results_df

    if len(successful) > 0:

        hd95_values = successful["hd95_mm"].to_numpy()

        if args.max_hd95_mm is not None:
            threshold = args.max_hd95_mm
            print(f"\nUsing fixed HD95 quality threshold: {threshold:.3f} mm")
        elif len(successful) >= MIN_SAMPLES_FOR_AUTO_THRESHOLD:
            threshold = compute_outlier_threshold(hd95_values, args.outlier_mad_multiplier)
            print(
                f"\nAuto HD95 quality threshold: {threshold:.3f} mm "
                f"(median={np.median(hd95_values):.3f} mm, multiplier={args.outlier_mad_multiplier})"
            )
        else:
            threshold = None
            print(
                f"\nOnly {len(successful)} sample(s) and no --max-hd95-mm given: "
                f"cannot determine a meaningful quality threshold. All fits marked "
                f"fit_quality_ok=True - determine a threshold from a validation batch "
                f"first (see hd95_summary.csv from a larger run) and pass it via "
                f"--max-hd95-mm for real filtering."
            )

        if threshold is not None:
            results_df.loc[results_df["success"] == True, "fit_quality_ok"] = (
                results_df.loc[results_df["success"] == True, "hd95_mm"] <= threshold
            )
        else:
            results_df.loc[results_df["success"] == True, "fit_quality_ok"] = True

        n_bad = (~results_df.loc[results_df["success"] == True, "fit_quality_ok"]).sum()
        if n_bad > 0:
            print(
                f"WARNING: {n_bad} fit(s) flagged fit_quality_ok=False - do not use "
                f"these to compute vermis height measurements"
            )

    results_df.to_csv(output_dir / "fit_results.csv", index=False)

    print("\n==========================================")
    print("Fitting complete")
    print("==========================================")
    print(f"Successful: {len(successful)}/{len(results_df)}")

    if len(successful) > 0:

        hd95_values = successful["hd95_mm"]

        summary = {
            "n_samples": len(successful),
            "hd95_mean_mm": hd95_values.mean(),
            "hd95_std_mm": hd95_values.std(),
            "hd95_median_mm": hd95_values.median(),
            "hd95_min_mm": hd95_values.min(),
            "hd95_max_mm": hd95_values.max(),
            "quality_threshold_mm": threshold if threshold is not None else np.nan,
            "n_fit_quality_ok": int(results_df["fit_quality_ok"].sum()) if "fit_quality_ok" in results_df else len(successful),
        }

        pd.DataFrame([summary]).to_csv(output_dir / "hd95_summary.csv", index=False)

        print(
            f"HD95 (mm): mean={summary['hd95_mean_mm']:.3f}  "
            f"std={summary['hd95_std_mm']:.3f}  "
            f"median={summary['hd95_median_mm']:.3f}  "
            f"min={summary['hd95_min_mm']:.3f}  "
            f"max={summary['hd95_max_mm']:.3f}"
        )


if __name__ == "__main__":
    main()