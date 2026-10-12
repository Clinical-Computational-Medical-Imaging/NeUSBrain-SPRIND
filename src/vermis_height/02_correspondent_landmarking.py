"""
Build point-corresponding landmarks over segmentation borders.

Pipeline, run once per class ("sagittal", "coronal"):

    1. Pick a REFERENCE TEMPLATE: one sample's border, resampled to N
       points by arc length (just to fix a point count/order - CPD
       below does not require this resampling to give real
       correspondence). The reference is chosen as the sample whose
       border length is closest to the population median, so an
       atypical/outlier shape isn't accidentally used as the anchor.
       (--reference-sample-id can override this choice.)

    2. For every sample, in two stages:
         a. RIGID CPD: pycpd.RigidRegistration finds the best
            rotation + scale + translation placing the template onto
            that sample's own dense border - directly, via the same
            robust probabilistic point-matching CPD uses, with no
            need for any pre-established point correspondence (unlike
            arc-length parameterization, which silently breaks if two
            contours' starting points/directions don't happen to
            line up).
         b. DEFORMABLE CPD ("normal" CPD): starting from that rigid
            placement, non-rigidly deforms the template's N points
            onto the sample's actual border. The result: N points per
            sample, always in the same order - i.e. corresponding
            landmarks.

    3. Optionally (--build-atlas): treat step 2's results as a first
       pass. Run GPA (Generalized Procrustes Analysis) over the
       well-converged samples from that pass to build a proper
       population mean-shape atlas, then repeat step 2 using the
       atlas as the template instead of the single reference sample.
       This gives a more representative template than an arbitrary
       single sample, at the cost of a second full CPD pass.

    4. Quality control: for every sample, measure how far its
       corresponded points actually ended up from its own true dense
       border. Samples with a poor residual (a robust median+MAD
       outlier rule, or a fixed --max-correspondence-residual) are
       excluded from the final output - this is exactly what catches
       a wrong-plane or otherwise misclassified image: CPD simply
       cannot converge such a case onto the template well, and it
       shows up as a clear outlier.

Requires: pip install pycpd
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
    from pycpd import DeformableRegistration, RigidRegistration
except ImportError as e:
    raise ImportError(
        "This script requires pycpd. Install with:\n"
        "    pip install pycpd --break-system-packages"
    ) from e


# ============================================================
# Configuration
# ============================================================

CLASSES = ("sagittal", "coronal")

# Only label value 2 defines the border we build correspondences over.
TARGET_LABEL_VALUE = 2

# A valid labelmap should have at most 3 unique values (e.g. background=0
# plus up to 2 foreground classes). More than that indicates a labeling
# error, so such samples are skipped and logged.
MAX_EXPECTED_UNIQUE_LABELS = 3

N_LANDMARKS_DEFAULT = 100          # points on the template
TARGET_DENSITY_DEFAULT = 400       # points describing each sample's own border, used as the CPD target

GPA_MAX_ITERS = 100
GPA_TOL = 1e-8

RIGID_MAX_ITERATIONS_DEFAULT = 100
CPD_ALPHA_DEFAULT = 2.0            # motion coherence (higher = smoother/stiffer deformation)
CPD_BETA_DEFAULT = 2.0             # width of the smoothness Gaussian kernel
CPD_MAX_ITERATIONS = 200

# Outlier rule for excluding poorly-converged samples: exclude if
# residual > median + OUTLIER_MAD_MULTIPLIER * MAD (median absolute
# deviation). Robust to the outliers themselves skewing the threshold,
# unlike a mean/std-based rule. Can be overridden with a fixed
# --max-correspondence-residual instead.
OUTLIER_MAD_MULTIPLIER_DEFAULT = 3.0


# ============================================================
# Loading labelmaps (matches the preprocessing script's (H, W, 1),
# no-transpose/flip convention)
# ============================================================

def load_label_2d(nii_path: Path) -> Optional[np.ndarray]:

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


# ============================================================
# Border / contour extraction
# ============================================================

def resample_contour(
    contour: np.ndarray,
    n_points: int,
) -> Optional[np.ndarray]:
    """
    Resample a closed polyline (M, 2) to n_points, equally spaced by
    arc length, starting from the contour's first point.
    """

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


def validate_label_count(
    label_2d: np.ndarray,
    max_unique_labels: int = MAX_EXPECTED_UNIQUE_LABELS,
) -> Tuple[bool, np.ndarray]:
    """
    A valid labelmap should have at most `max_unique_labels` distinct
    values (e.g. background=0 plus up to 2 foreground classes).
    More than that means something went wrong upstream (e.g. stray
    pixels, wrong dtype/scaling, a corrupted mask) - such labelmaps
    are treated as erroneous and should be skipped rather than
    silently processed.

    Returns (is_valid, unique_labels).
    """

    unique_labels = np.unique(label_2d)

    is_valid = len(unique_labels) <= max_unique_labels

    return is_valid, unique_labels


def extract_border_points(
    label_2d: np.ndarray,
    n_points: int,
    target_label: int = TARGET_LABEL_VALUE,
) -> Optional[np.ndarray]:
    """
    Extract the outer border of the region matching `target_label`
    (only label value 2, by default) as an ordered, closed set of
    n_points points, equally spaced by arc length. Points are (x, y)
    in the labelmap's own pixel coordinates.
    """

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


def contour_arc_length(contour: np.ndarray) -> float:

    closed = np.vstack([contour, contour[0]])
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


# ============================================================
# Reference template selection
# ============================================================

def select_reference_sample(
    sample_ids: List[str],
    rough_landmarks: List[np.ndarray],
    forced_sample_id: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """
    Choose one sample's border as the initial template for CPD. By
    default: the sample whose border arc length is closest to the
    population median - a simple, robust way to avoid picking an
    atypically small/large/oddly-shaped sample as the anchor.
    """

    if forced_sample_id is not None:

        if forced_sample_id not in sample_ids:
            raise ValueError(f"--reference-sample-id {forced_sample_id} not found among loaded samples")

        idx = sample_ids.index(forced_sample_id)
        print(f"    Using forced reference sample: {forced_sample_id}")
        return rough_landmarks[idx], forced_sample_id

    lengths = np.array([contour_arc_length(r) for r in rough_landmarks])
    median_length = np.median(lengths)

    best_idx = int(np.argmin(np.abs(lengths - median_length)))

    print(
        f"    Auto-selected reference sample: {sample_ids[best_idx]} "
        f"(border length {lengths[best_idx]:.1f}px, population median {median_length:.1f}px)"
    )

    return rough_landmarks[best_idx], sample_ids[best_idx]


# ============================================================
# Procrustes alignment / GPA (used only for optional atlas building)
# ============================================================

def procrustes_align(
    shape: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Similarity-align `shape` onto `reference` (translation, rotation,
    uniform scale), matched point-for-point by index. No reflection
    is allowed, since a mirrored segmentation border isn't a valid
    anatomical alignment.
    """

    shape_c = shape - shape.mean(axis=0)
    ref_c = reference - reference.mean(axis=0)

    norm_shape = np.linalg.norm(shape_c)
    norm_ref = np.linalg.norm(ref_c)

    if norm_shape == 0 or norm_ref == 0:
        return shape.copy()

    shape_n = shape_c / norm_shape
    ref_n = ref_c / norm_ref

    U, S, Vt = np.linalg.svd(shape_n.T @ ref_n)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    scale = S.sum() * (norm_ref / norm_shape)

    aligned = (shape_n @ R) * scale * norm_shape + reference.mean(axis=0)

    return aligned


def generalized_procrustes_analysis(
    shapes: List[np.ndarray],
    max_iters: int = GPA_MAX_ITERS,
    tol: float = GPA_TOL,
) -> np.ndarray:
    """
    Iteratively align every shape (same point count / index order) to
    the common mean shape until the mean stabilizes. Returns the
    final mean shape (N, 2), centered at the origin. Used only to
    build the optional GPA atlas, from already CPD-corresponded
    shapes (which have real, not just index-based, correspondence).
    """

    mean_shape = shapes[0] - shapes[0].mean(axis=0)
    norm = np.linalg.norm(mean_shape)
    if norm > 0:
        mean_shape = mean_shape / norm

    for iteration in range(max_iters):

        aligned = [procrustes_align(s, mean_shape) for s in shapes]

        new_mean = np.mean(aligned, axis=0)
        new_mean = new_mean - new_mean.mean(axis=0)

        norm = np.linalg.norm(new_mean)
        if norm > 0:
            new_mean = new_mean / norm

        diff = np.linalg.norm(new_mean - mean_shape)
        mean_shape = new_mean

        if diff < tol:
            print(f"    GPA converged after {iteration + 1} iterations (delta={diff:.2e})")
            break

    else:
        print(f"    GPA reached max_iters ({max_iters}) without fully converging")

    return mean_shape


# ============================================================
# Two-stage CPD correspondence: rigid, then deformable
# ============================================================

def rigid_cpd_align(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int,
) -> np.ndarray:
    """
    Rigidly align source_points (the template) onto target_points
    (a sample's dense border) - rotation + scale + translation only,
    no point correspondence assumed beforehand. Returns the
    transformed source points (same count/order as source_points).
    """

    reg = RigidRegistration(X=target_points, Y=source_points, max_iterations=max_iterations)
    aligned_points, _ = reg.register()

    return aligned_points


def deformable_cpd_align(
    source_points: np.ndarray,
    target_points: np.ndarray,
    alpha: float,
    beta: float,
    max_iterations: int,
) -> np.ndarray:
    """
    Non-rigidly deform source_points (already rigidly placed near
    target_points) onto target_points. Returns the deformed points -
    same count/order as source_points, i.e. the corresponding
    landmarks for this sample.
    """

    reg = DeformableRegistration(
        X=target_points, Y=source_points,
        alpha=alpha, beta=beta, max_iterations=max_iterations,
    )
    deformed_points, _ = reg.register()

    return deformed_points


def compute_correspondence_quality(
    corresponding: np.ndarray,
    target_points: np.ndarray,
) -> dict:
    """
    How well did CPD actually converge for this sample? Measures the
    distance from each corresponded (deformed) point to the nearest
    point on the sample's own true dense border. Low residual = the
    corresponded points really do lie on the border = good
    convergence. High residual = CPD failed to fit this sample well
    (e.g. wrong plane, a badly cropped image, a corrupted label) and
    should not be trusted.

    Distances are in pixels (all samples share the same processed
    resolution, so they're directly comparable to each other, even
    without physical spacing).
    """

    tree = cKDTree(target_points)
    distances, _ = tree.query(corresponding)

    return {
        "residual_mean_px": float(distances.mean()),
        "residual_max_px": float(distances.max()),
    }


def compute_outlier_threshold(
    values: np.ndarray,
    mad_multiplier: float,
) -> float:
    """
    Robust outlier threshold: median + mad_multiplier * MAD (median
    absolute deviation), scaled by 1.4826 so MAD approximates the
    standard deviation for normally-distributed data. Robust to the
    outliers themselves inflating the threshold, unlike mean/std.
    """

    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826

    if mad == 0:
        # Degenerate case (all values identical/near-identical):
        # fall back to a small relative margin instead of a zero-width
        # threshold that would flag everything as an outlier.
        return median * 1.5 + 1e-6

    return median + mad_multiplier * mad


def run_correspondence_pass(
    sample_ids: List[str],
    label_arrays: List[np.ndarray],
    template: np.ndarray,
    target_density: int,
    rigid_max_iterations: int,
    cpd_alpha: float,
    cpd_beta: float,
    cpd_max_iterations: int,
) -> List[dict]:
    """
    Runs rigid CPD then deformable CPD for every sample against the
    given template. Returns one dict per sample: either
    {"sample_id", "success": True, "corresponding", "residual_mean_px",
    "residual_max_px"} or {"sample_id", "success": False, "reason"}.
    """

    results = []

    for i, (sample_id, label_2d) in enumerate(zip(sample_ids, label_arrays)):

        target_points = extract_border_points(label_2d, target_density)

        if target_points is None:
            results.append({
                "sample_id": sample_id, "success": False,
                "reason": "dense border extraction failed (label 2 not found)",
            })
            continue

        try:
            rigidly_aligned = rigid_cpd_align(template, target_points, max_iterations=rigid_max_iterations)
            corresponding = deformable_cpd_align(
                rigidly_aligned, target_points,
                alpha=cpd_alpha, beta=cpd_beta, max_iterations=cpd_max_iterations,
            )
        except Exception as e:
            results.append({"sample_id": sample_id, "success": False, "reason": f"CPD failed: {e}"})
            continue

        quality = compute_correspondence_quality(corresponding, target_points)

        results.append({
            "sample_id": sample_id, "success": True,
            "corresponding": corresponding, **quality,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(sample_ids):
            print(f"\r    CPD: {i + 1:6d}/{len(sample_ids)}", end="", flush=True)

    print()

    return results


# ============================================================
# Per-class pipeline
# ============================================================

def process_class(
    labels_dir: Path,
    output_dir: Path,
    class_name: str,
    n_landmarks: int,
    target_density: int,
    rigid_max_iterations: int,
    cpd_alpha: float,
    cpd_beta: float,
    cpd_max_iterations: int,
    outlier_mad_multiplier: float,
    max_correspondence_residual: Optional[float],
    build_atlas: bool,
    reference_sample_id: Optional[str],
):

    print(f"\n=== Processing class: {class_name} ===")

    label_files = sorted(labels_dir.glob("*.nii.gz")) + sorted(labels_dir.glob("*.nii"))

    if not label_files:
        print(f"    No labelmaps found in {labels_dir}")
        return

    # --------------------------------------------------------
    # Step 1: load + validate + a fixed-count border per sample
    # (needed to pick the reference template; not used for
    # correspondence itself - that's CPD's job).
    # --------------------------------------------------------

    sample_ids: List[str] = []
    label_arrays: List[np.ndarray] = []
    rough_landmarks: List[np.ndarray] = []
    failures: List[dict] = []

    for path in label_files:

        label_2d = load_label_2d(path)

        if label_2d is None:
            failures.append({"sample_id": path.stem, "reason": "could not load label"})
            continue

        is_valid, unique_labels = validate_label_count(label_2d)

        if not is_valid:

            print(
                f"    ERROR: {path.stem} has {len(unique_labels)} unique labels "
                f"{unique_labels.tolist()} (max expected: {MAX_EXPECTED_UNIQUE_LABELS}); "
                f"this indicates an error in the labelmap - skipping"
            )

            failures.append(
                {
                    "sample_id": path.stem,
                    "reason": (
                        f"labelmap has {len(unique_labels)} unique labels "
                        f"{unique_labels.tolist()}, expected at most "
                        f"{MAX_EXPECTED_UNIQUE_LABELS}"
                    ),
                }
            )
            continue

        landmarks = extract_border_points(label_2d, n_landmarks)

        if landmarks is None:
            failures.append({"sample_id": path.stem, "reason": "border extraction failed (label 2 not found)"})
            continue

        sample_ids.append(path.stem.replace(".nii", ""))
        label_arrays.append(label_2d)
        rough_landmarks.append(landmarks)

    print(f"    Loaded {len(sample_ids)}/{len(label_files)} samples")

    if len(rough_landmarks) < 2:
        print("    Not enough samples to build correspondence; skipping class")
        return

    # --------------------------------------------------------
    # Step 2: reference template
    # --------------------------------------------------------

    template, reference_id = select_reference_sample(sample_ids, rough_landmarks, reference_sample_id)

    # --------------------------------------------------------
    # Step 3: pass 1 - rigid CPD then deformable CPD, every sample
    # against the reference template.
    # --------------------------------------------------------

    print("    Pass 1: rigid + deformable CPD against the reference template...")

    pass1_results = run_correspondence_pass(
        sample_ids, label_arrays, template, target_density,
        rigid_max_iterations, cpd_alpha, cpd_beta, cpd_max_iterations,
    )

    final_results = pass1_results
    template_used = template
    used_atlas = False

    # --------------------------------------------------------
    # Step 4 (optional): build a GPA atlas from pass 1's
    # well-converged samples, then re-run against the atlas.
    # --------------------------------------------------------

    if build_atlas:

        pass1_successes = [r for r in pass1_results if r["success"]]

        if len(pass1_successes) < 2:

            print("    Not enough successful pass-1 correspondences to build an atlas; keeping single-reference template")

        else:

            pass1_residuals = np.array([r["residual_mean_px"] for r in pass1_successes])
            pass1_threshold = (
                max_correspondence_residual if max_correspondence_residual is not None
                else compute_outlier_threshold(pass1_residuals, outlier_mad_multiplier)
            )

            atlas_inputs = [r["corresponding"] for r in pass1_successes if r["residual_mean_px"] <= pass1_threshold]

            print(
                f"    Building GPA atlas from {len(atlas_inputs)}/{len(pass1_successes)} "
                f"well-converged pass-1 samples (residual threshold {pass1_threshold:.2f}px)..."
            )

            if len(atlas_inputs) < 2:

                print("    Not enough well-converged pass-1 samples to build an atlas; keeping single-reference template")

            else:

                atlas = generalized_procrustes_analysis(atlas_inputs)

                # GPA's mean shape is unit-norm and centered at the
                # origin (normalized "shape space"), while the actual
                # samples are in real pixel scale/position. Rescale
                # and retranslate the atlas to match the average
                # size/position of the samples that built it, so
                # rigid CPD starts from a template that's actually
                # close to the real data rather than needing to solve
                # for both a huge scale factor and a huge translation.
                mean_size = np.mean(
                    [np.linalg.norm(s - s.mean(axis=0)) for s in atlas_inputs]
                )
                mean_centroid = np.mean([s.mean(axis=0) for s in atlas_inputs], axis=0)
                atlas = atlas * mean_size + mean_centroid

                print("    Pass 2: rigid + deformable CPD against the GPA atlas...")

                pass2_results = run_correspondence_pass(
                    sample_ids, label_arrays, atlas, target_density,
                    rigid_max_iterations, cpd_alpha, cpd_beta, cpd_max_iterations,
                )

                final_results = pass2_results
                template_used = atlas
                used_atlas = True

    # --------------------------------------------------------
    # Step 5: final quality control - exclude poorly-converged
    # samples (wrong-plane images, corrupted labels, etc. all show
    # up here as residual outliers) before saving training data.
    # --------------------------------------------------------

    successes = [r for r in final_results if r["success"]]

    for r in final_results:
        if not r["success"]:
            failures.append({"sample_id": r["sample_id"], "reason": r["reason"]})

    class_output_dir = output_dir / class_name
    landmarks_dir = class_output_dir / "landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    if not successes:
        print("    No samples survived CPD; skipping class")
        pd.DataFrame(failures).to_csv(class_output_dir / "failures.csv", index=False)
        return

    residuals = np.array([r["residual_mean_px"] for r in successes])

    if max_correspondence_residual is not None:
        threshold = max_correspondence_residual
        print(f"    Using fixed correspondence residual threshold: {threshold:.2f}px")
    else:
        threshold = compute_outlier_threshold(residuals, outlier_mad_multiplier)
        print(
            f"    Auto correspondence residual threshold: {threshold:.2f}px "
            f"(median={np.median(residuals):.2f}px, multiplier={outlier_mad_multiplier})"
        )

    all_rows = []
    quality_rows = []

    for result in successes:

        sample_id = result["sample_id"]
        corresponding = result["corresponding"]
        included = result["residual_mean_px"] <= threshold

        quality_rows.append(
            {
                "sample_id": sample_id,
                "residual_mean_px": result["residual_mean_px"],
                "residual_max_px": result["residual_max_px"],
                "threshold_px": threshold,
                "included": included,
            }
        )

        if not included:

            print(
                f"    EXCLUDING {sample_id}: correspondence residual "
                f"{result['residual_mean_px']:.2f}px exceeds threshold "
                f"{threshold:.2f}px (poor CPD convergence - check for a "
                f"wrong-plane image, bad label, or other data issue)"
            )

            failures.append(
                {
                    "sample_id": sample_id,
                    "reason": (
                        f"poor CPD convergence (residual={result['residual_mean_px']:.2f}px, "
                        f"threshold={threshold:.2f}px)"
                    ),
                }
            )
            continue

        df = pd.DataFrame(corresponding, columns=["x", "y"])
        df.insert(0, "landmark_index", np.arange(len(df)))
        df.to_csv(landmarks_dir / f"{sample_id}.csv", index=False)

        row = {"sample_id": sample_id}
        for idx, (x, y) in enumerate(corresponding):
            row[f"x{idx}"] = x
            row[f"y{idx}"] = y
        all_rows.append(row)

    # --------------------------------------------------------
    # Save template/mean shape + combined landmark table + QC + failures
    # --------------------------------------------------------

    class_output_dir.mkdir(parents=True, exist_ok=True)

    mean_df = pd.DataFrame(template_used, columns=["x", "y"])
    mean_df.insert(0, "landmark_index", np.arange(len(mean_df)))
    mean_df.to_csv(class_output_dir / "mean_shape.csv", index=False)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            class_output_dir / "landmarks_all_samples.csv", index=False
        )

    pd.DataFrame(quality_rows).to_csv(class_output_dir / "correspondence_quality.csv", index=False)
    pd.DataFrame(failures).to_csv(class_output_dir / "failures.csv", index=False)

    template_note = f"GPA atlas (built from pass-1 results)" if used_atlas else f"single reference sample ({reference_id})"

    print(
        f"    Done: {len(all_rows)} samples included, "
        f"{len(successes) - len(all_rows)} excluded for poor convergence, "
        f"template used: {template_note}"
    )


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-dir",
        required=True,
        help="Root output dir from the preprocessing script (contains labels/sagittal, labels/coronal)",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-landmarks", type=int, default=N_LANDMARKS_DEFAULT)
    ap.add_argument("--target-density", type=int, default=TARGET_DENSITY_DEFAULT)
    ap.add_argument("--rigid-max-iterations", type=int, default=RIGID_MAX_ITERATIONS_DEFAULT)
    ap.add_argument("--cpd-alpha", type=float, default=CPD_ALPHA_DEFAULT)
    ap.add_argument("--cpd-beta", type=float, default=CPD_BETA_DEFAULT)
    ap.add_argument("--cpd-max-iterations", type=int, default=CPD_MAX_ITERATIONS)
    ap.add_argument(
        "--outlier-mad-multiplier", type=float, default=OUTLIER_MAD_MULTIPLIER_DEFAULT,
        help="Auto exclusion threshold = median + this * MAD of CPD residuals (default 3.0). Ignored if --max-correspondence-residual is given.",
    )
    ap.add_argument(
        "--max-correspondence-residual", type=float, default=1,
        help="Fixed max CPD residual (pixels) to keep a sample; overrides the automatic MAD-based threshold if given.",
    )
    ap.add_argument(
        "--build-atlas", action="store_true",
        help="After an initial CPD pass against a single reference sample, build a GPA mean-shape atlas from the well-converged results and re-run CPD against that atlas instead.",
    )
    ap.add_argument(
        "--reference-sample-id", default=None,
        help="Force a specific sample (by its sample_id) as the initial reference template, instead of the automatic median-border-length choice.",
    )

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:

        labels_dir = data_dir / "labels" / class_name

        if not labels_dir.exists():
            print(f"Skipping {class_name}: {labels_dir} does not exist")
            continue

        process_class(
            labels_dir=labels_dir,
            output_dir=output_dir,
            class_name=class_name,
            n_landmarks=args.n_landmarks,
            target_density=args.target_density,
            rigid_max_iterations=args.rigid_max_iterations,
            cpd_alpha=args.cpd_alpha,
            cpd_beta=args.cpd_beta,
            cpd_max_iterations=args.cpd_max_iterations,
            outlier_mad_multiplier=args.outlier_mad_multiplier,
            max_correspondence_residual=args.max_correspondence_residual,
            build_atlas=args.build_atlas,
            reference_sample_id=args.reference_sample_id,
        )


if __name__ == "__main__":
    main()