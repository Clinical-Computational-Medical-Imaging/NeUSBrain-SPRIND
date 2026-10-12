"""
Build a linear-PCA statistical shape model (Point Distribution Model)
from the point-corresponding landmarks produced by
build_shape_correspondence.py.

One independent model is built per plane (sagittal, coronal):

    1. Load that plane's landmarks_all_samples.csv (N corresponding
       points per sample, already in point-for-point correspondence
       thanks to the earlier GPA+CPD step).

    2. Re-align every sample's landmarks with Generalized Procrustes
       Analysis (GPA): remove translation, rotation, and scale, so
       PCA models pure SHAPE variation rather than trivial pose
       differences. Each sample's alignment transform (rotation,
       scale, translation) is kept, so an aligned/generated shape can
       always be mapped back into that sample's own pixel space.

    3. Run linear PCA (via SVD) on the aligned, flattened landmark
       vectors. This gives:
         - a mean shape
         - a set of orthogonal modes of variation (eigenvectors)
         - the variance explained by each mode (eigenvalues)
         - each sample's position along every mode ("shape scores" /
           PCA parameters b)

    A shape can then be regenerated as:
        shape = mean_shape + sum_i ( b_i * sqrt(eigenvalue_i) * mode_i )
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


CLASSES = ("sagittal", "coronal")

GPA_MAX_ITERS = 100
GPA_TOL = 1e-8

VARIANCE_THRESHOLD_DEFAULT = 0.95


# ============================================================
# Loading correspondence landmarks
# ============================================================

def load_landmarks_matrix(
    csv_path: Path,
) -> Tuple[List[str], np.ndarray]:
    """
    Load a landmarks_all_samples.csv (sample_id, x0, y0, x1, y1, ...)
    and return (sample_ids, landmarks) where landmarks has shape
    (n_samples, n_landmarks, 2).
    """

    df = pd.read_csv(csv_path)

    sample_ids = df["sample_id"].astype(str).tolist()

    x_cols = sorted(
        [c for c in df.columns if c.startswith("x") and c[1:].isdigit()],
        key=lambda c: int(c[1:]),
    )
    y_cols = sorted(
        [c for c in df.columns if c.startswith("y") and c[1:].isdigit()],
        key=lambda c: int(c[1:]),
    )

    if len(x_cols) != len(y_cols) or len(x_cols) == 0:
        raise ValueError(
            f"Could not find matching x/y landmark columns in {csv_path}"
        )

    xs = df[x_cols].to_numpy(dtype=np.float64)
    ys = df[y_cols].to_numpy(dtype=np.float64)

    landmarks = np.stack([xs, ys], axis=-1)  # (n_samples, n_landmarks, 2)

    return sample_ids, landmarks


# ============================================================
# Procrustes alignment / GPA (with recoverable transforms)
# ============================================================

def procrustes_align(
    shape: np.ndarray,
    reference: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Similarity-align `shape` onto `reference` (translation, rotation,
    uniform scale, no reflection), matched point-for-point by index.

    Returns (aligned_shape, transform) where transform contains
    everything needed to invert the alignment:
        aligned = scale * (shape - shape_mean) @ R + reference_mean
    """

    shape_mean = shape.mean(axis=0)
    reference_mean = reference.mean(axis=0)

    shape_c = shape - shape_mean
    ref_c = reference - reference_mean

    norm_shape = np.linalg.norm(shape_c)
    norm_ref = np.linalg.norm(ref_c)

    if norm_shape == 0 or norm_ref == 0:
        identity_transform = {
            "R": np.eye(2),
            "scale": 1.0,
            "shape_mean": shape_mean,
            "reference_mean": reference_mean,
        }
        return shape.copy(), identity_transform

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
        "R": R,
        "scale": scale,
        "shape_mean": shape_mean,
        "reference_mean": reference_mean,
    }

    return aligned, transform


def generalized_procrustes_analysis(
    shapes: List[np.ndarray],
    max_iters: int = GPA_MAX_ITERS,
    tol: float = GPA_TOL,
) -> Tuple[np.ndarray, List[np.ndarray], List[Dict[str, np.ndarray]]]:
    """
    Iteratively align every shape to the common mean shape until it
    stabilizes.

    Returns (mean_shape, aligned_shapes, transforms) - transforms[i]
    maps shapes[i] onto the final mean_shape via procrustes_align's
    convention, so it can be inverted to go back to original space.
    """

    mean_shape = shapes[0] - shapes[0].mean(axis=0)
    norm = np.linalg.norm(mean_shape)
    if norm > 0:
        mean_shape = mean_shape / norm

    aligned = shapes
    transforms: List[Dict[str, np.ndarray]] = []

    for iteration in range(max_iters):

        aligned = []
        transforms = []

        for s in shapes:
            a, t = procrustes_align(s, mean_shape)
            aligned.append(a)
            transforms.append(t)

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

    return mean_shape, aligned, transforms


# ============================================================
# Linear PCA (via SVD)
# ============================================================

def fit_pca(
    data_matrix: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Fit linear PCA on `data_matrix` (n_samples, n_features) via SVD.

    Returns a dict with:
        mean:                    (n_features,)
        components:              (K, n_features) - unit-norm eigenvectors, rows
        eigenvalues:             (K,) - variance explained by each component
        explained_variance_ratio:(K,)
        scores:                  (n_samples, K) - each sample's coordinate
                                  along every component
    K = min(n_samples - 1, n_features), since with fewer samples than
    dimensions (the usual case for shape models), only n_samples - 1
    components carry any variance.
    """

    n_samples = data_matrix.shape[0]

    mean = data_matrix.mean(axis=0)
    centered = data_matrix - mean

    # full_matrices=False keeps this tractable even for
    # high-dimensional landmark vectors.
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    k = min(n_samples - 1, Vt.shape[0])

    components = Vt[:k]
    eigenvalues = (S[:k] ** 2) / (n_samples - 1)

    total_variance = eigenvalues.sum()
    explained_variance_ratio = (
        eigenvalues / total_variance if total_variance > 0 else eigenvalues
    )

    scores = centered @ components.T

    return {
        "mean": mean,
        "components": components,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_variance_ratio,
        "scores": scores,
    }


def reconstruct(
    pca_model: Dict[str, np.ndarray],
    scores: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct shape vector(s) from PCA scores:
        shape = mean + scores @ components
    """

    return pca_model["mean"] + scores @ pca_model["components"]


# ============================================================
# Per-class pipeline
# ============================================================

def process_class(
    correspondence_dir: Path,
    output_dir: Path,
    class_name: str,
    variance_threshold: float,
):

    print(f"\n=== Building PCA shape model: {class_name} ===")

    csv_path = correspondence_dir / class_name / "landmarks_all_samples.csv"

    if not csv_path.exists():
        print(f"    {csv_path} not found; skipping")
        return

    sample_ids, landmarks = load_landmarks_matrix(csv_path)

    n_samples, n_landmarks, _ = landmarks.shape

    print(f"    Loaded {n_samples} samples, {n_landmarks} landmarks each")

    if n_samples < 3:
        print("    Not enough samples to build a meaningful PCA model; skipping")
        return

    # --------------------------------------------------------
    # GPA: remove translation/rotation/scale before PCA, so PCA
    # captures pure shape variation.
    # --------------------------------------------------------

    print("    Running GPA to align samples before PCA...")

    shapes = [landmarks[i] for i in range(n_samples)]

    mean_shape, aligned_shapes, transforms = generalized_procrustes_analysis(shapes)

    # --------------------------------------------------------
    # PCA on the flattened, GPA-aligned landmark vectors
    # --------------------------------------------------------

    print("    Running PCA...")

    data_matrix = np.stack(
        [s.reshape(-1) for s in aligned_shapes], axis=0,
    )  # (n_samples, 2 * n_landmarks)

    pca_model = fit_pca(data_matrix)

    cumulative = np.cumsum(pca_model["explained_variance_ratio"])
    n_for_threshold = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_for_threshold = min(n_for_threshold, len(cumulative))

    print(
        f"    PCA: {len(pca_model['eigenvalues'])} components computed; "
        f"{n_for_threshold} needed to explain {variance_threshold:.0%} of variance"
    )

    # Sanity check: with all components, reconstruction should match
    # the (GPA-aligned) input almost exactly.
    reconstructed = reconstruct(pca_model, pca_model["scores"])
    recon_error = np.abs(reconstructed - data_matrix).max()
    print(f"    Reconstruction sanity check (max abs error): {recon_error:.2e}")

    # --------------------------------------------------------
    # Save everything
    # --------------------------------------------------------

    class_output_dir = output_dir / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    # Compact, load-anywhere model file
    np.savez(
        class_output_dir / "pca_shape_model.npz",
        sample_ids=np.array(sample_ids),
        n_landmarks=n_landmarks,
        mean=pca_model["mean"],
        components=pca_model["components"],
        eigenvalues=pca_model["eigenvalues"],
        explained_variance_ratio=pca_model["explained_variance_ratio"],
        scores=pca_model["scores"],
    )

    # Human-readable / easy-to-inspect companions

    mean_shape_df = pd.DataFrame(
        pca_model["mean"].reshape(n_landmarks, 2), columns=["x", "y"],
    )
    mean_shape_df.insert(0, "landmark_index", np.arange(n_landmarks))
    mean_shape_df.to_csv(class_output_dir / "mean_shape_aligned.csv", index=False)

    variance_df = pd.DataFrame(
        {
            "component_index": np.arange(len(pca_model["eigenvalues"])),
            "eigenvalue": pca_model["eigenvalues"],
            "explained_variance_ratio": pca_model["explained_variance_ratio"],
            "cumulative_variance_ratio": cumulative,
        }
    )
    variance_df.to_csv(class_output_dir / "explained_variance.csv", index=False)

    scores_df = pd.DataFrame(
        pca_model["scores"],
        columns=[f"b{i}" for i in range(pca_model["scores"].shape[1])],
    )
    scores_df.insert(0, "sample_id", sample_ids)
    scores_df.to_csv(class_output_dir / "shape_scores.csv", index=False)

    # Per-sample alignment transforms, so an aligned/generated shape
    # can be mapped back into that sample's own pixel space:
    #   original_space_shape = scale * (aligned_shape - 0) @ R.T + shape_mean
    # (since aligned = scale * (shape - shape_mean) @ R + reference_mean,
    #  and reference_mean == 0 for the zero-centered GPA mean shape)
    transform_rows = []
    for sample_id, t in zip(sample_ids, transforms):
        transform_rows.append(
            {
                "sample_id": sample_id,
                "scale": t["scale"],
                "r00": t["R"][0, 0], "r01": t["R"][0, 1],
                "r10": t["R"][1, 0], "r11": t["R"][1, 1],
                "shape_mean_x": t["shape_mean"][0],
                "shape_mean_y": t["shape_mean"][1],
            }
        )
    pd.DataFrame(transform_rows).to_csv(
        class_output_dir / "alignment_transforms.csv", index=False
    )

    print(f"    Saved PCA shape model to {class_output_dir}")


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--correspondence-dir",
        required=True,
        help="Output dir from build_shape_correspondence.py (contains sagittal/, coronal/)",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--variance-threshold",
        type=float,
        default=VARIANCE_THRESHOLD_DEFAULT,
        help="Fraction of variance used to report how many components are 'needed' (informational; the full model is always saved)",
    )

    args = ap.parse_args()

    correspondence_dir = Path(args.correspondence_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:

        process_class(
            correspondence_dir=correspondence_dir,
            output_dir=output_dir,
            class_name=class_name,
            variance_threshold=args.variance_threshold,
        )


if __name__ == "__main__":
    main()