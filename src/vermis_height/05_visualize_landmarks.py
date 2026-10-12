"""
Visualize numbered landmarks on real images, to determine which
landmark indices to use for a specific measurement (e.g. vermis
height).

Why not just look at the PCA/GPA mean shape directly: GPA's
Procrustes alignment is rotation-invariant by design, so the mean
shape's orientation in its own coordinate space is arbitrary - it
does NOT necessarily correspond to anatomical up/down. Landmark
INDEX correspondence is what's guaranteed (index i always means the
same anatomical location across every sample), so the reliable way
to identify "which index is the top of the vermis" is to look at
numbered landmarks on real images where you can actually see the
anatomy, then reuse that index pair everywhere.

Recommended usage: point this at the FITTED landmarks (output of
04_fit_pca.py), since that's the exact coordinate space/format the
final measurement step will consume, on a handful of your best-
fitting samples (lowest HD95) plus a couple of average ones, to make
sure the chosen index pair looks right consistently and isn't an
artifact of one unusual case.
"""

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_image_2d(nii_path: Path) -> Optional[np.ndarray]:

    try:
        nii = nib.load(str(nii_path))
        data = np.squeeze(nii.get_fdata())
    except Exception as e:
        print(f"    Cannot read {nii_path}: {e}")
        return None

    if data.ndim != 2:
        print(f"    Unexpected shape {data.shape} in {nii_path}")
        return None

    return data


def plot_sample(
    sample_id: str,
    image_2d: Optional[np.ndarray],
    label_2d: Optional[np.ndarray],
    landmarks_df: pd.DataFrame,
    output_path: Path,
    highlight_every: int = 5,
    dpi: int = 150,
):

    # Swap x/y for display: transpose the background arrays and swap
    # the landmark columns together, so image and overlay stay
    # consistent with each other while the whole picture rotates.
    if image_2d is not None:
        image_2d = image_2d.T
    if label_2d is not None:
        label_2d = label_2d.T

    fig, ax = plt.subplots(figsize=(8, 8))

    if image_2d is not None:
        ax.imshow(image_2d, cmap="gray")
    elif label_2d is not None:
        ax.imshow(np.zeros_like(label_2d), cmap="gray", vmin=0, vmax=1)

    if label_2d is not None:
        ax.contour(label_2d == 2, levels=[0.5], colors="red", linewidths=1.0)

    n = len(landmarks_df)
    colors = cm.hsv(np.linspace(0, 1, n, endpoint=False))

    xs = landmarks_df["y"].to_numpy()
    ys = landmarks_df["x"].to_numpy()
    idxs = landmarks_df["landmark_index"].to_numpy()

    ax.scatter(xs, ys, c=colors, s=10, zorder=3)

    for x, y, idx in zip(xs, ys, idxs):

        # Number every point, but make every Nth one bold/bigger so
        # dense clusters of points stay readable.
        is_major = (idx % highlight_every == 0)

        ax.annotate(
            str(int(idx)),
            (x, y),
            fontsize=7 if is_major else 4,
            fontweight="bold" if is_major else "normal",
            color="yellow" if is_major else "white",
            xytext=(2, 2),
            textcoords="offset points",
        )

    ax.set_title(f"{sample_id} - landmark indices")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--landmarks-dir", required=True, help="Dir of <sample_id>.csv landmark files")
    ap.add_argument("--images-dir", default=None, help="Dir of processed images/<class>/*.nii.gz (optional but recommended)")
    ap.add_argument("--labels-dir", default=None, help="Dir of labels/<class>/*.nii.gz (optional, draws the label-2 border)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--sample-ids", nargs="*", default=None,
        help="Specific sample IDs to render. If omitted, renders --n-samples from --landmarks-dir",
    )
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--highlight-every", type=int, default=5)

    args = ap.parse_args()

    landmarks_dir = Path(args.landmarks_dir)
    images_dir = Path(args.images_dir) if args.images_dir else None
    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    output_dir = Path(args.output_dir)

    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        csv_files = sorted(landmarks_dir.glob("*.csv"))[: args.n_samples]
        sample_ids = [p.stem for p in csv_files]

    if not sample_ids:
        print(f"No landmark CSVs found in {landmarks_dir}")
        return

    for sample_id in sample_ids:

        csv_path = landmarks_dir / f"{sample_id}.csv"

        if not csv_path.exists():
            print(f"    Skipping {sample_id}: {csv_path} not found")
            continue

        landmarks_df = pd.read_csv(csv_path)

        image_2d = None
        if images_dir is not None:
            for ext in (".nii.gz", ".nii"):
                candidate = images_dir / f"{sample_id}{ext}"
                if candidate.exists():
                    image_2d = load_image_2d(candidate)
                    break

        label_2d = None
        if labels_dir is not None:
            for ext in (".nii.gz", ".nii"):
                candidate = labels_dir / f"{sample_id}{ext}"
                if candidate.exists():
                    label_2d = load_image_2d(candidate)
                    break

        output_path = output_dir / f"{sample_id}_landmarks.png"

        plot_sample(
            sample_id, image_2d, label_2d, landmarks_df, output_path,
            highlight_every=args.highlight_every,
        )

        print(f"    Saved {output_path}")


if __name__ == "__main__":
    main()