"""
Measure a landmark-defined distance (e.g. vermis height) on fitted
shapes (output of 04_fit_pca.py).

Rather than trusting a single landmark pair - which can be slightly
off due to correspondence imprecision, especially near a
high-curvature anatomical tip - this measures across a small window
of MIRRORED neighbor pairs around the two chosen indices:

    pair(k) = (landmark_a + k, landmark_b - k),   k = -R .. +R

Mirroring (rather than shifting both indices the same direction)
tests whether the true measurement axis sits slightly to one side of
the chosen anchor pair, without introducing a directional/diagonal
bias. With the default offset range of 2, this gives 5 candidate
measurements per sample:

    (a-2, b+2), (a-1, b+1), (a, b), (a+1, b-1), (a+2, b-2)

For each sample, all 5 distances (converted to mm using that
sample's own pixel spacing) are reported individually, plus:

    mean            - uses all 5, but one bad point can pull it off
    median          - robust to up to 2 bad points, but coarse (just
                      the 3rd-ranked value out of 5)
    trimmed_mean    - drop the single min and max, average the
                      middle 3: a middle ground between the two
    std / min / max - to see how much the 5 candidates actually
                      agree; if they're tight, the choice of
                      statistic barely matters, if they're not,
                      this is what to look at

Landmark indices wrap around (mod n_landmarks), since these are
points on a closed contour.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_spacing_mm(nii_path: Path) -> Optional[Tuple[float, float]]:
    """
    Returns (spacing_x_mm, spacing_y_mm) read from the NIfTI affine
    saved by the preprocessing script (affine[0,0] = row/y spacing,
    affine[1,1] = col/x spacing - see preprocess_ultrasound.py's
    build_affine, no rotation/flip applied).
    """

    try:
        nii = nib.load(str(nii_path))
        affine = nii.affine
        spacing_y_mm = abs(affine[0, 0])
        spacing_x_mm = abs(affine[1, 1])
        return spacing_x_mm, spacing_y_mm
    except Exception as e:
        print(f"    Cannot read spacing from {nii_path}: {e}")
        return None


def find_nii_file(spacing_dir: Path, sample_id: str) -> Optional[Path]:

    for ext in (".nii.gz", ".nii"):
        candidate = spacing_dir / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate

    return None


def angle_from_vertical_deg(point_a: pd.Series, point_b: pd.Series) -> float:
    """
    Angle between the line point_a-point_b and vertical (the image's
    y-axis), in degrees: 0 = perfectly vertical, 90 = perfectly
    horizontal. Direction-agnostic (doesn't matter which point is
    "a" vs "b").
    """

    dx = abs(point_a["x"] - point_b["x"])
    dy = abs(point_a["y"] - point_b["y"])

    return float(np.degrees(np.arctan2(dx, dy)))


def resolve_adaptive_anchors(
    coords: pd.DataFrame,
    landmark_a: int,
    landmark_b: int,
    n_landmarks: int,
    adaptive_search_window: int,
) -> Tuple[int, int]:
    """
    A fixed landmark index assumes the true anatomical extremum
    (topmost/bottommost point) always falls at exactly the same
    place along the shape's parameterization - but different SSM
    shape variants can shift where that true extremum actually lands
    by a few indices. Rather than trusting landmark_a/landmark_b
    exactly, search a small window of neighboring indices around
    each and pick whichever one is the true local geometric extremum
    (max/min y) in THIS sample's own fitted shape.

    Which direction to search (max-y vs min-y) for each anchor is
    determined from this same sample: whichever of the two nominal
    indices has the larger y is "bottom" (search for max-y nearby),
    the other is "top" (search for min-y nearby) - image y increases
    downward, so no separate reference/atlas is needed.

    Returns (resolved_landmark_a, resolved_landmark_b). With
    adaptive_search_window <= 0, returns (landmark_a, landmark_b)
    unchanged.
    """

    if adaptive_search_window <= 0:
        return landmark_a, landmark_b

    y_a = coords.loc[landmark_a, "y"]
    y_b = coords.loc[landmark_b, "y"]

    direction_a = "max" if y_a > y_b else "min"
    direction_b = "min" if direction_a == "max" else "max"

    def find_local_extremum(nominal_index: int, direction: str) -> int:

        candidates = [
            (nominal_index + offset) % n_landmarks
            for offset in range(-adaptive_search_window, adaptive_search_window + 1)
        ]
        ys = np.array([coords.loc[idx, "y"] for idx in candidates])

        best_local = int(np.argmax(ys)) if direction == "max" else int(np.argmin(ys))

        return candidates[best_local]

    resolved_a = find_local_extremum(landmark_a, direction_a)
    resolved_b = find_local_extremum(landmark_b, direction_b)

    return resolved_a, resolved_b


def measure_sample(
    landmarks_df: pd.DataFrame,
    landmark_a: int,
    landmark_b: int,
    offset_range: int,
    spacing_xy_mm: Tuple[float, float],
    adaptive_search_window: int = 0,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Returns (measurements_df, resolved_landmark_a, resolved_landmark_b).
    measurements_df has one row per offset k, containing the landmark
    indices used, the resulting distance in mm, and the line's angle
    from vertical (a mid-sagittal vermis measurement should be close
    to vertical; a candidate pair far from vertical usually means one
    of its two endpoints landed somewhere it shouldn't have, even if
    the overall shape fit looked fine).

    If adaptive_search_window > 0, landmark_a/landmark_b are first
    adaptively resolved to the true local extremum near each nominal
    index (see resolve_adaptive_anchors) before building the +/-
    offset_range mirrored candidate pairs around them.
    """

    n_landmarks = len(landmarks_df)

    coords = landmarks_df.set_index("landmark_index")[["x", "y"]]

    resolved_a, resolved_b = resolve_adaptive_anchors(
        coords, landmark_a, landmark_b, n_landmarks, adaptive_search_window,
    )

    spacing_x_mm, spacing_y_mm = spacing_xy_mm

    rows = []

    for k in range(-offset_range, offset_range + 1):

        idx_a = (resolved_a + k) % n_landmarks
        idx_b = (resolved_b - k) % n_landmarks

        point_a = coords.loc[idx_a]
        point_b = coords.loc[idx_b]

        dx_mm = (point_a["x"] - point_b["x"]) * spacing_x_mm
        dy_mm = (point_a["y"] - point_b["y"]) * spacing_y_mm

        distance_mm = float(np.sqrt(dx_mm ** 2 + dy_mm ** 2))
        angle_deg = angle_from_vertical_deg(point_a, point_b)

        rows.append(
            {
                "k": k,
                "landmark_a_index": idx_a,
                "landmark_b_index": idx_b,
                "distance_mm": distance_mm,
                "angle_from_vertical_deg": angle_deg,
            }
        )

    return pd.DataFrame(rows), resolved_a, resolved_b


def summarize_measurements(
    measurements_df: pd.DataFrame,
    max_angle_deviation_deg: float,
) -> dict:
    """
    Summarizes the 5 candidate measurements, but first drops any
    candidate whose angle from vertical exceeds
    max_angle_deviation_deg - those are almost always caused by one
    endpoint landing off the true anatomical tip, not real anatomical
    tilt. If NONE of the 5 pass (i.e. something is more seriously
    wrong with this fit), falls back to using all 5 anyway, but flags
    measurement_reliable=False so this sample can be caught and
    excluded downstream rather than silently trusted.
    """

    angle_ok_mask = measurements_df["angle_from_vertical_deg"] <= max_angle_deviation_deg
    n_angle_ok = int(angle_ok_mask.sum())

    if n_angle_ok > 0:
        used_df = measurements_df[angle_ok_mask]
        measurement_reliable = True
    else:
        used_df = measurements_df
        measurement_reliable = False

    distances = used_df["distance_mm"].to_numpy()

    sorted_d = np.sort(distances)

    if len(sorted_d) > 2:
        trimmed_mean = float(sorted_d[1:-1].mean())
    else:
        trimmed_mean = float(sorted_d.mean())

    return {
        "n_measurements": len(measurements_df),
        "n_angle_ok": n_angle_ok,
        "measurement_reliable": measurement_reliable,
        "min_angle_from_vertical_deg": float(measurements_df["angle_from_vertical_deg"].min()),
        "mean_mm": float(distances.mean()),
        "median_mm": float(np.median(distances)),
        "trimmed_mean_mm": trimmed_mean,
        "std_mm": float(distances.std()),
        "min_mm": float(distances.min()),
        "max_mm": float(distances.max()),
    }


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


def plot_measurement(
    sample_id: str,
    image_2d: Optional[np.ndarray],
    label_2d: Optional[np.ndarray],
    landmarks_df: pd.DataFrame,
    measurements_df: pd.DataFrame,
    stats: dict,
    output_path: Path,
    dpi: int = 150,
):

    # Swap x/y for display: transpose the background arrays and swap
    # the landmark columns together, so image and overlay stay
    # consistent with each other while the whole picture rotates.
    if image_2d is not None:
        image_2d = image_2d.T
    if label_2d is not None:
        label_2d = label_2d.T

    landmarks_df = landmarks_df.rename(columns={"x": "_orig_x", "y": "_orig_y"})
    landmarks_df["x"] = landmarks_df["_orig_y"]
    landmarks_df["y"] = landmarks_df["_orig_x"]

    fig, ax = plt.subplots(figsize=(8, 8))

    if image_2d is not None:
        ax.imshow(image_2d, cmap="gray")
    elif label_2d is not None:
        ax.imshow(np.zeros_like(label_2d), cmap="gray", vmin=0, vmax=1)

    if label_2d is not None:
        ax.contour(label_2d == 2, levels=[0.5], colors="red", linewidths=0.8, alpha=0.6)

    # All fitted landmarks, faint, just for spatial context.
    ax.scatter(
        landmarks_df["x"], landmarks_df["y"],
        s=4, c="cyan", alpha=0.3, zorder=2,
    )

    coords = landmarks_df.set_index("landmark_index")[["x", "y"]]

    n = len(measurements_df)
    colors = cm.cool(np.linspace(0, 1, n))

    for i, row in measurements_df.iterrows():

        k = int(row["k"])
        idx_a = int(row["landmark_a_index"])
        idx_b = int(row["landmark_b_index"])

        point_a = coords.loc[idx_a]
        point_b = coords.loc[idx_b]

        is_anchor = (k == 0)

        color = "yellow" if is_anchor else colors[i]
        linewidth = 2.5 if is_anchor else 1.2
        alpha = 1.0 if is_anchor else 0.7

        # Angle for the legend text is computed from the DISPLAYED
        # (swapped) coordinates, so it matches what's visually shown
        # here. The angle stored in measurement_detail CSVs / used for
        # QC (row["angle_from_vertical_deg"]) is computed from the
        # true, unswapped data - that's the clinically correct one;
        # this is a display-only label so it isn't confusing on screen.
        display_angle_deg = angle_from_vertical_deg(point_a, point_b)

        ax.plot(
            [point_a["x"], point_b["x"]], [point_a["y"], point_b["y"]],
            color=color, linewidth=linewidth, alpha=alpha, zorder=4,
            label=(
                f"k={k:+d}: {row['distance_mm']:.2f} mm, {display_angle_deg:.0f} deg from vertical (as shown)"
                + (" (anchor)" if is_anchor else "")
            ),
        )

        ax.scatter(
            [point_a["x"], point_b["x"]], [point_a["y"], point_b["y"]],
            color=color, s=25 if is_anchor else 12, zorder=5,
        )

    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

    reliability_note = "" if stats["measurement_reliable"] else "  [UNRELIABLE: all candidates off-vertical]"

    title = (
        f"{sample_id}{reliability_note}\n"
        f"mean={stats['mean_mm']:.2f}mm   median={stats['median_mm']:.2f}mm   "
        f"trimmed_mean={stats['trimmed_mean_mm']:.2f}mm   std={stats['std_mm']:.2f}mm"
    )
    ax.set_title(title, fontsize=10, color="red" if not stats["measurement_reliable"] else "black")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_plane(
    fitted_landmarks_dir: Path,
    spacing_dir: Path,
    output_dir: Path,
    landmark_a: int,
    landmark_b: int,
    offset_range: int,
    max_angle_deviation_deg: float,
    adaptive_search_window: int = 0,
    images_dir: Optional[Path] = None,
    labels_dir: Optional[Path] = None,
    fit_quality: Optional[dict] = None,
):
    """
    fit_quality: optional {sample_id: bool} map, e.g. loaded from
    04_fit_pca.py's fit_results.csv (fit_quality_ok column). Samples
    mapping to False are skipped entirely - a bad fit cannot be
    trusted to compute a real measurement from.
    """

    csv_files = sorted(fitted_landmarks_dir.glob("*.csv"))

    if not csv_files:
        print(f"    No fitted landmark files found in {fitted_landmarks_dir}")
        return

    detail_dir = output_dir / "measurement_detail"
    detail_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    failures = []

    for csv_path in csv_files:

        sample_id = csv_path.stem

        if fit_quality is not None and fit_quality.get(sample_id) is False:

            print(f"    Skipping {sample_id}: flagged fit_quality_ok=False, cannot trust this fit for measurement")
            failures.append({"sample_id": sample_id, "reason": "fit_quality_ok=False (poor PCA fit, see fit_results.csv)"})
            continue

        landmarks_df = pd.read_csv(csv_path)

        spacing_path = find_nii_file(spacing_dir, sample_id)

        if spacing_path is None:
            print(f"    Skipping {sample_id}: no matching file in {spacing_dir}")
            failures.append({"sample_id": sample_id, "reason": "spacing source file not found"})
            continue

        spacing_xy_mm = get_spacing_mm(spacing_path)

        if spacing_xy_mm is None:
            failures.append({"sample_id": sample_id, "reason": "could not read spacing"})
            continue

        n_landmarks = len(landmarks_df)

        if landmark_a >= n_landmarks or landmark_b >= n_landmarks:
            failures.append(
                {
                    "sample_id": sample_id,
                    "reason": (
                        f"landmark index out of range (n_landmarks={n_landmarks}, "
                        f"landmark_a={landmark_a}, landmark_b={landmark_b})"
                    ),
                }
            )
            continue

        measurements_df, resolved_a, resolved_b = measure_sample(
            landmarks_df, landmark_a, landmark_b, offset_range, spacing_xy_mm,
            adaptive_search_window=adaptive_search_window,
        )

        measurements_df.to_csv(detail_dir / f"{sample_id}.csv", index=False)

        stats = summarize_measurements(measurements_df, max_angle_deviation_deg)
        stats["resolved_landmark_a"] = resolved_a
        stats["resolved_landmark_b"] = resolved_b

        anchor_shift_note = (
            f"  anchors adaptively shifted: a {landmark_a}->{resolved_a}, b {landmark_b}->{resolved_b}"
            if (resolved_a != landmark_a or resolved_b != landmark_b) else ""
        )

        print(
            f"    {sample_id}: mean={stats['mean_mm']:.2f}mm  "
            f"median={stats['median_mm']:.2f}mm  "
            f"trimmed_mean={stats['trimmed_mean_mm']:.2f}mm  "
            f"std={stats['std_mm']:.2f}mm  "
            f"angle_ok={stats['n_angle_ok']}/{stats['n_measurements']}"
            f"{anchor_shift_note}"
        )

        if not stats["measurement_reliable"]:
            print(
                f"    WARNING: {sample_id} - ALL {stats['n_measurements']} candidate pairs exceed "
                f"{max_angle_deviation_deg:.0f} degrees from vertical (closest: "
                f"{stats['min_angle_from_vertical_deg']:.1f} deg). This usually means the SSM fit "
                f"placed the top or bottom landmark somewhere it shouldn't be - flagged "
                f"measurement_reliable=False; treat this measurement with caution"
            )

        summary_rows.append({"sample_id": sample_id, **stats})

        if images_dir is not None or labels_dir is not None:

            image_2d = None
            if images_dir is not None:
                image_path = find_nii_file(images_dir, sample_id)
                if image_path is not None:
                    image_2d = load_image_2d(image_path)

            label_2d = None
            if labels_dir is not None:
                label_path = find_nii_file(labels_dir, sample_id)
                if label_path is not None:
                    label_2d = load_image_2d(label_path)

            if image_2d is None and label_2d is None:
                print(f"    (no image/label found for {sample_id}; skipping visualization)")
            else:
                viz_path = output_dir / "visualizations" / f"{sample_id}.png"
                plot_measurement(
                    sample_id, image_2d, label_2d, landmarks_df,
                    measurements_df, stats, viz_path,
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "measurements_summary.csv", index=False)

    pd.DataFrame(failures).to_csv(output_dir / "measurement_failures.csv", index=False)

    print(f"\n    Measured {len(summary_rows)}/{len(csv_files)} samples")

    if len(summary_df) > 0:

        for col in ("mean_mm", "median_mm", "trimmed_mean_mm"):

            print(
                f"    Across all samples ({col}): "
                f"mean={summary_df[col].mean():.2f}mm  "
                f"std={summary_df[col].std():.2f}mm  "
                f"min={summary_df[col].min():.2f}mm  "
                f"max={summary_df[col].max():.2f}mm"
            )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--fitted-landmarks-dir", required=True,
        help="Dir of <sample_id>.csv fitted landmarks (output of 04_fit_pca.py) for ONE plane",
    )
    ap.add_argument(
        "--spacing-dir", required=True,
        help="Dir of the corresponding images/ or labels/ *.nii.gz for that plane, used to read pixel spacing",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--landmark-a", type=int, required=True)
    ap.add_argument("--landmark-b", type=int, required=True)
    ap.add_argument(
        "--offset-range", type=int, default=2,
        help="Use mirrored neighbor pairs (a+k, b-k) for k in [-range, range] (default 2 -> 5 measurements)",
    )
    ap.add_argument(
        "--max-angle-deviation-deg", type=float, default=20.0,
        help="A mid-sagittal/coronal vermis measurement should be close to vertical. Candidate pairs whose line deviates more than this from vertical are dropped from the summary stats; if ALL 5 candidates fail this, the sample is flagged measurement_reliable=False (default 20 degrees).",
    )
    ap.add_argument(
        "--adaptive-search-window", type=int, default=0,
        help="If > 0, don't trust landmark-a/landmark-b exactly: search this many indices on either side of each for the true local geometric extremum (topmost/bottommost point) in that sample's own fitted shape, and center the +/- offset-range stabilization window on that instead. Compensates for the true anatomical extremum shifting to a different index for different SSM shape variants. Default 0 (disabled, uses landmark-a/landmark-b exactly).",
    )
    ap.add_argument(
        "--images-dir", default=None,
        help="Dir of images/<class>/*.nii.gz for that plane. If given (with or without --labels-dir), a PNG visualizing the measurement lines is saved per sample.",
    )
    ap.add_argument(
        "--labels-dir", default=None,
        help="Dir of labels/<class>/*.nii.gz for that plane. Used to draw the label-2 border in visualizations; can be used instead of or alongside --images-dir.",
    )
    ap.add_argument(
        "--fit-results-csv", default=None,
        help="fit_results.csv from 04_fit_pca.py for this plane. If given, samples flagged fit_quality_ok=False are skipped entirely rather than measured.",
    )

    args = ap.parse_args()

    fit_quality = None
    if args.fit_results_csv:
        fit_results_df = pd.read_csv(args.fit_results_csv)
        if "fit_quality_ok" in fit_results_df.columns:
            fit_quality = dict(zip(fit_results_df["sample_id"], fit_results_df["fit_quality_ok"]))
        else:
            print(f"    Warning: {args.fit_results_csv} has no 'fit_quality_ok' column; not filtering by fit quality")

    process_plane(
        fitted_landmarks_dir=Path(args.fitted_landmarks_dir),
        spacing_dir=Path(args.spacing_dir),
        output_dir=Path(args.output_dir),
        landmark_a=args.landmark_a,
        landmark_b=args.landmark_b,
        offset_range=args.offset_range,
        max_angle_deviation_deg=args.max_angle_deviation_deg,
        adaptive_search_window=args.adaptive_search_window,
        images_dir=Path(args.images_dir) if args.images_dir else None,
        labels_dir=Path(args.labels_dir) if args.labels_dir else None,
        fit_quality=fit_quality,
    )


if __name__ == "__main__":
    main()