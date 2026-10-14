"""
Compute the performance numbers for the README on held-out data and write
them as Markdown tables (append-only, so the three parts can run in
different environments).

    classifier  Keras classifier vs. reviewer labels
                (CSV with columns image_path,user_label)
    seg         nnU-Net predictions vs. ground truth: Dice + HD95 (mm)
    vermis      vermis height from predicted vs. from ground-truth
                segmentations (both measurements_summary.csv of
                06_measure_vermis.py): bias, MAE, limits of agreement, r

Usage:
    python tests/evaluate.py classifier --csv labels.csv --model models/classifiers/mid_sagittal_classifier/model.keras \\
        --class-names not_mid_saggital mid_saggital \\
        --label-map mid_saggital=mid_saggital relevant_coronal=not_mid_saggital not_relevant=not_mid_saggital \\
        --name "Mid-sagittal" --out results.md
    python tests/evaluate.py seg --pred-dir preds/ --gt-dir labelsTs/ --name "Cerebellum" --out results.md
    python tests/evaluate.py vermis --pred-csv pred/measurements_summary.csv \\
        --gt-csv gt/measurements_summary.csv --out results.md
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def append_markdown(out: Path, text: str) -> None:
    with open(out, "a") as f:
        f.write(text.rstrip() + "\n\n")
    print(text)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def load_classify_module():
    spec = importlib.util.spec_from_file_location("classify", REPO / "src/classification/classify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_gray(path: Path, classify) -> np.ndarray:
    """Same input handling as classify.py: 2D slice -> RGB uint8."""
    import cv2
    name = path.name.lower()
    if name.endswith((".png", ".jpg", ".jpeg")):
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError("cannot read image")
    elif name.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        arr = np.asanyarray(nib.load(str(path)).dataobj)
    else:  # single DICOM instance
        import SimpleITK as sitk
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        if arr.ndim == 4 or (arr.ndim == 3 and arr.shape[-1] in (3, 4)):
            arr = arr[..., :3].mean(axis=-1)  # RGB(A) -> gray
        arr = np.squeeze(arr)
        if arr.ndim == 3:  # multi-frame: (frames, H, W) -> (H, W, frames)
            arr = np.moveaxis(arr, 0, -1)
    return classify.to_rgb_uint8(classify.pick_representative_slice(np.asarray(arr), "middle"))


def eval_classifier(args) -> None:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    classify = load_classify_module()
    model = classify.InclExclClassifier(args.model, args.class_names)

    df = pd.read_csv(args.csv)
    df["user_label"] = df["user_label"].astype(str).str.strip()
    label_map = dict(item.split("=", 1) for item in args.label_map) if args.label_map else {}
    if label_map:
        df["target"] = df["user_label"].map(label_map)
    else:
        df["target"] = df["user_label"].where(df["user_label"].isin(args.class_names))
    df = df.dropna(subset=["target"])
    if args.max_images and len(df) > args.max_images:
        df = df.sample(args.max_images, random_state=0)
    print(f"Evaluating {len(df)} labeled images ...")

    preds, failed = [], 0
    for path in df["image_path"]:
        try:
            label, _ = model.predict(load_gray(Path(path), classify))
        except Exception as e:  # noqa: BLE001
            print(f"    {path}: {e}")
            label, failed = None, failed + 1
        preds.append(label)
    df["pred"] = preds
    df = df.dropna(subset=["pred"])

    y_true, y_pred = df["target"], df["pred"]
    report = classification_report(y_true, y_pred, labels=args.class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=args.class_names)

    lines = [f"#### {args.name}", "",
             f"Model `{Path(args.model).relative_to(REPO) if Path(args.model).is_relative_to(REPO) else args.model}`, "
             f"n = {len(df)} images ({failed} unreadable), accuracy **{accuracy_score(y_true, y_pred):.3f}**, "
             f"macro F1 **{report['macro avg']['f1-score']:.3f}**", "",
             "| Class | Precision | Recall | F1 | n |", "|---|---|---|---|---|"]
    for c in args.class_names:
        r = report[c]
        lines.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    lines += ["", "Confusion matrix (rows = reviewer, columns = model): " +
              "; ".join(f"{c}: {list(map(int, row))}" for c, row in zip(args.class_names, cm))]
    append_markdown(args.out, "\n".join(lines))


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def surface_distances_mm(a: np.ndarray, b: np.ndarray, spacing) -> np.ndarray:
    from scipy import ndimage
    struct = ndimage.generate_binary_structure(a.ndim, 1)
    border_a = a ^ ndimage.binary_erosion(a, struct, border_value=0)
    border_b = b ^ ndimage.binary_erosion(b, struct, border_value=0)
    dt_b = ndimage.distance_transform_edt(~border_b, sampling=spacing)
    dt_a = ndimage.distance_transform_edt(~border_a, sampling=spacing)
    return np.concatenate([dt_b[border_a], dt_a[border_b]])


def eval_segmentation(args) -> None:
    import nibabel as nib

    rows = []
    preds = sorted(args.pred_dir.glob("*.nii.gz")) + sorted(args.pred_dir.glob("*.nii"))
    for pred_path in preds:
        gt_path = args.gt_dir / pred_path.name
        if not gt_path.exists():
            continue
        pred_nii, gt_nii = nib.load(str(pred_path)), nib.load(str(gt_path))
        pred = np.squeeze(np.asanyarray(pred_nii.dataobj)) == args.label
        gt = np.squeeze(np.asanyarray(gt_nii.dataobj)) == args.label
        if pred.shape != gt.shape:
            print(f"    {pred_path.name}: shape mismatch, skipped")
            continue
        spacing = [float(s) for s in gt_nii.header.get_zooms()[:gt.ndim]]
        denom = pred.sum() + gt.sum()
        dice = np.nan if denom == 0 else 2 * np.logical_and(pred, gt).sum() / denom
        hd95 = np.nan
        if pred.any() and gt.any():
            hd95 = float(np.percentile(surface_distances_mm(pred, gt, spacing), 95))
        rows.append({"case": pred_path.name, "dice": dice, "hd95_mm": hd95,
                     "empty_prediction": not pred.any(), "gt_present": bool(gt.any())})

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No prediction/ground-truth pairs found")
    df.to_csv(args.out.with_name(f"seg_{args.name.lower().replace(' ', '_')}_per_case.csv"), index=False)
    d = df[df["gt_present"]]
    lines = [f"#### {args.name}", "",
             f"n = {len(d)} test images with ground truth, {int(d['empty_prediction'].sum())} empty predictions", "",
             "| Metric | Mean ± SD | Median [IQR] |", "|---|---|---|"]
    for col, label in [("dice", "Dice"), ("hd95_mm", "HD95 (mm)")]:
        v = d[col].dropna()
        lines.append(f"| {label} | {v.mean():.3f} ± {v.std():.3f} | "
                     f"{v.median():.3f} [{v.quantile(.25):.3f}–{v.quantile(.75):.3f}] |")
    append_markdown(args.out, "\n".join(lines))


# ---------------------------------------------------------------------------
# Vermis height agreement
# ---------------------------------------------------------------------------

def eval_vermis(args) -> None:
    pred = pd.read_csv(args.pred_csv)
    gt = pd.read_csv(args.gt_csv)
    col = args.value_column

    def reliable(df):
        return df[df["measurement_reliable"] == True] if "measurement_reliable" in df else df  # noqa: E712

    m = reliable(pred)[["sample_id", col]].merge(reliable(gt)[["sample_id", col]],
                                                 on="sample_id", suffixes=("_pred", "_gt"))
    m = m.dropna()
    diff = m[f"{col}_pred"] - m[f"{col}_gt"]
    rel = (diff.abs() / m[f"{col}_gt"]) * 100
    r = np.corrcoef(m[f"{col}_pred"], m[f"{col}_gt"])[0, 1] if len(m) > 2 else np.nan
    m.to_csv(args.out.with_name("vermis_height_pairs.csv"), index=False)

    lines = ["#### Vermis height: automatic (nnU-Net + SSM) vs. ground-truth segmentation", "",
             f"Both measured with the same SSM + landmark method (`{col}`). "
             f"n = {len(m)} paired images (reliable in both; "
             f"{int(reliable(pred).shape[0])}/{len(pred)} predicted, {int(reliable(gt).shape[0])}/{len(gt)} ground truth "
             "measurements reliable)", "",
             "| Metric | Value |", "|---|---|",
             f"| Ground-truth height, mean ± SD | {m[f'{col}_gt'].mean():.2f} ± {m[f'{col}_gt'].std():.2f} mm |",
             f"| Bias (pred − GT) | {diff.mean():+.2f} mm |",
             f"| MAE | {diff.abs().mean():.2f} mm ({rel.mean():.1f} %) |",
             f"| 95 % limits of agreement | {diff.mean() - 1.96 * diff.std():+.2f} to "
             f"{diff.mean() + 1.96 * diff.std():+.2f} mm |",
             f"| Pearson r | {r:.3f} |"]
    append_markdown(args.out, "\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="part", required=True)

    c = sub.add_parser("classifier")
    c.add_argument("--csv", type=Path, required=True, help="CSV with image_path,user_label")
    c.add_argument("--model", type=Path, required=True)
    c.add_argument("--class-names", nargs="+", required=True, help="Model output order")
    c.add_argument("--label-map", nargs="*", default=None,
                   help="user_label=class pairs; labels not listed are ignored")
    c.add_argument("--max-images", type=int, default=0, help="Random subsample (0 = all)")
    c.add_argument("--name", required=True)
    c.add_argument("--out", type=Path, default=Path("results.md"))

    s = sub.add_parser("seg")
    s.add_argument("--pred-dir", type=Path, required=True)
    s.add_argument("--gt-dir", type=Path, required=True)
    s.add_argument("--label", type=int, default=1)
    s.add_argument("--name", required=True)
    s.add_argument("--out", type=Path, default=Path("results.md"))

    v = sub.add_parser("vermis")
    v.add_argument("--pred-csv", type=Path, required=True)
    v.add_argument("--gt-csv", type=Path, required=True)
    v.add_argument("--value-column", default="trimmed_mean_mm")
    v.add_argument("--out", type=Path, default=Path("results.md"))

    args = ap.parse_args()
    {"classifier": eval_classifier, "seg": eval_segmentation, "vermis": eval_vermis}[args.part](args)


if __name__ == "__main__":
    main()
