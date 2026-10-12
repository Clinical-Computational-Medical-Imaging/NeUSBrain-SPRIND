"""
train_tf_simple.py

3-class classifier (not_relevant / mid_saggital / relevant_coronal).
Based on the original working EfficientNetB0 model, with:
- Proper US ROI preprocessing from find_ultrasound_crop (DICOM metadata → contour fallback)
- Per-class weight boosting (relevant_coronal boosted extra)
- Focal loss gamma=3 (was 2) — harder push on missed coronal examples
- Cosine LR, F1 monitoring, leaderboard

Usage:
    python3 train_tf_simple.py \
        --csv      /path/to/labels.csv \
        --output   /path/to/models_dir \
        [--img_height 224] [--img_width 224] \
        [--batch_size 32]  [--val_split 0.2] [--seed 42] \
        [--coronal_boost 2.0]
"""

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",           required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--img_height",    type=int,   default=224)
    p.add_argument("--img_width",     type=int,   default=224)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--val_split",     type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--backbone",      type=str,   default="B3",
                   choices=["B0","B1","B2","B3","B4"],
                   help="EfficientNet variant (default B3)")
    p.add_argument("--coronal_boost", type=float, default=2.0,
                   help="Extra weight multiplier for relevant_coronal on top of "
                        "inverse-frequency weighting. Use >1 to boost coronal recall.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "not_relevant":     0,
    "mid_saggital":     1,
    "relevant_coronal": 2,
}
CLASS_NAMES = ["not_relevant", "mid_saggital", "relevant_coronal"]


# ---------------------------------------------------------------------------
# DICOM / ROI preprocessing  (from your preprocessing pipeline)
# ---------------------------------------------------------------------------

class UltrasoundRegionBase(BaseModel):
    ultrasound_region_uid: str
    spatial_format:      Optional[int] = None
    data_type:           Optional[int] = None
    flags:               Optional[int] = None
    location_min_x0:     Optional[int] = None
    location_min_y0:     Optional[int] = None
    location_max_x1:     Optional[int] = None
    location_max_y1:     Optional[int] = None
    reference_pixel_x0:  Optional[int] = None
    reference_pixel_y0:  Optional[int] = None
    units_x:             Optional[str] = None
    units_y:             Optional[str] = None
    physical_value_x:    Optional[float] = None
    physical_value_y:    Optional[float] = None
    delta_x:             Optional[float] = None
    delta_y:             Optional[float] = None
    frequency:           Optional[int] = None
    instance_id:         Optional[int] = None
    model_config = ConfigDict(from_attributes=True)


def parse_ultrasound_regions(dcm: pydicom.FileDataset) -> Optional[List[UltrasoundRegionBase]]:
    try:
        if not hasattr(dcm, "SequenceOfUltrasoundRegions"):
            return None
        regions = []
        for i, region in enumerate(dcm.SequenceOfUltrasoundRegions):
            regions.append(UltrasoundRegionBase(
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
            ))
        return regions
    except Exception as e:
        raise ValueError(f"Failed to parse DICOM UltrasoundRegion: {e}") from e


def extract_ultrasound_image(whole_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Contour-based fallback crop — finds the largest ultrasound region."""
    contours, _ = cv2.findContours(whole_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 10000:
            img_contour.append((x, y, w, h))
    if len(img_contour) != 1:
        return None
    return img_contour[0]


def find_ultrasound_crop(
    dcm: pydicom.FileDataset,
    gray: np.ndarray,
    regions: Optional[List[UltrasoundRegionBase]],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Two-stage crop:
    1. DICOM region metadata (spatial_format==1, 2D tissue region)
    2. Contour-based fallback
    """
    if regions:
        for r in regions:
            if (
                r.spatial_format == 1
                and None not in (
                    r.location_min_x0, r.location_min_y0,
                    r.location_max_x1, r.location_max_y1,
                )
            ):
                x0 = max(0, min(int(r.location_min_x0), gray.shape[1]))
                y0 = max(0, min(int(r.location_min_y0), gray.shape[0]))
                x1 = max(0, min(int(r.location_max_x1), gray.shape[1]))
                y1 = max(0, min(int(r.location_max_y1), gray.shape[0]))
                if x1 > x0 and y1 > y0:
                    return (x0, y0, x1 - x0, y1 - y0)

    # Contour fallback
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return extract_ultrasound_image(thresh)


def select_single_frame(dcm: pydicom.FileDataset, pixel_array: np.ndarray) -> np.ndarray:
    """Handle cine/multi-frame DICOMs — use frame 0."""
    n_frames = int(getattr(dcm, "NumberOfFrames", 1) or 1)
    if n_frames > 1 and pixel_array.ndim >= 3 and pixel_array.shape[0] == n_frames:
        return pixel_array[0]
    return pixel_array


def load_image(path: str, img_h: int, img_w: int) -> np.ndarray:
    """
    Load DICOM → select frame → grayscale → find US crop (metadata or contour)
    → resize → convert to 3-channel RGB for EfficientNetB0 → float32.
    Never returns None — returns blank image on any failure so tf.data stays stable.
    """
    try:
        dcm = pydicom.dcmread(path)
        pixel_array = dcm.pixel_array
        arr = select_single_frame(dcm, pixel_array)

        # Squeeze redundant channel axis
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        # → 2D grayscale
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        gray = cv2.normalize(
            arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Find US crop
        regions = parse_ultrasound_regions(dcm)
        bbox    = find_ultrasound_crop(dcm, gray, regions)

        if bbox is not None:
            x, y, w, h = bbox
            cropped = gray[y:y + h, x:x + w]
            if cropped.size > 0 and 0 not in cropped.shape:
                gray = cropped

        resized = cv2.resize(gray, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        # EfficientNetB0 expects 3 channels
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return rgb.astype(np.float32)

    except Exception as e:
        print(f"  Could not load {path}: {e}")
        return np.zeros((img_h, img_w, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if missing := ({"image_path", "user_label"} - set(df.columns)):
        raise ValueError(f"CSV missing columns: {missing}")
    before = len(df)
    df = df[df["user_label"].isin(LABEL_MAP)].copy()
    if dropped := before - len(df):
        print(f"  Dropped {dropped} rows with unknown labels.")
    df["label_int"] = df["user_label"].map(LABEL_MAP).astype(int)
    print("\nLabel distribution:")
    for lbl, n in df["user_label"].value_counts().items():
        print(f"  {lbl:20s}: {n:5d}")
    return df


def compute_class_weights(labels: np.ndarray, coronal_boost: float) -> dict:
    """
    Inverse-frequency weights + optional extra boost for relevant_coronal.
    coronal_boost=2.0 means coronal gets 2× its already-computed weight,
    trading some precision for better recall on the hardest class.
    """
    counts  = Counter(labels.tolist())
    total   = sum(counts.values())
    n_cls   = len(counts)
    weights = {int(c): float(total / (n_cls * cnt)) for c, cnt in counts.items()}

    # Boost coronal (class 2) specifically
    if 2 in weights and coronal_boost != 1.0:
        weights[2] *= coronal_boost
        print(f"  relevant_coronal weight boosted ×{coronal_boost}")

    print("\nClass weights:")
    for c, w in sorted(weights.items()):
        print(f"  class {c} ({CLASS_NAMES[c]:20s}): {w:.4f}")
    return weights


def build_dataset(paths, labels, img_h, img_w, batch_size, augment) -> tf.data.Dataset:
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
    ]) if augment else None

    def load_tf(path, label):
        img = tf.numpy_function(
            lambda p: load_image(p.decode(), img_h, img_w),
            [path], tf.float32
        )
        img = tf.ensure_shape(img, (img_h, img_w, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_tf, num_parallel_calls=tf.data.AUTOTUNE)

    if augment and data_augmentation:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.cache().shuffle(2000, seed=42).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Focal loss — gamma=3 puts more pressure on hard/missed examples
# ---------------------------------------------------------------------------

def focal_loss(gamma=3.0):
    def loss_fn(y_true, y_pred):
        y_true   = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        probs    = tf.nn.softmax(y_pred, axis=-1)
        y_onehot = tf.one_hot(y_true, tf.shape(y_pred)[-1])
        p_t      = tf.reduce_sum(probs * y_onehot, axis=-1)
        ce       = tf.nn.sparse_softmax_cross_entropy_with_logits(
                       labels=y_true, logits=y_pred)
        return tf.reduce_mean(tf.pow(1.0 - p_t, gamma) * ce)
    loss_fn.__name__ = f"focal_loss_gamma{gamma}"
    return loss_fn


# ---------------------------------------------------------------------------
# F1 callback
# ---------------------------------------------------------------------------

class ValF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for x_batch, y_batch in self.val_ds:
            probs = tf.nn.softmax(self.model(x_batch, training=False), axis=-1).numpy()
            y_pred.extend(np.argmax(probs, axis=1))
            y_true.extend(y_batch.numpy())
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        (logs or {})["val_f1_macro"] = f1
        print(f"  — val_f1_macro: {f1:.4f}")


# ---------------------------------------------------------------------------
# Plotting / leaderboard
# ---------------------------------------------------------------------------

def plot_history(h1, h2, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, m, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        e1 = range(1, len(h1.history[m]) + 1)
        ax.plot(e1, h1.history[m],          label="P1 train")
        ax.plot(e1, h1.history[f"val_{m}"], label="P1 val")
        off = len(e1)
        e2 = range(off + 1, off + len(h2.history[m]) + 1)
        ax.plot(e2, h2.history[m],          label="P2 train", linestyle="--")
        ax.plot(e2, h2.history[f"val_{m}"], label="P2 val",   linestyle="--")
        ax.axvline(off, color="gray", linestyle=":", linewidth=1)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


def plot_confusion(cm, out_dir, filename, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()


LEADERBOARD = "leaderboard.csv"

def update_leaderboard(output_dir, run_name, metrics):
    lb_path = output_dir / LEADERBOARD
    lb = pd.concat([pd.read_csv(lb_path), pd.DataFrame([{"run": run_name, **metrics}])],
                   ignore_index=True) if lb_path.exists() \
         else pd.DataFrame([{"run": run_name, **metrics}])
    lb = lb.sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
    lb.to_csv(lb_path, index=False)
    best_run = lb.iloc[0]["run"]
    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    src = output_dir / run_name / "model.keras"
    dst = best_dir / "model.keras"
    if src.exists() and best_run == run_name:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)
        print(f"  New best model → {dst}")
    return lb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n{'GPU: ' + str([g.name for g in gpus]) if gpus else 'No GPU — CPU only'}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir  = output_dir / run_name
    run_dir.mkdir()
    print(f"Run: {run_dir}\n")

    # ---- data ----
    df = load_csv(args.csv)
    num_classes = len(CLASS_NAMES)

    train_df, val_df = train_test_split(
        df, test_size=args.val_split,
        random_state=args.seed, stratify=df["label_int"],
    )
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

    class_weights = compute_class_weights(
        train_df["label_int"].values, args.coronal_boost
    )

    train_ds = build_dataset(
        train_df["image_path"].values,
        train_df["label_int"].values.astype(np.int32),
        args.img_height, args.img_width, args.batch_size, augment=True,
    )
    val_ds = build_dataset(
        val_df["image_path"].values,
        val_df["label_int"].values.astype(np.int32),
        args.img_height, args.img_width, args.batch_size, augment=False,
    )

    # ---- model ----
    backbone_map = {
        "B0": tf.keras.applications.EfficientNetB0,
        "B1": tf.keras.applications.EfficientNetB1,
        "B2": tf.keras.applications.EfficientNetB2,
        "B3": tf.keras.applications.EfficientNetB3,
        "B4": tf.keras.applications.EfficientNetB4,
    }
    # Native input sizes per variant — B3=300, B4=380; using user-specified
    # img_height/width so set --img_height 300 --img_width 300 for B3,
    # or --img_height 380 --img_width 380 for B4.
    print(f"\nBackbone: EfficientNet{args.backbone}  "
          f"input: {args.img_height}×{args.img_width}")

    base_model = backbone_map[args.backbone](
        input_shape=(args.img_height, args.img_width, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs  = tf.keras.Input(shape=(args.img_height, args.img_width, 3))
    x       = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x       = base_model(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes)(x)
    model   = tf.keras.Model(inputs, outputs)
    model.summary()

    checkpoint_path = str(run_dir / "model.keras")
    f1_cb = ValF1Callback(val_ds)
    cp_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_best_only=True,
        monitor="val_f1_macro", mode="max", verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_macro", mode="max",
        patience=20, restore_best_weights=True, verbose=1,
    )

    steps = max(1, len(train_df) // args.batch_size)
    lr_p1 = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=steps * 10,
        t_mul=2.0, m_mul=0.9, alpha=1e-6,
    )

    # ---- Phase 1: frozen backbone ----
    print("\n--- Phase 1: frozen backbone ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_p1),
        loss=focal_loss(gamma=3.0),
        metrics=["accuracy"],
    )
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=500,
        class_weight=class_weights,
        callbacks=[f1_cb, cp_cb, early_stop],
    )

    # ---- Phase 2: unfreeze top 20 layers ----
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    lr_p2 = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-5,
        first_decay_steps=steps * 5,
        t_mul=2.0, m_mul=0.9, alpha=1e-8,
    )
    early_stop_p2 = tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_macro", mode="max",
        patience=10, restore_best_weights=True, verbose=1,
    )

    print("\n--- Phase 2: top-20 layers unfrozen ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_p2),
        loss=focal_loss(gamma=3.0),
        metrics=["accuracy"],
    )
    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=100,
        class_weight=class_weights,
        callbacks=[f1_cb, cp_cb, early_stop_p2],
    )

    # ---- evaluation ----
    print("\n--- Evaluation ---")
    imgs, true_labels = [], []
    for path, lbl in zip(val_df["image_path"].values, val_df["label_int"].values):
        imgs.append(load_image(path, args.img_height, args.img_width))
        true_labels.append(int(lbl))

    probs       = tf.nn.softmax(
        model.predict(np.stack(imgs).astype(np.float32), verbose=0), axis=-1
    ).numpy()
    preds       = np.argmax(probs, axis=1)
    report_str  = classification_report(true_labels, preds,
                                        target_names=CLASS_NAMES, zero_division=0)
    report_dict = classification_report(true_labels, preds,
                                        target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0)
    print(report_str)

    val_acc = float(report_dict["accuracy"])
    val_f1  = float(report_dict["macro avg"]["f1-score"])

    plot_history(h1, h2, run_dir)
    plot_confusion(confusion_matrix(true_labels, preds),
                   run_dir, "confusion_matrix.png", "Confusion matrix (val)")

    with open(run_dir / "classification_report.txt", "w") as f:
        f.write(report_str)

    meta = {
        "run":            run_name,
        "val_accuracy":   round(val_acc, 4),
        "val_f1_macro":   round(val_f1, 4),
        "train_samples":  int(len(train_df)),
        "val_samples":    int(len(val_df)),
        "class_weights":  {int(k): float(v) for k, v in class_weights.items()},
        "label_map":      {k: int(v) for k, v in LABEL_MAP.items()},
        "img_height":     args.img_height,
        "img_width":      args.img_width,
        "phase1_epochs":  int(len(h1.history["accuracy"])),
        "phase2_epochs":  int(len(h2.history["accuracy"])),
        "backbone":       f"EfficientNet{args.backbone}",
        "loss":           "focal_loss_gamma3",
        "lr_schedule":    "CosineDecayRestarts",
        "coronal_boost":  args.coronal_boost,
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    lb = update_leaderboard(output_dir, run_name,
                            {"val_accuracy": val_acc, "val_f1_macro": val_f1,
                             "train_samples": len(train_df), "val_samples": len(val_df)})

    print(f"\n{'='*50}")
    print(f"Val accuracy   : {val_acc:.4f}")
    print(f"Val F1 (macro) : {val_f1:.4f}")
    print(f"\nLeaderboard (top 5):\n{lb.head(5).to_string(index=False)}")


if __name__ == "__main__":
    main()