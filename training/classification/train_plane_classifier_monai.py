"""
train_incl_excl_monai.py

Trains a 3-class classifier (not_relevant / mid_saggital / relevant_coronal)
from a CSV label table of DICOM files — using MONAI.

MONAI advantages over plain TF/Keras for medical imaging:
- DenseNet121 with native grayscale input (in_channels=1)
- MONAI transforms handle medical image specifics (intensity normalisation,
  spatial transforms) more robustly than generic CV augmentation
- CacheDataset keeps preprocessed images in RAM — fast repeated access
- Built-in focal loss with class weights

Install:
    pip install monai[all] torch torchvision

Usage:
    python3 train_incl_excl_monai.py \
        --csv       /path/to/labels.csv \
        --output    /path/to/models_dir \
        [--img_height 224] [--img_width 224] \
        [--batch_size 16]  [--val_split 0.2] [--seed 42] \
        [--threshold 0.6]  [--epochs_p1 200] [--epochs_p2 100]
"""

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from monai.data import CacheDataset, DataLoader
from monai.losses import FocalLoss
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    EnsureTyped,
    RandFlipd,
    RandRotated,
    RandZoomd,
    ScaleIntensityd,
    ResizeD,
)
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--img_height", type=int,   default=224)
    p.add_argument("--img_width",  type=int,   default=224)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--val_split",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--threshold",  type=float, default=0.6)
    p.add_argument("--epochs_p1",  type=int,   default=200)
    p.add_argument("--epochs_p2",  type=int,   default=100)
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
NUM_CLASSES  = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# DICOM / ROI preprocessing
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


def parse_ultrasound_regions(dcm: pydicom.FileDataset):
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


def load_dicom_array(path: str) -> Optional[np.ndarray]:
    """
    Load DICOM, extract ultrasound ROI via region metadata (fallback: full frame),
    normalise to float32 in [0, 1], return shape (H, W) grayscale.
    """
    try:
        dcm = pydicom.dcmread(path)
        arr = None

        try:
            regions = parse_ultrasound_regions(dcm)
            if regions and all(v is not None for v in [
                regions[0].location_min_x0, regions[0].location_min_y0,
                regions[0].location_max_x1, regions[0].location_max_y1,
            ]):
                r   = regions[0]
                arr = dcm.pixel_array[
                    r.location_min_y0:r.location_max_y1,
                    r.location_min_x0:r.location_max_x1,
                ]
        except Exception:
            pass

        if arr is None or arr.size == 0:
            arr = dcm.pixel_array

        # Normalise channel layout → 2D grayscale
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if arr.ndim == 2:
            pass   # already grayscale
        else:
            print(f"  Unexpected shape {arr.shape} in {path}")
            return None

        # Float32 in [0, 1] — MONAI ScaleIntensityd will handle per-image normalisation
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr   # (H, W)

    except Exception as e:
        print(f"  Could not load {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# MONAI dataset items  (dict-based, as MONAI transforms expect)
# ---------------------------------------------------------------------------

def make_items(df: pd.DataFrame) -> List[Dict]:
    """
    Each MONAI item is a dict with 'image' (np array) and 'label' (int).
    We load all images here so CacheDataset can cache them.
    """
    items = []
    skipped = 0
    for _, row in df.iterrows():
        arr = load_dicom_array(row["image_path"])
        if arr is None:
            skipped += 1
            continue
        items.append({
            "image": arr[np.newaxis, ...],   # (1, H, W) — channel first for MONAI
            "label": int(row["label_int"]),
        })
    if skipped:
        print(f"  Skipped {skipped} unreadable files.")
    return items


# ---------------------------------------------------------------------------
# MONAI transforms
# ---------------------------------------------------------------------------

def get_transforms(img_h: int, img_w: int, augment: bool) -> Compose:
    spatial_size = (img_h, img_w)

    base = [
        EnsureTyped(keys=["image"], dtype=torch.float32),
        # Resize to target resolution
        ResizeD(keys=["image"], spatial_size=spatial_size,
                   mode="bilinear"),
        # Per-image intensity normalisation to [0, 1]
        ScaleIntensityd(keys=["image"]),
    ]

    aug = [
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),       # horizontal flip
        RandRotated(keys=["image"], prob=0.8,
                    range_x=0.0, range_y=0.0,
                    range_z=np.deg2rad(10),                         # ±10°
                    mode="bilinear", padding_mode="reflection"),
        RandZoomd(keys=["image"], prob=0.3,
                  min_zoom=0.95, max_zoom=1.05,                     # subtle zoom only
                  mode="bilinear"),
    ] if augment else []

    return Compose(base + aug)


# ---------------------------------------------------------------------------
# Data loading & balancing
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"image_path", "user_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    before = len(df)
    df = df[df["user_label"].isin(LABEL_MAP)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with unknown labels.")

    df["label_int"] = df["user_label"].map(LABEL_MAP).astype(int)

    print("\nLabel distribution (raw):")
    for lbl, n in df["user_label"].value_counts().items():
        print(f"  {lbl:20s}: {n:5d}")

    return df




def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = Counter(labels.tolist())
    total  = sum(counts.values())
    n_cls  = len(counts)
    weights = torch.tensor(
        [float(total / (n_cls * counts[i])) for i in range(n_cls)],
        dtype=torch.float32, device=device
    )
    print("\nClass weights:")
    for i, w in enumerate(weights):
        print(f"  class {i} ({CLASS_NAMES[i]:20s}): {w:.4f}")
    return weights


# ---------------------------------------------------------------------------
# Model — DenseNet121 (pure MONAI, plug-and-play)
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> nn.Module:
    """
    MONAI DenseNet121 — the simplest plug-and-play MONAI classifier.
    - in_channels=1: native grayscale, no projection needed
    - pretrained=True: ImageNet weights
    - No custom head, no tricks — just the standard MONAI network
    """
    model = DenseNet121(
        spatial_dims=2,
        in_channels=1,
        out_channels=num_classes,
        pretrained=True,
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)          # (B,) integer class indices

            logits = model(imgs)                        # (B, num_classes)

            # MONAI FocalLoss expects one-hot labels: (B, num_classes)
            labels_oh = torch.nn.functional.one_hot(
                labels.long(), num_classes=NUM_CLASSES
            ).float()
            loss = loss_fn(logits, labels_oh)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def predict_with_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    preds = []
    for p in probs:
        best = np.argmax(p)
        if best != 0 and p[best] < threshold:
            preds.append(0)
        else:
            preds.append(best)
    return np.array(preds)


def threshold_sweep(probs: np.ndarray, true_labels: list, out_dir: Path):
    thresholds = np.arange(0.3, 0.96, 0.05)
    f1s = [
        f1_score(true_labels,
                 predict_with_threshold(probs, t),
                 average="macro", zero_division=0)
        for t in thresholds
    ]
    best_t  = thresholds[np.argmax(f1s)]
    best_f1 = max(f1s)

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, f1s, marker="o")
    plt.axvline(best_t, color="red", linestyle="--",
                label=f"best={best_t:.2f} (F1={best_f1:.3f})")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Val F1 macro")
    plt.title("Threshold sweep")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweep.png", dpi=150)
    plt.close()

    print(f"\nBest threshold: {best_t:.2f} → F1 macro={best_f1:.4f}")
    return float(best_t), best_f1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(train_losses, val_losses, train_accs, val_accs,
                 phase_boundary: int, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    for ax, train_vals, val_vals, title in zip(
        axes,
        [train_accs, train_losses],
        [val_accs,   val_losses],
        ["Accuracy", "Loss"],
    ):
        ax.plot(epochs, train_vals, label="train")
        ax.plot(epochs, val_vals,   label="val")
        ax.axvline(phase_boundary, color="gray", linestyle=":",
                   linewidth=1, label="phase boundary")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


def plot_confusion(cm, class_names, title, out_dir, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
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


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

LEADERBOARD = "leaderboard.csv"

def update_leaderboard(output_dir: Path, run_name: str, metrics: dict):
    lb_path = output_dir / LEADERBOARD
    entry   = {"run": run_name, **metrics}
    lb = pd.concat(
        [pd.read_csv(lb_path), pd.DataFrame([entry])], ignore_index=True
    ) if lb_path.exists() else pd.DataFrame([entry])
    lb = lb.sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
    lb.to_csv(lb_path, index=False)

    best_run = lb.iloc[0]["run"]
    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    src = output_dir / run_name / "model.pt"
    dst = best_dir / "model.pt"
    if src.exists() and best_run == run_name:
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        print(f"  New best model copied → {dst}")
    return lb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir  = output_dir / run_name
    run_dir.mkdir()
    print(f"\nRun directory: {run_dir}")

    # ---- data ----
    df = load_csv(args.csv)

    train_df, val_df = train_test_split(
        df, test_size=args.val_split,
        random_state=args.seed, stratify=df["label_int"],
    )
    print(f"\nSplit — Train: {len(train_df)}  |  Val: {len(val_df)}")

    class_weights = compute_class_weights(train_df["label_int"].values, device)

    # Load pixel arrays (done once; CacheDataset caches transforms)
    print("\nLoading training images...")
    train_items = make_items(train_df)
    print("Loading validation images...")
    val_items   = make_items(val_df)

    train_transforms = get_transforms(args.img_height, args.img_width, augment=True)
    val_transforms   = get_transforms(args.img_height, args.img_width, augment=False)

    train_ds = CacheDataset(train_items, transform=train_transforms, cache_rate=1.0)
    val_ds   = CacheDataset(val_items,   transform=val_transforms,   cache_rate=1.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ---- model ----
    model = build_model(NUM_CLASSES).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # FocalLoss with class weights — MONAI's implementation handles logits natively
    loss_fn = FocalLoss(
        include_background=True,
        gamma=2.0,
        weight=class_weights,
        reduction="mean",
    )

    checkpoint_path = run_dir / "model.pt"
    best_val_f1 = -1.0

    # ---- phase 1: frozen backbone ----
    print("\n--- Phase 1: frozen backbone ---")

    # Freeze all layers except the classifier head
    for name, param in model.named_parameters():
        if "class_layers" not in name:   # DenseNet121 classifier head
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=8, min_lr=1e-7)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    no_improve, patience_p1  = 0, 30

    for epoch in range(1, args.epochs_p1 + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   loss_fn, None,      device, train=False)

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);   val_accs.append(va_acc)
        scheduler.step(va_loss)

        # Quick F1 check every 5 epochs for early stopping
        if epoch % 5 == 0 or epoch == 1:
            all_preds, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch["image"].to(device))
                    all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    all_labels.extend(batch["label"].numpy())
            val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            print(f"Epoch {epoch:3d} | loss {tr_loss:.4f}/{va_loss:.4f} | "
                  f"acc {tr_acc:.3f}/{va_acc:.3f} | val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✓ Saved (val_f1={val_f1:.4f})")
                no_improve = 0
            else:
                no_improve += 5
                if no_improve >= patience_p1:
                    print(f"Early stopping phase 1 at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:3d} | loss {tr_loss:.4f}/{va_loss:.4f} | "
                  f"acc {tr_acc:.3f}/{va_acc:.3f}")

    phase_boundary = len(train_losses)

    # ---- phase 2: partial fine-tune ----
    print("\n--- Phase 2: partial fine-tune ---")

    # 1. Reload best phase 1 weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 2. Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # 3. Unfreeze only the last 10 children (top of backbone + classifier)
    children = list(model.children())
    for child in children[-10:]:
        for param in child.parameters():
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters in phase 2: {n_trainable:,}")

    optimizer_p2 = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler_p2 = ReduceLROnPlateau(optimizer_p2, mode="min", factor=0.5,
                                     patience=8, min_lr=1e-8)
    no_improve_p2, patience_p2 = 0, 30

    for epoch in range(1, args.epochs_p2 + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer_p2, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   loss_fn, None,          device, train=False)

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);   val_accs.append(va_acc)
        scheduler_p2.step(va_loss)

        if epoch % 5 == 0 or epoch == 1:
            all_preds, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch["image"].to(device))
                    all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    all_labels.extend(batch["label"].numpy())
            val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            print(f"Epoch {epoch:3d} | loss {tr_loss:.4f}/{va_loss:.4f} | "
                  f"acc {tr_acc:.3f}/{va_acc:.3f} | val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✓ Saved (val_f1={val_f1:.4f})")
                no_improve_p2 = 0
            else:
                no_improve_p2 += 5
                if no_improve_p2 >= patience_p2:
                    print(f"Early stopping phase 2 at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:3d} | loss {tr_loss:.4f}/{va_loss:.4f} | "
                  f"acc {tr_acc:.3f}/{va_acc:.3f}")

    # ---- evaluation ----
    print("\n--- Evaluation on validation set ---")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["image"].to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].numpy())

    all_probs  = np.array(all_probs)
    preds_argmax = np.argmax(all_probs, axis=1)

    print("\n[Argmax predictions]")
    print(classification_report(all_labels, preds_argmax,
                                target_names=CLASS_NAMES, zero_division=0))

    best_t, _ = threshold_sweep(all_probs, all_labels, run_dir)

    preds_thresh = predict_with_threshold(all_probs, args.threshold)
    print(f"\n[Threshold={args.threshold:.2f} predictions]")
    report_str  = classification_report(all_labels, preds_thresh,
                                        target_names=CLASS_NAMES, zero_division=0)
    report_dict = classification_report(all_labels, preds_thresh,
                                        target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0)
    print(report_str)

    val_acc = float(report_dict["accuracy"])
    val_f1  = float(report_dict["macro avg"]["f1-score"])

    # ---- save artefacts ----
    plot_history(train_losses, val_losses, train_accs, val_accs,
                 phase_boundary, run_dir)
    plot_confusion(confusion_matrix(all_labels, preds_argmax),
                   CLASS_NAMES, "Confusion matrix — argmax",
                   run_dir, "confusion_argmax.png")
    plot_confusion(confusion_matrix(all_labels, preds_thresh),
                   CLASS_NAMES, f"Confusion matrix — threshold={args.threshold:.2f}",
                   run_dir, "confusion_threshold.png")

    with open(run_dir / "classification_report.txt", "w") as f:
        f.write("=== Argmax ===\n")
        f.write(classification_report(all_labels, preds_argmax,
                                      target_names=CLASS_NAMES, zero_division=0))
        f.write(f"\n=== Threshold={args.threshold:.2f} ===\n")
        f.write(report_str)
        f.write(f"\n=== Best threshold from sweep: {best_t:.2f} ===\n")

    meta = {
        "run":             run_name,
        "val_accuracy":    round(val_acc, 4),
        "val_f1_macro":    round(val_f1, 4),
        "threshold_used":  args.threshold,
        "best_threshold":  round(best_t, 2),
        "train_samples":   int(len(train_df)),
        "val_samples":     int(len(val_df)),
        "label_map":       {k: int(v) for k, v in LABEL_MAP.items()},
        "class_weights":   [float(w) for w in class_weights.cpu().numpy()],
        "img_height":      args.img_height,
        "img_width":       args.img_width,
        "epochs_p1":       args.epochs_p1,
        "epochs_p2":       args.epochs_p2,
        "backbone":        "MONAI_DenseNet121_grayscale",
        "loss":            "FocalLoss_gamma2_weighted",
        "framework":       "MONAI+PyTorch",
        "augmentation":    "class_weights + RandFlip + RandRotate±10° + RandZoom",
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    lb = update_leaderboard(output_dir, run_name, {
        "val_accuracy":   val_acc,
        "val_f1_macro":   val_f1,
        "best_threshold": best_t,
        "train_samples":  len(train_df),
        "val_samples":    len(val_df),
    })

    print(f"\n{'='*50}")
    print(f"Run complete   : {run_name}")
    print(f"Val accuracy   : {val_acc:.4f}")
    print(f"Val F1 (macro) : {val_f1:.4f}")
    print(f"Best threshold : {best_t:.2f}")
    print(f"Artefacts      : {run_dir}")
    print(f"\nLeaderboard (top 5):")
    print(lb.head(5).to_string(index=False))


if __name__ == "__main__":
    main()