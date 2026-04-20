"""
Visualizer.py — Plotting utilities for FSS training and evaluation
===================================================================
Reference: Pytorch training visualization patterns from
  - He et al. (2016), Deep Residual Learning (loss/acc curve conventions)
  - Shaban et al. (2017), One-Shot Learning for Semantic Segmentation (FSS eval plots)

All plots are saved to:   plots/fold_{N}/
  - training_curves.png       Phase 1: loss + mIoU + LR per epoch
  - phase3_per_class_iou.png  Phase 3: per-class mIoU + pixel acc bar chart
  - segmentation_samples.png  Phase 3: image | GT mask | pred mask (up to N_SAMPLES)
  - fold_summary.png          After all folds: grouped bar chart

Usage (called from main_seg.py):
    import Visualizer
    Visualizer.plot_training_curves(fold, train_losses, val_losses,
                                    train_mious, val_mious, lr_history)
    Visualizer.plot_per_class_iou(fold, class_names, per_class_ious, per_class_accs)
    Visualizer.plot_segmentation_samples(fold, samples)
    Visualizer.plot_fold_summary(fold_results)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server/cluster runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# ─────────────────────────────────────────────────────────────────
# Output directory helper
# ─────────────────────────────────────────────────────────────────
PLOT_ROOT = "plots"

def _fold_dir(fold):
    """Return (and create) the per-fold output directory."""
    path = os.path.join(PLOT_ROOT, f"fold_{fold}")
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Visualizer] Saved → {path}")


# ─────────────────────────────────────────────────────────────────
# 1. Training curves  (Phase 1, called at end of phase1_train)
# ─────────────────────────────────────────────────────────────────
def plot_training_curves(fold, train_losses, val_losses,
                         train_mious, val_mious, lr_history=None):
    """
    3-panel figure (or 2-panel if no LR history):
      Panel 1 — Train / Val Loss vs Epoch
      Panel 2 — Train / Val mIoU vs Epoch
      Panel 3 — Learning rate vs Epoch  (optional)

    Parameters
    ----------
    fold         : int
    train_losses : list[float]  — mean training loss per epoch
    val_losses   : list[float]  — mean validation loss per epoch
    train_mious  : list[float]  — training mIoU per epoch
    val_mious    : list[float]  — validation mIoU per epoch
    lr_history   : list[float] | None  — backbone LR per epoch
    """
    n_panels = 3 if lr_history else 2
    epochs   = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    fig.suptitle(f"Phase 1 Training — Fold {fold}", fontsize=13, fontweight="bold")

    # Panel 1 — Loss
    ax = axes[0]
    ax.plot(epochs, train_losses, "o-", label="Train", color="#2196F3", linewidth=1.8, markersize=4)
    ax.plot(epochs, val_losses,   "s--", label="Val",  color="#FF5722", linewidth=1.8, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Cross-Entropy)")
    ax.set_title("Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _annotate_best(ax, list(epochs), val_losses, mode="min", label="best val")

    # Panel 2 — mIoU
    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in train_mious], "o-",  label="Train",
            color="#2196F3", linewidth=1.8, markersize=4)
    ax.plot(epochs, [v * 100 for v in val_mious],   "s--", label="Val",
            color="#FF5722", linewidth=1.8, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU (%)")
    ax.set_title("mIoU Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _annotate_best(ax, list(epochs), [v * 100 for v in val_mious],
                   mode="max", label="best val")

    # Panel 3 — LR (optional)
    if lr_history:
        ax = axes[2]
        ax.plot(epochs, lr_history, "^-", color="#4CAF50", linewidth=1.8, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule (backbone)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "training_curves.png"))


def _annotate_best(ax, epochs, values, mode="max", label="best"):
    """Place a small annotation at the best (max or min) epoch."""
    fn    = max if mode == "max" else min
    best  = fn(values)
    idx   = values.index(best)
    ax.annotate(f"{label}\n{best:.2f}",
                xy=(epochs[idx], best),
                xytext=(0, 14 if mode == "max" else -22),
                textcoords="offset points",
                ha="center", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                color="gray")


# ─────────────────────────────────────────────────────────────────
# 2. Per-class IoU bar chart  (Phase 3)
# ─────────────────────────────────────────────────────────────────
def plot_per_class_iou(fold, class_names, per_class_ious, per_class_accs=None):
    """
    Grouped (or single) horizontal bar chart of per-class mIoU.
    Optionally adds pixel accuracy as a secondary bar.

    Parameters
    ----------
    fold            : int
    class_names     : list[str]
    per_class_ious  : list[float]   — mIoU per novel class (0–1)
    per_class_accs  : list[float]   — pixel acc per novel class (0–1), optional
    """
    n       = len(class_names)
    y_pos   = np.arange(n)
    has_acc = per_class_accs is not None

    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.6)))
    fig.suptitle(f"Phase 3 — Novel Class Performance  (Fold {fold})",
                 fontsize=13, fontweight="bold")

    bar_h = 0.35 if has_acc else 0.6

    bars_iou = ax.barh(y_pos + (bar_h / 2 if has_acc else 0),
                       [v * 100 for v in per_class_ious],
                       height=bar_h, label="mIoU (%)", color="#2196F3", alpha=0.85)

    if has_acc:
        bars_acc = ax.barh(y_pos - bar_h / 2,
                           [v * 100 for v in per_class_accs],
                           height=bar_h, label="Pixel Acc (%)", color="#FF9800", alpha=0.85)
        # Value labels on acc bars
        for bar in bars_acc:
            w = bar.get_width()
            ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%", va="center", fontsize=7.5, color="#555")

    # Value labels on IoU bars
    for bar in bars_iou:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%", va="center", fontsize=7.5, color="#333")

    mean_iou = np.mean(per_class_ious) * 100
    ax.axvline(mean_iou, color="red", linestyle="--", linewidth=1.2,
               label=f"Mean mIoU = {mean_iou:.1f}%")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Score (%)")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "phase3_per_class_iou.png"))


# ─────────────────────────────────────────────────────────────────
# 3. Segmentation sample grid  (Phase 3)
# ─────────────────────────────────────────────────────────────────
def plot_segmentation_samples(fold, samples, n_samples=6):
    """
    Grid of (image | GT mask | pred mask) for up to n_samples query images.

    Parameters
    ----------
    fold      : int
    samples   : list of dict with keys:
                  'image'     : FloatTensor [3, H, W]  (normalised)
                  'gt_mask'   : LongTensor  [H, W]
                  'pred_mask' : LongTensor  [H, W]
                  'class_name': str
                  'iou'       : float
    n_samples : int  — max number of rows to plot
    """
    samples = samples[:n_samples]
    n       = len(samples)
    if n == 0:
        return

    fig = plt.figure(figsize=(10, 3.2 * n))
    fig.suptitle(f"Segmentation Samples — Fold {fold}", fontsize=13, fontweight="bold")

    # ImageNet denormalization constants
    _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for row, s in enumerate(samples):
        img_disp = (s["image"].cpu() * _std + _mean).clamp(0, 1).permute(1, 2, 0).numpy()
        gt   = s["gt_mask"].cpu().numpy()
        pred = s["pred_mask"].cpu().numpy()

        ax_img  = fig.add_subplot(n, 3, row * 3 + 1)
        ax_gt   = fig.add_subplot(n, 3, row * 3 + 2)
        ax_pred = fig.add_subplot(n, 3, row * 3 + 3)

        ax_img.imshow(img_disp)
        ax_img.set_title(f"{s['class_name']} — query image", fontsize=8)
        ax_img.axis("off")

        ax_gt.imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax_gt.set_title("Ground truth", fontsize=8)
        ax_gt.axis("off")

        ax_pred.imshow(pred, cmap="gray", vmin=0, vmax=1)
        ax_pred.set_title(f"Prediction  (IoU={s['iou']*100:.1f}%)", fontsize=8)
        ax_pred.axis("off")

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "segmentation_samples.png"))


# ─────────────────────────────────────────────────────────────────
# 4. Cross-fold summary bar chart  (called once after all folds)
# ─────────────────────────────────────────────────────────────────
def plot_fold_summary(fold_results):
    """
    Grouped bar chart comparing Phase 1 (base) and Phase 3 (novel) mIoU
    across all folds, with mean ± std annotations.

    Parameters
    ----------
    fold_results : list of dict
                   Each dict must have: 'fold', 'phase1_miou', 'phase3_miou'
    """
    os.makedirs(PLOT_ROOT, exist_ok=True)

    folds  = [r["fold"] for r in fold_results]
    p1vals = [r["phase1_miou"] * 100 for r in fold_results]
    p3vals = [r["phase3_miou"] * 100 for r in fold_results]

    x   = np.arange(len(folds))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(folds) * 1.8), 5))
    fig.suptitle("Cross-Fold Summary — mIoU (%)", fontsize=14, fontweight="bold")

    bars1 = ax.bar(x - w / 2, p1vals, width=w, label="Phase 1 (base)",
                   color="#2196F3", alpha=0.85, zorder=3)
    bars3 = ax.bar(x + w / 2, p3vals, width=w, label="Phase 3 (novel)",
                   color="#FF5722", alpha=0.85, zorder=3)

    # Value labels above each bar
    for bar in list(bars1) + list(bars3):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    # Mean ± std reference lines
    m1, s1 = np.mean(p1vals), np.std(p1vals)
    m3, s3 = np.mean(p3vals), np.std(p3vals)
    ax.axhline(m1, color="#1565C0", linestyle="--", linewidth=1.1,
               label=f"Mean P1 = {m1:.1f} ± {s1:.1f}%")
    ax.axhline(m3, color="#BF360C", linestyle="--", linewidth=1.1,
               label=f"Mean P3 = {m3:.1f} ± {s3:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("mIoU (%)")
    ax.set_ylim(0, max(max(p1vals), max(p3vals)) * 1.18)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    _save(fig, os.path.join(PLOT_ROOT, "fold_summary.png"))


# ─────────────────────────────────────────────────────────────────
# 5. Confusion matrix heatmap  (optional, per fold Phase 1)
# ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(fold, confusion, class_names=("background", "foreground")):
    """
    Normalized confusion matrix heatmap.

    Parameters
    ----------
    fold        : int
    confusion   : np.ndarray [C, C]   — raw counts from SegMetrics
    class_names : tuple[str]
    """
    cm_norm = confusion.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_norm, row_sums,
                         out=np.zeros_like(cm_norm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle(f"Confusion Matrix — Fold {fold} (Phase 1 val)", fontsize=11)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks);  ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(ticks);  ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=11)

    plt.tight_layout()
    _save(fig, os.path.join(_fold_dir(fold), "confusion_matrix.png"))
