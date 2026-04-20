"""
main_seg.py  —  FSS with FPN Decoder (updated version)
====================================================
CHANGES from the no-decoder version:
  1. load_backbone() now returns feat_dims dict instead of single int
  2. SegAPM now takes decoder_out_channels instead of feature_dim
  3. Optimizer now includes BOTH backbone.layer4 AND decoder parameters
  4. Upsample target is still 224×224 but source is now 56×56 (not 7×7)
  5. Phase 2: we pass images through backbone+decoder to get fused features
     for novel prototype extraction (not just raw backbone features)

VISUALIZATION ADDITIONS (v2):
  - Visualizer.py is imported; plots are saved to plots/fold_N/
  - phase1_train now tracks val_losses, train_mious, lr_history
  - phase1_validate now returns (val_loss, val_miou, val_acc)
  - phase3_test now saves per-class IoU, pixel acc, and sample images/masks
  - After all folds: cross-fold summary bar chart is produced
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os

import Data_Loader
import Models
import APM
import Metrics
import Visualizer          # ← NEW: visualization module

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
VOC_ROOT            =  "./data/fss-data/VOCdevkit/VOC2012"
SBD_ROOT            =  "./data/fss-data/sbd/benchmark_RELEASE/dataset"
NUM_FOLDS           = 4  # ← Run all 4 folds
K_SHOT              = 5
BACKBONE_NAME       = "resnet50"
DECODER_CHANNELS    = 256   # FPN output channels — 256 is standard
BATCH_SIZE          = 8
NUM_EPOCHS          = 10
LEARNING_RATE       = 0.001
DECODER_LR          = 0.001   # can set higher e.g. 0.005 since decoder is random init
IMG_SIZE            = 473

# Number of query-image samples to save per fold for visual inspection
N_VIS_SAMPLES       = 6    # ← NEW: how many segmentation samples to plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Backbone: {BACKBONE_NAME} | {K_SHOT}-shot")
print(f"Decoder: FPN, out_channels={DECODER_CHANNELS}")
print(f"Running {NUM_FOLDS} folds...")

# Global loss criterion (used in compute_batch_loss)
criterion = nn.CrossEntropyLoss(ignore_index=255)

# ─────────────────────────────────────────────────────────────────
# NOTE: 'fold' is now passed explicitly to phase functions so that
# Visualizer can save into the correct fold sub-directory.
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# Shared helper — compute loss for one batch
# ─────────────────────────────────────────────────────────────────
def compute_batch_loss(model, images, masks, class_labels, novel_cls_id=None):
    logits, fused = model(images, novel_cls_id)   # [B, S, 56, 56]

    logits_full = F.interpolate(
        logits, size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear", align_corners=False
    )

    B    = images.shape[0]
    loss = torch.tensor(0.0, device=device)
    preds = []

    for i in range(B):
        if novel_cls_id is None:
             cls_idx  = class_labels[i].item()
             fg_slot  = cls_idx + 1
             bg_slot  = model.memory_module._bg_slot(cls_idx)
             logits_i = torch.stack(
                [logits_full[i, bg_slot], logits_full[i, fg_slot]], dim=0
            ).unsqueeze(0)
        else:
            logits_i = logits_full[i].unsqueeze(0)

        mask_i = masks[i].unsqueeze(0)
        loss  += criterion(logits_i, mask_i)
        preds.append(logits_i.argmax(dim=1).squeeze(0))

    return loss / B, preds, fused


# ─────────────────────────────────────────────────────────────────
# PHASE 1 — Train on base classes
# ─────────────────────────────────────────────────────────────────
def phase1_train(fold):                            # ← CHANGED: added 'fold' param
    print("\n" + "="*60)
    print("  PHASE 1 — Learning on BASE classes (with FPN decoder)")
    print("="*60)

    best_val_miou = 0.0

    # ── NEW: initialise per-epoch history lists ──────────────────
    train_losses = []
    val_losses   = []
    train_mious  = []
    val_mious    = []
    lr_history   = []         # backbone LR per epoch
    # ────────────────────────────────────────────────────────────

    for epoch in range(NUM_EPOCHS):
        model.train()
        metrics    = Metrics.SegMetrics(num_classes=2)
        epoch_loss = 0.0

        for batch_idx, (images, masks, labels) in enumerate(train_loader):
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            loss, preds, fused = compute_batch_loss(model, images, masks, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.memory_module.update_from_batch(
                    fused.detach(), masks, labels.tolist()
                )

            for i in range(images.shape[0]):
                metrics.update(preds[i].unsqueeze(0), masks[i].unsqueeze(0))

            epoch_loss += loss.item()

            if batch_idx % 30 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss {loss.item():.4f}")

        # ── CHANGED: phase1_validate now also returns val_loss ───
        val_loss, val_miou, val_acc = phase1_validate()
        _, train_miou, _            = metrics.compute()
        avg_loss                    = epoch_loss / len(train_loader)

        # ── NEW: append to history ───────────────────────────────
        train_losses.append(avg_loss)
        val_losses.append(val_loss)
        train_mious.append(float(train_miou))
        val_mious.append(float(val_miou))
        lr_history.append(optimizer.param_groups[0]["lr"])
        # ────────────────────────────────────────────────────────

        lrs = [g["lr"] for g in optimizer.param_groups]
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | "
              f"LR backbone={lrs[0]:.5f} decoder={lrs[1]:.5f}")
        print(f"  Train mIoU={train_miou*100:.2f}%  "
              f"Val mIoU={val_miou*100:.2f}%  PixAcc={val_acc*100:.2f}%")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), "phase1_best_decoder.pth")
            print(f"  ★ Saved (val mIoU={best_val_miou*100:.2f}%)")

        scheduler.step()

    print(f"\n[Phase 1] Best val mIoU = {best_val_miou*100:.2f}%")

    # ── NEW: plot training curves for this fold ──────────────────
    Visualizer.plot_training_curves(
        fold        = fold,
        train_losses= train_losses,
        val_losses  = val_losses,
        train_mious = train_mious,
        val_mious   = val_mious,
        lr_history  = lr_history,
    )
    # ────────────────────────────────────────────────────────────

    return best_val_miou


def phase1_validate():
    """
    CHANGED: now returns (val_loss, val_miou, val_acc) — added val_loss
    so it can be recorded for the training-curve plot.
    """
    model.eval()
    metrics    = Metrics.SegMetrics(num_classes=2)
    total_loss = 0.0

    with torch.no_grad():
        for images, masks, labels in val_loader:
            images = images.to(device)
            masks  = masks.to(device)
            loss, preds, _ = compute_batch_loss(model, images, masks, labels)
            total_loss += loss.item()
            for i in range(images.shape[0]):
                metrics.update(preds[i].unsqueeze(0), masks[i].unsqueeze(0))

    val_loss = total_loss / len(val_loader)          # ← NEW: compute mean val loss
    _, miou, acc = metrics.compute()
    return val_loss, miou, acc                        # ← CHANGED: returns 3 values


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — Adapt to novel classes  (unchanged)
# ─────────────────────────────────────────────────────────────────
def phase2_adapt(novel_dataset, novel_classes, k_shot):
    print("\n" + "="*60)
    print(f"  PHASE 2 — {k_shot}-shot adaptation (with FPN decoder)")
    print("="*60)

    model.load_state_dict(
        torch.load("phase1_best_decoder.pth", map_location=device)
    )
    model.freeze_everything()
    model.eval()

    query_data = {}

    for cls_id in novel_classes:
        cls_name = Data_Loader.VOC_CLASS_NAMES[cls_id]
        print(f"\n  Adapting: {cls_name} (class {cls_id})")

        support, queries = novel_dataset.get_support_and_queries(
            cls_id, k_shot=k_shot, seed=42
        )
        query_data[cls_id] = queries

        support_feats, support_masks_list = [], []

        with torch.no_grad():
            for img, msk in support:
                img_t = img.unsqueeze(0).to(device)
                feat2, feat3, feat4 = model.backbone(img_t)
                fused = model.decoder(feat2, feat3, feat4)
                support_feats.append(fused)
                support_masks_list.append(msk.unsqueeze(0).to(device))

        model.memory_module.build_novel_prototype(
            support_feats, support_masks_list, cls_id
        )

    print("\n[Phase 2] Novel prototypes built in decoder feature space.")
    return query_data


# ─────────────────────────────────────────────────────────────────
# PHASE 3 — Test on novel classes
# ─────────────────────────────────────────────────────────────────
def phase3_test(fold, novel_classes, query_data):     # ← CHANGED: added 'fold'
    print("\n" + "="*60)
    print("  PHASE 3 — Testing on NOVEL classes (with FPN decoder)")
    print("="*60)

    model.eval()
    all_mious = []

    # ── NEW: containers for visualization ───────────────────────
    per_class_ious  = []
    per_class_accs  = []
    class_name_list = []
    vis_samples     = []    # dicts: {image, gt_mask, pred_mask, class_name, iou}
    # ────────────────────────────────────────────────────────────

    with torch.no_grad():
        for cls_id in novel_classes:
            cls_name = Data_Loader.VOC_CLASS_NAMES[cls_id]
            queries  = query_data[cls_id]
            metrics  = Metrics.SegMetrics(num_classes=2)

            for q_img, q_mask in queries:
                img_t  = q_img.unsqueeze(0).to(device)
                mask_t = q_mask.unsqueeze(0).to(device)

                logits, _ = model(img_t, novel_cls_id=cls_id)
                logits_full = F.interpolate(
                    logits, size=(IMG_SIZE, IMG_SIZE),
                    mode="bilinear", align_corners=False
                )
                pred = logits_full.argmax(dim=1)
                metrics.update(pred, mask_t)

            _, cls_miou, cls_acc = metrics.compute()
            all_mious.append(cls_miou)

            # ── NEW: record for plots ────────────────────────────
            per_class_ious.append(float(cls_miou))
            per_class_accs.append(float(cls_acc))
            class_name_list.append(cls_name)
            # ────────────────────────────────────────────────────

            print(f"  {cls_name:15s} (class {cls_id:2d}) | "
                  f"mIoU={cls_miou*100:.2f}%  PixAcc={cls_acc*100:.2f}%  "
                  f"({len(queries)} query images)")

            # ── NEW: collect segmentation samples ────────────────
            if len(vis_samples) < N_VIS_SAMPLES:
                for q_img, q_mask in queries:
                    if len(vis_samples) >= N_VIS_SAMPLES:
                        break
                    img_t  = q_img.unsqueeze(0).to(device)
                    mask_t = q_mask.unsqueeze(0).to(device)

                    logits, _ = model(img_t, novel_cls_id=cls_id)
                    logits_full = F.interpolate(
                        logits, size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False
                    )
                    pred_mask = logits_full.argmax(dim=1).squeeze(0)

                    # quick per-sample IoU for the subplot title
                    from Metrics import SegMetrics
                    sm = SegMetrics(num_classes=2)
                    sm.update(pred_mask.unsqueeze(0), q_mask.unsqueeze(0))
                    _, sample_iou, _ = sm.compute()

                    vis_samples.append({
                        "image"     : q_img,
                        "gt_mask"   : q_mask,
                        "pred_mask" : pred_mask.cpu(),
                        "class_name": cls_name,
                        "iou"       : float(sample_iou),
                    })
            # ────────────────────────────────────────────────────

    mean_novel_miou = sum(all_mious) / len(all_mious)
    print(f"\n[Phase 3] Mean novel mIoU = {mean_novel_miou*100:.2f}%")

    # ── NEW: save Phase 3 plots ──────────────────────────────────
    Visualizer.plot_per_class_iou(
        fold           = fold,
        class_names    = class_name_list,
        per_class_ious = per_class_ious,
        per_class_accs = per_class_accs,
    )
    Visualizer.plot_segmentation_samples(
        fold      = fold,
        samples   = vis_samples,
        n_samples = N_VIS_SAMPLES,
    )
    # ────────────────────────────────────────────────────────────

    return mean_novel_miou


# ─────────────────────────────────────────────────────────────────
# RUN — Loop over all folds
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fold_results = []

    for fold in range(NUM_FOLDS):
        print(f"\n\n{'#'*70}")
        print(f"#  STARTING FOLD {fold}")
        print(f"{'#'*70}\n")

        # Reload data for this fold
        train_loader, val_loader, NUM_BASE = Data_Loader.prepare_base_loaders(
            voc_root=VOC_ROOT, sbd_root=SBD_ROOT, fold=fold, batch_size=BATCH_SIZE
        )
        novel_dataset, novel_classes = Data_Loader.prepare_novel_dataset(
            voc_root=VOC_ROOT, fold=fold
        )

        # Recreate model for this fold
        backbone, feat_dims = Models.load_backbone(BACKBONE_NAME)
        model = APM.SegAPM(
            backbone            = backbone,
            num_base_classes    = NUM_BASE,
            decoder_out_channels= DECODER_CHANNELS,
        ).to(device)

        optimizer = optim.Adam([
            {
                "params": model.backbone.layer4.parameters(),
                "lr": LEARNING_RATE,
                "name": "backbone_layer4"
            },
            {
                "params": model.decoder.parameters(),
                "lr": DECODER_LR,
                "name": "decoder",
                "weight_decay": 1e-4
            },
        ])

        scheduler = StepLR(optimizer, step_size=1, gamma=0.30)

        # ── CHANGED: pass fold to phase functions ────────────────
        phase1_val_miou = phase1_train(fold)                            # ← pass fold
        query_data      = phase2_adapt(novel_dataset, novel_classes, K_SHOT)
        novel_miou      = phase3_test(fold, novel_classes, query_data)  # ← pass fold
        # ────────────────────────────────────────────────────────

        result = {
            "fold"       : fold,
            "phase1_miou": phase1_val_miou,
            "phase3_miou": novel_miou
        }
        fold_results.append(result)

        print("\n" + "="*60)
        print(f"  FOLD {fold} RESULTS (with FPN decoder)")
        print("="*60)
        print(f"  Phase 1 val mIoU  (base)  = {phase1_val_miou*100:.2f}%")
        print(f"  Phase 3 mIoU      (novel) = {novel_miou*100:.2f}%")
        print(f"  Setting: Fold={fold} | {K_SHOT}-shot | {BACKBONE_NAME} + FPN")

    # Print summary across all folds
    print(f"\n\n{'='*60}")
    print("  SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*60}")
    for res in fold_results:
        print(f"  Fold {res['fold']} | Phase1={res['phase1_miou']*100:.2f}% | Phase3={res['phase3_miou']*100:.2f}%")

    avg_phase1 = sum(r['phase1_miou'] for r in fold_results) / len(fold_results)
    avg_phase3 = sum(r['phase3_miou'] for r in fold_results) / len(fold_results)
    print(f"\n  Average Phase 1 mIoU (base)  = {avg_phase1*100:.2f}%")
    print(f"  Average Phase 3 mIoU (novel) = {avg_phase3*100:.2f}%")

    # ── NEW: cross-fold summary plot (runs once, after all folds) ──
    Visualizer.plot_fold_summary(fold_results)
    # ──────────────────────────────────────────────────────────────
    print(f"\n[Visualizer] All plots saved to ./plots/")
