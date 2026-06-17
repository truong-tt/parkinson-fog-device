"""LOSO training loop.

Per fold:
  - Train HopeGaitTCN with focal loss on the last-step head plus a weighted
    dense per-timestep auxiliary loss (denser supervision -> better gradient
    signal than predicting only the last sample of each window).
  - Track an EMA of the model weights and validate on the EMA copy.
  - Select the best epoch by val MCC on the EMA copy (not val loss — focal
    minima don't always coincide with MCC maxima on imbalanced data).
  - Save the EMA weights, the scaler, and a per-fold operating threshold
    chosen on the inner val set by Youden's J. Test-time evaluation reuses
    that threshold so we never tune the threshold on the test fold.

Cloud-GPU friendly: optional AMP, CLI overrides for subjects/windows, runs
without a GPU (falls back to CPU).
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.optim as optim

from sklearn.metrics import matthews_corrcoef, roc_curve

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import (BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                    WINDOW_SIZES, PROCESSED_DATA_DIR, MODELS_DIR,
                    CLASS_WEIGHTS, SEED, NUM_CHANNELS, KERNEL_SIZE, DROPOUT,
                    NUM_INPUTS, NUM_CLASSES, FOCAL_GAMMA,
                    EARLY_STOP_PATIENCE, LR_PATIENCE, NUM_WORKERS,
                    DROP_PATH, USE_SE, DENSE_LOSS_WEIGHT, EMA_DECAY,
                    ROTATION_MAX_DEG, ROTATION_PROB, USE_AMP, DEVICE)
from data_pipeline.dataset import create_loso_dataloaders, get_all_subjects
from models.tcn_model import HopeGaitTCN
from models.focal_loss import FocalLoss
from training.ema import ModelEMA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _train_epoch(model, loader, criterion, optimizer, device, scaler_amp,
                 dense_weight, use_amp):
    model.train()
    total_loss, total_n = 0.0, 0
    for x, y_dense in loader:
        x = x.to(device, non_blocking=True)
        y_dense = y_dense.to(device, non_blocking=True)
        y_last = y_dense[:, -1]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            last_logits, dense_logits = model.forward_dense(x)
            loss_last = criterion(last_logits, y_last)
            loss_dense = criterion(dense_logits, y_dense)
            loss = loss_last + dense_weight * loss_dense

        if use_amp:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def _eval_probs(model, loader, device):
    """Return (probs, last-step targets) for the val/test set using last-step head."""
    model.eval()
    probs, targets = [], []
    for x, y_dense in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        targets.append(y_dense[:, -1].numpy())
    if not probs:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(targets)


def _focal_alpha(pos_rate, fallback=CLASS_WEIGHTS):
    """Inverse-frequency focal-loss alpha from the fold's actual FoG rate.

    Normalized inverse-frequency for 2 classes reduces to ``[p, 1 - p]``, so
    the minority (freeze) class gets the larger weight automatically. The old
    hardcoded ``[0.2, 0.8]`` was just this for an assumed p=0.2 — deriving it
    per fold tracks each subject pool's real imbalance. Degenerate folds (no
    positives or all positives) fall back to the configured constant.
    """
    p = float(pos_rate)
    if not (0.0 < p < 1.0):
        return list(fallback)
    return [p, 1.0 - p]


def _val_mcc_and_threshold(probs, targets):
    if len(np.unique(targets)) < 2:
        return 0.0, 0.5
    fpr, tpr, thr = roc_curve(targets, probs)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    best_thr = float(np.clip(thr[best_idx], 1e-4, 1 - 1e-4))
    preds = (probs >= best_thr).astype(np.int64)
    mcc = float(matthews_corrcoef(targets, preds)) if len(np.unique(preds)) > 1 else 0.0
    return mcc, best_thr


def train_fold(test_subject, seq_length, args):
    set_seed(args.seed)
    data_dir = os.path.join(PROCESSED_DATA_DIR, f'win_{seq_length}')
    device = resolve_device(args.device)
    use_amp = args.use_amp and device.type == 'cuda'

    train_loader, val_loader, _, scaler, meta = create_loso_dataloaders(
        data_dir, test_subject=test_subject, batch_size=args.batch_size,
        augment_train=True, num_workers=args.num_workers, seed=args.seed,
        rotation_max_deg=ROTATION_MAX_DEG, rotation_prob=ROTATION_PROB)

    model = HopeGaitTCN(num_inputs=NUM_INPUTS, num_channels=NUM_CHANNELS,
                        kernel_size=KERNEL_SIZE, num_classes=NUM_CLASSES,
                        dropout=DROPOUT, drop_path=DROP_PATH, use_se=USE_SE).to(device)
    ema = ModelEMA(model, decay=EMA_DECAY)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=LR_PATIENCE, factor=0.5)
    alpha = _focal_alpha(meta['train_pos_rate'])
    class_weights = torch.tensor(alpha, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    target_dir = os.path.join(MODELS_DIR, f'win_{seq_length}')
    os.makedirs(target_dir, exist_ok=True)
    model_path = os.path.join(target_dir, f'hopegait_tcn_best_subj{test_subject}.pth')
    scaler_path = os.path.join(target_dir, f'scaler_subj{test_subject}.npz')
    meta_path = os.path.join(target_dir, f'fold_meta_subj{test_subject}.json')
    history_path = os.path.join(target_dir, f'history_subj{test_subject}.json')

    best_mcc, bad_epochs = -float('inf'), 0
    best_threshold = 0.5
    history = []

    if args.verbose:
        print(f"[win={seq_length} test={test_subject} val={meta['val_subject']} "
              f"device={device} amp={use_amp}] "
              f"train_w={meta['train_windows']} val_w={meta['val_windows']} "
              f"test_w={meta['test_windows']} pos_rate={meta['train_pos_rate']:.3f} "
              f"focal_alpha=[{alpha[0]:.3f}, {alpha[1]:.3f}]")

    for epoch in range(args.epochs):
        tr_loss = _train_epoch(model, train_loader, criterion, optimizer, device,
                               scaler_amp, DENSE_LOSS_WEIGHT, use_amp)
        ema.update(model)

        # Validate on the EMA copy: smoother curves, usually a bit higher.
        probs, targets = _eval_probs(ema.shadow, val_loader, device)
        val_mcc, val_thr = _val_mcc_and_threshold(probs, targets)
        scheduler.step(val_mcc)

        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_mcc': val_mcc,
                        'val_threshold': val_thr,
                        'lr': optimizer.param_groups[0]['lr']})

        improved = val_mcc > best_mcc + 1e-6
        if improved:
            best_mcc = val_mcc
            best_threshold = val_thr
            bad_epochs = 0
            torch.save(ema.state_dict(), model_path)
            scaler.save(scaler_path)
            with open(meta_path, 'w') as f:
                json.dump({**meta, 'best_epoch': epoch, 'best_val_mcc': best_mcc,
                           'val_threshold': best_threshold, 'focal_alpha': alpha,
                           'seed': args.seed, 'seq_length': seq_length,
                           'use_se': USE_SE, 'drop_path': DROP_PATH}, f, indent=2)
        else:
            bad_epochs += 1

        if args.verbose:
            flag = '*' if improved else ' '
            print(f"  {flag} epoch {epoch+1:03d}/{args.epochs}  "
                  f"train L={tr_loss:.4f}  val MCC={val_mcc:+.3f}  thr={val_thr:.3f}")

        if bad_epochs >= EARLY_STOP_PATIENCE:
            if args.verbose:
                print(f"  early stop at epoch {epoch+1} (best val MCC {best_mcc:+.3f})")
            break

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="HopeGait LOSO training.")
    p.add_argument('--window', type=int, default=None,
                   help="Train only this window size (default: all in WINDOW_SIZES).")
    p.add_argument('--subject', type=str, default=None,
                   help="Train only this test subject (default: all).")
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--learning-rate', type=float, default=LEARNING_RATE)
    p.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--device', type=str, default=DEVICE,
                   help="'cuda', 'cpu', or 'auto' (default).")
    p.add_argument('--use-amp', action='store_true', default=USE_AMP)
    p.add_argument('--no-amp', dest='use_amp', action='store_false')
    p.add_argument('--quiet', dest='verbose', action='store_false', default=True)
    return p.parse_args()


def main():
    args = parse_args()
    windows = [args.window] if args.window is not None else WINDOW_SIZES
    for seq in windows:
        data_dir = os.path.join(PROCESSED_DATA_DIR, f'win_{seq}')
        subjects = get_all_subjects(data_dir)
        if len(subjects) < 3:
            print(f"Skipping win_{seq}: need >=3 subjects for train/val/test, found {len(subjects)}.")
            continue
        target_subjects = [args.subject] if args.subject is not None else subjects
        for subj in target_subjects:
            if subj not in subjects:
                print(f"Subject {subj} not found in win_{seq}, skipping.")
                continue
            train_fold(subj, seq, args)


if __name__ == "__main__":
    main()
