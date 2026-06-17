"""Non-deep LOSO baselines — the "why deep learning" control.

Two classical baselines on the SAME LOSO protocol, scaler-on-train-only and
inner-val Youden-J threshold as the TCN, so the comparison is apples-to-apples:

  1. Freeze-Index threshold (Bachlin 2010 style): a single FI scalar per window,
     thresholded. The minimal domain baseline — if the TCN can't beat this it
     isn't earning its parameters.
  2. Shallow ML (LogisticRegression / DecisionTree) on a small hand-built
     feature vector: FI + STFT band power + per-channel mean/std/energy.

Reuses the existing DSP (`freeze_index_window`, `stft_band_power_window`), the
LOSO grouping (`_group_files_by_subject`), and the SAME scoring helpers as the
TCN evaluator (`_metrics_from_preds`, per-recording event metrics) so the
numbers drop straight into the report's comparison table.

Run:
    python src/baselines/freeze_index_baseline.py
"""

import os
import sys
import json
import random
import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import SAMPLING_RATE, SEED, WINDOW_SIZES, PROCESSED_DATA_DIR, MODELS_DIR
from data_pipeline.dsp import freeze_index_window, stft_band_power_window
from data_pipeline.dataset import (_group_files_by_subject, _last_step,
                                   recording_lengths)
from training.evaluate import (_metrics_from_preds, event_metrics_from_segments,
                               _split_by_lengths, _prediction_rate_hz)

LIN_ACC = slice(0, 3)   # linear-acc channels in the 9-channel layout
FI_INDEX = 0            # index of the freeze-index feature in the vector below
REPORTS_DIR = os.path.join(os.path.dirname(SRC_DIR), 'reports')


def window_features(win):
    """Hand-built feature vector for one (T, 9) raw window.

    [freeze_index, stft_band_power, mean(9), std(9), energy(9)] -> 29 dims.
    FI/band-power are computed on the linear-acc channels (the gait band lives
    there); the per-channel stats use all 9 channels.
    """
    lin = np.asarray(win[:, LIN_ACC])
    fi = freeze_index_window(lin, fs=SAMPLING_RATE)
    bp = stft_band_power_window(lin, fs=SAMPLING_RATE)
    mean = win.mean(axis=0)
    std = win.std(axis=0)
    energy = (win ** 2).mean(axis=0)
    return np.concatenate([[fi, bp], mean, std, energy]).astype(np.float64)


def _features_for(windows):
    return np.stack([window_features(w) for w in windows], axis=0)


def _load_subject_windows(groups, subj):
    """Concatenated (X raw, y last-step) for a subject, in the same file order
    and with the same skip rule as `recording_lengths` / the dataloaders."""
    xs, ys = [], []
    for xf in groups[subj]:
        yf = xf.replace('_x.npy', '_y.npy')
        if not os.path.exists(yf):
            continue
        xs.append(np.load(xf))
        ys.append(_last_step(np.load(yf)))
    if not xs:
        return None, None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _youden_threshold(scores, y):
    """Threshold maximizing tpr - fpr on the inner-val scores (no test peeking)."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float(np.median(scores))
    fpr, tpr, thr = roc_curve(y, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def _score_subject(scores_test, y_test, thr, data_dir, subj, seq_length):
    """Sample-level + seam-safe event metrics for one test subject's scores."""
    preds = (np.asarray(scores_test) >= thr).astype(np.int64)
    sample = _metrics_from_preds(y_test, preds, thr)
    rec_lengths = recording_lengths(data_dir, subj)
    pred_segs = _split_by_lengths(preds, rec_lengths)
    tgt_segs = _split_by_lengths(y_test, rec_lengths)
    events = event_metrics_from_segments(
        list(zip(tgt_segs, pred_segs)),
        prediction_rate_hz=_prediction_rate_hz(seq_length))
    return {'sample': sample, 'events': events}


def run_baselines(data_dir, seq_length, seed=SEED):
    groups = _group_files_by_subject(data_dir)
    subjects = sorted(groups)
    if len(subjects) < 3:
        return None

    models = ('freeze_index', 'logreg', 'tree')
    per_subject = {m: {} for m in models}

    for test_subj in subjects:
        non_test = [s for s in subjects if s != test_subj]
        rng = random.Random(f"{seed}:{test_subj}")
        val_subj = rng.choice(non_test) if len(non_test) > 1 else non_test[0]
        train_subj = [s for s in non_test if s != val_subj]

        X_tr = np.concatenate([_load_subject_windows(groups, s)[0] for s in train_subj], axis=0)
        y_tr = np.concatenate([_load_subject_windows(groups, s)[1] for s in train_subj], axis=0)
        X_va, y_va = _load_subject_windows(groups, val_subj)
        X_te, y_te = _load_subject_windows(groups, test_subj)
        if X_va is None or X_te is None:
            continue

        F_tr, F_va, F_te = _features_for(X_tr), _features_for(X_va), _features_for(X_te)

        # 1) Freeze-Index threshold — single feature, Youden-J on inner val.
        thr_fi = _youden_threshold(F_va[:, FI_INDEX], y_va)
        per_subject['freeze_index'][test_subj] = _score_subject(
            F_te[:, FI_INDEX], y_te, thr_fi, data_dir, test_subj, seq_length)

        # 2) Shallow ML — standardize on train only, Youden-J on inner-val proba.
        scaler = StandardScaler().fit(F_tr)
        Zt, Zv, Ze = scaler.transform(F_tr), scaler.transform(F_va), scaler.transform(F_te)
        for name, clf in (('logreg', LogisticRegression(max_iter=1000,
                                                        class_weight='balanced')),
                          ('tree', DecisionTreeClassifier(max_depth=6,
                                                          class_weight='balanced',
                                                          random_state=seed))):
            clf.fit(Zt, y_tr)
            thr = _youden_threshold(clf.predict_proba(Zv)[:, 1], y_va)
            per_subject[name][test_subj] = _score_subject(
                clf.predict_proba(Ze)[:, 1], y_te, thr, data_dir, test_subj, seq_length)

    summary = {'window': seq_length, 'n_subjects': len(subjects), 'models': {}}
    for m in models:
        mccs = [v['sample']['mcc'] for v in per_subject[m].values()]
        if not mccs:
            continue
        summary['models'][m] = {
            'mcc_mean': float(np.mean(mccs)),
            'mcc_std': float(np.std(mccs)),
            'per_subject': per_subject[m],
        }
    return summary


def _print_summary(summary):
    print(f"\n=== Baselines, window {summary['window']} "
          f"({summary['n_subjects']} subjects) ===")
    for name, m in summary['models'].items():
        print(f"  {name:13s} LOSO MCC mean±std = "
              f"{m['mcc_mean']:+.3f} ± {m['mcc_std']:.3f}")


def main():
    p = argparse.ArgumentParser(description="HopeGait non-deep LOSO baselines.")
    p.add_argument('--window', type=int, default=None)
    args = p.parse_args()
    windows = [args.window] if args.window is not None else WINDOW_SIZES

    out = {}
    for seq in windows:
        data_dir = os.path.join(PROCESSED_DATA_DIR, f'win_{seq}')
        r = run_baselines(data_dir, seq)
        if r is None:
            print(f"Skipping win_{seq}: need >=3 subjects with processed data.")
            continue
        out[f'win_{seq}'] = r
        _print_summary(r)

    if not out:
        print("No baselines computed. Is data/processed/win_*/ populated?")
        return
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, 'baseline_summary.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote baseline summary -> {path}")


if __name__ == "__main__":
    main()
