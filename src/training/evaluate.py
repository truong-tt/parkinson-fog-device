"""LOSO evaluation.

Reports three operating points per fold — @0.5, @fold-Youden (the honest
threshold chosen on the inner val set, no test peeking), and @post-processed
(per-recording smoothing + hysteresis) — plus sample-level metrics
(sens/spec/F1/MCC/PR-AUC/ROC-AUC) and event-level metrics (episode detection
rate, latency, false alarms/h) that describe the cueing experience.
"""

import os
import sys
import json
import numpy as np
import torch

from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             average_precision_score, roc_auc_score)

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import (WINDOW_SIZES, PROCESSED_DATA_DIR, MODELS_DIR, BATCH_SIZE,
                    NUM_CHANNELS, KERNEL_SIZE, DROPOUT, NUM_INPUTS, NUM_CLASSES, SEED,
                    DROP_PATH, USE_SE, SMOOTH_WINDOW, HYSTERESIS_LOW, HYSTERESIS_HIGH,
                    SAMPLING_RATE, WINDOW_OVERLAP)
from data_pipeline.dataset import (create_loso_dataloaders, get_all_subjects,
                                   recording_lengths)
from data_pipeline.dsp import RobustScaler
from models.tcn_model import HopeGaitTCN
from inference.postprocess import postprocess_predictions


def _collect_probs(model, loader, device):
    probs, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y_dense in loader:
            x = x.to(device)
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            # y_dense is (B, T) — last column is the causal target.
            targets.append(y_dense[:, -1].numpy() if y_dense.dim() == 2 else y_dense.numpy())
    if not probs:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(targets)


def _metrics_from_preds(targets, preds, threshold):
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = f1_score(targets, preds, zero_division=0)
    mcc_defined = (tp + fn) > 0 and (tn + fp) > 0 and (tp + fp) > 0 and (tn + fn) > 0
    mcc = matthews_corrcoef(targets, preds) if mcc_defined else 0.0
    return {'threshold': float(threshold),
            'sensitivity': float(sens), 'specificity': float(spec),
            'f1': float(f1), 'mcc': float(mcc),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}


def _metrics_at_threshold(targets, probs, thr):
    preds = (probs >= thr).astype(np.int64)
    return _metrics_from_preds(targets, preds, thr)


def _find_runs(arr, value):
    """Indices of contiguous runs of `value` in 1-D array. Returns list of
    (start, end) inclusive index pairs.
    """
    arr = np.asarray(arr).astype(np.int64)
    runs = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == value:
            j = i
            while j < n and arr[j] == value:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _event_counts(targets, preds):
    """Raw episode counts for ONE contiguous recording.

    Returns (n_episodes, n_detected, latencies_samples, false_alarms,
    non_fog_samples). Aggregating these summed counts across recordings is the
    only correct way to compute event metrics — concatenating recordings first
    would merge episodes across seams and miscount detections/false alarms.
    """
    targets = np.asarray(targets).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    target_episodes = _find_runs(targets, 1)
    pred_alarms = _find_runs(preds, 1)

    detected = 0
    latencies = []
    for start, end in target_episodes:
        # First positive prediction inside the episode counts as detection.
        # Any positive *before* the episode start is a false alarm, not a
        # detection — we only look in [start, end].
        hits = np.where(preds[start:end + 1] == 1)[0]
        if hits.size > 0:
            detected += 1
            latencies.append(int(hits[0]))

    false_alarms = 0
    for ps, pe in pred_alarms:
        # Overlap = not (pe < target_start OR ps > target_end). A pred run
        # overlapping any target episode is not an alarm.
        overlapped = any(not (pe < ts or ps > te) for ts, te in target_episodes)
        if not overlapped:
            false_alarms += 1

    non_fog_samples = int((targets == 0).sum())
    return len(target_episodes), detected, latencies, false_alarms, non_fog_samples


def _format_event_metrics(n_episodes, detected, latencies, false_alarms,
                          non_fog_samples, prediction_rate_hz):
    detection_rate = detected / n_episodes if n_episodes > 0 else 0.0
    mean_lat_s = float(np.mean(latencies)) / prediction_rate_hz if latencies else None
    median_lat_s = float(np.median(latencies)) / prediction_rate_hz if latencies else None
    non_fog_hours = non_fog_samples / prediction_rate_hz / 3600.0
    fa_per_hour = false_alarms / non_fog_hours if non_fog_hours > 0 else None
    return {
        'n_episodes': int(n_episodes),
        'episode_detection_rate': float(detection_rate),
        'mean_detection_latency_s': mean_lat_s,
        'median_detection_latency_s': median_lat_s,
        'false_alarms': int(false_alarms),
        'false_alarms_per_hour': fa_per_hour,
    }


def event_level_metrics(targets, preds, prediction_rate_hz=1.0):
    """Episode-level metrics for a single contiguous stream.

    Each contiguous run of 1s in `targets` is one FoG episode. An episode
    counts as detected if any positive prediction lands inside [start, end];
    detection latency is the index of the first such positive divided by
    `prediction_rate_hz`. A predicted run not overlapping any target episode
    is a false alarm, normalized to alarms per non-FoG hour. For multiple
    recordings use `event_metrics_from_segments` instead.
    """
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    if len(targets) == 0 or len(preds) != len(targets):
        return _format_event_metrics(0, 0, [], 0, 0, prediction_rate_hz)
    n_epi, det, lat, fa, nf = _event_counts(targets, preds)
    return _format_event_metrics(n_epi, det, lat, fa, nf, prediction_rate_hz)


def event_metrics_from_segments(segments, prediction_rate_hz=1.0):
    """Aggregate event metrics across recordings without crossing seams.

    `segments` is an iterable of (targets, preds) pairs, one per recording.
    Episode counts, detections, latencies and false alarms are summed/pooled
    across segments so a FoG episode split by a recording boundary is never
    merged with the next recording's.
    """
    tot_epi = tot_det = tot_fa = tot_nonfog = 0
    latencies = []
    for t, p in segments:
        t = np.asarray(t)
        p = np.asarray(p)
        if len(t) == 0 or len(p) != len(t):
            continue
        n_epi, det, lat, fa, nf = _event_counts(t, p)
        tot_epi += n_epi
        tot_det += det
        latencies.extend(lat)
        tot_fa += fa
        tot_nonfog += nf
    return _format_event_metrics(tot_epi, tot_det, latencies, tot_fa,
                                 tot_nonfog, prediction_rate_hz)


def _prediction_rate_hz(seq_length):
    """Predictions arrive at fs / step where step = window * (1 - overlap)."""
    step_samples = max(1, int(seq_length * (1.0 - WINDOW_OVERLAP)))
    return float(SAMPLING_RATE) / step_samples


def _split_by_lengths(arr, lengths):
    """Split a subject's concatenated stream into per-recording segments.

    Falls back to a single segment if `lengths` is missing or inconsistent
    with the array length, so a metadata mismatch degrades gracefully instead
    of crashing (it just reverts to the old, seam-crossing behavior).
    """
    arr = np.asarray(arr)
    if not lengths or int(sum(lengths)) != len(arr):
        return [arr]
    return np.split(arr, np.cumsum(lengths)[:-1])


def _load_fold_threshold(meta_path, fallback=0.5):
    if not os.path.exists(meta_path):
        return fallback
    with open(meta_path) as f:
        meta = json.load(f)
    return float(meta.get('val_threshold', fallback))


def evaluate_fold(test_subject, seq_length):
    """Load a fold's model/scaler and score its test split.

    Returns:
        ``(probs, targets, meta, fold_threshold, rec_lengths)``, or ``None`` if
        the checkpoint or scaler is missing.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_dir = os.path.join(MODELS_DIR, f'win_{seq_length}')
    model_path = os.path.join(target_dir, f'hopegait_tcn_best_subj{test_subject}.pth')
    scaler_path = os.path.join(target_dir, f'scaler_subj{test_subject}.npz')
    meta_path = os.path.join(target_dir, f'fold_meta_subj{test_subject}.json')
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None

    data_dir = os.path.join(PROCESSED_DATA_DIR, f'win_{seq_length}')
    scaler = RobustScaler.load(scaler_path)
    _, _, test_loader, _, meta = create_loso_dataloaders(
        data_dir, test_subject=test_subject, batch_size=BATCH_SIZE,
        scaler=scaler, augment_train=False, seed=SEED)

    model = HopeGaitTCN(num_inputs=NUM_INPUTS, num_channels=NUM_CHANNELS,
                        kernel_size=KERNEL_SIZE, num_classes=NUM_CLASSES,
                        dropout=DROPOUT, drop_path=DROP_PATH, use_se=USE_SE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    probs, targets = _collect_probs(model, test_loader, device)
    fold_threshold = _load_fold_threshold(meta_path)
    # Per-recording window counts (same order as the concatenated test stream)
    # so post-processing/event metrics never cross a recording seam.
    rec_lengths = recording_lengths(data_dir, test_subject)
    return probs, targets, meta, fold_threshold, rec_lengths


def evaluate_window(seq_length):
    """Aggregate per-subject LOSO metrics for one window size.

    Returns:
        A summary dict (pooled and per-subject metrics, event metrics, AUCs,
        mean +/- std), or ``None`` if no folds were evaluated.
    """
    data_dir = os.path.join(PROCESSED_DATA_DIR, f'win_{seq_length}')
    subjects = get_all_subjects(data_dir)
    if not subjects:
        return None

    per_subject = {}
    agg_targets, agg_probs = [], []
    pp_preds_all, pp_targets_all = [], []
    # (targets, preds) per recording across all subjects, for seam-safe
    # aggregate event metrics.
    seg_pairs_all = []

    band = max(0.0, HYSTERESIS_HIGH - HYSTERESIS_LOW)
    pred_rate_hz = _prediction_rate_hz(seq_length)

    for subj in subjects:
        r = evaluate_fold(subj, seq_length)
        if r is None:
            continue
        probs, targets, _, fold_thr, rec_lengths = r
        if len(targets) == 0:
            continue

        raw = _metrics_at_threshold(targets, probs, fold_thr)

        # Post-process per recording so the hysteresis state machine and the
        # episode counting never cross a recording boundary (postprocess.py
        # contract). Sample-level metrics are still computed on the concat.
        prob_segs = _split_by_lengths(probs, rec_lengths)
        tgt_segs = _split_by_lengths(targets, rec_lengths)
        pp_segs = [postprocess_predictions(
            ps, threshold=fold_thr, smooth_window=SMOOTH_WINDOW,
            hysteresis_band=band)[0] for ps in prob_segs]
        pp_preds = (np.concatenate(pp_segs) if pp_segs
                    else np.array([], dtype=np.int64))
        pp = _metrics_from_preds(targets, pp_preds, fold_thr)

        subj_seg_pairs = list(zip(tgt_segs, pp_segs))
        events_pp = event_metrics_from_segments(subj_seg_pairs,
                                                prediction_rate_hz=pred_rate_hz)

        per_subject[subj] = {
            'at_fold_threshold': raw,
            'at_postprocessed': pp,
            'events_at_postprocessed': events_pp,
        }
        agg_targets.append(targets)
        agg_probs.append(probs)
        pp_preds_all.append(pp_preds)
        pp_targets_all.append(targets)
        seg_pairs_all.extend(subj_seg_pairs)

    if not per_subject:
        return None

    t = np.concatenate(agg_targets)
    p = np.concatenate(agg_probs)
    at_05 = _metrics_at_threshold(t, p, 0.5)
    pp_preds_cat = np.concatenate(pp_preds_all)
    pp_targets_cat = np.concatenate(pp_targets_all)
    at_pp = _metrics_from_preds(pp_targets_cat, pp_preds_cat, threshold=float('nan'))

    pr_auc = float(average_precision_score(t, p)) if len(np.unique(t)) > 1 else 0.0
    roc_auc = float(roc_auc_score(t, p)) if len(np.unique(t)) > 1 else 0.0

    # Aggregate event-level metrics across every recording (seam-safe): sums
    # per-recording episode counts rather than scanning a concatenated stream.
    events_agg = event_metrics_from_segments(seg_pairs_all,
                                             prediction_rate_hz=pred_rate_hz)

    # LOSO variance: the spread across folds matters as much as the pooled
    # number (one near-zero subject is invisible in a pooled MCC).
    pp_mccs = [v['at_postprocessed']['mcc'] for v in per_subject.values()]
    fold_mccs = [v['at_fold_threshold']['mcc'] for v in per_subject.values()]

    return {
        'window': seq_length,
        'n_subjects': len(per_subject),
        'prediction_rate_hz': pred_rate_hz,
        'at_0.5': at_05,
        'at_postprocessed': at_pp,
        'events_at_postprocessed': events_agg,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'mcc_mean': float(np.mean(pp_mccs)),
        'mcc_std': float(np.std(pp_mccs)),
        'fold_mcc_mean': float(np.mean(fold_mccs)),
        'fold_mcc_std': float(np.std(fold_mccs)),
        'smooth_window': SMOOTH_WINDOW,
        'hysteresis_band': band,
        'per_subject': per_subject,
    }


def _print_summary(r):
    w = r['window']
    print(f"\n=== Window {w} ({r['n_subjects']} subjects, "
          f"prediction_rate={r['prediction_rate_hz']:.2f} Hz) ===")
    a = r['at_0.5']
    print(f"  @0.5      sens={a['sensitivity']*100:5.1f}%  spec={a['specificity']*100:5.1f}%  "
          f"F1={a['f1']:.3f}  MCC={a['mcc']:+.3f}")
    b = r['at_postprocessed']
    print(f"  @post-pp  sens={b['sensitivity']*100:5.1f}%  spec={b['specificity']*100:5.1f}%  "
          f"F1={b['f1']:.3f}  MCC={b['mcc']:+.3f}  (smooth={r['smooth_window']}, band={r['hysteresis_band']:.2f})")
    print(f"  PR-AUC={r['pr_auc']:.3f}  ROC-AUC={r['roc_auc']:.3f}")
    print(f"  LOSO MCC (pp) mean±std = {r['mcc_mean']:+.3f} ± {r['mcc_std']:.3f}  "
          f"(fold-thr {r['fold_mcc_mean']:+.3f} ± {r['fold_mcc_std']:.3f})")
    e = r['events_at_postprocessed']
    mean_lat = f"{e['mean_detection_latency_s']:.2f} s" if e['mean_detection_latency_s'] is not None else "n/a"
    fa_hr = f"{e['false_alarms_per_hour']:.2f}" if e['false_alarms_per_hour'] is not None else "n/a"
    print(f"  events    n_episodes={e['n_episodes']}  detected={e['episode_detection_rate']*100:5.1f}%  "
          f"mean_latency={mean_lat}  false_alarms/h={fa_hr}")


REPORTS_DIR = os.path.join(os.path.dirname(SRC_DIR), 'reports')


def _write_results_table(summary, path):
    """Per-subject LOSO results table (markdown) for the technical report."""
    lines = ["# HopeGait — LOSO Results (post-processed operating point)", ""]
    for key, r in summary.items():
        lines.append(f"## Window {r['window']} — {r['n_subjects']} subjects")
        lines.append("")
        lines.append(f"Post-processed MCC across folds: "
                     f"**{r['mcc_mean']:+.3f} ± {r['mcc_std']:.3f}** "
                     f"(PR-AUC {r['pr_auc']:.3f}, ROC-AUC {r['roc_auc']:.3f}).")
        lines.append("")
        lines.append("| Subject | MCC (fold-thr) | MCC (pp) | Sens (pp) | "
                     "Spec (pp) | Episodes | Detected | FA/h |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for subj, v in sorted(r['per_subject'].items()):
            ft = v['at_fold_threshold']
            pp = v['at_postprocessed']
            ev = v['events_at_postprocessed']
            fa = ev['false_alarms_per_hour']
            fa_s = f"{fa:.2f}" if fa is not None else "n/a"
            lines.append(
                f"| {subj} | {ft['mcc']:+.3f} | {pp['mcc']:+.3f} | "
                f"{pp['sensitivity']*100:.1f}% | {pp['specificity']*100:.1f}% | "
                f"{ev['n_episodes']} | {ev['episode_detection_rate']*100:.1f}% | "
                f"{fa_s} |")
        lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def sweep_postproc(seq_length, smooth_windows=(1, 3, 5), bands=(0.0, 0.05, 0.10)):
    """Re-score post-processing over a (smooth_window x hysteresis_band) grid.

    No retrain: reuses each fold's saved probs + frozen threshold, recomputing
    only the cheap smoothing + hysteresis. Smoothing longer than a FoG episode
    washes out short ones (the dominant lever at 1 Hz / ~4 s episodes); the band
    trades specificity (sticky FoG) for flicker debouncing.
    """
    data_dir = os.path.join(PROCESSED_DATA_DIR, f'win_{seq_length}')
    folds = []
    for subj in get_all_subjects(data_dir):
        r = evaluate_fold(subj, seq_length)
        if r is None:
            continue
        probs, targets, _, fold_thr, rec_lengths = r
        if len(targets) == 0:
            continue
        folds.append((subj, probs, targets, fold_thr, rec_lengths))
    if not folds:
        return

    print(f"\n=== Post-proc sweep (win {seq_length}) — per-subject mean pp MCC ===")
    print("  smooth\\band | " + " | ".join(f"{b:.2f}" for b in bands))
    for sw in smooth_windows:
        cells = []
        for band in bands:
            per_subj = []
            for _, probs, targets, fold_thr, rec_lengths in folds:
                prob_segs = _split_by_lengths(probs, rec_lengths)
                pp_segs = [postprocess_predictions(
                    ps, threshold=fold_thr, smooth_window=sw,
                    hysteresis_band=band)[0] for ps in prob_segs]
                pp_preds = (np.concatenate(pp_segs) if pp_segs
                            else np.array([], dtype=np.int64))
                per_subj.append(_metrics_from_preds(targets, pp_preds, fold_thr)['mcc'])
            cells.append(f"{np.mean(per_subj):+.3f}")
        print(f"  sw={sw:<8} | " + " | ".join(cells))


def main():
    summary = {}
    for seq in WINDOW_SIZES:
        r = evaluate_window(seq)
        if r is None:
            continue
        summary[f'win_{seq}'] = r
        _print_summary(r)
        sweep_postproc(seq)

    if not summary:
        print("No models evaluated. Did training finish?")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    out = os.path.join(MODELS_DIR, 'evaluation_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary -> {out}")

    table_path = os.path.join(REPORTS_DIR, 'results_table.md')
    _write_results_table(summary, table_path)
    print(f"Wrote per-subject results table -> {table_path}")

    best = max(summary.values(), key=lambda r: r['at_postprocessed']['mcc'])
    print(f"Best window by post-processed MCC: {best['window']} "
          f"(MCC={best['at_postprocessed']['mcc']:+.3f})")


if __name__ == "__main__":
    main()
