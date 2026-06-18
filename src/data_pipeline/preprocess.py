"""Segment raw IMU recordings into labeled windows.

The upstream Stanford NMBL recordings carry many body-worn IMUs at 128 Hz; this
pipeline uses only the lumbar sensor (6 of the ``imu_*`` columns) plus ``time``,
``subject_ID`` and ``freeze_label``. We resample to 64 Hz and expand the 3 accel
channels into ``linear_acc + gravity`` via a 0.3 Hz lowpass — a deliberate
inductive bias giving the model a clean orientation signal. Output channels:

    0..2  linear_acc (xyz, m/s^2, gravity removed)
    3..5  gravity    (xyz, m/s^2, low-frequency accel)
    6..8  gyro       (xyz, rad/s)

A 2 s window at 64 Hz is 128 samples. Labels are saved per-timestep ``(N, T)``;
training derives the last-step label (causal head) and uses the full vector for
the dense auxiliary loss. ``main`` writes a ``preprocess_summary.json`` manifest
(per-file rows, windows, pos_rate, NaNs filled, skips) for reproducibility.
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import (WINDOW_SIZES, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLING_RATE,
                    WINDOW_OVERLAP, RAW_SAMPLING_RATE)

try:
    from .dsp import IMUFilter
except ImportError:
    from dsp import IMUFilter


NUM_FEATURE_CHANNELS = 9

LUMBAR_ACC_COLS = ['imu_lumbar_ax', 'imu_lumbar_ay', 'imu_lumbar_az']
LUMBAR_GYRO_COLS = ['imu_lumbar_gx', 'imu_lumbar_gy', 'imu_lumbar_gz']
LUMBAR_COLS = LUMBAR_ACC_COLS + LUMBAR_GYRO_COLS
REQUIRED_COLS = ['subject_ID', 'freeze_label'] + LUMBAR_COLS


def _read_recording(path):
    if path.endswith('.xlsx'):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _validate_columns(df, path):
    """Raise a clear error naming any missing required columns."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)}: missing required columns {missing}. "
            f"Expected the lumbar IMU columns plus subject_ID/freeze_label."
        )


def _extract_lumbar(df):
    """Pull the 6 lumbar channels, bridging NaN dropouts. Returns (acc, gyro, n_nan).

    Sensor dropouts leave NaNs that would propagate through the filters and
    poison every downstream window. Index-based linear interpolation bridges
    short gaps; leading/trailing NaNs are back/forward filled.
    """
    block = df[LUMBAR_COLS]
    n_nan = int(block.isna().to_numpy().sum())
    if n_nan:
        block = block.interpolate(limit_direction='both').bfill().ffill()
    acc = block[LUMBAR_ACC_COLS].to_numpy(dtype=np.float32)
    gyro = block[LUMBAR_GYRO_COLS].to_numpy(dtype=np.float32)
    return acc, gyro, n_nan


def _stack_features(linear_acc, gravity, gyro, start, win_size):
    la = linear_acc[start:start + win_size]
    gr = gravity[start:start + win_size]
    gy = gyro[start:start + win_size]
    return np.hstack((la, gr, gy)).astype(np.float32)


def _resample_labels(timestamps, labels, fs):
    new_ts = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    f = interp1d(timestamps, labels.astype(np.float32), kind='nearest', fill_value='extrapolate')
    return f(new_ts).astype(np.int64)


def segment_file(raw_data_path, output_base_dir, window_sizes, overlap=WINDOW_OVERLAP, fs=SAMPLING_RATE):
    """Preprocess one recording into windowed ``_x.npy``/``_y.npy`` per size.

    Args:
        raw_data_path: CSV/XLSX recording with the expected ``imu_lumbar_*`` columns.
        output_base_dir: Root under which ``win_<size>/`` directories are written.
        window_sizes: Window lengths in samples to emit.
        overlap: Fractional window overlap.
        fs: Target sampling rate in Hz.

    Returns:
        Summary dict for the manifest: subject id, raw/resampled row counts, NaNs
        filled, per-window counts + pos_rate, and any window sizes skipped.

    Raises:
        ValueError: If a required column is absent.
    """
    df = _read_recording(raw_data_path)
    _validate_columns(df, raw_data_path)

    subject_id = int(df['subject_ID'].iloc[0])
    file_id = os.path.basename(raw_data_path).rsplit('.', 1)[0]

    acc, gyro, n_nan = _extract_lumbar(df)
    labels = df['freeze_label'].to_numpy(dtype=np.int64)
    timestamps = df['time'].to_numpy() if 'time' in df.columns else None

    imu = IMUFilter(fs=fs, raw_fs=RAW_SAMPLING_RATE)
    linear_acc, gravity, gyro_f, _ = imu.process_signal(acc, gyro, timestamps)

    if timestamps is not None and len(linear_acc) != len(labels):
        labels = _resample_labels(timestamps, labels, fs)

    n = min(len(linear_acc), len(labels))
    linear_acc, gravity, gyro_f, labels = linear_acc[:n], gravity[:n], gyro_f[:n], labels[:n]

    summary = {'file': file_id, 'subject_id': subject_id,
               'raw_rows': int(len(df)), 'resampled_rows': int(n),
               'lumbar_nans_filled': n_nan, 'windows': {}, 'skipped': []}

    for win_size in window_sizes:
        if n < win_size:
            summary['skipped'].append(win_size)
            print(f"  win_{win_size}: SKIP ({n} samples < window {win_size})")
            continue

        win_dir = os.path.join(output_base_dir, f'win_{win_size}')
        os.makedirs(win_dir, exist_ok=True)
        step = max(1, int(win_size * (1 - overlap)))

        x_windows, y_windows = [], []
        for i in range(0, n - win_size + 1, step):
            x_windows.append(_stack_features(linear_acc, gravity, gyro_f, i, win_size))
            y_windows.append(labels[i:i + win_size].astype(np.int64))

        if not x_windows:
            summary['skipped'].append(win_size)
            continue

        x_npy = np.stack(x_windows).astype(np.float32)            # (N, T, 9)
        y_npy = np.stack(y_windows).astype(np.int64)              # (N, T)
        np.save(os.path.join(win_dir, f'subj_{subject_id}_{file_id}_x.npy'), x_npy)
        np.save(os.path.join(win_dir, f'subj_{subject_id}_{file_id}_y.npy'), y_npy)
        last_pos = float(y_npy[:, -1].mean())
        summary['windows'][str(win_size)] = {'n': len(x_windows), 'pos_rate': round(last_pos, 4)}
        print(f"  win_{win_size}: saved {len(x_windows)} windows (last-step pos_rate={last_pos:.3f})")

    return summary


def _write_manifest(summaries, n_files, n_ok, n_err, out_dir):
    """Aggregate per-file summaries into preprocess_summary.json."""
    totals = {}
    for w in WINDOW_SIZES:
        tot, pos_acc = 0, 0.0
        for s in summaries:
            wd = s.get('windows', {}).get(str(w))
            if wd:
                tot += wd['n']
                pos_acc += wd['n'] * wd['pos_rate']
        totals[f'win_{w}'] = {'windows': tot,
                              'pos_rate': round(pos_acc / tot, 4) if tot else 0.0}

    subjects = sorted({s['subject_id'] for s in summaries if 'subject_id' in s})
    manifest = {
        'n_files': n_files, 'n_ok': n_ok, 'n_errors': n_err,
        'n_subjects': len(subjects), 'subjects': subjects,
        'raw_sampling_rate': RAW_SAMPLING_RATE, 'target_sampling_rate': SAMPLING_RATE,
        'window_sizes': WINDOW_SIZES, 'overlap': WINDOW_OVERLAP,
        'totals': totals, 'files': summaries,
    }
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'preprocess_summary.json')
    with open(out, 'w') as f:
        json.dump(manifest, f, indent=2)
    return out, totals


def main():
    # Search recursively so both flat (data/raw/*.xlsx) and a nested
    # (data/raw/<subject>/*.csv) layout are picked up. Subject identity comes
    # from the subject_ID column, not the path.
    raw_files = sorted(
        glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.xlsx"), recursive=True) +
        glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.csv"), recursive=True))
    if not raw_files:
        print(f"No raw data files found under {RAW_DATA_DIR} (searched recursively). "
              "Place the Stanford NMBL CSV/XLSX recordings there first.")
        return

    print(f"Found {len(raw_files)} raw files. Starting preprocessing...")
    summaries, n_ok, n_err = [], 0, 0
    for file_path in raw_files:
        print(f"Processing: {os.path.basename(file_path)}")
        try:
            summaries.append(segment_file(file_path, PROCESSED_DATA_DIR, WINDOW_SIZES))
            n_ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            summaries.append({'file': os.path.basename(file_path), 'error': str(e)})
            n_err += 1

    out, totals = _write_manifest(summaries, len(raw_files), n_ok, n_err, PROCESSED_DATA_DIR)
    print(f"\nPreprocessing complete: {n_ok} ok, {n_err} errors.")
    for win, t in totals.items():
        print(f"  {win}: {t['windows']} windows, pos_rate={t['pos_rate']}")
    print(f"Manifest -> {out}")


if __name__ == "__main__":
    main()
