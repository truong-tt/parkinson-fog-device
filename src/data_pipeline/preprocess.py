"""Segment raw IMU recordings into labeled windows.

The upstream stanfordnmbl/imu-fog-detection dataset provides 6 raw channels
per IMU at 128 Hz:

    ax, ay, az  (accelerometer, m/s^2)
    gx, gy, gz  (gyroscope,    rad/s)

We resample to FREQ_DESIRED=64 Hz then EXPAND those 6 raw channels into 9
input channels by splitting accelerometer into linear_acc + gravity via a
0.3 Hz lowpass (the gravity component is the slow part of acc; subtracting
it yields motion-only linear_acc). The TCN consumes these 9 channels; the
expansion is a deliberate inductive bias — it gives the model a clean
gravity-orientation signal without forcing it to learn the lowpass split:

  In (raw, 6):       ax  ay  az  gx  gy  gz                    @ 128 Hz raw
  Resample:          ax  ay  az  gx  gy  gz                    @ 64 Hz
  DSP split:         linear_acc[3] | gravity[3] | gyro[3]      @ 64 Hz

Output channel layout (9):
  0..2  linear_acc  (xyz, m/s^2, gravity removed)
  3..5  gravity     (xyz, m/s^2, slow component of acc)
  6..8  gyro        (xyz, rad/s)

A 2-second window at 64 Hz is 128 samples — matches WINDOW_DUR * FREQ_DESIRED
in the upstream dataset code.

Per-window scalar DSP features (Freeze Index, STFT band power) used to live
in the input tensor but were broadcast as constants across all timesteps —
flat columns that wasted model capacity. They're gone. The TCN learns its
own frequency cues from the linear_acc channels, given the dilated
receptive field.

Labels are saved per-timestep (shape (N, T)). Training derives the last-step
label on the fly when needed (causal real-time head) and uses the full vector
for the dense auxiliary loss.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import WINDOW_SIZES, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLING_RATE, WINDOW_OVERLAP

try:
    from .dsp import IMUFilter
except ImportError:
    from dsp import IMUFilter


NUM_FEATURE_CHANNELS = 9


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
    if raw_data_path.endswith('.xlsx'):
        df = pd.read_excel(raw_data_path)
    else:
        df = pd.read_csv(raw_data_path)

    subject_id = df['subject_ID'].iloc[0]
    file_id = os.path.basename(raw_data_path).rsplit('.', 1)[0]

    acc = df[['imu_lumbar_ax', 'imu_lumbar_ay', 'imu_lumbar_az']].values.astype(np.float32)
    gyro = df[['imu_lumbar_gx', 'imu_lumbar_gy', 'imu_lumbar_gz']].values.astype(np.float32)
    labels = df['freeze_label'].values.astype(np.int64)
    timestamps = df['time'].values if 'time' in df.columns else None

    imu = IMUFilter(fs=fs)
    linear_acc, gravity, gyro_f, _ = imu.process_signal(acc, gyro, timestamps)

    if timestamps is not None and len(linear_acc) != len(labels):
        labels = _resample_labels(timestamps, labels, fs)

    n = min(len(linear_acc), len(labels))
    linear_acc, gravity, gyro_f, labels = linear_acc[:n], gravity[:n], gyro_f[:n], labels[:n]

    for win_size in window_sizes:
        win_dir = os.path.join(output_base_dir, f'win_{win_size}')
        os.makedirs(win_dir, exist_ok=True)
        step = max(1, int(win_size * (1 - overlap)))

        x_windows, y_windows = [], []
        for i in range(0, n - win_size + 1, step):
            x_windows.append(_stack_features(linear_acc, gravity, gyro_f, i, win_size))
            y_windows.append(labels[i:i + win_size].astype(np.int64))

        if not x_windows:
            continue

        x_npy = np.stack(x_windows).astype(np.float32)            # (N, T, 9)
        y_npy = np.stack(y_windows).astype(np.int64)              # (N, T)
        np.save(os.path.join(win_dir, f'subj_{subject_id}_{file_id}_x.npy'), x_npy)
        np.save(os.path.join(win_dir, f'subj_{subject_id}_{file_id}_y.npy'), y_npy)
        last_pos = float(y_npy[:, -1].mean())
        print(f"  win_{win_size}: saved {len(x_windows)} windows (last-step pos_rate={last_pos:.3f})")


def main():
    # Search recursively so both flat (data/raw/*.csv) and the documented
    # nested layout (data/raw/subject_XX/*.csv) are picked up.
    raw_files = sorted(
        glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.xlsx"), recursive=True) +
        glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.csv"), recursive=True))
    if not raw_files:
        print(f"No raw data files found under {RAW_DATA_DIR} (searched recursively). "
              "Place the Stanford NMBL CSV/XLSX recordings there first.")
        return

    print(f"Found {len(raw_files)} raw files. Starting preprocessing...")
    for file_path in raw_files:
        print(f"Processing: {os.path.basename(file_path)}")
        try:
            segment_file(file_path, PROCESSED_DATA_DIR, WINDOW_SIZES)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
