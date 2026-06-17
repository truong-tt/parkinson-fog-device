"""A2: the non-deep LOSO baselines run end-to-end on synthetic data and return
well-formed per-model MCC summaries."""

import numpy as np

from baselines.freeze_index_baseline import run_baselines, window_features, FI_INDEX


def _make_data(tmp_path, fs=64, T=32):
    win = tmp_path / "win_32"
    win.mkdir()
    rng = np.random.default_rng(0)
    t = np.arange(T) / fs
    for subj in ('A', 'B', 'C'):
        for fid in (1, 2):
            n = 8
            X = np.zeros((n, T, 9), np.float32)
            y = np.zeros((n, T), np.int64)
            for i in range(n):
                label = i % 2
                # FoG -> 6 Hz (freeze band); walking -> 1.5 Hz (loco band).
                freq = 6.0 if label else 1.5
                sig = np.sin(2 * np.pi * freq * t).astype(np.float32)
                X[i, :, 0] = sig
                X[i, :, 1] = 0.5 * sig
                X[i] += rng.standard_normal((T, 9)).astype(np.float32) * 0.05
                y[i, :] = label
            np.save(win / f"subj_{subj}_run{fid}_x.npy", X)
            np.save(win / f"subj_{subj}_run{fid}_y.npy", y)
    return str(win)


def test_window_features_shape_and_fi_separates():
    fs, T = 64, 32
    t = np.arange(T) / fs
    walk = np.zeros((T, 9), np.float32); walk[:, 0] = np.sin(2 * np.pi * 1.5 * t)
    fog = np.zeros((T, 9), np.float32); fog[:, 0] = np.sin(2 * np.pi * 6.0 * t)
    f_walk, f_fog = window_features(walk), window_features(fog)
    assert f_walk.shape == (29,)
    # Freeze index is higher for the freeze-band signal.
    assert f_fog[FI_INDEX] > f_walk[FI_INDEX]


def test_baselines_run(tmp_path):
    data_dir = _make_data(tmp_path)
    r = run_baselines(data_dir, 32, seed=42)
    assert r is not None
    assert set(r['models']).issubset({'freeze_index', 'logreg', 'tree'})
    assert r['models'], "expected at least one model scored"
    for m in r['models'].values():
        assert -1.0 <= m['mcc_mean'] <= 1.0
        assert m['mcc_std'] >= 0.0
