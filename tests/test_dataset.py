"""LOSO splitter never leaks subjects; rotation aug preserves vector norms;
labels are accepted in both dense and legacy 1D shapes."""

import os
import numpy as np
import torch

from data_pipeline.dataset import (FoGDataset, _random_rotation_matrix,
                                   _apply_rotation, ACC_SLICE, GRAV_SLICE,
                                   GYRO_SLICE, _group_files_by_subject,
                                   create_loso_dataloaders)
from data_pipeline.dsp import RobustScaler


def test_rotation_preserves_norms():
    rng = np.random.default_rng(0)
    win = rng.standard_normal((64, 9)).astype(np.float32)
    R = _random_rotation_matrix(np.deg2rad(15), rng)
    out = _apply_rotation(win, R)
    for sl in (ACC_SLICE, GRAV_SLICE, GYRO_SLICE):
        n_in = np.linalg.norm(win[:, sl], axis=1)
        n_out = np.linalg.norm(out[:, sl], axis=1)
        np.testing.assert_allclose(n_in, n_out, atol=1e-4)


def test_rotation_matrix_is_orthonormal():
    rng = np.random.default_rng(2)
    for _ in range(20):
        R = _random_rotation_matrix(np.deg2rad(30), rng)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
        assert abs(np.linalg.det(R) - 1.0) < 1e-5


def test_dataset_accepts_legacy_1d_labels():
    X = np.zeros((4, 32, 9), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    ds = FoGDataset(X, y, augment=False)
    x_t, y_t = ds[0]
    assert y_t.shape == (32,)
    assert int(y_t[-1].item()) == 0


def test_dataset_accepts_dense_labels_and_returns_tensors():
    X = np.zeros((3, 16, 9), dtype=np.float32)
    y = np.zeros((3, 16), dtype=np.int64)
    y[1, -1] = 1
    ds = FoGDataset(X, y, augment=False)
    x_t, y_t = ds[1]
    assert isinstance(x_t, torch.Tensor)
    assert y_t.shape == (16,)
    assert int(y_t[-1].item()) == 1


def test_augmentation_runs_in_physical_space_before_scaling():
    # B1: rotation must precede the (anisotropic) scaler so it stays a rigid
    # SO(3) transform in sensor space. Pin the contract by replaying the exact
    # RNG draw order of __getitem__ and asserting output == scale(rotate(raw)).
    raw = np.random.default_rng(7).standard_normal((16, 9)).astype(np.float32)
    X = raw[None]  # (1, 16, 9)
    y = np.zeros((1, 16), dtype=np.int64)
    scaler = RobustScaler(median=np.zeros(9, np.float32),
                          iqr=np.arange(1, 10, dtype=np.float32))  # anisotropic
    ds = FoGDataset(X, y, scaler=scaler, augment=True, augment_prob=0.0,
                    rotation_prob=1.0, seed=123, rotation_max_deg=20.0)
    x_out, _ = ds[0]

    # Replicate the exact RNG sequence: rotation gate, rotation draw, scaling
    # gate (skipped, p=0), time-shift gate (skipped), scale, jitter gate.
    rng = np.random.default_rng(123)
    assert rng.random() < 1.0
    R = _random_rotation_matrix(np.deg2rad(20.0), rng)
    expected = _apply_rotation(raw, R)
    rng.random(); rng.random()
    expected = scaler.transform(expected)
    rng.random()
    np.testing.assert_allclose(x_out.numpy(), expected, atol=1e-5)


def test_augmentation_is_reproducible_with_seed():
    # B2: same seed -> identical augmented windows across fresh dataset instances.
    X = np.random.default_rng(1).standard_normal((6, 16, 9)).astype(np.float32)
    y = np.zeros((6, 16), dtype=np.int64)

    def run():
        ds = FoGDataset(X, y, augment=True, seed=99)
        return [ds[i][0].numpy().copy() for i in range(len(ds))]

    for xa, xb in zip(run(), run()):
        np.testing.assert_array_equal(xa, xb)


def test_loso_no_subject_leak(tmp_path):
    # Synthesize processed data for 4 fake subjects, 2 files each.
    win_dir = tmp_path / "win_32"
    win_dir.mkdir()
    rng = np.random.default_rng(0)
    for subj in ('A', 'B', 'C', 'D'):
        for fid in (1, 2):
            X = rng.standard_normal((5, 32, 9)).astype(np.float32)
            y = rng.integers(0, 2, size=(5, 32)).astype(np.int64)
            np.save(win_dir / f"subj_{subj}_run{fid}_x.npy", X)
            np.save(win_dir / f"subj_{subj}_run{fid}_y.npy", y)

    train_loader, val_loader, test_loader, _, meta = create_loso_dataloaders(
        str(win_dir), test_subject='B', batch_size=4, augment_train=False, seed=42)

    assert meta['test_subject'] == 'B'
    assert meta['val_subject'] != 'B'
    assert 'B' not in meta['train_subjects']
    assert meta['val_subject'] not in meta['train_subjects']
