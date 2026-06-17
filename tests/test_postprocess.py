"""Hysteresis suppresses near-threshold flicker; smoothing is causal."""

import numpy as np

from inference.postprocess import smooth_probs, apply_hysteresis, postprocess_predictions


def test_smooth_probs_causal():
    p = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    s = smooth_probs(p, window=3)
    # Output at index 2 only sees p[0..2], which are all zero.
    assert s[2] == 0.0
    # Output at last index averages last three (all 1.0).
    assert abs(s[-1] - 1.0) < 1e-9


def test_hysteresis_suppresses_flicker():
    # Probability oscillates around 0.5 -> raw threshold gives chattering.
    p = np.array([0.4, 0.6, 0.4, 0.6, 0.4, 0.6])
    raw = (p >= 0.5).astype(np.int64)
    assert raw.tolist() == [0, 1, 0, 1, 0, 1]

    # With low=0.4, high=0.6: enters at first 0.6, never drops *below* 0.4.
    out = apply_hysteresis(p, low=0.4, high=0.6, initial_state=0)
    assert out.tolist() == [0, 1, 1, 1, 1, 1]


def test_hysteresis_clean_walk_to_freeze():
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.7, 0.2, 0.1])
    out = apply_hysteresis(p, low=0.4, high=0.6, initial_state=0)
    assert out.tolist() == [0, 0, 0, 1, 1, 1, 0, 0]


def test_postprocess_enters_at_threshold():
    # Probs above threshold but below the OLD straddled high (thr + band/2 = 0.6).
    # Asymmetric entry at threshold must still detect these (the bug we fixed:
    # low-prevalence folds collapsing to all-negative).
    p = np.full(10, 0.55)
    decisions, _ = postprocess_predictions(p, threshold=0.5, smooth_window=1,
                                           hysteresis_band=0.2)
    assert decisions.sum() == 10


def test_postprocess_exit_is_debounced():
    # thr 0.5, band 0.2 -> exit only below 0.3. 0.45 stays FoG; 0.2 exits.
    p = np.array([0.7, 0.7, 0.45, 0.45, 0.2])
    decisions, _ = postprocess_predictions(p, threshold=0.5, smooth_window=1,
                                           hysteresis_band=0.2)
    assert decisions.tolist() == [1, 1, 1, 1, 0]


def test_postprocess_wrapper_runs():
    rng = np.random.default_rng(0)
    p = np.clip(rng.normal(0.5, 0.2, 100), 0, 1)
    decisions, smoothed = postprocess_predictions(p, threshold=0.5, smooth_window=5,
                                                  hysteresis_band=0.2)
    assert decisions.shape == p.shape
    assert smoothed.shape == p.shape
    assert set(np.unique(decisions)).issubset({0, 1})
