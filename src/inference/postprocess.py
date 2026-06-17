"""Streaming post-processing for FoG probabilities (causal, MCU-portable).

Two stages: ``smooth_probs`` (causal boxcar moving average that suppresses
near-threshold flicker) and ``apply_hysteresis`` (Schmitt trigger that debounces
the decision so the cueing belt does not pulse on/off). Both take a 1-D array
for a single recording in chronological order; call them once per recording so
the state machine never crosses a recording boundary.
"""

from __future__ import annotations

import numpy as np


def smooth_probs(probs: np.ndarray, window: int = 5) -> np.ndarray:
    """Causal boxcar moving average. Output length matches input."""
    p = np.asarray(probs, dtype=np.float64)
    if window <= 1 or p.size == 0:
        return p.astype(np.float64)
    # Cumulative sum trick: smoothed[i] = mean(p[max(0, i-window+1) : i+1]).
    cs = np.concatenate(([0.0], np.cumsum(p)))
    idx = np.arange(p.size)
    lo = np.maximum(0, idx - window + 1)
    counts = (idx - lo + 1).astype(np.float64)
    return (cs[idx + 1] - cs[lo]) / counts


def apply_hysteresis(probs: np.ndarray, low: float = 0.4, high: float = 0.6,
                     initial_state: int = 0) -> np.ndarray:
    """Schmitt-trigger gate over `probs` — enter FREEZE at `high`, leave at `low`.

    Operates on a 1-D array of (already-smoothed) probabilities and returns
    int64 binary decisions of the same length. `initial_state` seeds the
    machine before the first sample (0 = walking, 1 = freeze) — pass the
    last state of the previous chunk for streaming-style continuity across
    recording boundaries.
    """
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"Invalid thresholds low={low} high={high}.")
    p = np.asarray(probs, dtype=np.float64)
    out = np.empty_like(p, dtype=np.int64)
    state = int(initial_state)
    for i, v in enumerate(p):
        if state == 0 and v >= high:
            state = 1
        elif state == 1 and v < low:
            state = 0
        out[i] = state
    return out


def postprocess_predictions(probs: np.ndarray, threshold: float,
                            smooth_window: int = 5, hysteresis_band: float = 0.1):
    """Convenience wrapper: smooth, then asymmetric hysteresis at the threshold.

    `threshold` is the per-fold operating point (e.g. Youden's J on inner val).
    Hysteresis is ASYMMETRIC: enter FREEZE at `threshold` and leave only when the
    smoothed prob drops below `threshold - hysteresis_band`.

    Why not straddle (high = threshold + band/2)? Straddling raises the entry bar
    above the chosen operating point, so low-prevalence folds whose smoothed
    probs never reach threshold + band/2 collapse to all-negative (observed:
    subjects with high fold-threshold MCC dropping to 0 after post-processing).
    Entering exactly at `threshold` is never stricter than plain thresholding;
    the band only debounces the exit, killing flicker without losing detections.
    """
    p_smooth = smooth_probs(probs, window=smooth_window)
    high = float(np.clip(threshold, 0.0, 1.0))
    low = float(np.clip(threshold - hysteresis_band, 0.0, 1.0))
    decisions = apply_hysteresis(p_smooth, low=low, high=high)
    return decisions, p_smooth
