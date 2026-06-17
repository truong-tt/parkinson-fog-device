"""B3: event metrics must not cross recording seams, and per-recording window
counts are recovered in the same order the dataloaders concatenate them."""

import numpy as np

from training.evaluate import (event_level_metrics, event_metrics_from_segments,
                               _split_by_lengths)
from data_pipeline.dataset import recording_lengths


def test_event_metrics_seam_safe():
    # Recording 1 ends in FoG; recording 2 begins in FoG. Concatenated, the two
    # episodes merge into one; segmented, they stay two.
    t1 = np.array([0, 0, 1, 1]); p1 = np.array([0, 0, 1, 1])
    t2 = np.array([1, 1, 0, 0]); p2 = np.array([1, 1, 0, 0])

    seg = event_metrics_from_segments([(t1, p1), (t2, p2)])
    assert seg['n_episodes'] == 2
    assert seg['episode_detection_rate'] == 1.0

    concat = event_level_metrics(np.concatenate([t1, t2]),
                                 np.concatenate([p1, p2]))
    assert concat['n_episodes'] == 1  # the seam bug we are avoiding


def test_split_by_lengths_fallback():
    arr = np.arange(10)
    # Good split.
    segs = _split_by_lengths(arr, [4, 6])
    assert [len(s) for s in segs] == [4, 6]
    # Mismatched lengths -> single fallback segment (graceful degrade).
    segs = _split_by_lengths(arr, [4, 5])
    assert len(segs) == 1 and len(segs[0]) == 10
    # Empty lengths -> single segment.
    assert len(_split_by_lengths(arr, [])) == 1


def test_recording_lengths_order_and_skip(tmp_path):
    win = tmp_path / "win_32"
    win.mkdir()
    rng = np.random.default_rng(0)
    # run1: 5 windows, run2: 3 windows, both with labels.
    for fid, n in (("run1", 5), ("run2", 3)):
        np.save(win / f"subj_A_{fid}_x.npy",
                rng.standard_normal((n, 32, 9)).astype(np.float32))
        np.save(win / f"subj_A_{fid}_y.npy",
                rng.integers(0, 2, size=(n, 32)).astype(np.int64))
    # run3 has no labels -> must be skipped (matches _load_files).
    np.save(win / "subj_A_run3_x.npy",
            rng.standard_normal((7, 32, 9)).astype(np.float32))

    assert recording_lengths(str(win), 'A') == [5, 3]
