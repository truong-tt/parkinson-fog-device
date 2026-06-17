# HopeGait — Decision Log

Dated record of design decisions and their rationale. Solo project under
lecturer supervision; this log is the process evidence for the course's
teamwork/supervision requirement. Newest first.

---

## 2026-06-17 — Project advancement plan adopted

**Decision:** Execute the 2-week advancement plan
(`~/.claude/plans/make-a-plan-to-dynamic-popcorn.md`) scoped to Stanford NMBL
only, PTQ accuracy-delta only (no QAT/pruning), solo, Colab GPU.

**Why:** Course requires the three stages (problem analysis → solution design →
prototyping) with a defensible result. The codebase is already well-engineered;
the missing pieces are (1) a baseline to justify the TCN, (2) correctness fixes
that make the LOSO numbers honest, and (3) a measured int8-vs-fp32 delta to
back the "edge deployment" claim.

**Scope boundary:** second dataset, QAT, pruning, and on-hardware MCU profiling
are explicitly deferred to "future work" — named so the boundary is deliberate,
not an omission.

---

## 2026-06-18 — Post-processing collapsed low-prevalence folds; made hysteresis asymmetric

**Finding:** First good LOSO run (EMA 0.99) gave pooled post-pp MCC +0.337 but
per-subject mean only +0.071 (vs +0.204 at the fold threshold). Subjects 2/5/6
dropped to MCC 0.000 / 0% sensitivity after post-processing — e.g. subj 2 fell
from +0.356 @fold-threshold to 0.000.

**Cause:** the hysteresis straddled the operating point (`high = threshold +
band/2`), so entry was *stricter* than the threshold itself. Low-prevalence
folds whose smoothed probs never reached `threshold + 0.1` never triggered.

**Decision:** asymmetric hysteresis — enter at `threshold`, leave below
`threshold - band`. Entry is never stricter than plain thresholding; the band
only debounces the exit. Also surfaced the broader point: **report per-subject
mean, not the prevalence-weighted pooled MCC** (pooling hid a 0.27 gap).

---

## 2026-06-18 — quantize_model.py looked for the checkpoint in the wrong dir

**Fix:** default checkpoint path now `MODELS_DIR/win_<window>/hopegait_tcn_best_subj<id>.pth`
(training writes there); it previously looked in the MODELS_DIR root and failed
with "checkpoint not found". Also: the edge stack (TF 2.15) needs Python ≤3.11,
so it can't run on current Colab (3.12) — the int8 delta is produced locally.

---

## 2026-06-17 — EMA decay 0.999 too slow for this dataset (training finding)

**Finding:** A first LOSO run (25 epochs, EMA decay 0.999) gave LOSO ROC-AUC
≈ 0.52 (chance) and post-processed MCC −0.13 — worse than the +0.20 Decision
Tree baseline. The raw model WAS learning (train focal loss 0.092 → 0.064), but
validation/selection/checkpointing all use the EMA shadow copy.

**Why it fails:** With ~5k train windows / batch 64 ≈ 78 steps/epoch × 25 ≈ 2k
steps, EMA decay 0.999 (≈1000-step horizon) leaves the shadow weights roughly
half-way to init for the whole run. We then save those near-init EMA weights, so
the deployed model scores at chance.

**Decision:** Drop EMA decay to **0.99** and train **60 epochs**. A single-fold
test (subj 3) lifted best val MCC from 0.178 → 0.22 and still climbing. Full
retrain via `notebooks/colab_runner.ipynb` (`HOPEGAIT_EMA_DECAY=0.99`). The
Colab GPU notebook (removed earlier when CPU looked sufficient) is restored for
these longer convergence runs.

---

## 2026-06-17 — Augmentation must happen in physical sensor space (B1)

**Decision:** Restructure `FoGDataset` to hold *raw* (unscaled) windows plus the
fitted `RobustScaler`. Apply geometric/gain augmentations (rotation, per-channel
gain, time-shift) in physical units, then scale, then add jitter in normalized
space.

**Why:** The scaler is per-channel anisotropic (independent IQR per axis).
Rotating the 3 accel axes *after* anisotropic scaling is not a rigid SO(3)
rotation in sensor space — it teaches the network a wrong invariance. Rotation
only commutes with isotropic scaling, so the rotation must precede the scaler.

**Trade-off considered:** alternative was making `RobustScaler` isotropic per
sensor-block; rejected because per-axis centering (median subtraction) still
breaks rotation-equivariance, and it would change the scaling semantics for the
whole pipeline. Reordering augmentation is the smaller, cleaner fix.

---

## 2026-06-17 — Event metrics must not cross recording seams (B3)

**Decision:** Post-processing and episode counting now run per recording, not
per concatenated subject stream. Added `recording_lengths` (dataset) and
`event_metrics_from_segments` (evaluate) and split each subject's predictions
back into recordings before scoring.

**Why:** `postprocess.py` documents that its hysteresis state machine must not
cross recording boundaries, but the evaluator was feeding it a subject's
recordings concatenated end to end. A freeze episode at the end of one walk and
one at the start of the next were merged into a single episode, miscounting
detections and false alarms. Sample-level metrics are unaffected (per-sample),
so those stay on the concatenated array.

---

## 2026-06-17 — Class weights derived from data, not hardcoded (B4)

**Decision:** Focal-loss alpha is now `[p, 1-p]` from each fold's actual FoG
rate (`meta['train_pos_rate']`), falling back to the config constant only for
degenerate folds.

**Why:** The hardcoded `[0.2, 0.8]` is exactly normalized inverse-frequency for
an assumed 20% FoG rate; deriving it per fold tracks each subject pool's real
imbalance instead of assuming one number for all folds.

---

## 2026-06-17 — int8 calibration was on the wrong distribution (C3 finding)

**Decision:** `quantize_model.py` now scales calibration windows with the
fold's persisted `RobustScaler` before feeding them to the TFLite converter,
and a new `evaluate_tflite.py` measures the int8-vs-fp32 MCC delta on
identically-scaled inputs.

**Why:** The model trains and infers on RobustScaler-normalized windows, but
PTQ calibration was reading RAW `*_x.npy` windows. The int8 input range was
therefore set from the wrong distribution, which would have shown up as a large
(but spurious) quantization error. Calibrating on scaled windows makes the
measured delta attributable to quantization alone. This is the key edge-phase
finding to write up in the report.
