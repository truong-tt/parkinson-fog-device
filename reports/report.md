# HopeGait: Real-Time Freezing-of-Gait Detection on a Lumbar IMU

**Technical report — university ML course.** Solo project under lecturer
supervision. Dataset: Stanford NMBL IMU FoG. Not a medical device.

> Numbers in `{{...}}` are filled after the Colab run from `models/` and
> `reports/` artifacts. Figures referenced below are produced by `scripts/eda.py`
> and the evaluator.

---

## Abstract

Freezing of Gait (FoG) is a sudden, involuntary gait arrest and a leading fall
risk in Parkinson's disease. We build a causal Temporal Convolutional Network
(TCN) that detects FoG from a single lumbar IMU in real time, suitable for an
on-body cueing device. We (1) characterize the Stanford NMBL dataset, (2)
establish classical baselines to justify a deep model, (3) train and evaluate
under leave-one-subject-out (LOSO) cross-validation with honest, leakage-free
thresholds, and (4) quantize to int8 and measure the accuracy cost of edge
deployment. On 7 subjects (7,846 windows, 26.0% FoG), the TCN reaches ROC-AUC **0.77** and a
per-subject LOSO MCC of **+0.20 ± 0.13** at its operating threshold — matching
the best classical baseline (+0.20 Decision Tree) while ranking better — and with
streaming post-processing detects **78% of freeze episodes at 0.17 s latency**
(3.7 false alarms/h). Honest per-subject evaluation surfaced two corrections:
the prevalence-pooled MCC masked a 0.27 per-subject gap, and symmetric hysteresis
collapsed low-prevalence folds until switched to asymmetric. int8 quantization is
**effectively lossless (ΔMCC +0.001)** at a **270 KiB**, MCU-sized model.

---

## 1. Problem Analysis (stage 1)

### 1.1 Clinical problem
FoG episodes last from <1 s to tens of seconds and respond to external cues
(rhythmic audio/haptic). A wearable that detects onset within ~1 s and cues the
patient can reduce falls. Constraints: **causal** (no future samples),
**low-latency**, and small enough for a microcontroller (MCU).

### 1.2 Dataset characterization
Stanford NMBL lumbar IMU, 6 channels @ 128 Hz → resampled 64 Hz, 2 s windows
(128 samples), expanded to 9 channels (linear-acc + gravity + gyro). Key facts
from `scripts/eda.py` (`reports/eda_summary.md`):

- Subjects: **7**, windows: **7,846**.
- Class imbalance — overall FoG rate **26.0%**
  (Figure: `figures/class_balance_win128.png`).
- Per-subject prevalence varies from **5.7% (subj 6) to 63.0% (subj 4)**
  (Figure: `figures/per_subject_prevalence_win128.png`) — motivates LOSO and
  per-fold class weighting.
- FoG episode lengths: median **7.0 s** (mean 9.9 s, max 200 s)
  (Figure: `figures/episode_lengths_win128.png`) — motivates event-level
  metrics, not just per-window accuracy.

### 1.3 Baseline — why a deep model?
Two classical LOSO baselines (`src/baselines/freeze_index_baseline.py`):

| Baseline | LOSO MCC (mean ± std) |
|---|---|
| Freeze Index threshold (Bächlin 2010) | −0.05 ± 0.07 |
| Logistic Regression on hand features | +0.16 ± 0.12 |
| Decision Tree on hand features | **+0.20 ± 0.12** |

The single Freeze-Index threshold is no better than chance on this lumbar data;
the best classical model (Decision Tree on FI + STFT band power + per-channel
stats) reaches **+0.20** MCC. This sets the bar the TCN must clear to justify
its complexity (§3).

---

## 2. Solution Design (stage 2)

### 2.1 Architecture
Causal TCN: 4 dilated residual blocks `(32, 64, 96, 128)`, kernel 3, dilations
`1/2/4/8` → receptive field 61 samples (~0.95 s). Streaming-friendly building
blocks: per-timestep LayerNorm (batch-1 stable), causal Squeeze-Excite
(cumulative mean, no future leak), stochastic depth. Two heads share one 1×1
conv: a last-step head (MCU runtime) and a dense per-timestep head (training-time
auxiliary supervision). See `src/models/tcn_model.py` and README §3 for the full
rationale and citations.

### 2.2 Evaluation protocol & leakage controls
LOSO by true subject identity; inner-val subject drawn from the training pool
only; threshold chosen by Youden-J on inner-val and frozen before the test fold;
`RobustScaler` fit on training data only. Primary metric **MCC** (imbalance-safe);
also report sensitivity/specificity, PR/ROC-AUC, and **event-level** metrics
(episode detection rate, latency, false alarms/hour).

### 2.3 Methodology corrections made in this work
| Fix | Problem | Resolution |
|---|---|---|
| B1 | Rotation augmentation ran after anisotropic scaling → not a rigid SO(3) transform | Augment in physical units, scale last (`FoGDataset`) |
| B2 | Augmentation RNG unseeded → non-reproducible | Seeded generator + per-worker init |
| B3 | Post-processing/event metrics crossed recording seams | Per-recording splitting (`recording_lengths`, `event_metrics_from_segments`) |
| B4 | Class weights hardcoded | Derived per fold from actual FoG rate |
| B5 | Only a pooled MCC reported | Per-subject table + mean ± std |

Each is recorded with rationale in `reports/decision_log.md` and covered by a
unit test.

### 2.4 Edge design (PTQ)
Post-training int8 quantization via a representative dataset. **Finding:**
calibration was reading raw windows while the model consumes scaled windows;
we now calibrate on scaled windows (the fold's `RobustScaler`) so the measured
int8 error reflects quantization, not a distribution mismatch.

---

## 3. Implementation & Results (stage 3)

### 3.1 LOSO results
Per-subject table: `reports/results_table.md` (generated by the evaluator).
Trained with EMA decay 0.99, 60 epochs (the default 0.999 left the saved EMA
weights near-init on this small set — see decision log).

**Threshold-agnostic ranking is strong:** PR-AUC **0.593**, ROC-AUC **0.772**.

Pooled vs per-subject (the distinction matters — pooling is prevalence-weighted
and flattered by the one high-prevalence subject):

| | MCC | Sens | Spec |
|---|---|---|---|
| Pooled @0.5 | +0.281 | 84.7% | 46.4% |
| Pooled @post-processed | +0.277 | 87.6% | 42.4% |
| **Per-subject mean @fold-threshold** | **+0.204 ± 0.129** | — | — |
| **Per-subject mean @post-processed** | **+0.182 ± 0.129** | — | — |

Event level (post-processed): detection rate **78.2%**, mean latency
**0.17 s**, false alarms/h **3.72**.

**TCN vs baseline:** at the honest per-subject operating point the TCN **matches**
the Decision-Tree baseline (0.204 vs 0.204 MCC) and ranks markedly better
(ROC-AUC 0.772).

A post-processing finding drove a fix. The original symmetric hysteresis (enter
at `threshold + band/2`) was *stricter* than the operating threshold and
collapsed low-prevalence folds to all-negative — subject 2 fell from +0.356
@fold-threshold to 0.000, dragging the per-subject post-pp mean to +0.071.
Switching to **asymmetric hysteresis** (enter at the threshold, debounce only the
exit) recovered it: per-subject post-pp mean **+0.071 → +0.182** (≈ the
fold-threshold), and episode detection **32% → 78%** at 0.17 s latency. The
trade-off is now visible and clinically tunable: sensitivity rose at the cost of
specificity and **3.7 false alarms/h** (subject 5 worst at 14/h). Subject 6 (5.7%
FoG, the rarest) stays undetectable — a data-scale limit, not a pipeline bug. Two
takeaways for the report: **report per-subject mean, not the prevalence-pooled
MCC** (pooling hid a 0.27 gap), and **streaming post-processing must enter at the
operating point**, not above it.

### 3.2 Edge int8 delta
Measured on the subject-3 fold (`models/int8_delta.json`), int8 PTQ calibrated
on real scaled windows. Conversion runs on Python ≤3.11 (TF 2.15); not on current
Colab (3.12), so it is produced locally.

| | fp32 | int8 | Δ |
|---|---|---|---|
| MCC | +0.195 | +0.197 | **+0.001** |
| Sensitivity | 39.3% | 39.6% | — |
| Specificity | 79.7% | 79.6% | — |
| Model size | — | **269.9 KiB** | — |
| Parameters | 185,770 | — | — |

Probability MAE fp32↔int8: **0.0015**. int8 quantization is effectively
lossless here (ΔMCC +0.001, within run-to-run noise) while the model is MCU-sized
at 270 KiB — the central edge-deployment claim, now measured rather than asserted.

---

## 4. Limitations & Future Work
Deliberately out of scope here (constraints: Stanford-only, solo, 2 weeks, PTQ
only): cross-dataset/external validation (e.g. Daphnet), quantization-aware
training, magnitude pruning, on-hardware MCU latency/RAM/flash profiling, and
architecture search. The int8 numbers are host-side TFLite, not on-device.

---

## 5. Reproducibility
```bash
pytest                                   # all unit tests
python scripts/eda.py                    # dataset figures + summary
python src/baselines/freeze_index_baseline.py
python src/main.py                       # preprocess + LOSO train + eval
python src/training/evaluate.py          # per-subject table + mean±std
# edge venv:
python src/edge_conversion/quantize_model.py --subject <id>
python src/edge_conversion/evaluate_tflite.py --subject <id>
```
The full pipeline runs on **CPU** in minutes (small model, ~8k windows); a
**Colab GPU** runner (`notebooks/colab_runner.ipynb`) is provided for faster
convergence runs / hyperparameter sweeps. Design decisions:
`reports/decision_log.md`.

## References
See README §12 (TCN, SE, LayerNorm, stochastic depth, focal loss, AdamW, MCC,
int8 quantization) and the Stanford NMBL dataset.
