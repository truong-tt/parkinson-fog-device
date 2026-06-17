# HopeGait — Presentation Outline

~10 slides. Render with any Markdown-to-slides tool (Marp, reveal.js); `---`
separates slides. Fill `{{...}}` from the run artifacts.

---

## 1. Title
**HopeGait: Real-Time Freezing-of-Gait Detection on a Lumbar IMU**
Causal TCN → int8 edge model. Stanford NMBL dataset. Solo, supervised.
_Not a medical device._

---

## 2. The problem
- FoG = sudden gait arrest in Parkinson's → fall risk.
- External cues stop episodes → need **onset detection in ~1 s**.
- Wearable → constraints: **causal, low-latency, MCU-sized**.

---

## 3. Data
- Lumbar IMU, 6ch @128Hz → 64Hz, 2s windows, 9 channels.
- **7 subjects**, 7,846 windows, FoG **26.0%** → imbalance.
- Prevalence 5.7%–63% per subject → LOSO + per-fold weighting.
- _Figures: class balance, per-subject prevalence, episode lengths._

---

## 4. Why not something simple? (Baseline)
- Freeze-Index threshold: MCC **{{fi_mcc}}**.
- Shallow ML on hand features: MCC **{{lr_mcc}}** / **{{dt_mcc}}**.
- Sets the bar the deep model must clear.

---

## 5. Model
- Causal TCN, 4 dilated blocks, RF ≈ 0.95 s.
- Streaming-safe: per-timestep LayerNorm, causal Squeeze-Excite, stochastic depth.
- Two heads: last-step (deploy) + dense aux (training signal).

---

## 6. Honest evaluation (what I fixed)
- LOSO, inner-val Youden-J threshold, scaler on train only.
- **Fixes:** rotation in physical space (B1), seeded aug (B2),
  per-recording event metrics (B3), data-derived class weights (B4),
  per-subject reporting (B5).
- Primary metric **MCC** + event-level (latency, false alarms/h).

---

## 7. Results
- ROC-AUC **0.77**, PR-AUC **0.59** (strong ranking).
- Per-subject LOSO MCC @threshold **+0.20 ± 0.13** — ties baseline, ranks better.
- Pooled MCC reads +0.34 but is prevalence-weighted (one subject dominates).
- Events: detection **32%**, latency **0.45 s**, FA/h **0.0**.
- Honest-eval finding: post-processing collapses low-prevalence folds → fix it.

---

## 8. Edge: int8 delta
- PTQ int8; calibration fixed to use scaled windows.
- MCC fp32 **{{fp32_mcc}}** → int8 **{{int8_mcc}}** (Δ **{{delta_mcc}}**).
- Model **{{tflite_kib}} KiB**, {{n_params}} params.

---

## 9. Limitations / future work
- Stanford-only; no cross-dataset validation yet.
- PTQ only — no QAT, no pruning.
- No on-hardware MCU latency/RAM numbers yet.

---

## 10. Takeaways
- Defensible pipeline: baseline-justified, leakage-controlled, reproducible.
- TCN matches the baseline (MCC 0.20) and ranks better (ROC-AUC 0.77), with
  headroom (more epochs/data, post-pp calibration).
- Honest evaluation surfaced two findings: pooled-vs-per-subject, post-pp collapse.
- Edge cost quantified: **{{delta_mcc}} MCC** for an MCU-sized int8 model.
