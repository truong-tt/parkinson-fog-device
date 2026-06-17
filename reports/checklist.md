# HopeGait — Implementation Checklist

Mirrors the 2-week plan (`~/.claude/plans/...`). Process evidence for the
course deliverable. Code-complete items are checked; run-dependent items
(need the dataset + Colab GPU) are left open with their command.

## Stage 1 — Problem analysis
- [x] A1 — EDA script → figures + summary (`scripts/eda.py`)
- [x] A2 — Freeze-Index + shallow-ML LOSO baseline (`src/baselines/freeze_index_baseline.py`)
- [ ] A1/A2 run on real data → `reports/eda_summary.md`, `models/baseline_summary.json`

## Stage 2 — Solution design (correctness fixes, each with a test)
- [x] B1 — augment in physical space before scaling (`FoGDataset`)
- [x] B2 — seeded augmentation RNG + per-worker init
- [x] B3 — per-recording post-processing & event metrics
- [x] B4 — focal-loss class weights derived from fold FoG rate
- [x] B5 — per-subject mean ± std + `results_table.md`
- [x] Edge calibration fix — scale representative windows (`quantize_model.py`)

## Stage 3 — Implementation / prototyping
- [x] C1 — Colab GPU runner (`notebooks/colab_runner.ipynb`); also runs on CPU
- [ ] C2 — full LOSO run with EMA 0.99 + 60 epochs → per-subject table + figures
- [x] C3 — int8 vs fp32 delta script (`src/edge_conversion/evaluate_tflite.py`)
- [ ] C3 run → `models/int8_delta.json`

## Deliverables
- [x] Implementation plan + this checklist
- [x] Decision log (`reports/decision_log.md`)
- [x] Technical report draft (`reports/report.md`) — fill `{{...}}` after run
- [x] Slides outline (`reports/slides.md`)
- [ ] Lecturer checkpoint per stage (sign-off)

## Run commands
```bash
pytest                                          # unit tests (all green)
python scripts/eda.py                           # stage-1 figures
python src/baselines/freeze_index_baseline.py   # stage-1 baseline
python src/main.py                              # preprocess + LOSO + eval
python src/edge_conversion/quantize_model.py --subject <id>   # edge venv
python src/edge_conversion/evaluate_tflite.py --subject <id>  # edge venv
```
