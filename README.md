# AI Anomaly Detection (SKAB)

End-to-end ML lifecycle for sensor anomaly detection using the SKAB dataset, with a baseline Isolation Forest model.

**Requirements**
- Python 3.10+

**Setup**
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -e .`

**Prepare Data**
- `bash scripts/run_prepare.sh`

**Train Baseline**
- `bash scripts/run_train.sh`
- Or full pipeline: `bash scripts/run_train_pipeline.sh`

**Artifacts**
- Metrics: `artifacts/metrics/baseline_isolation_forest.json`
- Model: `artifacts/models/isolation_forest.joblib`
- Prepare report: `artifacts/reports/prepare_report.json`

**Notebooks**
- `notebooks/01_eda.ipynb`
- `notebooks/02_error_analysis.ipynb`

**Configuration**
- Dataset: `configs/dataset.yaml`
- Training: `configs/train.yaml`
- `contamination` can be `auto`, a float, or `from_data` (use train anomaly rate)
- Predict/Monitoring configs are placeholders: `configs/predict.yaml`, `configs/monitoring.yaml`

**Scripts**
- `scripts/run_prepare.sh`
- `scripts/run_train.sh`
- `scripts/run_train_pipeline.sh`
- `scripts/run_predict.sh` (placeholder)
- `scripts/run_monitoring.sh` (placeholder)

**Training Approach**

Training is per-file: a separate Isolation Forest is fitted on the first 70 % of each experiment file (temporal split, no look-ahead) and evaluated on the remaining 30 %. Global training across all files was evaluated as an alternative and produced substantially lower F1 (0.355 vs 0.665).

A likely explanation is that feature distributions differ between experiments, so a single global model loses the local calibration needed to separate normal from anomalous behaviour. This remains a hypothesis: the next step would be per-file feature normalisation before concatenation, which would distinguish a pure scaling effect from genuinely context-specific anomaly patterns.

**Project Layout**
- `src/anomaly_detection`: core package
- `data/`: raw, processed, interim
- `artifacts/`: models, metrics, reports
- `notebooks/`: analysis
- `tests/`: tests


***Development notes*** Development suported by Claude Code (Anthropic)
