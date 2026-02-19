#!/usr/bin/env bash
set -euo pipefail

echo "==> Running baseline training..."
python -m anomaly_detection.models.train
echo "==> Training finished."