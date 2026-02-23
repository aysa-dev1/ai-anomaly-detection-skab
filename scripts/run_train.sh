#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train.yaml}"

echo "==> Running baseline training..."
python -m anomaly_detection.models.train --config "$CONFIG_PATH"
echo "==> Training finished."
