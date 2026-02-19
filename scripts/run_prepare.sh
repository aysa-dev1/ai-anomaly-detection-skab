#!/usr/bin/env bash
set -euo pipefail

echo "==> Running data preparation..."
python -m anomaly_detection.data.prepare
echo "==> Data preparation finished."