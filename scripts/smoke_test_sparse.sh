#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "=== Sparse assertion tests ==="
uv run python scripts/smoke_test_sparse_assertions.py

for model in yolov8_sparse_stem:n yolov8_sparse_backbone:n yolov8_sparse_full:n; do
    echo ""
    echo "=== Training smoke: ${model} ==="
    uv run python main.py train \
        --model "${model}" \
        --pretrained-weights yolov8n.pt \
        --max-epochs 1 \
        --batch-size 2 \
        --max-train-samples 4 \
        --max-eval-samples 4 \
        --checkpoint-out ""
done

echo ""
echo "=== All sparse smoke tests passed ==="
