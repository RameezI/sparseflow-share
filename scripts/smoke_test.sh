#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

uv run python main.py train \
    --pretrained-weights yolov8n.pt \
    --max-epochs 5 \
    --batch-size 8 \
    --max-train-samples 200 \
    --max-eval-samples 200 \
    --checkpoint-out checkpoints/smoke.ckpt
