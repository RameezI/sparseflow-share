# SparseFlow

SparseFlow provides a standalone PyTorch implementation of sparse neuronal activation (`AxonHillock`) and sparse convolutional blocks, plus Lightning-based training protocols for comparing dense and sparse YOLOv8-nano-style models.

## Getting started

### 1) Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2) Create environment and install dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 3) Create `.env` for faster Hugging Face downloads

Create `.env` in the project root and paste:

```bash
cat > .env << 'EOF'
HF_TOKEN=<your_hf_token>
HF_HUB_ENABLE_HF_TRANSFER=1
EOF
```

Notes:

- `HF_TOKEN` enables authenticated requests to the Hugging Face Hub.
- `HF_HUB_ENABLE_HF_TRANSFER=1` enables faster transfer mode when `hf_transfer` is installed.
- `.env` is git-ignored and should stay local.

Load these variables with `uv` when running commands:

```bash
uv run --env-file .env python scripts/check_dataset.py --split train --num-samples 3
uv run --env-file .env python main.py train --model yolov8:n --pretrained-weights yolov8n.pt --max-epochs 1
```

You can use the same `--env-file .env` pattern for `evaluate` commands as well.

### 4) Train, evaluate, and predict via CLI

`--pretrained-weights` accepts either:

- an Ultralytics YOLO-style `.pt` file, such as `yolov8n.pt`
- a SparseFlow Lightning `.ckpt` checkpoint, such as `checkpoints/baseline.ckpt`

The `--device` option accepts `auto`, `cpu`, or `cuda`. Use `auto` unless you
need to force CPU or CUDA.

Full-scale baseline training (COCO loaded automatically from Hugging Face):

```bash
uv run python main.py train \
  --model yolov8:n \
  --pretrained-weights yolov8n.pt \
  --max-epochs 100 \
  --batch-size 16 \
  --device auto \
  --checkpoint-out checkpoints/baseline.ckpt
```

Evaluate either a `.pt` file or a `.ckpt` checkpoint:

```bash
uv run python main.py evaluate --model yolov8:n --pretrained-weights yolov8n.pt --batch-size 16
uv run python main.py evaluate --model yolov8:n --pretrained-weights checkpoints/baseline.ckpt --batch-size 16
```

Run prediction with either weight format:

```bash
uv run python main.py predict --model yolov8:n --pretrained-weights yolov8n.pt --batch-size 16
uv run python main.py predict --model yolov8:n --pretrained-weights checkpoints/baseline.ckpt --batch-size 16
```

Sample-limit options are command-specific:

- `train`: supports `--max-train-samples` and `--max-eval-samples`
- `evaluate`: supports `--max-eval-samples`
- `predict`: does not support `--max-eval-samples`

Validate both `--pretrained-weights` loading paths without running COCO:

```bash
uv run python scripts/check_pretrained_weights_paths.py --pretrained-weights yolov8n.pt
```

### 5) Monitor training with TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

Logs are written to `logs/tensorboard` by default. Override with `--log-dir <path>` at training time.

## Linting

Python lint configuration lives in `pyproject.toml` under `[tool.pylint.*]`.
Markdown lint configuration lives in `pyproject.toml` under `[tool.pymarkdown]`.
Format Python files with Black before running lint.

```bash
uv run black main.py detection_utils.py step_module.py export.py benchmark.py ./models ./data ./scripts
uv run pylint main.py detection_utils.py step_module.py ./models ./data
uv run pymarkdown scan README.md AGENTS.md CLAUDE.md .ai/*.md
```
