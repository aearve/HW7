#!/bin/bash
#SBATCH --job-name=hw7_bdb
#SBATCH --partition=Centaurus
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G
#SBATCH --output=hw7_run_%j.out

set -e
cd ~/hw7/SportsTrackingTransformer

echo "=== Step 1: prep_data.py ==="
uv run python src/prep_data.py

echo "=== Step 2: datasets.py (zoo only) ==="
uv run python src/datasets.py

echo "=== Step 3: train.py - Zoo M32_L1 then M32_L2 ==="
uv run python src/train.py --model_type zoo --device -1 --hparam_search_iters 2

echo "=== DONE ==="
