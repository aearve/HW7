#!/bin/bash
#SBATCH --job-name=hw7_train
#SBATCH --partition=Centaurus
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G
#SBATCH --output=hw7_train_%j.out

cd ~/hw7/SportsTrackingTransformer
echo "=== train.py - Zoo M32_L1 then M32_L2 ==="
uv run python src/train.py --model_type zoo --device -1 --hparam_search_iters 2
echo "=== DONE ==="
