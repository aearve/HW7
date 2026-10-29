# HW7 - 

I chose to run: "Attention is All You Need, for Sports Tracking Data" -

**Files to look at:** the README has both answers. `run_log.txt` is the full terminal/SLURM
output showing that I ran both parts ran on student. `hw7_changes.diff` is the exact code change needed to
make training work on a CPU node (no GPU was available on the allocated node). `run_hw7.slurm` /
`run_hw7_train.slurm` are the batch job scripts actually submitted.

## How to run it
1. On student: `git clone https://github.com/SumerSports/SportsTrackingTransformer.git`
2. `ln -s /projects/class/spoa4001_u01/SportsTrackingTransformer/data/bdb_2024 data/bdb_2024`
3. `uv sync`
4. Apply `hw7_changes.diff` (`git apply hw7_changes.diff`) - fixes a hardcoded
   `accelerator="gpu"` in `src/train.py` that otherwise crashes on a CPU-only node, and limits
   `src/datasets.py` to the Zoo architecture only (we only trained Zoo models here).
5. `sbatch run_hw7.slurm` (data prep + dataset precompute), then `sbatch run_hw7_train.slurm`
   (training). Run via SLURM, not directly in the terminal.

## Part 1 [30 pts] - Running the submission
Trained the Zoo architecture, model_dim=32, num_layers=1, lr=1e-4, batch_size=256, dropout=0.3
(the repo's own hyperparameter grid, config #1) for 25 epochs. Best checkpoint: epoch 24,
**val_loss = 4.075**. Full training log, including every epoch's val_loss/train_loss, is in
`run_log.txt`.

## Part 2 [15 pts] - Parameter tweak
Changed **num_layers: 1 -> 2** (same model_dim=32, lr=1e-4, batch_size=256, dropout=0.3 - only
one parameter changed). 
This is the repo's own config #2, which I did by passing
`--hparam_search_iters 2` to `src/train.py`.

**Result:** The 2-layer model actually did better — 4.005 vs 4.075 — and it got there way faster, by epoch 10 instead of epoch 24 for the 1-layer one. I had to cut the 2-layer run off around epoch 11 because of the deadline, so I just used the best checkpoint it had saved at that point. Even without letting it fully finish, this matches what the paper’s own results show (Zoo M32_L2 beats M32_L1 in their table). Makes sense — giving it a second layer means it has more room to actually learn how players are interacting with each other, so it gets a lower loss and gets there quicker. 
This is just a small step though, as going from 1 layer to 2 — their README shows the Zoo architecture eventually stops benefiting from more capacity once you scale it up a lot more than that.

## Citation
"Attention is All You Need, for Sports Tracking Data" - Ranasaria, U. & Vabishchevich, P. (2024).
https://github.com/SumerSports/SportsTrackingTransformer
