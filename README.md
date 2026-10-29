# HW7 - Sports Modeling Class

Submission run: "Attention is All You Need, for Sports Tracking Data" -
https://github.com/SumerSports/SportsTrackingTransformer, run on hpc-student.charlotte.edu.

**Files to look at:** this README has both answers. `run_log.txt` is the full terminal/SLURM
output proving both parts ran on student. `hw7_changes.diff` is the exact code change needed to
make training work on a CPU node (no GPU was available on the allocated node). `run_hw7.slurm` /
`run_hw7_train.slurm` are the batch job scripts actually submitted.

**Data note:** per the assignment hint, no data is stored in this repo. The pipeline reads
directly from the class's shared folder on student
(`/projects/class/spoa4001_u01/SportsTrackingTransformer/data/bdb_2024`, symlinked into a local
clone of the submission's repo rather than re-downloaded).

## How to run it
1. On student: `git clone https://github.com/SumerSports/SportsTrackingTransformer.git`
2. `ln -s /projects/class/spoa4001_u01/SportsTrackingTransformer/data/bdb_2024 data/bdb_2024`
3. `uv sync`
4. Apply `hw7_changes.diff` (`git apply hw7_changes.diff`) - fixes a hardcoded
   `accelerator="gpu"` in `src/train.py` that otherwise crashes on a CPU-only node, and limits
   `src/datasets.py` to the Zoo architecture only (we only trained Zoo models here).
5. `sbatch run_hw7.slurm` (data prep + dataset precompute), then `sbatch run_hw7_train.slurm`
   (training). Run via SLURM, not directly in the terminal - this is a multi-stage,
   compute-heavy pipeline, and the login node kills long-running interactive processes on
   disconnect.

## Part 1 [30 pts] - Running the submission
Trained the Zoo architecture, model_dim=32, num_layers=1, lr=1e-4, batch_size=256, dropout=0.3
(the repo's own hyperparameter grid, config #1) for 25 epochs. Best checkpoint: epoch 24,
**val_loss = 4.075**. Full training log, including every epoch's val_loss/train_loss, is in
`run_log.txt`.

## Part 2 [15 pts] - Parameter tweak
Changed **num_layers: 1 -> 2** (same model_dim=32, lr=1e-4, batch_size=256, dropout=0.3 - only
one parameter changed). This is the repo's own config #2, reached by passing
`--hparam_search_iters 2` to `src/train.py`.

**Result:** the 2-layer model reached a *better* validation loss (**4.005** vs 4.075) in *fewer*
epochs (best checkpoint at epoch 10, vs epoch 24 for the 1-layer model) - training was stopped
manually around epoch 11 due to the 7pm deadline, using the best checkpoint saved up to that
point. Even without full convergence, this matches the direction of the pattern in the
submission's own published results (their table shows Zoo M32_L2 outperforming M32_L1): adding
a second layer gives the Zoo architecture more capacity to model player interactions, which both
lowers the loss and gets there faster, before the architecture's capacity limit (per the
README, Zoo stops benefiting from more capacity at much larger sizes - this run only tests the
first small step, 1 layer to 2).

## Citation
"Attention is All You Need, for Sports Tracking Data" - Ranasaria, U. & Vabishchevich, P. (2024).
https://github.com/SumerSports/SportsTrackingTransformer
