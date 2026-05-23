#!/usr/bin/env bash
#SBATCH -c 24
#SBATCH --mem=512G
#SBATCH --gres=gpu:a100l:4
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable
#SBATCH --exclude=cn-k003
#SBATCH --job-name=hugsim-eval-suite
#SBATCH --output=sbatch_eval_suite-config_may22_1.out
#SBATCH --error=sbatch_eval_suite-config_may22_1.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CHECKPOINT_PATH="/network/scratch/l/luke.rowe/experiments/config_may22_1/checkpoints/drivor_compatible_last.ckpt"
OUTPUT_BASE="/network/scratch/l/luke.rowe/hugsim_output/benchmark/config_may22_1"

export CHECKPOINT_PATH
export OUTPUT_BASE

# Gigapixel-checkpoint compat flags. Unset -> use drivoR.yaml defaults (all true).
# To evaluate a checkpoint trained WITHOUT these, uncomment and set to 0.
# Values "1"/"true"/"yes" -> on, anything else -> off. Each SLURM job is isolated.
# export DRIVOR_PAD_LW=0
# export DRIVOR_REWARD_COND=0
# export DRIVOR_REAR_AXLE_SHIFT=0
# export DRIVOR_ORIGINAL_CAMERA_ORDER=1
export DRIVOR_PROPOSAL_NUM=64

# Run the four eval scripts in parallel, one per GPU.
# SIM_CUDA picks the physical GPU; per-dataset logs avoid interleaved output.
declare -a pids
SIM_CUDA=0 pixi run bash run_eval_kitti_434_252.sh    > eval_kitti.log    2>&1 &
pids+=($!)
SIM_CUDA=1 pixi run bash run_eval_waymo_434_252.sh    > eval_waymo.log    2>&1 &
pids+=($!)
SIM_CUDA=2 pixi run bash run_eval_nuscenes_434_252.sh > eval_nuscenes.log 2>&1 &
pids+=($!)
SIM_CUDA=3 pixi run bash run_eval_pandaset_434_252.sh > eval_pandaset.log 2>&1 &
pids+=($!)

# Wait for all jobs; exit non-zero if any failed.
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
exit "$fail"
