#!/usr/bin/env bash
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=48:00:00
#SBATCH --partition=unkillable
#SBATCH --exclude=cn-k003
#SBATCH --job-name=hugsim-eval-suite
#SBATCH --output=sbatch_eval_suite-config_may18_2.out
#SBATCH --error=sbatch_eval_suite-config_may18_2.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CHECKPOINT_PATH="/network/scratch/l/luke.rowe/experiments/config_may_18_2/checkpoints/last.ckpt"
OUTPUT_BASE="/network/scratch/l/luke.rowe/hugsim_output/benchmark/config_may18_2"

export CHECKPOINT_PATH
export OUTPUT_BASE

# Gigapixel-checkpoint compat flags. Unset -> use drivoR.yaml defaults (all true).
# To evaluate a checkpoint trained WITHOUT these, uncomment and set to 0.
# Values "1"/"true"/"yes" -> on, anything else -> off. Each SLURM job is isolated.
# export DRIVOR_PAD_LW=0
# export DRIVOR_REWARD_COND=0
# export DRIVOR_REAR_AXLE_SHIFT=0
# export DRIVOR_ORIGINAL_CAMERA_ORDER=1

pixi run bash run_eval_kitti.sh
pixi run bash run_eval_waymo.sh
pixi run bash run_eval_nuscenes.sh
pixi run bash run_eval_pandaset.sh
