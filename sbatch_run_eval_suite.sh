#!/usr/bin/env bash
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=24:00:00
#SBATCH --partition=main
#SBATCH --job-name=hugsim-eval-suite
#SBATCH --output=sbatch_eval_suite-config_may13_3.out
#SBATCH --error=sbatch_eval_suite-config_may13_3.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CHECKPOINT_PATH="/network/scratch/l/luke.rowe/experiments/config_may13_3/checkpoints/last_drivor_compatible.ckpt"
OUTPUT_BASE="/network/scratch/l/luke.rowe/hugsim_output/benchmark/config_may13_3"

export CHECKPOINT_PATH
export OUTPUT_BASE

# Gigapixel-checkpoint compat flags. Unset -> use drivoR.yaml defaults (all true).
# To evaluate a checkpoint trained WITHOUT these, uncomment and set to 0.
# Values "1"/"true"/"yes" -> on, anything else -> off. Each SLURM job is isolated.
# export DRIVOR_PAD_LW=0
# export DRIVOR_REWARD_COND=0
# export DRIVOR_REAR_AXLE_SHIFT=0

pixi run bash run_eval_kitti.sh
pixi run bash run_eval_waymo.sh
pixi run bash run_eval_nuscenes.sh
pixi run bash run_eval_pandaset.sh
