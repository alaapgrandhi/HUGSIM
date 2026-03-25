#!/usr/bin/env bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=48:00:00
#SBATCH --partition=unkillable
#SBATCH --job-name=hugsim-eval-suite
#SBATCH --output=sbatch_eval_suite-drivor_mini_config_mar24_1.out
#SBATCH --error=sbatch_eval_suite-drivor_mini_config_mar24_1.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CHECKPOINT_PATH="/network/scratch/l/luke.rowe/experiments/config_mar24_1/checkpoints/last.ckpt"
OUTPUT_BASE="/network/scratch/l/luke.rowe/hugsim_output/benchmark/out_drivor_mini_config_mar24_1"

export CHECKPOINT_PATH
export OUTPUT_BASE

pixi run bash run_eval_kitti.sh
pixi run bash run_eval_waymo.sh
pixi run bash run_eval_nuscenes.sh
pixi run bash run_eval_pandaset.sh
