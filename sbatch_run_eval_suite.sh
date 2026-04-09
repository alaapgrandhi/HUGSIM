#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=48:00:00
#SBATCH --partition=main
#SBATCH --job-name=hugsim-eval-suite
#SBATCH --output=sbatch_eval_suite-config_apr_5_6500.out
#SBATCH --error=sbatch_eval_suite-config_apr_5_6500.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CHECKPOINT_PATH="/network/scratch/l/luke.rowe/experiments/config_apr_5/checkpoints/last_drivor_compatible_6500.ckpt"
OUTPUT_BASE="/network/scratch/l/luke.rowe/hugsim_output/benchmark/config_apr_5_6500"

export CHECKPOINT_PATH
export OUTPUT_BASE

pixi run bash run_eval_kitti.sh
pixi run bash run_eval_waymo.sh
pixi run bash run_eval_nuscenes.sh
pixi run bash run_eval_pandaset.sh
