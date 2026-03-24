#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=48:00:00
#SBATCH --partition=main
#SBATCH --job-name=hugsim-eval-suite
#SBATCH --output=sbatch_eval_suite-drivor_pretrained_1142_672.out
#SBATCH --error=sbatch_eval_suite-drivor_pretrained_1142_672.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

pixi run bash run_eval_kitti.sh
pixi run bash run_eval_waymo.sh
pixi run bash run_eval_nuscenes.sh
pixi run bash run_eval_pandaset.sh
