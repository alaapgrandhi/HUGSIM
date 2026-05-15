#!/usr/bin/env bash
# Dump DrivoR input-feature statistics + camera images during HUGSIM eval.
# Mirrors run_eval_nuscenes.sh / sbatch_run_eval_suite.sh, but runs closed_loop.py
# with --base_path <dataset>_base_dump.yaml so the AD side is dump_e2e.sh ->
# dump_drivor_features_hugsim.py. Each scenario writes <output>/<scene>/feat_dump/
# (feature_stats.log + frame_<NNN>.png).
#
# Run from the HUGSIM dir in the HUGSIM pixi env, e.g.:
#   pixi run bash dump_drivor_features_hugsim.sh [CHECKPOINT] [OUTPUT_BASE] [N_SCENARIOS]
#
# Dataset defaults to nuscenes; override with DATASET=kitti360 (or waymo/pandaset).
# Per-frame dump depth is controlled by DRIVOR_DUMP_N_FRAMES (default 5), which
# inherits down the launch chain into dump_drivor_features_hugsim.py.

module load cuda/11.8/cudnn/8.9
export PYTHONNOUSERSITE=1
export DRIVOR_DUMP_N_FRAMES="${DRIVOR_DUMP_N_FRAMES:--1}"

DATASET="${DATASET:-nuscenes}"
sim_cuda=0
ad_cuda=0
scenario_dir=/network/scratch/g/grandhia/hugsim_data_old/scenarios/${DATASET}/

ad_checkpoint_path="${1:-/network/scratch/l/luke.rowe/experiments/config_may13_3/checkpoints/last_drivor_compatible.ckpt}"
output_base="${2:-/network/scratch/l/luke.rowe/hugsim_output/feature_dump/config_may13_3_${DATASET}}"
n_scenarios="${3:-1}"

echo "DATASET=${DATASET}"
echo "ad_checkpoint_path=${ad_checkpoint_path}"
echo "output_base=${output_base}"
echo "n_scenarios=${n_scenarios}  DRIVOR_DUMP_N_FRAMES=${DRIVOR_DUMP_N_FRAMES}"

count=0
for cfg in ${scenario_dir}/*.yaml; do
    if [ "${count}" -ge "${n_scenarios}" ]; then break; fi
    count=$((count + 1))
    echo "[${count}/${n_scenarios}] ${cfg}"
    CUDA_VISIBLE_DEVICES=${sim_cuda} \
    python3 closed_loop.py --scenario_path ${cfg} \
                        --base_path ./configs/sim/${DATASET}_base_dump.yaml \
                        --camera_path ./configs/sim/${DATASET}_camera.yaml \
                        --kinematic_path ./configs/sim/kinematic.yaml \
                        --ad ltf \
                        --ad_cuda ${ad_cuda} \
                        --ad_checkpoint_path ${ad_checkpoint_path} \
                        --output_dir ${output_base} \
                        --image_size 434 252
done

echo "Done. Per-scenario dumps under ${output_base}/<scene>/feat_dump/"
