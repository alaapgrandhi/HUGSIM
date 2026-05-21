# pixi shell
module load cuda/11.8/cudnn/8.9
export PYTHONNOUSERSITE=1
sim_cuda=0
ad_cuda=0

# change this variable as the scenario path on your machine
scenario_dir=/network/scratch/g/grandhia/hugsim_data_old/scenarios/pandaset/
default_output_base=/network/scratch/l/luke.rowe/hugsim_output/benchmark/out_test_setup/pandaset_ltf
default_ad_checkpoint_path=/network/scratch/g/grandhia/hugsim_data/drivor_Nav2_10epochs.pth

# Optional overrides:
# - args: run_eval_pandaset.sh [/path/to.ckpt] [/path/to/output_base]
# - env:  CHECKPOINT_PATH=... OUTPUT_BASE=...
ad_checkpoint_path="${1:-${CHECKPOINT_PATH:-$default_ad_checkpoint_path}}"
output_base="${2:-${OUTPUT_BASE:-$default_output_base}}"

echo "ad_checkpoint_path=${ad_checkpoint_path}"
echo "output_base=${output_base}"

for cfg in ${scenario_dir}/*.yaml; do
    basename=$(basename ${cfg} .yaml)           # scene-021-easy-00
    dirname=${basename#scene-}                   # 021-easy-00
    dirname=${dirname//-/_}                      # 021_easy_00

    if [ -f "${output_base}/${dirname}/eval.json" ]; then
        echo "SKIP (already complete): ${cfg}"
        continue
    fi

    echo ${cfg}
    CUDA_VISIBLE_DEVICES=${sim_cuda} \
    python3 closed_loop.py --scenario_path ${cfg} \
                        --base_path ./configs/sim/pandaset_base.yaml \
                        --camera_path ./configs/sim/pandaset_camera.yaml \
                        --kinematic_path ./configs/sim/kinematic.yaml \
                        --ad ltf \
                        --ad_cuda ${ad_cuda} \
                        --ad_checkpoint_path ${ad_checkpoint_path} \
                        --output_dir ${output_base} \
                        --image_size 1148 672

done