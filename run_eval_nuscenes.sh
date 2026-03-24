# pixi shell
module load cuda/11.8/cudnn/8.9
export PYTHONNOUSERSITE=1
sim_cuda=0
ad_cuda=0

# change this variable as the scenario path on your machine
scenario_dir=/network/scratch/g/grandhia/hugsim_data_old/scenarios/nuscenes/
output_base=/network/scratch/l/luke.rowe/hugsim_output/benchmark/out_test_setup/nuscenes_ltf
ad_checkpoint_path=/network/scratch/g/grandhia/hugsim_data/drivor_Nav2_10epochs.pth

for cfg in ${scenario_dir}/*.yaml; do
    basename=$(basename ${cfg} .yaml)                                  # scene-021-easy-00 or scene-021_easy_00
    dirname_prefix=${basename%%-*}; dirname_suffix=${basename#*-}      # dirname_prefix=scene, dirname_suffix=021-easy-00
    dirname="${dirname_prefix}-${dirname_suffix//-/_}"                 # scene-021_easy_00

    if [[ -f "${output_base}/${dirname}/eval.json" || -f "${output_base}/0000_${dirname}/eval.json" ]]; then        
        echo "SKIP (already complete): ${cfg}"
        continue
    fi

    echo "not skip"
    echo ${cfg}
    CUDA_VISIBLE_DEVICES=${sim_cuda} \
    python3 closed_loop.py --scenario_path ${cfg} \
                        --base_path ./configs/sim/nuscenes_base.yaml \
                        --camera_path ./configs/sim/nuscenes_camera.yaml \
                        --kinematic_path ./configs/sim/kinematic.yaml \
                        --ad ltf \
                        --ad_cuda ${ad_cuda} \
                        --ad_checkpoint_path ${ad_checkpoint_path} \
                        --output_dir ${output_base} \
                        --image_size 1142 672
done