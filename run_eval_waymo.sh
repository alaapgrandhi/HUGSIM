pixi shell
module load cuda/11.8/cudnn/8.9
sim_cuda=0
ad_cuda=1

# change this variable as the scenario path on your machine
scenario_dir=/network/scratch/g/grandhia/hugsim_data_old/scenarios/waymo/
output_base=/network/scratch/g/grandhia/hugsim_output/benchmark/out_debug_mini/waymo_ltf
ad_checkpoint_path=/home/mila/g/grandhia/NAVSIM/ckpts/drivor_mini.ckpt

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
                        --base_path ./configs/sim/waymo_base.yaml \
                        --camera_path ./configs/sim/waymo_camera.yaml \
                        --kinematic_path ./configs/sim/kinematic.yaml \
                        --ad ltf \
                        --ad_cuda ${ad_cuda} \
                        --ad_checkpoint_path ${ad_checkpoint_path} \
                        --output_dir ${output_base} \
                        --image_size 434 252
done