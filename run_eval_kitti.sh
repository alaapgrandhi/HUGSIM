pixi shell
module load cuda/11.8/cudnn/8.9
sim_cuda=0
ad_cuda=1

# change this variable as the scenario path on your machine
scenario_dir=/network/scratch/g/grandhia/hugsim_data/kitti360
output_base=/network/scratch/g/grandhia/hugsim_output/benchmark/out_official/kitti360_ltf


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
                        --base_path ./configs/sim/kitti360_base.yaml \
                        --camera_path ./configs/sim/kitti360_camera.yaml \
                        --kinematic_path ./configs/sim/kinematic.yaml \
                        --ad ltf \
                        --ad_cuda ${ad_cuda}
done