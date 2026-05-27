CHECKPOINT_PATH="/network/scratch/l/luke.rowe/experiments/drivor_checkpoints/drivoR_Nav1_103k_25epochs_p1_1148x672_last_checkpoint.ckpt"
OUTPUT_BASE="/network/scratch/l/luke.rowe/hugsim_output/benchmark/nav1_25_epochs_1_proposal_rerun"

export CHECKPOINT_PATH
export OUTPUT_BASE

# Gigapixel-checkpoint compat flags. Unset -> use drivoR.yaml defaults (all true).
# To evaluate a checkpoint trained WITHOUT these, uncomment and set to 0.
# Values "1"/"true"/"yes" -> on, anything else -> off. Each SLURM job is isolated.
export DRIVOR_PAD_LW=0
export DRIVOR_REWARD_COND=0
export DRIVOR_REAR_AXLE_SHIFT=0
export DRIVOR_ORIGINAL_CAMERA_ORDER=1
export DRIVOR_PROPOSAL_NUM=1

pixi run bash run_eval_kitti.sh
pixi run bash run_eval_waymo.sh
pixi run bash run_eval_nuscenes.sh
pixi run bash run_eval_pandaset.sh