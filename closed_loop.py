import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import hugsim_env
from argparse import ArgumentParser
from sim.utils.sim_utils import traj2control, traj_transform_to_global
import pickle
import json
import pickle
from sim.utils.launch_ad import launch, check_alive
from omegaconf import OmegaConf
import open3d as o3d
from sim.utils.score_calculator import hugsim_evaluate
from sim.utils.viz_utils import render_overlaid_video, render_bev_video
import numpy as np
from moviepy import ImageSequenceClip

# Diagnostic: when set, bypass the iLQR controller and teleport the ego directly
# onto the model's predicted trajectory (one sim step per frame). Isolates
# whether closed-loop failure is the controller's fault or the model's.
TELEPORT = os.getenv('DRIVOR_TELEPORT', '').strip().lower() in ('1', 'true', 'yes')

def to_video(observations, output_path):
    frames = []
    for obs in observations:
        row1 = np.concatenate([obs['CAM_FRONT_LEFT'], obs['CAM_FRONT'], obs['CAM_FRONT_RIGHT']], axis=1)
        row2 = np.concatenate([obs['CAM_BACK_RIGHT'], obs['CAM_BACK'], obs['CAM_BACK_LEFT']], axis=1)
        frame = np.concatenate([row1, row2], axis=0)
        frames.append(frame)
    clip = ImageSequenceClip(frames, fps=4)
    clip.write_videofile(output_path)


def create_gym_env(cfg, output):

    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)

    observations_save, infos_save, plan_trajs_save = [], [], []
    obs, info = env.reset()
    done = False
    cnt = 0
    save_data = {'type': 'closeloop', 'frames': []}

    obs_pipe = os.path.join(output, 'obs_pipe')
    plan_pipe = os.path.join(output, 'plan_pipe')
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print('Ready for simulation')

    obs, info = None, None
    while not done:

        if obs is None or info is None:
            obs, info = env.reset()
        observations_save.append(obs['rgb'])
        infos_save.append(info)

        print('ego pose', info['ego_pos'])

        with open(obs_pipe, "wb") as pipe:
            pipe.write(pickle.dumps((obs, info)))
        with open(plan_pipe, "rb") as pipe:
            plan_traj = pickle.loads(pipe.read())
        plan_trajs_save.append(plan_traj)

        if plan_traj is not None:
            if TELEPORT:
                action = {'teleport': plan_traj}
            else:
                acc, steer_rate = traj2control(plan_traj, info)
                action = {'acc': acc, 'steer_rate': steer_rate}
            obs, reward, terminated, truncated, info = env.step(action)
            cnt += 1
            done = terminated or truncated or cnt > 400

        else:  # AD Side Crushed
            done = True

        if plan_traj is None:
            continue
        imu_plan_traj = plan_traj[:, [1, 0]]
        imu_plan_traj[:, 1] *= -1
        global_traj = traj_transform_to_global(imu_plan_traj, info['ego_box'])
        save_data['frames'].append({
            'time_stamp': info['timestamp'],
            'is_key_frame': True,
            'ego_box': info['ego_box'],
            'obj_boxes': info['obj_boxes'],
            'obj_names': ['car' for _ in info['obj_boxes']],
            'planned_traj': {
                'traj': global_traj,
                'timestep': 0.5
            },
            'collision': info['collision'],
            'rc': info['rc']
        })

    with open(obs_pipe, "wb") as pipe:
        pipe.write(pickle.dumps('Done'))

    # Skip the full eval+viz pipeline when no frames were stepped (e.g. the AD
    # side sent None on the first iteration for a one-frame diagnostic sweep).
    # hugsim_evaluate crashes on empty save_data['frames'], so guard explicitly.
    if not save_data['frames']:
        print('[closed_loop] no frames stepped; skipping eval + viz')
        return

    with open(os.path.join(output, 'data.pkl'), 'wb') as wf:
        pickle.dump([save_data], wf)
    with open(os.path.join(output, 'infos.pkl'), 'wb') as wf:
        pickle.dump(infos_save, wf)

    ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'ground.ply')).points)
    scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'scene.ply')).points)
    results = hugsim_evaluate([save_data], ground_xyz, scene_xyz)
    with open(os.path.join(output, 'eval.json'), 'w') as f:
        json.dump(results, f)

    # Visualization -- purely additive; wrapped so a viz failure can never
    # block the eval outputs (data.pkl / infos.pkl / eval.json) saved above.
    try:
        render_overlaid_video(observations_save, infos_save, plan_trajs_save,
                              os.path.join(output, 'video.mp4'))
    except Exception as e:
        print(f'[viz] overlaid video.mp4 failed: {e}')
    try:
        render_bev_video(save_data, ground_xyz, os.path.join(output, 'video_bev.mp4'))
    except Exception as e:
        print(f'[viz] video_bev.mp4 failed: {e}')


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--kinematic_path", type=str, required=True)
    parser.add_argument('--ad', default="uniad")
    parser.add_argument('--ad_cuda', default="1")
    parser.add_argument('--ad_checkpoint_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--image_size', type=int, nargs=2, default=None)
    args = parser.parse_args()

    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    camera_config = OmegaConf.load(args.camera_path)
    kinematic_config = OmegaConf.load(args.kinematic_path)
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config}
    )
    cfg.base.output_dir = cfg.base.output_dir + args.ad

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
    cfg.update(model_config)
    if args.output_dir is not None:
        output = os.path.join(args.output_dir, cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    else:
        output = os.path.join(cfg.base.output_dir, cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    os.makedirs(output, exist_ok=True)

    if args.ad == 'uniad':
        ad_path = cfg.base.uniad_path
    elif args.ad == 'vad':
        ad_path = cfg.base.vad_path
    elif args.ad == 'ltf':
        ad_path = cfg.base.ltf_path
    else:
        raise NotImplementedError
    
    process = launch(ad_path, args.ad_cuda, output, args.ad_checkpoint_path, args.image_size)
    try:
        create_gym_env(cfg, output)
        check_alive(process)
    except Exception as e:
        import traceback
        traceback.print_exc()
        process.kill()
    
    # # For debug
    # create_gym_env(cfg, output)
