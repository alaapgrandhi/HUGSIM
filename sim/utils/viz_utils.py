"""Visualization helpers for HUGSIM closed-loop eval.

Two outputs, both purely additive (never touch eval/scoring logic):
  - render_overlaid_video: the 6-camera grid (video.mp4) with the predicted
    trajectory projected onto every camera.
  - render_bev_video: an ego-centric bird's-eye-view (video_bev.mp4) with the
    ground point cloud, other agents, the ego box, and the predicted trajectory.
"""

import math

import cv2
import numpy as np
from moviepy import ImageSequenceClip

# Height of the lidar/IMU origin above the ground plane (metres). The predicted
# trajectory lives on the ground, so in the lidar frame (z up) its z = -LIDAR_HEIGHT.
# 1.4 matches the constant the AD-side overlay (hugsim/visualize.py) already uses.
LIDAR_HEIGHT = 1.4

# Camera layout of the stitched grid written to video.mp4 (matches closed_loop.to_video).
_CAM_GRID = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"],
]

_TRAJ_COLOR = (255, 191, 0)  # amber, RGB


def _fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def _intrinsic_matrix(intr):
    """Build a 3x3 K from a HUGSIM intrinsic dict (fovx, fovy, cx, cy, H, W)."""
    K = np.eye(3)
    K[0, 0] = _fov2focal(intr["fovx"], intr["W"])
    K[1, 1] = _fov2focal(intr["fovy"], intr["H"])
    K[0, 2] = intr["cx"]
    K[1, 2] = intr["cy"]
    return K


def _rect_corners(cx, cy, width, length, yaw):
    """Four corners (4, 2) of a box; length is along yaw. Matches score_calculator."""
    c, s = math.cos(yaw), math.sin(yaw)
    xo = np.array([length / 2, length / 2, -length / 2, -length / 2])
    yo = np.array([width / 2, -width / 2, -width / 2, width / 2])
    x = cx + xo * c - yo * s
    y = cy + xo * s + yo * c
    return np.stack([x, y], axis=1)


def project_lidar_traj_to_image(traj_xy, cam_param):
    """Project a lidar-frame 2D trajectory onto one camera image.

    :param traj_xy: (N, 2) waypoints in lidar coords (x right, y forward) -- the
        same `plan_traj` HUGSIM receives from the AD side.
    :param cam_param: dict with 'l2c' (4x4 lidar->camera) and 'intrinsic'.
    :return: (uv (N, 2) pixel coords, valid (N,) bool mask of points in front).
    """
    traj_xy = np.asarray(traj_xy, dtype=np.float64).reshape(-1, 2)
    n = len(traj_xy)
    pts_lidar = np.concatenate(
        [traj_xy, np.full((n, 1), -LIDAR_HEIGHT), np.ones((n, 1))], axis=1
    )  # (N, 4)
    l2c = np.asarray(cam_param["l2c"], dtype=np.float64)
    pts_cam = (l2c @ pts_lidar.T).T[:, :3]  # (N, 3)
    depth = pts_cam[:, 2]
    K = _intrinsic_matrix(cam_param["intrinsic"])
    uvw = (K @ pts_cam.T).T  # (N, 3)
    valid = depth > 1e-3
    uv = np.full((n, 2), -1e9)
    uv[valid] = uvw[valid, :2] / depth[valid, None]
    return uv, valid


def _draw_traj_on_image(img, uv, valid, color=_TRAJ_COLOR):
    """Draw the projected trajectory (polyline + dots) on an RGB image in place."""
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    pts = [
        (int(round(p[0])), int(round(p[1])))
        for p, v in zip(uv, valid)
        if v and -2 * w < p[0] < 3 * w and -2 * h < p[1] < 3 * h
    ]
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(img, p, 6 if i == 0 else 4, color, -1, cv2.LINE_AA)
    return img


def render_overlaid_video(observations, infos, plan_trajs, output_path, fps=4):
    """Stitch the 6-camera grid (as closed_loop.to_video does) with the predicted
    trajectory projected onto every camera, and write it to output_path.

    observations / infos / plan_trajs are per-frame lists; plan_trajs[i] is the
    lidar-frame waypoint array the AD side returned for observations[i] (or None).
    """
    frames = []
    for obs, info, traj in zip(observations, infos, plan_trajs):
        cam_params = info.get("cam_params", {}) if isinstance(info, dict) else {}
        cam_imgs = {}
        for cam in (c for row in _CAM_GRID for c in row):
            img = np.ascontiguousarray(np.asarray(obs[cam]).copy())
            if traj is not None and cam in cam_params:
                uv, valid = project_lidar_traj_to_image(traj, cam_params[cam])
                img = _draw_traj_on_image(img, uv, valid)
            cam_imgs[cam] = img
        row_imgs = [np.concatenate([cam_imgs[c] for c in row], axis=1) for row in _CAM_GRID]
        frames.append(np.concatenate(row_imgs, axis=0))
    if not frames:
        print("[viz] render_overlaid_video: no frames, skipping")
        return
    ImageSequenceClip(frames, fps=fps).write_videofile(output_path, logger=None)


def render_bev_video(
    save_data,
    ground_xyz,
    output_path,
    fps=4,
    x_range=(-20.0, 40.0),
    y_range=(-20.0, 20.0),
    px_per_m=10,
    max_ground_pts=80000,
):
    """Ego-centric bird's-eye-view video: ground point cloud + other agents +
    ego box + predicted trajectory, all transformed into the ego frame each frame.

    :param save_data: the `save_data` dict built in closed_loop (has 'frames').
    :param ground_xyz: (M, 3) global ground points (from ground.ply), or None.
    """
    frames_data = save_data.get("frames", [])
    if not frames_data:
        print("[viz] render_bev_video: no frames, skipping")
        return

    if ground_xyz is not None and len(ground_xyz) > max_ground_pts:
        idx = np.random.RandomState(0).choice(len(ground_xyz), max_ground_pts, replace=False)
        ground_xyz = ground_xyz[idx]
    ground_xy = None if ground_xyz is None else np.asarray(ground_xyz)[:, :2].astype(np.float64)

    W = int(round((y_range[1] - y_range[0]) * px_per_m))
    H = int(round((x_range[1] - x_range[0]) * px_per_m))

    def to_px(pts_ego):
        """(..., 2) ego-frame (x forward, y left) -> (..., 2) pixel (col, row).
        Forward is up, left is left."""
        pts_ego = np.asarray(pts_ego, dtype=np.float64).reshape(-1, 2)
        col = (y_range[1] - pts_ego[:, 1]) * px_per_m
        row = (x_range[1] - pts_ego[:, 0]) * px_per_m
        return np.stack([col, row], axis=1)

    frames = []
    for fr in frames_data:
        canvas = np.full((H, W, 3), 28, dtype=np.uint8)

        ego_x, ego_y, _, ego_w, ego_l, _, ego_yaw = fr["ego_box"][:7]
        c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)
        R = np.array([[c, -s], [s, c]])

        def g2e(xy):
            d = np.asarray(xy, dtype=np.float64).reshape(-1, 2) - np.array([ego_x, ego_y])
            return (R @ d.T).T

        # metric grid lines every 10 m
        for gx in range(int(math.ceil(x_range[0] / 10) * 10), int(x_range[1]) + 1, 10):
            r = int((x_range[1] - gx) * px_per_m)
            cv2.line(canvas, (0, r), (W - 1, r), (45, 45, 45), 1)
        for gy in range(int(math.ceil(y_range[0] / 10) * 10), int(y_range[1]) + 1, 10):
            cc = int((y_range[1] - gy) * px_per_m)
            cv2.line(canvas, (cc, 0), (cc, H - 1), (45, 45, 45), 1)

        # ground point cloud
        if ground_xy is not None and len(ground_xy):
            ge = g2e(ground_xy)
            m = (
                (ge[:, 0] > x_range[0])
                & (ge[:, 0] < x_range[1])
                & (ge[:, 1] > y_range[0])
                & (ge[:, 1] < y_range[1])
            )
            if np.any(m):
                pix = to_px(ge[m]).astype(np.int32)
                canvas[np.clip(pix[:, 1], 0, H - 1), np.clip(pix[:, 0], 0, W - 1)] = (95, 95, 100)

        # other agents
        for ob in fr.get("obj_boxes", []):
            ob = np.asarray(ob).reshape(-1)
            if ob.shape[0] < 7:
                continue
            ox, oy, _, ow, ol, _, oyaw = ob[:7]
            corners_e = g2e(_rect_corners(ox, oy, ow, ol, oyaw))
            cv2.fillPoly(canvas, [to_px(corners_e).astype(np.int32)], (240, 140, 40))

        # predicted trajectory (global (x, y, yaw))
        traj = np.asarray(
            [(p[0], p[1]) for p in fr["planned_traj"]["traj"]], dtype=np.float64
        ).reshape(-1, 2)
        if len(traj):
            tp = to_px(g2e(traj)).astype(np.int32)
            for i in range(1, len(tp)):
                cv2.line(canvas, tuple(tp[i - 1]), tuple(tp[i]), (0, 200, 255), 2, cv2.LINE_AA)
            for p in tp:
                cv2.circle(canvas, tuple(p), 3, (0, 200, 255), -1, cv2.LINE_AA)

        # ego box at origin, plus heading arrow (forward = up)
        ego_corners_e = _rect_corners(0.0, 0.0, ego_w, ego_l, 0.0)
        cv2.fillPoly(canvas, [to_px(ego_corners_e).astype(np.int32)], (60, 220, 90))
        head = to_px([[0.0, 0.0], [ego_l, 0.0]]).astype(np.int32)
        cv2.arrowedLine(canvas, tuple(head[0]), tuple(head[1]), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)

        frames.append(canvas)

    ImageSequenceClip(frames, fps=fps).write_videofile(output_path, logger=None)
