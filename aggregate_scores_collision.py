#!/usr/bin/env python3
"""
Aggregate HUGSIM closed-loop collision rates.

Mirrors `aggregate_scores_pdms.py`, but reports the closed-loop collision rate
recorded per-frame in each scenario's `data.pkl` (the per-step `info['collision']`
produced by `HUGSimEnv.step`). The env terminates on either background or
foreground collision; the colliding step is the last frame in `data.pkl`.

For each scenario this script reports:
- merged collision (any cause)
- foreground collision rate (ego vs other vehicles) -- requires the
  per-frame `fg_collision` field added to `closed_loop.py`
- ego speed at the foreground collision frame, in m/s -- requires `ego_velo`

Scenarios produced before those fields were added are still counted toward the
merged collision rate; their fg-only and velocity stats are reported as N/A.

Difficulty is inferred from the scenario output folder name containing one of:
`_easy_`, `_medium_`, `_hard_`, `_extreme_`.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import aggregate_scores as base


DIFF_ORDER: Tuple[str, ...] = base.DIFF_ORDER
DIFF_LABEL: Dict[str, str] = base.DIFF_LABEL
DIFF_RE = base.DIFF_RE


@dataclass(frozen=True)
class ScenarioCollision:
    dataset: str
    scenario_dir: str
    difficulty: str
    collided: bool
    n_frames: int
    # The three fields below are None for rollouts produced before the
    # split-flag / ego_velo additions to closed_loop.py.
    fg_collided: Optional[bool]
    bg_collided: Optional[bool]
    fg_collision_speed: Optional[float]  # |ego_velo| at the colliding frame, m/s


def _iter_eval_json_paths(dataset_dir: str) -> Iterable[str]:
    yield from base._iter_eval_json_paths(dataset_dir)  # type: ignore[attr-defined]


def _infer_difficulty_from_path(path: str) -> Optional[str]:
    return base._infer_difficulty_from_path(path)  # type: ignore[attr-defined]


def _read_collision(
    data_pkl_path: str,
) -> Tuple[bool, int, Optional[bool], Optional[bool], Optional[float]]:
    """Read collision info from a closed_loop data.pkl.

    Returns (collided, n_frames, fg_collided, bg_collided, fg_collision_speed).
    The last three are None when the data.pkl predates the split-flag /
    ego_velo additions in closed_loop.py.
    """
    with open(data_pkl_path, "rb") as f:
        data = pickle.load(f)
    # closed_loop.py writes `pickle.dump([save_data], wf)` -> list of one dict
    if not isinstance(data, list) or not data:
        raise ValueError(f"Unexpected data.pkl layout in {data_pkl_path}")
    frames = data[0].get("frames", [])
    if not frames:
        return False, 0, None, None, None
    collided = any(bool(fr.get("collision", False)) for fr in frames)

    has_split = any("fg_collision" in fr or "bg_collision" in fr for fr in frames)
    if not has_split:
        return collided, len(frames), None, None, None

    fg_collided = any(bool(fr.get("fg_collision", False)) for fr in frames)
    bg_collided = any(bool(fr.get("bg_collision", False)) for fr in frames)

    # Env terminates on the colliding step, so a foreground crash is on the
    # last frame; pick that frame's ego_velo as the impact speed. Fall back to
    # the first frame flagged fg_collision (shouldn't differ in practice).
    fg_speed: Optional[float] = None
    if fg_collided:
        impact_frame = None
        if bool(frames[-1].get("fg_collision", False)):
            impact_frame = frames[-1]
        else:
            for fr in frames:
                if bool(fr.get("fg_collision", False)):
                    impact_frame = fr
                    break
        if impact_frame is not None and "ego_velo" in impact_frame:
            fg_speed = abs(float(impact_frame["ego_velo"]))

    return collided, len(frames), fg_collided, bg_collided, fg_speed


def collect_collisions(
    output_root: str, dataset_subdirs: Optional[List[str]] = None
) -> List[ScenarioCollision]:
    if not os.path.isdir(output_root):
        raise FileNotFoundError(
            f"output_root does not exist or is not a directory: {output_root}"
        )

    if dataset_subdirs is None:
        dataset_subdirs = sorted(
            d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))
        )

    results: List[ScenarioCollision] = []
    for ds in dataset_subdirs:
        ds_dir = os.path.join(output_root, ds)
        if not os.path.isdir(ds_dir):
            continue
        # Index by eval.json so we only count scenarios whose eval actually
        # finished (matches aggregate_scores_pdms.py's notion of "valid run").
        for eval_path in _iter_eval_json_paths(ds_dir):
            diff = _infer_difficulty_from_path(eval_path)
            if diff is None:
                continue
            scenario_dir = os.path.dirname(eval_path)
            data_pkl = os.path.join(scenario_dir, "data.pkl")
            if not os.path.isfile(data_pkl):
                print(f"[warn] missing data.pkl in {scenario_dir}", file=sys.stderr)
                continue
            try:
                collided, n_frames, fg_coll, bg_coll, fg_speed = _read_collision(data_pkl)
            except Exception as e:
                print(f"[warn] failed to read collision from {data_pkl}: {e}", file=sys.stderr)
                continue
            results.append(
                ScenarioCollision(
                    dataset=ds,
                    scenario_dir=scenario_dir,
                    difficulty=diff,
                    collided=collided,
                    n_frames=n_frames,
                    fg_collided=fg_coll,
                    bg_collided=bg_coll,
                    fg_collision_speed=fg_speed,
                )
            )
    return results


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def compute_collision_summary(
    scores: List[ScenarioCollision],
) -> Dict[str, Dict[str, float]]:
    by_diff: Dict[str, List[ScenarioCollision]] = {d: [] for d in DIFF_ORDER}
    for s in scores:
        if s.difficulty in by_diff:
            by_diff[s.difficulty].append(s)

    summary: Dict[str, Dict[str, float]] = {
        "collision_rate": {},
        "fg_collision_rate": {},
        "bg_collision_rate": {},
        "fg_count": {},  # denominator for fg/bg rates (scenarios with split fields)
        "n_fg_collided": {},
        "fg_speed_mean": {},
        "fg_speed_median": {},
        "fg_speed_max": {},
        "count": {},
        "n_collided": {},
    }

    def _fg_bg_split_subset(ds: List[ScenarioCollision]) -> List[ScenarioCollision]:
        return [x for x in ds if x.fg_collided is not None]

    def _fg_speeds(ds: List[ScenarioCollision]) -> List[float]:
        return [x.fg_collision_speed for x in ds
                if x.fg_collided and x.fg_collision_speed is not None]

    for d in DIFF_ORDER:
        ds = by_diff[d]
        summary["count"][d] = float(len(ds))
        summary["n_collided"][d] = float(sum(1 for x in ds if x.collided))
        summary["collision_rate"][d] = _mean([1.0 if x.collided else 0.0 for x in ds])

        split = _fg_bg_split_subset(ds)
        summary["fg_count"][d] = float(len(split))
        summary["n_fg_collided"][d] = float(sum(1 for x in split if x.fg_collided))
        summary["fg_collision_rate"][d] = _mean(
            [1.0 if x.fg_collided else 0.0 for x in split]
        ) if split else float("nan")
        summary["bg_collision_rate"][d] = _mean(
            [1.0 if x.bg_collided else 0.0 for x in split]
        ) if split else float("nan")

        speeds = _fg_speeds(ds)
        summary["fg_speed_mean"][d] = _mean(speeds) if speeds else float("nan")
        summary["fg_speed_median"][d] = _median(speeds) if speeds else float("nan")
        summary["fg_speed_max"][d] = max(speeds) if speeds else float("nan")

    summary["count"]["avg"] = float(len(scores))
    summary["n_collided"]["avg"] = float(sum(1 for x in scores if x.collided))
    summary["collision_rate"]["avg"] = _mean([1.0 if x.collided else 0.0 for x in scores])

    split_all = _fg_bg_split_subset(scores)
    summary["fg_count"]["avg"] = float(len(split_all))
    summary["n_fg_collided"]["avg"] = float(sum(1 for x in split_all if x.fg_collided))
    summary["fg_collision_rate"]["avg"] = _mean(
        [1.0 if x.fg_collided else 0.0 for x in split_all]
    ) if split_all else float("nan")
    summary["bg_collision_rate"]["avg"] = _mean(
        [1.0 if x.bg_collided else 0.0 for x in split_all]
    ) if split_all else float("nan")

    speeds_all = _fg_speeds(scores)
    summary["fg_speed_mean"]["avg"] = _mean(speeds_all) if speeds_all else float("nan")
    summary["fg_speed_median"]["avg"] = _median(speeds_all) if speeds_all else float("nan")
    summary["fg_speed_max"]["avg"] = max(speeds_all) if speeds_all else float("nan")
    return summary


def _median(values: List[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _fmt_pct(x: float, decimals: int = 1) -> str:
    return base._fmt_pct(x, decimals=decimals)  # type: ignore[attr-defined]


def _fmt_speed(x: float, decimals: int = 2) -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{x:.{decimals}f}"


def print_collision_table(summary: Dict[str, Dict[str, float]], decimals: int = 1) -> None:
    cols = [DIFF_LABEL[d] for d in DIFF_ORDER] + ["Avg."]
    keys = list(DIFF_ORDER) + ["avg"]

    def _pct_row(label: str, metric: str) -> None:
        vals = [_fmt_pct(summary[metric][k], decimals) for k in keys]
        print(f"{label:<10} " + "  ".join(cols))
        print(f"{'':<10} " + "  ".join(vals))

    _pct_row("CollRate", "collision_rate")

    nc = [int(summary["n_collided"][k]) for k in keys]
    cnts = [int(summary["count"][k]) for k in keys]
    print(f"{'Collided':<10} " + "  ".join(cols))
    print(f"{'':<10} " + "  ".join(f"{c}/{t}" for c, t in zip(nc, cnts)))

    _pct_row("FgCollRate", "fg_collision_rate")
    _pct_row("BgCollRate", "bg_collision_rate")

    n_fg = [int(summary["n_fg_collided"][k]) for k in keys]
    fg_cnts = [int(summary["fg_count"][k]) for k in keys]
    print(f"{'FgCollided':<10} " + "  ".join(cols))
    print(f"{'':<10} " + "  ".join(f"{c}/{t}" for c, t in zip(n_fg, fg_cnts)))

    # Foreground collision speed (|ego_velo|, m/s) over fg-collided scenarios.
    print(f"{'FgSpeed':<10} " + "  ".join(cols) + "    (m/s, |ego_velo| at fg collision)")
    for label, metric in (("  mean", "fg_speed_mean"),
                          ("  med", "fg_speed_median"),
                          ("  max", "fg_speed_max")):
        vals = [_fmt_speed(summary[metric][k]) for k in keys]
        print(f"{label:<10} " + "  ".join(vals))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate HUGSIM closed-loop collision rates from data.pkl files."
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="/network/scratch/g/grandhia/hugsim_output/benchmark/out_mini",
        help="Root containing dataset output folders (e.g., waymo_ltf, nuscenes_ltf, ...).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated list of dataset subdirs to include. Default: include all subdirs found.",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=1,
        help="Decimal places for percentage printing.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    dataset_subdirs = [d.strip() for d in args.datasets.split(",") if d.strip()] or None

    scores = collect_collisions(args.output_root, dataset_subdirs=dataset_subdirs)
    if not scores:
        print(
            "No scenarios with both eval.json and data.pkl found. "
            "Check --output_root and that evaluations finished.",
            file=sys.stderr,
        )
        return 2

    summary = compute_collision_summary(scores)
    print_collision_table(summary, decimals=args.decimals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
