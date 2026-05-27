#!/usr/bin/env python3
"""
Aggregate HUGSIM closed-loop collision rates.

Mirrors `aggregate_scores_pdms.py`, but reports the closed-loop collision rate
recorded per-frame in each scenario's `data.pkl` (the per-step `info['collision']`
produced by `HUGSimEnv.step`). The env terminates on either background or
foreground collision and writes the same boolean into every frame, so per-scenario
"collision" reduces to: did any frame in the rollout report `collision=True`.

Note: foreground (vehicle-vehicle) and background (scene) collisions are merged
into one boolean inside the env, so this script cannot split them.

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


def _iter_eval_json_paths(dataset_dir: str) -> Iterable[str]:
    yield from base._iter_eval_json_paths(dataset_dir)  # type: ignore[attr-defined]


def _infer_difficulty_from_path(path: str) -> Optional[str]:
    return base._infer_difficulty_from_path(path)  # type: ignore[attr-defined]


def _read_collision(data_pkl_path: str) -> Tuple[bool, int]:
    """Return (any_frame_collided, n_frames) from a closed_loop data.pkl."""
    with open(data_pkl_path, "rb") as f:
        data = pickle.load(f)
    # closed_loop.py writes `pickle.dump([save_data], wf)` -> list of one dict
    if not isinstance(data, list) or not data:
        raise ValueError(f"Unexpected data.pkl layout in {data_pkl_path}")
    frames = data[0].get("frames", [])
    if not frames:
        return False, 0
    collided = any(bool(fr.get("collision", False)) for fr in frames)
    return collided, len(frames)


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
                collided, n_frames = _read_collision(data_pkl)
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
        "count": {},
        "n_collided": {},
    }
    for d in DIFF_ORDER:
        ds = by_diff[d]
        summary["count"][d] = float(len(ds))
        summary["n_collided"][d] = float(sum(1 for x in ds if x.collided))
        summary["collision_rate"][d] = _mean([1.0 if x.collided else 0.0 for x in ds])

    summary["count"]["avg"] = float(len(scores))
    summary["n_collided"]["avg"] = float(sum(1 for x in scores if x.collided))
    summary["collision_rate"]["avg"] = _mean([1.0 if x.collided else 0.0 for x in scores])
    return summary


def _fmt_pct(x: float, decimals: int = 1) -> str:
    return base._fmt_pct(x, decimals=decimals)  # type: ignore[attr-defined]


def print_collision_table(summary: Dict[str, Dict[str, float]], decimals: int = 1) -> None:
    cols = [DIFF_LABEL[d] for d in DIFF_ORDER] + ["Avg."]

    rate_vals = [_fmt_pct(summary["collision_rate"][d], decimals) for d in DIFF_ORDER] + [
        _fmt_pct(summary["collision_rate"]["avg"], decimals)
    ]
    print("CollRate " + "  ".join(cols))
    print("         " + "  ".join(rate_vals))

    nc = [int(summary["n_collided"][d]) for d in DIFF_ORDER] + [int(summary["n_collided"]["avg"])]
    cnts = [int(summary["count"][d]) for d in DIFF_ORDER] + [int(summary["count"]["avg"])]
    print("Collided " + "  ".join(cols))
    print("         " + "  ".join(f"{c}/{t}" for c, t in zip(nc, cnts)))


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
