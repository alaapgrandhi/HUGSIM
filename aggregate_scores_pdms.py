#!/usr/bin/env python3
"""
Aggregate HUGSIM closed-loop evaluation results into PDMS scores.

This script mirrors `aggregate_scores.py` but focuses on the PDMS metric
stored in each scenario's `eval.json`. It reports mean PDMS per difficulty
bucket (Easy / Medium / Hard / Extreme) plus an overall average.

Difficulty is inferred from the scenario output folder name containing one of:
`_easy_`, `_medium_`, `_hard_`, `_extreme_`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import aggregate_scores as base


DIFF_ORDER: Tuple[str, ...] = base.DIFF_ORDER
DIFF_LABEL: Dict[str, str] = base.DIFF_LABEL
DIFF_RE = base.DIFF_RE


@dataclass(frozen=True)
class ScenarioPDMS:
    dataset: str
    scenario_dir: str
    difficulty: str
    pdms: float


def _iter_eval_json_paths(dataset_dir: str) -> Iterable[str]:
    # Reuse the traversal logic from the base module.
    yield from base._iter_eval_json_paths(dataset_dir)  # type: ignore[attr-defined]


def _infer_difficulty_from_path(path: str) -> Optional[str]:
    # Reuse difficulty inference from the base module.
    return base._infer_difficulty_from_path(path)  # type: ignore[attr-defined]


def _read_pdms(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "pdms" not in data:
        raise KeyError(f"Missing required key in {path}: need 'pdms'")
    return float(data["pdms"])


def collect_pdms(
    output_root: str, dataset_subdirs: Optional[List[str]] = None
) -> List[ScenarioPDMS]:
    if not os.path.isdir(output_root):
        raise FileNotFoundError(
            f"output_root does not exist or is not a directory: {output_root}"
        )

    if dataset_subdirs is None:
        dataset_subdirs = sorted(
            d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))
        )

    results: List[ScenarioPDMS] = []
    for ds in dataset_subdirs:
        ds_dir = os.path.join(output_root, ds)
        if not os.path.isdir(ds_dir):
            continue
        for eval_path in _iter_eval_json_paths(ds_dir):
            diff = _infer_difficulty_from_path(eval_path)
            if diff is None:
                # Skip files we can't bucket reliably.
                continue
            try:
                pdms = _read_pdms(eval_path)
            except Exception:
                # If a single eval is malformed, keep going but report later via stderr.
                print(f"[warn] failed to read PDMS from {eval_path}", file=sys.stderr)
                continue
            scenario_dir = os.path.dirname(eval_path)
            results.append(
                ScenarioPDMS(dataset=ds, scenario_dir=scenario_dir, difficulty=diff, pdms=pdms)
            )
    return results


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def compute_pdms_summary(scores: List[ScenarioPDMS]) -> Dict[str, Dict[str, float]]:
    by_diff: Dict[str, List[ScenarioPDMS]] = {d: [] for d in DIFF_ORDER}
    for s in scores:
        if s.difficulty in by_diff:
            by_diff[s.difficulty].append(s)

    summary: Dict[str, Dict[str, float]] = {"pdms": {}, "count": {}}
    for d in DIFF_ORDER:
        ds = by_diff[d]
        summary["count"][d] = float(len(ds))
        summary["pdms"][d] = _mean([x.pdms for x in ds])

    # Overall averages across all scenarios (not mean-of-means).
    summary["count"]["avg"] = float(len(scores))
    summary["pdms"]["avg"] = _mean([x.pdms for x in scores])
    return summary


def _fmt_pct(x: float, decimals: int = 1) -> str:
    # Reuse the same formatting style as aggregate_scores.
    return base._fmt_pct(x, decimals=decimals)  # type: ignore[attr-defined]


def print_pdms_table(summary: Dict[str, Dict[str, float]], decimals: int = 1) -> None:
    cols = [DIFF_LABEL[d] for d in DIFF_ORDER] + ["Avg."]

    pdms_vals = [_fmt_pct(summary["pdms"][d], decimals) for d in DIFF_ORDER] + [
        _fmt_pct(summary["pdms"]["avg"], decimals)
    ]

    print("PDMS   " + "  ".join(cols))
    print("       " + "  ".join(pdms_vals))

    cnts = [int(summary["count"][d]) for d in DIFF_ORDER] + [int(summary["count"]["avg"])]
    print("Count  " + "  ".join(cols))
    print("       " + "  ".join(str(c) for c in cnts))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate HUGSIM eval.json PDMS scores.")
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
        help="Decimal places for percentage printing (paper tables commonly use 1).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    dataset_subdirs = [d.strip() for d in args.datasets.split(",") if d.strip()] or None

    scores = collect_pdms(args.output_root, dataset_subdirs=dataset_subdirs)
    if not scores:
        print(
            "No eval.json files with PDMS found. Check --output_root and that evaluations finished.",
            file=sys.stderr,
        )
        return 2

    summary = compute_pdms_summary(scores)
    print_pdms_table(summary, decimals=args.decimals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

