#!/usr/bin/env python3
"""
Aggregate HUGSIM closed-loop evaluation results into unified RC / HD-Score table.

This script reads per-scenario `eval.json` files produced by `closed_loop.py`
and reports mean Road Completion (RC) and HD-Score per difficulty bucket:
Easy / Medium / Hard / Extreme, plus overall average across all scenarios.

Difficulty is inferred from the scenario output folder name containing one of:
`_easy_`, `_medium_`, `_hard_`, `_extreme_`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


DIFF_ORDER: Tuple[str, ...] = ("easy", "medium", "hard", "extreme")
DIFF_LABEL: Dict[str, str] = {"easy": "E", "medium": "M", "hard": "H", "extreme": "X"}
DIFF_RE = re.compile(r"_(easy|medium|hard|extreme)_\d+\b")


@dataclass(frozen=True)
class ScenarioScore:
    dataset: str
    scenario_dir: str
    difficulty: str
    rc: float
    hdscore: float


def _iter_eval_json_paths(dataset_dir: str) -> Iterable[str]:
    for root, _dirs, files in os.walk(dataset_dir):
        if "eval.json" in files:
            yield os.path.join(root, "eval.json")


def _infer_difficulty_from_path(path: str) -> Optional[str]:
    # Prefer parent folder name (matches how outputs are structured).
    parent = os.path.basename(os.path.dirname(path))
    m = DIFF_RE.search(parent)
    if m:
        return m.group(1)
    # Fallback: scan full path (covers alternate layouts).
    m = DIFF_RE.search(path)
    if m:
        return m.group(1)
    return None


def _read_eval_json(path: str) -> Tuple[float, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "rc" not in data or "hdscore" not in data:
        raise KeyError(f"Missing required keys in {path}: need 'rc' and 'hdscore'")
    return float(data["rc"]), float(data["hdscore"])


def collect_scores(output_root: str, dataset_subdirs: Optional[List[str]] = None) -> List[ScenarioScore]:
    if not os.path.isdir(output_root):
        raise FileNotFoundError(f"output_root does not exist or is not a directory: {output_root}")

    if dataset_subdirs is None:
        dataset_subdirs = sorted(
            d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))
        )

    results: List[ScenarioScore] = []
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
                rc, hd = _read_eval_json(eval_path)
            except Exception:
                # If a single eval is malformed, keep going but report later via stderr.
                print(f"[warn] failed to read {eval_path}", file=sys.stderr)
                continue
            scenario_dir = os.path.dirname(eval_path)
            results.append(
                ScenarioScore(dataset=ds, scenario_dir=scenario_dir, difficulty=diff, rc=rc, hdscore=hd)
            )
    return results


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def compute_summary(scores: List[ScenarioScore]) -> Dict[str, Dict[str, float]]:
    by_diff: Dict[str, List[ScenarioScore]] = {d: [] for d in DIFF_ORDER}
    for s in scores:
        if s.difficulty in by_diff:
            by_diff[s.difficulty].append(s)

    summary: Dict[str, Dict[str, float]] = {"rc": {}, "hdscore": {}, "count": {}}
    for d in DIFF_ORDER:
        ds = by_diff[d]
        summary["count"][d] = float(len(ds))
        summary["rc"][d] = _mean([x.rc for x in ds])
        summary["hdscore"][d] = _mean([x.hdscore for x in ds])

    # Overall averages across all scenarios (not mean-of-means).
    summary["count"]["avg"] = float(len(scores))
    summary["rc"]["avg"] = _mean([x.rc for x in scores])
    summary["hdscore"]["avg"] = _mean([x.hdscore for x in scores])
    return summary


def _fmt_pct(x: float, decimals: int = 1) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x * 100:.{decimals}f}"


def print_table(summary: Dict[str, Dict[str, float]], decimals: int = 1) -> None:
    # Header approximates Table 2 format: RC and HD-Score blocks.
    cols = [DIFF_LABEL[d] for d in DIFF_ORDER] + ["Avg."]

    rc_vals = [_fmt_pct(summary["rc"][d], decimals) for d in DIFF_ORDER] + [
        _fmt_pct(summary["rc"]["avg"], decimals)
    ]
    hd_vals = [_fmt_pct(summary["hdscore"][d], decimals) for d in DIFF_ORDER] + [
        _fmt_pct(summary["hdscore"]["avg"], decimals)
    ]

    # Keep it compact for copy/paste into papers.
    print("RC     " + "  ".join(cols))
    print("       " + "  ".join(rc_vals))
    print("HD     " + "  ".join(cols))
    print("       " + "  ".join(hd_vals))

    # Counts help sanity-check that you covered the whole benchmark.
    cnts = [int(summary["count"][d]) for d in DIFF_ORDER] + [int(summary["count"]["avg"])]
    print("Count  " + "  ".join(cols))
    print("       " + "  ".join(str(c) for c in cnts))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate HUGSIM eval.json into unified scores.")
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

    scores = collect_scores(args.output_root, dataset_subdirs=dataset_subdirs)
    if not scores:
        print(
            "No eval.json files found. Check --output_root and that evaluations finished.",
            file=sys.stderr,
        )
        return 2

    summary = compute_summary(scores)
    print_table(summary, decimals=args.decimals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

