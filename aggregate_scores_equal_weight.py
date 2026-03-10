#!/usr/bin/env python3
"""
Aggregate HUGSIM closed-loop evaluation results with equal dataset weights.

This script is a companion to `aggregate_scores.py`. Instead of weighting each
scenario equally across all datasets, it:

- Computes per-dataset averages of Road Completion (RC) and HD-Score per
  difficulty bucket: Easy / Medium / Hard / Extreme, plus overall;
- Then averages those per-dataset means with **equal weight per dataset**,
  regardless of how many scenarios each dataset contributes.

Usage example:

    python3 aggregate_scores_equal_weight.py \\
      --output_root /network/scratch/g/grandhia/hugsim_output/benchmark/out_mini \\
      --datasets kitti360_ltf,nuscenes_ltf,pandaset_ltf,waymo_ltf
"""

from __future__ import annotations

import argparse
import importlib
import math
from typing import Dict, List, Optional, Tuple


def _load_base_module():
    """
    Import the existing `aggregate_scores` module so we can reuse its logic.

    This assumes `aggregate_scores.py` lives next to this script and can be
    imported as a top-level module when executed from the repo root.
    """

    return importlib.import_module("aggregate_scores")


def _mean(values: List[float]) -> float:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def compute_equal_weight_summary(
    per_ds_summary: Dict[str, Dict[str, Dict[str, float]]],
    diff_order: Tuple[str, ...],
) -> Dict[str, Dict[str, float]]:
    """
    Build an equal-weight combined summary over datasets.

    per_ds_summary: mapping dataset -> summary dict as returned by
        aggregate_scores.compute_summary (keys: "rc", "hdscore", "count").
    """

    summary_eq: Dict[str, Dict[str, float]] = {"rc": {}, "hdscore": {}, "count": {}}

    # We also include the 'avg' pseudo-difficulty.
    all_diffs = list(diff_order) + ["avg"]

    for d in all_diffs:
        rc_vals: List[float] = []
        hd_vals: List[float] = []

        for ds, s in per_ds_summary.items():
            # Skip datasets that don't have this difficulty key.
            if d not in s["rc"] or d not in s["hdscore"]:
                continue
            rc = s["rc"][d]
            hd = s["hdscore"][d]
            if isinstance(rc, float) and math.isnan(rc):
                pass
            else:
                rc_vals.append(rc)
            if isinstance(hd, float) and math.isnan(hd):
                pass
            else:
                hd_vals.append(hd)

        summary_eq["rc"][d] = _mean(rc_vals)
        summary_eq["hdscore"][d] = _mean(hd_vals)
        # For equal-weight aggregation, "count" is the number of datasets that
        # contributed a finite value for this difficulty.
        summary_eq["count"][d] = float(len(rc_vals))

    return summary_eq


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate HUGSIM eval.json into equal-weight per-dataset scores."
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
        help=(
            "Comma-separated list of dataset subdirs to include. "
            "Default: include all subdirs found."
        ),
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

    base = _load_base_module()
    # Reuse DIFF_ORDER and print_table from the base script.
    diff_order: Tuple[str, ...] = getattr(base, "DIFF_ORDER")
    collect_scores = getattr(base, "collect_scores")
    compute_summary = getattr(base, "compute_summary")
    print_table = getattr(base, "print_table")

    # Determine which dataset subdirs to use.
    dataset_subdirs: Optional[List[str]]
    if args.datasets.strip():
        dataset_subdirs = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        dataset_subdirs = None

    # Collect scores for all requested datasets at once, then split by dataset.
    all_scores = collect_scores(args.output_root, dataset_subdirs=dataset_subdirs)
    if not all_scores:
        import sys

        print(
            "No eval.json files found. Check --output_root and that evaluations finished.",
            file=sys.stderr,
        )
        return 2

    # Group scores by dataset name.
    by_dataset: Dict[str, List[base.ScenarioScore]] = {}
    for s in all_scores:
        by_dataset.setdefault(s.dataset, []).append(s)

    if not by_dataset:
        import sys

        print(
            "No per-dataset scores could be grouped. "
            "Check that eval.json files have valid dataset directories.",
            file=sys.stderr,
        )
        return 2

    # Compute per-dataset summaries.
    per_ds_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for ds in sorted(by_dataset.keys()):
        scores_ds = by_dataset[ds]
        if not scores_ds:
            continue
        summary_ds = compute_summary(scores_ds)
        per_ds_summary[ds] = summary_ds

    # Print per-dataset tables.
    for ds in sorted(per_ds_summary.keys()):
        print(f"==== Dataset: {ds} ====")
        print_table(per_ds_summary[ds], decimals=args.decimals)
        print()

    # Compute and print equal-weight combined summary.
    summary_eq = compute_equal_weight_summary(per_ds_summary, diff_order=diff_order)

    print("==== Equal-weight combined across datasets (each dataset weight = 1/N) ====")
    print_table(summary_eq, decimals=args.decimals)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

