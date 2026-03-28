#!/usr/bin/env python3
"""Compare HUGSIM eval RC scores and copy paired videos for large deltas.

Rules:
- Extreme: copy scenarios where drivor_mini rc - gigapixel_distilled rc > 0.5
- Medium: copy scenarios where gigapixel_distilled rc - drivor_mini rc > 0.5
- For each category, randomly sample up to N scenarios (default 10)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class ScenarioEval:
    name: str
    rc: float
    eval_path: Path
    video_path: Path


def _load_eval_json(eval_path: Path) -> Dict[str, object]:
    with eval_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_scenarios(root: Path) -> Dict[str, ScenarioEval]:
    """Collect scenario name -> ScenarioEval from an eval output 'all' directory."""
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    scenarios: Dict[str, ScenarioEval] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        eval_path = child / "eval.json"
        video_path = child / "video.mp4"
        if not eval_path.is_file():
            continue
        try:
            data = _load_eval_json(eval_path)
            rc_raw = data.get("rc")
            if rc_raw is None:
                continue
            rc = float(rc_raw)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            continue

        scenarios[child.name] = ScenarioEval(
            name=child.name,
            rc=rc,
            eval_path=eval_path,
            video_path=video_path,
        )
    return scenarios


def filter_by_level(names: Iterable[str], level: str) -> List[str]:
    level_lower = level.lower()
    return [name for name in names if level_lower in name.lower()]


def summarize(
    level: str,
    paired_names: List[str],
    drivor: Dict[str, ScenarioEval],
    giga: Dict[str, ScenarioEval],
) -> str:
    if not paired_names:
        return f"{level}: no paired scenarios"
    d_mean = sum(drivor[n].rc for n in paired_names) / len(paired_names)
    g_mean = sum(giga[n].rc for n in paired_names) / len(paired_names)
    return (
        f"{level}: paired={len(paired_names)}, "
        f"drivor_mean_rc={d_mean:.4f}, gigapixel_mean_rc={g_mean:.4f}, "
        f"mean_delta(drivor-gigapixel)={d_mean - g_mean:+.4f}"
    )


def copy_pair(
    name: str,
    dst_dir: Path,
    drivor_video: Path,
    giga_video: Path,
) -> Tuple[bool, str]:
    if not drivor_video.is_file():
        return False, f"missing drivor video: {drivor_video}"
    if not giga_video.is_file():
        return False, f"missing gigapixel video: {giga_video}"

    dst_dir.mkdir(parents=True, exist_ok=True)
    d_dst = dst_dir / f"video_{name}_drivor_mini.mp4"
    g_dst = dst_dir / f"video_{name}_gigapixel_distilled.mp4"
    shutil.copy2(drivor_video, d_dst)
    shutil.copy2(giga_video, g_dst)
    return True, f"copied pair for {name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drivor-root",
        type=Path,
        default=Path(
            "/home/mila/l/luke.rowe/scratch/hugsim_output/benchmark/"
            "out_drivor_mini_config_mar24_1/all"
        ),
        help="Path to drivor_mini eval 'all' directory.",
    )
    parser.add_argument(
        "--gigapixel-root",
        type=Path,
        default=Path(
            "/home/mila/l/luke.rowe/scratch/hugsim_output/benchmark/"
            "out_gigapixel_distilled/all"
        ),
        help="Path to gigapixel distilled eval 'all' directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd(),
        help="Directory where videos_medium/videos_extreme are created.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=10,
        help="Maximum paired scenarios to copy per category.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    drivor = collect_scenarios(args.drivor_root)
    giga = collect_scenarios(args.gigapixel_root)
    common_names = sorted(set(drivor) & set(giga))

    medium_names = filter_by_level(common_names, "medium")
    extreme_names = filter_by_level(common_names, "extreme")

    print(summarize("medium", medium_names, drivor, giga))
    print(summarize("extreme", extreme_names, drivor, giga))

    # (a) drivor_mini > 0.5 better in rc for extreme
    extreme_qualifying = [
        name for name in extreme_names if (drivor[name].rc - giga[name].rc) > 0.5
    ]
    # (b) gigapixel_distilled > 0.5 better in rc for medium
    medium_qualifying = [
        name for name in medium_names if (giga[name].rc - drivor[name].rc) > 0.5
    ]

    print(f"extreme qualifying (>+0.5 drivor-gigapixel): {len(extreme_qualifying)}")
    print(f"medium qualifying (>+0.5 gigapixel-drivor): {len(medium_qualifying)}")

    if len(extreme_qualifying) > args.max_pairs:
        extreme_selected = rng.sample(extreme_qualifying, k=args.max_pairs)
    else:
        extreme_selected = list(extreme_qualifying)

    if len(medium_qualifying) > args.max_pairs:
        medium_selected = rng.sample(medium_qualifying, k=args.max_pairs)
    else:
        medium_selected = list(medium_qualifying)

    videos_extreme_dir = args.output_root / "videos_extreme"
    videos_medium_dir = args.output_root / "videos_medium"

    copied_extreme = 0
    for name in sorted(extreme_selected):
        ok, msg = copy_pair(
            name=name,
            dst_dir=videos_extreme_dir,
            drivor_video=drivor[name].video_path,
            giga_video=giga[name].video_path,
        )
        if ok:
            copied_extreme += 1
        else:
            print(f"[warn] {msg}")

    copied_medium = 0
    for name in sorted(medium_selected):
        ok, msg = copy_pair(
            name=name,
            dst_dir=videos_medium_dir,
            drivor_video=drivor[name].video_path,
            giga_video=giga[name].video_path,
        )
        if ok:
            copied_medium += 1
        else:
            print(f"[warn] {msg}")

    print(f"copied extreme pairs: {copied_extreme} -> {videos_extreme_dir}")
    print(f"copied medium pairs: {copied_medium} -> {videos_medium_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
