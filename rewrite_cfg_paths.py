#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

try:
    import ruamel.yaml as yaml
except ImportError:
    import yaml  # fall back to pyyaml if available


ROOT_SCENES_OLD = Path("/network/scratch/g/grandhia/hugsim_data_old/scenes")


def compute_scene_path_in_old_dataset(cfg_path: Path) -> str:
    """
    Derive the canonical scene directory inside hugsim_data_old for this cfg.yaml
    based on its location.

    Example:
    - cfg_path = /.../hugsim_data_old/scenes/waymo/938501362409_0_200/cfg.yaml
      -> /network/scratch/g/grandhia/hugsim_data_old/scenes/waymo/938501362409_0_200
    """
    # cfg_path.parent already points to .../hugsim_data_old/scenes/{dataset}/{scene_dir}
    return str(cfg_path.parent)


def rewrite_value(cfg_path: Path, value: str) -> str:
    """
    For a given existing model_path/source_path value, return the updated value.
    We ignore the old absolute prefix and always map to the canonical directory
    under hugsim_data_old/scenes, unless it is already using that root.
    """
    if value.startswith(str(ROOT_SCENES_OLD)):
        # Already in the desired form (points to hugsim_data_old)
        return value

    return compute_scene_path_in_old_dataset(cfg_path)


def process_cfg(cfg_path: Path, apply: bool = False) -> None:
    with cfg_path.open("r") as f:
        data = yaml.safe_load(f)

    changed = False
    for key in ("model_path", "source_path"):
        if key in data and isinstance(data[key], str):
            old = data[key]
            new = rewrite_value(cfg_path, old)
            if new != old:
                changed = True
                print(f"[{cfg_path}] {key}:")
                print(f"  OLD: {old}")
                print(f"  NEW: {new}")
                if apply:
                    data[key] = new

    if changed and apply:
        with cfg_path.open("w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite outdated model_path/source_path in HUGSIM cfg.yaml files."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes to disk. Without this, just print what would change.",
    )
    args = parser.parse_args()

    if not ROOT_SCENES_OLD.exists():
        print(f"Root scenes directory does not exist: {ROOT_SCENES_OLD}")
        return

    cfg_files = list(ROOT_SCENES_OLD.rglob("cfg.yaml"))
    print(f"Found {len(cfg_files)} cfg.yaml files under {ROOT_SCENES_OLD}")

    for cfg_path in cfg_files:
        process_cfg(cfg_path, apply=args.apply)

    if not args.apply:
        print("\nDry run complete. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()

