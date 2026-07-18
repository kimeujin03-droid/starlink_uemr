#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_tle_blocks(path: Path) -> list[list[str]]:
    lines = [x.rstrip("\n") for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    blocks: list[list[str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            blocks.append([lines[i], lines[i + 1]])
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            blocks.append([lines[i], lines[i + 1], lines[i + 2]])
            i += 3
        else:
            i += 1
    return blocks


def write_subset_tle(src: Path, dst: Path, n_records: int, seed: int) -> int:
    blocks = parse_tle_blocks(src)
    if len(blocks) < n_records:
        raise ValueError(f"Requested {n_records} records, only {len(blocks)} available")
    rng = random.Random(seed)
    idx = list(range(len(blocks)))
    rng.shuffle(idx)
    chosen = sorted(idx[:n_records])
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i in chosen:
        lines.extend(blocks[i])
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--subset-size", type=int, default=1200)
    ap.add_argument("--tle", default="tle/starlink_jan2026.txt")
    ap.add_argument("--config", default="configs/coverage_robustness_all_tle.yaml")
    ap.add_argument("--out-prefix", default="outputs/random1200")
    ap.add_argument("--null-dir-prefix", default="outputs/random1200_null")
    args = ap.parse_args()

    tle_src = ROOT / args.tle
    subset_name = f"random1200_seed{args.seed}"
    subset_tle = ROOT / f"{args.out_prefix}_{subset_name}.tle"
    n_written = write_subset_tle(tle_src, subset_tle, args.subset_size, args.seed)
    if n_written != args.subset_size:
        raise RuntimeError("subset size mismatch")

    config_text = (ROOT / args.config).read_text(encoding="utf-8")
    config_path = ROOT / f"{args.out_prefix}_{subset_name}.yaml"
    config_path.write_text(
        config_text.replace("tle/starlink_jan2026.txt", subset_tle.as_posix()).replace(
            "max_tle_records: 6364", f"max_tle_records: {args.subset_size}"
        ),
        encoding="utf-8",
    )

    out_csv = ROOT / f"{args.out_prefix}_{subset_name}.csv"
    null_dir = ROOT / f"{args.null_dir_prefix}_{subset_name}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_coverage_grid.py"),
        "--config",
        str(config_path.relative_to(ROOT)),
        "--out",
        str(out_csv.relative_to(ROOT)),
        "--null-dir",
        str(null_dir.relative_to(ROOT)),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
