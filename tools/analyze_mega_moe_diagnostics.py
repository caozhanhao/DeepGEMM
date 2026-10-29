#!/usr/bin/env python3
"""Merge a complete all-rank statistics window on a CPU-only machine."""

import argparse
import csv
from pathlib import Path

from replay_mega_moe import load_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--block-m",
        type=int,
        help="Only supply BLOCK_M verified from the actual kernel",
    )
    args = parser.parse_args(argv)
    artifacts = load_artifacts()
    paths = sorted(args.window.glob("stats-rank-*.json"))
    result = artifacts.summarize_stats(paths, block_m=args.block_m)
    artifacts.write_json(args.output, result)
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["layers"][0]))
        writer.writeheader()
        writer.writerows(result["layers"])
    print(
        f"Validated {len(paths)} ranks, {len(result['calls'])} layer calls; wrote {args.output}"
    )


if __name__ == "__main__":
    main()
