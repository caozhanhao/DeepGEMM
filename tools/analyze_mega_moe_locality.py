#!/usr/bin/env python3
"""Inspect locality in a trusted coordinated capture on CPU; no CUDA/native load."""

import argparse
import csv
import importlib.util
from pathlib import Path

from replay_mega_moe import load_artifacts


def load_locality():
    path = Path(__file__).resolve().parents[1] / "deep_gemm/mega/locality.py"
    spec = importlib.util.spec_from_file_location("megamoe_locality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON path; also writes Markdown, edge CSV and per-token CSV",
    )
    args = parser.parse_args(argv)
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    for suffix in (".json", ".md", ".csv", ".tokens.csv"):
        if args.output.with_suffix(suffix).exists():
            raise FileExistsError(args.output.with_suffix(suffix))
    artifacts, locality = load_artifacts(), load_locality()
    # Validate the complete rank set and manifest hashes, but do not reread
    # multi-GB expert weights for an analysis that consumes only the inputs.
    records = artifacts.validate_capture(args.capture, verify_hashes=False)
    import torch

    routes = {}
    for record in records:
        path = artifacts._checked_file(args.capture, record["inputs"], True)
        inputs = torch.load(path, map_location="cpu", weights_only=True)
        ids = inputs["topk_idx"]
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Captured routes must be int32 or int64")
        routes[record["ep_rank"]] = ids.tolist()
        del inputs, ids
    result = locality.summarize_locality(records, routes)
    result["artifact_verification"] = (
        "All rank manifests and consumed input payload hashes verified; "
        "weights/reference checked for presence and size, not loaded or rehashed"
    )
    artifacts.write_json(args.output, result)
    args.output.with_suffix(".md").write_text(locality.render_markdown(result))
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["scope", "source_rank", "destination_rank", "assignments", "unique_tokens"]
        )
        for name, scope in result["scopes"].items():
            for s, row in enumerate(scope["source_destination_assignments"]):
                for d, count in enumerate(row):
                    writer.writerow(
                        [
                            name,
                            s,
                            d,
                            count,
                            scope["source_destination_unique_tokens"][s][d],
                        ]
                    )
    with args.output.with_suffix(".tokens.csv").open("w", newline="") as stream:
        fields = [
            "source_rank",
            "native_row",
            "row_kind",
            "assignments",
            "masked_slots",
            "destination_fanout",
            "remote_fanout",
            "has_local_destination",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(locality.iter_token_locality(records, routes))
    print(
        f"Wrote {args.output} and Markdown/CSV; row validity={result['row_classification']}; "
        f"native counters={result['validation']['native_receive_counters']}"
    )


if __name__ == "__main__":
    main()
