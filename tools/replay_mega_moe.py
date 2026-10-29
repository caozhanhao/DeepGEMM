#!/usr/bin/env python3
"""Replay a coordinated, real SGLang NVFP4 capture under torchrun.

--inspect validates an artifact without importing Torch or loading a GPU.
Profiling is a separate run; its event times are never labeled a baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import statistics
import sys
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path


def load_artifacts():
    path = Path(__file__).resolve().parents[1] / "deep_gemm/mega/diagnostics.py"
    spec = importlib.util.spec_from_file_location("megamoe_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Verify all ranks and SHA256s, without Torch/CUDA",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--deep-gemm-path",
        type=Path,
        help="Exact deep_gemm package directory, as used by SGLang",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--entry-mode", choices=("barrier", "free"), default="barrier")
    parser.add_argument("--profile", choices=("none", "torch", "ncu"), default="none")
    parser.add_argument(
        "--allow-native-change",
        action="store_true",
        help="Record an explicit source/binary A/B change",
    )
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if (
        args.warmup < 1
        or args.iterations < 1
        or args.timeout_seconds < 1
        or args.rtol < 0
        or args.atol < 0
    ):
        parser.error(
            "Require positive warmup/iterations/timeout and nonnegative tolerances"
        )
    if args.profile == "ncu" and args.iterations != 1:
        parser.error(
            "NCU runs require --iterations 1; compare against a separate --profile none run"
        )
    if not args.inspect and args.output is None:
        parser.error("--output is required for replay")
    return args


def load_native(path):
    if path is None:
        import deep_gemm

        return deep_gemm
    spec = importlib.util.spec_from_file_location(
        "deep_gemm_replay", path / "__init__.py", submodule_search_locations=[str(path)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return package


def main(argv=None):
    args = parse_args(argv)
    artifacts = load_artifacts()
    if args.inspect:
        records = artifacts.validate_capture(args.capture)
        counts = [
            sum(r["source_expert_counts"][e] for r in records)
            for e in range(records[0]["buffer"]["num_experts"])
        ]
        print(
            json.dumps(
                {
                    "status": "artifact_valid",
                    "ep_size": len(records),
                    "forward_id": records[0]["forward_id"],
                    "layer_id": records[0]["layer_id"],
                    "input_m": [r["input_m"] for r in records],
                    "buffer": records[0]["buffer"],
                    "workload": artifacts.workload_summary(counts, len(records)),
                },
                indent=2,
            )
        )
        return

    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))
    cpu_group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=args.timeout_seconds)
    )
    rank, size = dist.get_rank(), dist.get_world_size()
    args.output.mkdir(parents=True, exist_ok=True)

    def gather(record):
        records = [None] * size
        dist.all_gather_object(records, record, group=cpu_group)
        artifacts.require_consensus(records, ())
        return records

    def prepare():
        # Every rank verifies its own large payloads; every manifest is checked.
        records = artifacts.validate_capture(args.capture, verify_hashes=False)
        if len(records) != size:
            raise ValueError(
                f"Capture requires {len(records)} ranks, torchrun provided {size}"
            )
        record = records[rank]
        for key in ("inputs", "weights", "reference"):
            artifacts._checked_file(args.capture, record[key], True)
        package = load_native(args.deep_gemm_path)
        for getter, value in record["native"]["runtime"].items():
            getattr(package, getter.replace("get_", "set_", 1))(value)
        current = artifacts.native_identity(package)
        if (
            current["fingerprint"] != record["native"]["fingerprint"]
            and not args.allow_native_change
        ):
            raise ValueError(
                "Native binary/header identity differs; verify baseline or explicitly use --allow-native-change"
            )
        inputs = torch.load(
            args.capture / record["inputs"]["file"],
            map_location="cpu",
            weights_only=True,
        )
        weights = torch.load(
            args.capture / record["weights"]["file"],
            map_location="cpu",
            weights_only=True,
        )
        reference = torch.load(
            args.capture / record["reference"]["file"],
            map_location="cpu",
            weights_only=True,
        )
        actual_histogram, valid_rows = artifacts.source_counts(
            inputs["topk_idx"].tolist(), record["buffer"]["num_experts"]
        )
        if (
            actual_histogram != record["source_expert_counts"]
            or valid_rows != record["source_valid_rows"]
        ):
            raise ValueError(
                "Captured route histogram does not match the tensor payload"
            )
        if (
            list(reference.shape) != record["output_shape"]
            or str(reference.dtype) != record["output_dtype"]
        ):
            raise ValueError("Reference tensor shape/dtype mismatch")
        gpu_inputs = artifacts.tensor_tree(inputs, "cuda")
        gpu_weights = artifacts.tensor_tree(weights, "cuda")
        return (
            records,
            record,
            package,
            current,
            gpu_inputs,
            gpu_weights,
            reference.to("cuda"),
        )

    error = None
    prepared = None
    try:
        prepared = prepare()
    except Exception as caught:  # noqa: BLE001 - all ranks must learn the failure
        error = repr(caught)
    gather({"error": error})  # propagate preparation errors before rendezvous
    records, record, package, current_native, inputs, weights, reference = prepared
    settings = {
        "profile": args.profile,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "entry_mode": args.entry_mode,
        "rtol": args.rtol,
        "atol": args.atol,
        "capture_key": [
            record["run_id"],
            record["forward_id"],
            record["layer_id"],
            record["batch_fingerprint"],
        ],
    }
    artifacts.require_consensus(gather(settings), tuple(settings))
    # Output reservation: application replay may legitimately restart the process
    # under NCU. Keep each attempt's numerical checks and timing separately.
    import time

    attempt = f"rank-{rank:04d}-pid-{os.getpid()}-{time.time_ns()}"
    buffer = package.get_symm_buffer_for_mega_moe(dist.group.WORLD, **record["buffer"])
    y = torch.empty_like(reference)
    per_rank = buffer.num_experts // size
    total_counts = [
        sum(r["source_expert_counts"][e] for r in records)
        for e in range(buffer.num_experts)
    ]
    expected_counts = total_counts[rank * per_rank : (rank + 1) * per_rank]
    stats = torch.zeros(per_rank, dtype=torch.int32, device="cuda")

    def refill():
        n = y.shape[0]
        for key in ("x", "x_sf", "topk_idx", "topk_weights"):
            target, source = getattr(buffer, key)[:n], inputs[key]
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError(
                    f"Symmetric buffer {key} shape/dtype differs from capture"
                )
            target.copy_(source)

    def enter():
        refill()
        if args.entry_mode == "barrier":
            dist.barrier()

    def native(counts=None):
        package.fp4_fp4_mega_moe(
            y,
            weights["l1_weights"],
            weights["l2_weights"],
            buffer,
            cumulative_local_expert_recv_stats=counts,
            **inputs["arguments"],
        )

    error = None
    try:
        refill()
        inspect.signature(package.fp4_fp4_mega_moe).bind(
            y,
            weights["l1_weights"],
            weights["l2_weights"],
            buffer,
            cumulative_local_expert_recv_stats=stats,
            **inputs["arguments"],
        )
    except Exception as caught:  # noqa: BLE001 - propagate before collective kernel
        error = repr(caught)
    gather({"error": error})
    enter()
    native(stats)
    torch.cuda.synchronize()
    error = None
    try:
        if stats.cpu().tolist() != expected_counts:
            raise ValueError(
                "Native receive counters differ from exact captured routes"
            )
        torch.testing.assert_close(y, reference, rtol=args.rtol, atol=args.atol)
    except Exception as caught:  # noqa: BLE001 - all ranks must learn the failure
        error = repr(caught)
    diff = y.float() - reference.float()
    correctness = {
        "error": error,
        "max_abs": float(diff.abs().max().item()),
        "relative_l2": float(
            (diff.norm() / reference.float().norm().clamp_min(1e-30)).item()
        ),
        "expected_recv_counts": expected_counts,
        "actual_recv_counts": stats.cpu().tolist(),
    }
    artifacts.write_json(args.output / f"{attempt}-correctness.json", correctness)
    gather(correctness)

    events = [
        tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
        for _ in range(args.iterations)
    ]
    for pair in events:
        for event in pair:
            event.record()
    for _ in range(args.warmup):
        enter()
        native()
    torch.cuda.synchronize()
    # Correctness and counter validation above are outside the NCU NVTX filter.
    # No auxiliary counters in the profiled call: replay can repeat side effects.
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        )
        if args.profile == "torch"
        else nullcontext()
    )
    with profiler as trace:
        for begin, end in events:
            enter()  # copies/barrier outside the native event interval
            begin.record()
            if args.profile == "ncu":
                torch.cuda.nvtx.range_push("megamoe-replay")
            try:
                native()
            finally:
                if args.profile == "ncu":
                    torch.cuda.nvtx.range_pop()
            end.record()
        torch.cuda.synchronize()
    times = [begin.elapsed_time(end) for begin, end in events]
    # Check the final output too: profiler replay must not silently corrupt it.
    error = None
    try:
        torch.testing.assert_close(y, reference, rtol=args.rtol, atol=args.atol)
    except Exception as caught:  # noqa: BLE001 - all ranks must learn the failure
        error = repr(caught)
    gather({"error": error})
    if args.profile == "torch":
        trace.export_chrome_trace(str(args.output / f"{attempt}.trace.json"))
    report = {
        "schema": "megamoe.replay.v1",
        "status": "passed",
        "ep_rank": rank,
        "ep_size": size,
        "settings": settings,
        "native": current_native,
        "capture_native": record["native"],
        "allow_native_change": args.allow_native_change,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(),
        "native_ms": times,
        "median_native_ms": statistics.median(times),
        "correctness": correctness,
        "timing_class": "unprofiled_replay"
        if args.profile == "none"
        else "profiled_diagnostic_only",
        "scope": "Repeated isolated real-input native calls; not model throughput; cache not flushed; clocks not locked",
        "workload": artifacts.workload_summary(total_counts, size),
    }
    artifacts.write_json(args.output / f"{attempt}.json", report)
    dist.barrier()
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()
    print(
        json.dumps(
            {
                "rank": rank,
                "status": "passed",
                "median_native_ms": statistics.median(times),
                "profile": args.profile,
            }
        )
    )


if __name__ == "__main__":
    main()
