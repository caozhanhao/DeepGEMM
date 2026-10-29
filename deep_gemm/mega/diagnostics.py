"""Portable artifacts for opt-in, routed NVFP4 MegaMoE diagnostics.

No Torch/native imports at module import time: the manifest and statistics
readers also work on a CPU-only analysis machine. Tensor payloads are private
model data; load only captures produced by a trusted worker.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

CAPTURE_SCHEMA = "megamoe.capture.v1"
STATS_SCHEMA = "megamoe.stats.v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_name(value):
    if (
        not isinstance(value, str)
        or not value
        or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in value)
    ):
        raise ValueError(
            f"Expected an ASCII name using letters, digits, _ or -: {value!r}"
        )
    return value


def require_consensus(records, fields):
    """All ranks must call the collective BEFORE deciding to capture/skip."""
    if not records:
        raise ValueError("Empty rank consensus")
    for field in fields:
        if any(row.get(field) != records[0].get(field) for row in records):
            raise ValueError(
                f"Rank disagreement on {field}: {[r.get(field) for r in records]}"
            )
    errors = [r.get("error") for r in records if r.get("error")]
    if errors:
        raise RuntimeError(f"A diagnostic worker failed: {errors}")


def batch_identity(forward_id, batch):
    """CPU metadata only; never call .cpu() on a tensor in the timing path."""

    def integers(name):
        value = getattr(batch, name, None)
        if value is None:
            return None
        if getattr(getattr(value, "device", None), "type", "cpu") != "cpu":
            raise ValueError(f"{name} must already reside on CPU")
        return [int(x) for x in value]

    rids = getattr(batch, "rids", None)
    non_padded = getattr(batch, "num_token_non_padded_cpu", None)
    info = {
        "forward_id": int(forward_id),
        "batch_size": int(batch.batch_size),
        "forward_mode": str(batch.forward_mode),
        "request_ids_sha256": digest(list(rids)) if rids is not None else None,
        "extend_seq_lens": integers("extend_seq_lens_cpu"),
        "extend_prefix_lens": integers("extend_prefix_lens_cpu"),
        "global_num_tokens": integers("global_num_tokens_cpu"),
        "num_token_non_padded_cpu": int(non_padded) if non_padded is not None else None,
    }
    # A same-shaped batch alone is not an identity. Fail closed for unsupported
    # runners that do not carry scheduler request identities and chunk offsets.
    if (
        rids is None
        or info["extend_seq_lens"] is None
        or info["extend_prefix_lens"] is None
    ):
        raise ValueError(
            "Diagnostics require request IDs and CPU extend/prefix lengths"
        )
    if (
        len(rids) != info["batch_size"]
        or len(info["extend_seq_lens"]) != len(rids)
        or len(info["extend_prefix_lens"]) != len(rids)
    ):
        raise ValueError("Inconsistent request/extend metadata")
    info["fingerprint"] = digest(info)
    return info


def native_identity(package):
    root = Path(package.__file__).resolve().parent
    files = [root / "mega" / "__init__.py", Path(package._C.__file__).resolve()]
    files += sorted((root / "include").rglob("*.cuh"))
    files += sorted((root / "include").rglob("*.hpp"))
    hashes = {str(p.relative_to(root)): file_sha256(p) for p in files if p.is_file()}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    runtime = {
        name: getattr(package, name)()
        for name in ("get_num_sms", "get_tc_util", "get_pdl")
    }
    return {
        "package_path": str(root),
        "package_name": package.__name__,
        "git_head": commit,
        "files": hashes,
        "fingerprint": digest(hashes),
        "runtime": runtime,
        "jit_cubin_identity": "not_collected",
        "env": {
            key: os.environ[key]
            for key in (
                "DG_JIT_CACHE_DIR",
                "DG_JIT_WITH_LINEINFO",
                "DG_JIT_USE_NVRTC",
                "DG_USE_FP4_ACTS",
                "DG_USE_MXF4_KIND",
                "CUDA_VISIBLE_DEVICES",
            )
            if key in os.environ
        },
    }


def buffer_spec(buffer):
    return {
        key: getattr(buffer, key)
        for key in (
            "num_experts",
            "num_max_tokens_per_rank",
            "num_topk",
            "hidden",
            "intermediate_hidden",
            "num_shared_experts",
            "mma_type",
            "activation",
        )
    }


def tensor_tree(value, device="cpu"):
    """Copy values AND strides, including MN-major scale tensors.

    Zero the underlying storage as well as its visible view; serializing an
    empty_strided tensor must not copy uninitialized padding from host memory.
    """
    import torch

    if isinstance(value, torch.Tensor):
        shape, strides = tuple(value.shape), tuple(value.stride())
        if any(s < 0 for s in strides):
            raise ValueError("Negative tensor strides are unsupported")
        span = (
            0
            if value.numel() == 0
            else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
        )
        storage = torch.zeros(span, dtype=value.dtype, device=device)
        result = storage.as_strided(shape, strides)
        result.copy_(value.detach())
        return result
    if isinstance(value, tuple):
        return tuple(tensor_tree(x, device) for x in value)
    if isinstance(value, list):
        return [tensor_tree(x, device) for x in value]
    if isinstance(value, dict):
        return {k: tensor_tree(v, device) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported artifact type: {type(value)}")


def save_tensors(path, value):
    import torch

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def source_counts(topk_idx, num_experts):
    """Count physical IDs from a CPU capture, respecting masked -1 routes."""
    counts = [0] * num_experts
    valid_rows = 0
    for row in topk_idx:
        valid = [int(x) for x in row if int(x) != -1]
        if any(x < 0 or x >= num_experts for x in valid):
            raise ValueError("A route is outside the physical expert table")
        if len(set(valid)) != len(valid):
            raise ValueError("Duplicate expert IDs in one token's routed top-k")
        valid_rows += bool(valid)
        for expert in valid:
            counts[expert] += 1
    return counts, valid_rows


def capture_inputs(
    folder, *, package, native, y, l1_weights, l2_weights, buffer, kwargs, metadata
):
    """Untimed, blocking snapshot. Caller coordinates all ranks and completion."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rank = metadata["ep_rank"]
    prefix = f"rank-{rank:04d}"
    bound = inspect.signature(native).bind(y, l1_weights, l2_weights, buffer, **kwargs)
    bound.apply_defaults()
    arguments = {
        k: v
        for k, v in bound.arguments.items()
        if k
        not in (
            "y",
            "l1_weights",
            "l2_weights",
            "sym_buffer",
            "cumulative_local_expert_recv_stats",
        )
    }
    if arguments.get("shared_l1_weights") is not None or buffer.num_shared_experts != 0:
        raise ValueError("v1 capture supports routed-only NVFP4 calls")
    n = y.shape[0]
    inputs = tensor_tree(
        {
            "x": buffer.x[:n],
            "x_sf": buffer.x_sf[:n],
            "topk_idx": buffer.topk_idx[:n],
            "topk_weights": buffer.topk_weights[:n],
            "arguments": arguments,
        }
    )
    counts, valid_rows = source_counts(inputs["topk_idx"].tolist(), buffer.num_experts)
    record = dict(
        metadata,
        schema=CAPTURE_SCHEMA,
        status="inputs_saved",
        output_shape=list(y.shape),
        output_dtype=str(y.dtype),
        buffer=buffer_spec(buffer),
        native=native_identity(package),
        source_expert_counts=counts,
        source_valid_rows=valid_rows,
        source_masked_assignments=n * buffer.num_topk - sum(counts),
        host=socket.gethostname(),
        pid=os.getpid(),
        captured_ns=time.time_ns(),
    )
    record["inputs"] = save_tensors(folder / f"{prefix}-inputs.pt", inputs)
    del inputs
    record["weights"] = save_tensors(
        folder / f"{prefix}-weights.pt",
        tensor_tree({"l1_weights": l1_weights, "l2_weights": l2_weights}),
    )
    return record


def capture_reference(folder, record, y):
    folder = Path(folder)
    prefix = f"rank-{record['ep_rank']:04d}"
    record["reference"] = save_tensors(
        folder / f"{prefix}-reference.pt", tensor_tree(y)
    )
    record["status"] = "complete"
    write_json(folder / f"{prefix}.json", record)
    return {"file": f"{prefix}.json", "sha256": file_sha256(folder / f"{prefix}.json")}


def _checked_file(folder, entry, verify_hashes):
    name = entry["file"]
    if Path(name).name != name or name in (".", ".."):
        raise ValueError(f"Non-local artifact path: {name}")
    path = folder / name
    if not path.is_file():
        raise ValueError(f"Missing artifact: {path}")
    if "bytes" in entry and path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Artifact size mismatch: {path}")
    if verify_hashes and file_sha256(path) != entry["sha256"]:
        raise ValueError(f"Artifact SHA256 mismatch: {path}")
    return path


def validate_capture(folder, *, verify_hashes=True):
    folder = Path(folder)
    completion = json.loads((folder / "COMPLETE.json").read_text())
    if completion.get("schema") != CAPTURE_SCHEMA:
        raise ValueError("Unsupported capture schema")
    records = [
        json.loads(_checked_file(folder, item, True).read_text())
        for item in completion["ranks"]
    ]
    require_consensus(
        records,
        (
            "schema",
            "run_id",
            "forward_id",
            "batch_fingerprint",
            "layer_id",
            "ep_size",
            "buffer",
        ),
    )
    size = records[0]["ep_size"]
    if sorted(r["ep_rank"] for r in records) != list(range(size)):
        raise ValueError("Missing or duplicate EP ranks")
    for record in records:
        if record["status"] != "complete" or record["schema"] != CAPTURE_SCHEMA:
            raise ValueError("Incomplete capture")
        if (
            record["buffer"]["mma_type"] != "fp4xfp4"
            or record["buffer"]["num_shared_experts"]
        ):
            raise ValueError("Only routed NVFP4 captures are supported")
        for key in ("inputs", "weights", "reference"):
            _checked_file(folder, record[key], verify_hashes)
        if len(record["source_expert_counts"]) != record["buffer"]["num_experts"]:
            raise ValueError("Invalid route histogram length")
        if any(type(x) is not int or x < 0 for x in record["source_expert_counts"]):
            raise ValueError("Invalid route histogram")
        if record["output_shape"][0] <= 0:
            raise ValueError("Empty rank capture is unsupported in v1")
        if (
            record["input_m"] != record["native_m"]
            or record["output_shape"]
            != [record["native_m"], record["buffer"]["hidden"]]
            or record["native_m"] > record["buffer"]["num_max_tokens_per_rank"]
        ):
            raise ValueError("Invalid native M/output shape")
        slots = record["native_m"] * record["buffer"]["num_topk"]
        if (
            sum(record["source_expert_counts"]) + record["source_masked_assignments"]
            != slots
        ):
            raise ValueError("Source route histogram/mask does not cover native M")
    return sorted(records, key=lambda r: r["ep_rank"])


def workload_summary(counts, ep_size, *, block_m=None):
    if (
        not counts
        or len(counts) % ep_size
        or any(type(x) is not int or x < 0 for x in counts)
    ):
        raise ValueError("Invalid expert counts or EP size")
    local = len(counts) // ep_size
    rank_work = [sum(counts[r * local : (r + 1) * local]) for r in range(ep_size)]
    mean = statistics.mean(rank_work)
    result = {
        "physical_expert_counts": counts,
        "rank_assignments": rank_work,
        "assignments": sum(counts),
        "max_rank_over_mean": max(rank_work) / mean if mean else None,
    }
    if block_m is not None:
        if type(block_m) is not int or block_m <= 0:
            raise ValueError(
                "BLOCK_M must be a positive integer from the actual kernel"
            )
        result["m_blocks_per_rank"] = [
            sum(math.ceil(x / block_m) for x in counts[r * local : (r + 1) * local])
            for r in range(ep_size)
        ]
        result["block_m"] = block_m
        result["block_note"] = "M blocks only; not MMA FLOPs or tensor utilization"
    return result


def summarize_stats(paths, *, block_m=None):
    documents = [json.loads(Path(path).read_text()) for path in paths]
    require_consensus(documents, ("schema", "run_id", "config", "ep_size"))
    if documents[0]["schema"] != STATS_SCHEMA:
        raise ValueError("Unsupported statistics schema")
    size = documents[0]["ep_size"]
    documents.sort(key=lambda x: x["ep_rank"])
    if [d["ep_rank"] for d in documents] != list(range(size)):
        raise ValueError("Missing or duplicate EP ranks")
    if any(d.get("status") != "complete" for d in documents):
        raise ValueError("Incomplete statistics window")
    keys = [
        [(r["forward_id"], r["layer_id"], r["batch_fingerprint"]) for r in d["calls"]]
        for d in documents
    ]
    if any(k != keys[0] for k in keys) or len(set(keys[0])) != len(keys[0]):
        raise ValueError("Rank calls do not identify the same forwards/layers")
    expected = documents[0]["config"]["num_forwards"] * len(
        documents[0]["config"]["layers"]
    )
    if len(keys[0]) != expected:
        raise ValueError("Statistics window has missing calls")
    rows = []
    for index, (forward_id, layer_id, fingerprint) in enumerate(keys[0]):
        calls = [d["calls"][index] for d in documents]
        require_consensus(
            calls, ("num_experts", "hidden", "intermediate_hidden", "top_k")
        )
        counts = [n for call in calls for n in call["expert_recv_counts"]]
        if len(counts) != calls[0]["num_experts"]:
            raise ValueError("Expert histogram does not cover the full EP group")
        work = workload_summary(counts, size, block_m=block_m)
        actual_blocks = [c.get("block_m_from_native_api") for c in calls]
        if all(type(n) is int and n > 0 for n in actual_blocks):
            if block_m is not None and any(n != block_m for n in actual_blocks):
                raise ValueError(
                    "--block-m contradicts the native API's recorded selection"
                )
            work["block_m_by_rank"] = actual_blocks
            work["m_blocks_per_rank"] = [
                sum(math.ceil(n / b) for n in c["expert_recv_counts"])
                for c, b in zip(calls, actual_blocks)
            ]
            work["block_note"] = (
                "M blocks using native API selection; not MMA FLOPs or tensor utilization"
            )
        maximum = sum(c["native_m"] * c["top_k"] for c in calls)
        if work["assignments"] > maximum:
            raise ValueError(
                "Receive counts exceed route slots (cumulative/replayed counters?)"
            )
        durations = [c["native_ms"] for c in calls]
        if any(not math.isfinite(x) or x <= 0 for x in durations):
            raise ValueError("Invalid CUDA event duration")
        rows.append(
            dict(
                work,
                forward_id=forward_id,
                layer_id=layer_id,
                batch_fingerprint=fingerprint,
                native_m=[c["native_m"] for c in calls],
                native_ms_by_rank=durations,
                max_rank_native_ms=max(durations),
                valid_routed_flops=6
                * calls[0]["hidden"]
                * calls[0]["intermediate_hidden"]
                * work["assignments"],
                masked_or_unaccounted_route_slots=maximum - work["assignments"],
            )
        )
    layers = []
    for layer in sorted({r["layer_id"] for r in rows}):
        selected = [r for r in rows if r["layer_id"] == layer]
        ratios = [
            r["max_rank_over_mean"]
            for r in selected
            if r["max_rank_over_mean"] is not None
        ]
        layers.append(
            {
                "layer_id": layer,
                "samples": len(selected),
                "median_max_rank_native_ms": statistics.median(
                    r["max_rank_native_ms"] for r in selected
                ),
                "median_max_rank_over_mean": statistics.median(ratios)
                if ratios
                else None,
            }
        )
    ordered = sorted(layers, key=lambda r: r["median_max_rank_native_ms"])
    return {
        "schema": "megamoe.summary.v1",
        "run_id": documents[0]["run_id"],
        "calls": rows,
        "layers": layers,
        "representative_layers": {
            name: ordered[i]["layer_id"]
            for name, i in (
                ("fast", 0),
                ("typical", len(ordered) // 2),
                ("slow", len(ordered) - 1),
            )
        },
        "timing_scope": "Native API CUDA events include waiting and launch gaps; max rank duration is not a synchronized global span",
        "count_scope": "Physical expert assignments; matching route histograms require capture/replay validation",
        "input_files": [{"path": str(p), "sha256": file_sha256(p)} for p in paths],
    }
