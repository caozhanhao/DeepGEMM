"""CPU-only locality accounting for captured physical expert routes.

Counts describe routing, not observed transfers, cache hits, or link bandwidth.
Row validity is an independent producer annotation, never inferred from -1 IDs.
"""

from __future__ import annotations

LOCALITY_SCHEMA = "megamoe.locality.v1"


def _matrix(rows, columns):
    return [[0] * columns for _ in range(rows)]


def _scope(size, experts):
    return {
        "rows_by_source": [0] * size,
        "routed_rows_by_source": [0] * size,
        "masked_route_slots_by_source": [0] * size,
        "source_expert_assignments": _matrix(size, experts),
        "source_destination_assignments": _matrix(size, size),
        "source_destination_unique_tokens": _matrix(size, size),
        "destination_fanout_histogram_by_source": _matrix(size, size + 1),
        "remote_fanout_histogram_by_source": _matrix(size, size + 1),
    }


def _add(scope, rank, ids, width, local_experts):
    destinations = {expert // local_experts for expert in ids}
    scope["rows_by_source"][rank] += 1
    scope["routed_rows_by_source"][rank] += bool(ids)
    scope["masked_route_slots_by_source"][rank] += width - len(ids)
    scope["destination_fanout_histogram_by_source"][rank][len(destinations)] += 1
    scope["remote_fanout_histogram_by_source"][rank][len(destinations - {rank})] += 1
    for expert in ids:
        scope["source_expert_assignments"][rank][expert] += 1
        scope["source_destination_assignments"][rank][expert // local_experts] += 1
    for destination in destinations:
        scope["source_destination_unique_tokens"][rank][destination] += 1


def _finish(scope):
    matrix = scope["source_destination_assignments"]
    unique = scope["source_destination_unique_tokens"]
    size = len(matrix)
    total = sum(map(sum, matrix))
    local = sum(matrix[r][r] for r in range(size))
    destination_work = [sum(row[d] for row in matrix) for d in range(size)]
    unique_total = sum(map(sum, unique))
    scope.update(
        rows=sum(scope["rows_by_source"]),
        routed_rows=sum(scope["routed_rows_by_source"]),
        assignments=total,
        local_assignments=local,
        remote_assignments=total - local,
        local_assignment_fraction=local / total if total else None,
        remote_assignment_fraction=(total - local) / total if total else None,
        unique_token_destination_pairs=unique_total,
        local_unique_token_destination_pairs=sum(unique[r][r] for r in range(size)),
        assignments_per_token_destination_pair=total / unique_total
        if unique_total
        else None,
        assignments_by_destination=destination_work,
        max_destination_over_mean=max(destination_work) * size / total
        if total
        else None,
        physical_expert_assignments=[
            sum(column) for column in zip(*scope["source_expert_assignments"])
        ],
        destination_fanout_histogram=[
            sum(column)
            for column in zip(*scope["destination_fanout_histogram_by_source"])
        ],
    )
    scope["fully_masked_rows"] = scope["rows"] - scope["routed_rows"]


def _validity(record):
    annotation = record.get("row_validity")
    if annotation is None:
        return None, {
            "status": "unknown",
            "reason": "Legacy capture has no semantic row annotation",
        }
    if not isinstance(annotation, dict) or annotation.get("status") not in (
        "known",
        "unknown",
    ):
        raise ValueError("Invalid row-validity annotation")
    if annotation["status"] == "unknown":
        if annotation.get("valid_mask") is not None:
            raise ValueError("Unknown validity must not supply a mask")
        return None, annotation
    mask = annotation.get("valid_mask")
    if (
        not isinstance(mask, list)
        or len(mask) != record["native_m"]
        or any(type(value) is not bool for value in mask)
    ):
        raise ValueError("Row-validity mask must contain one boolean per native row")
    return mask, {k: v for k, v in annotation.items() if k != "valid_mask"}


def summarize_locality(records, routes_by_rank):
    """Summarize all, valid, padded and unknown rows without executing tensors.

    ``routes_by_rank`` maps EP rank to a nested list of physical expert IDs.
    Physical expert slots are contiguous and equally sized per EP rank, as
    required by the native MegaMoE buffer ABI. Logical expert placement is
    separate and cannot be inferred from these IDs.
    """
    if not records:
        raise ValueError("Empty capture")
    records = sorted(records, key=lambda r: r["ep_rank"])
    first = records[0]
    size = first["ep_size"]
    if type(size) is not int or size < 1:
        raise ValueError("Invalid EP size")
    if [r["ep_rank"] for r in records] != list(range(size)) or set(
        routes_by_rank
    ) != set(range(size)):
        raise ValueError("Missing or duplicate EP ranks")
    fields = (
        "run_id",
        "forward_id",
        "layer_id",
        "batch_fingerprint",
        "ep_size",
        "buffer",
    )
    if any(any(r.get(k) != first.get(k) for k in fields) for r in records):
        raise ValueError("Capture identity or buffer differs across ranks")
    for key in ("routing_overrides", "expert_placement"):
        if any(r.get(key) != first.get(key) for r in records):
            raise ValueError(f"Capture {key} differs across ranks")
    experts, width = first["buffer"]["num_experts"], first["buffer"]["num_topk"]
    capacity = first["buffer"]["num_max_tokens_per_rank"]
    if (
        any(type(n) is not int or n < 1 for n in (experts, width, capacity))
        or experts % size
    ):
        raise ValueError("Invalid expert, top-k or capacity dimensions")
    local_experts = experts // size
    scopes = {
        name: _scope(size, experts) for name in ("all", "valid", "padding", "unknown")
    }
    annotations = []
    for rank, record in enumerate(records):
        native_m = record["native_m"]
        if type(native_m) is not int or not 0 <= native_m <= capacity:
            raise ValueError("Invalid native M")
        rows = routes_by_rank[rank]
        if len(rows) != native_m:
            raise ValueError("Route row count differs from native M")
        mask, annotation = _validity(record)
        annotations.append(annotation)
        for index, row in enumerate(rows):
            if len(row) != width or any(
                type(e) is not int or e < -1 or e >= experts for e in row
            ):
                raise ValueError("Invalid route shape or physical expert ID")
            ids = [e for e in row if e != -1]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate physical expert in a token's top-k")
            category = (
                "unknown" if mask is None else ("valid" if mask[index] else "padding")
            )
            _add(scopes["all"], rank, ids, width, local_experts)
            _add(scopes[category], rank, ids, width, local_experts)
        if (
            scopes["all"]["source_expert_assignments"][rank]
            != record["source_expert_counts"]
        ):
            raise ValueError("Route tensor disagrees with captured source histogram")
        if (
            scopes["all"]["masked_route_slots_by_source"][rank]
            != record["source_masked_assignments"]
        ):
            raise ValueError("Route tensor disagrees with captured mask count")
        if (
            "source_valid_rows" in record
            and scopes["all"]["routed_rows_by_source"][rank]
            != record["source_valid_rows"]
        ):
            raise ValueError("Route tensor disagrees with legacy routed-row count")
    for scope in scopes.values():
        _finish(scope)

    recv = [r.get("expert_recv_counts") for r in records]
    native_check = "unavailable_in_capture"
    if any(value is not None for value in recv):
        if any(
            not isinstance(row, list)
            or len(row) != local_experts
            or any(type(n) is not int or n < 0 for n in row)
            for row in recv
        ):
            raise ValueError("Partial or invalid native receive counters")
        if [n for row in recv for n in row] != scopes["all"][
            "physical_expert_assignments"
        ]:
            raise ValueError(
                "Native receive counters disagree with captured source routes"
            )
        native_check = "matched"
    known = scopes["valid"]["rows"] + scopes["padding"]["rows"]
    unknown = scopes["unknown"]["rows"]
    return {
        "schema": LOCALITY_SCHEMA,
        "identity": {k: first[k] for k in fields if k != "buffer"},
        "dimensions": {
            "ep_size": size,
            "physical_experts": experts,
            "routed_top_k": width,
            "native_m_by_source": [r["native_m"] for r in records],
            "unused_capacity_rows_by_source": [
                capacity - r["native_m"] for r in records
            ],
        },
        "row_classification": "complete"
        if unknown == 0
        else ("partial" if known else "unknown"),
        "row_validity_by_source": annotations,
        "routing_overrides_by_source": [r.get("routing_overrides") for r in records],
        "expert_placement_by_source": [r.get("expert_placement") for r in records],
        "host_by_source": [r.get("host") for r in records],
        "scopes": scopes,
        "validation": {
            "source_histograms": "matched",
            "native_receive_counters": native_check,
        },
        "notes": [
            "Matrix rows are source EP ranks; columns are destination EP ranks. Expert IDs are physical slots.",
            "Valid/padding classification uses producer row metadata. Nonnegative routes and nonzero weights do not establish semantic validity.",
            "Unknown rows remain separate. Legacy source_valid_rows means rows with a nonmasked route, not semantic token rows.",
            "C counts assignments; U counts distinct source-token/destination pairs. Fan-out includes local and remote destinations.",
            "C/U and their ratio describe routing and potential reuse, not deduplicated wire bytes, cache hits or bandwidth.",
            "Local means the same GPU/rank. Host labels do not establish NVLink/RDMA topology or calibrated clocks.",
            "Unused buffer capacity is outside native M and is not included in padding or routing counts.",
        ],
    }


def iter_token_locality(records, routes_by_rank):
    """Yield individual native rows after summarize_locality has validated inputs."""
    for record in sorted(records, key=lambda r: r["ep_rank"]):
        rank = record["ep_rank"]
        local_experts = record["buffer"]["num_experts"] // record["ep_size"]
        mask, _ = _validity(record)
        for index, row in enumerate(routes_by_rank[rank]):
            ids = [e for e in row if e != -1]
            destinations = {e // local_experts for e in ids}
            yield {
                "source_rank": rank,
                "native_row": index,
                "row_kind": "unknown"
                if mask is None
                else ("valid" if mask[index] else "padding"),
                "assignments": len(ids),
                "masked_slots": len(row) - len(ids),
                "destination_fanout": len(destinations),
                "remote_fanout": len(destinations - {rank}),
                "has_local_destination": rank in destinations,
            }


def render_markdown(report):
    """An inspectable report without tensor/native dependencies or network assets."""
    size = report["dimensions"]["ep_size"]
    lines = [
        "# MegaMoE locality",
        "",
        f"Run: `{report['identity']['run_id']}`; forward {report['identity']['forward_id']}; layer {report['identity']['layer_id']}.",
        "",
        f"Row classification: **{report['row_classification']}**.",
        f"Native receive-counter check: **{report['validation']['native_receive_counters']}**.",
        "",
        "| Scope | Rows | Routed rows | Assignments | Local | Remote | Fully masked rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, scope in report["scopes"].items():
        lines.append(
            f"| {name} | {scope['rows']} | {scope['routed_rows']} | {scope['assignments']} | "
            f"{scope['local_assignments']} | {scope['remote_assignments']} | {scope['fully_masked_rows']} |"
        )
    lines += [
        "",
        "## Input rows",
        "",
        "| Source | Native M | Unused capacity | Validity evidence |",
        "| --- | ---: | ---: | --- |",
    ]
    for rank, annotation in enumerate(report["row_validity_by_source"]):
        reason = (
            str(
                annotation.get("reason", annotation.get("source", annotation["status"]))
            )
            .replace("|", "\\|")
            .replace("\n", " ")
        )
        lines.append(
            f"| {rank} | {report['dimensions']['native_m_by_source'][rank]} | "
            f"{report['dimensions']['unused_capacity_rows_by_source'][rank]} | {reason} |"
        )
    for name, scope in report["scopes"].items():
        lines += ["", f"## {name}", ""]
        for key, title in (
            ("source_destination_assignments", "C: source to destination assignments"),
            (
                "source_destination_unique_tokens",
                "U: unique source-token/destination pairs",
            ),
            ("destination_fanout_histogram_by_source", "Rows by destination fan-out"),
        ):
            matrix = scope[key]
            columns = len(matrix[0])
            lines += [
                f"### {title}",
                "",
                "| Source | " + " | ".join(map(str, range(columns))) + " |",
                "| --- | " + " | ".join(["---:"] * columns) + " |",
            ]
            lines.extend(
                f"| {r} | " + " | ".join(map(str, matrix[r])) + " |"
                for r in range(size)
            )
    lines += ["", "## Scope", ""] + [f"- {note}" for note in report["notes"]]
    return "\n".join(lines) + "\n"
