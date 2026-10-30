"""Analytic routing counterexamples and a CPU-only capture-to-report round trip."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


locality = load("locality", ROOT / "deep_gemm/mega/locality.py")
artifacts = load("artifacts", ROOT / "deep_gemm/mega/diagnostics.py")
try:
    import torch
except ImportError:
    torch = None


def fixture(routes=None, experts=4):
    routes = routes or {
        0: [[0, 1], [0, 2], [2, -1], [-1, -1]],
        1: [[0, 2], [2, 3], [1, 3], [-1, -1]],
    }
    size, width = len(routes), len(routes[0][0])
    records = []
    for rank, rows in routes.items():
        counts, active_rows = artifacts.source_counts(rows, experts)
        records.append(
            {
                "schema": artifacts.CAPTURE_SCHEMA,
                "status": "complete",
                "run_id": "synthetic_fixture",
                "forward_id": 2,
                "layer_id": 3,
                "batch_fingerprint": "synthetic",
                "ep_size": size,
                "ep_rank": rank,
                "native_m": len(rows),
                "input_m": len(rows),
                "output_shape": [len(rows), 4],
                "buffer": {
                    "num_experts": experts,
                    "num_topk": width,
                    "num_max_tokens_per_rank": len(rows) + 2,
                    "hidden": 4,
                    "mma_type": "fp4xfp4",
                    "num_shared_experts": 0,
                },
                "source_expert_counts": counts,
                "source_valid_rows": active_rows,
                "source_masked_assignments": len(rows) * width - sum(counts),
            }
        )
    total = [sum(r["source_expert_counts"][e] for r in records) for e in range(experts)]
    for rank, record in enumerate(records):
        record["expert_recv_counts"] = total[
            rank * (experts // size) : (rank + 1) * (experts // size)
        ]
    return records, routes


class LocalityAccounting(unittest.TestCase):
    def test_physical_counts_unique_tokens_and_semantic_padding_are_distinct(self):
        records, routes = fixture()
        for record, mask in zip(
            records, ([True, True, False, False], [True, False, True, False])
        ):
            record["row_validity"] = {
                "status": "known",
                "valid_mask": mask,
                "source": "synthetic fixture",
            }
        result = locality.summarize_locality(records, routes)
        all_rows = result["scopes"]["all"]
        self.assertEqual(all_rows["source_destination_assignments"], [[3, 2], [2, 4]])
        self.assertEqual(all_rows["source_destination_unique_tokens"], [[2, 2], [2, 3]])
        self.assertEqual(all_rows["destination_fanout_histogram"], [2, 3, 3])
        self.assertEqual(result["scopes"]["valid"]["assignments"], 8)
        self.assertEqual(result["scopes"]["padding"]["assignments"], 3)
        self.assertEqual(result["scopes"]["padding"]["fully_masked_rows"], 2)
        self.assertEqual(result["dimensions"]["unused_capacity_rows_by_source"], [2, 2])
        self.assertEqual(result["row_classification"], "complete")
        self.assertEqual(result["validation"]["native_receive_counters"], "matched")
        tokens = list(locality.iter_token_locality(records, routes))
        self.assertEqual(tokens[0]["destination_fanout"], 1)
        self.assertEqual(tokens[1]["destination_fanout"], 2)
        self.assertEqual(tokens[2]["row_kind"], "padding")
        self.assertEqual(tokens[2]["remote_fanout"], 1)
        for key in (
            "rows",
            "assignments",
            "local_assignments",
            "remote_assignments",
            "unique_token_destination_pairs",
        ):
            self.assertEqual(
                all_rows[key],
                sum(
                    result["scopes"][name][key]
                    for name in ("valid", "padding", "unknown")
                ),
            )

    def test_equal_compute_load_can_be_entirely_local_or_entirely_remote(self):
        a = locality.summarize_locality(*fixture({0: [[0], [0]], 1: [[1], [1]]}, 2))
        b = locality.summarize_locality(*fixture({0: [[1], [1]], 1: [[0], [0]]}, 2))
        x, y = a["scopes"]["all"], b["scopes"]["all"]
        self.assertEqual(
            x["physical_expert_assignments"], y["physical_expert_assignments"]
        )
        self.assertEqual(
            x["assignments_by_destination"], y["assignments_by_destination"]
        )
        self.assertEqual(x["remote_assignment_fraction"], 0)
        self.assertEqual(y["remote_assignment_fraction"], 1)

    def test_equal_expert_and_rank_marginals_do_not_preserve_token_fanout(self):
        a = locality.summarize_locality(
            *fixture({0: [[0, 1], [2, 3]], 1: [[0, 1], [2, 3]]})
        )
        b = locality.summarize_locality(
            *fixture({0: [[0, 2], [1, 3]], 1: [[0, 2], [1, 3]]})
        )
        x, y = a["scopes"]["all"], b["scopes"]["all"]
        self.assertEqual(x["source_expert_assignments"], y["source_expert_assignments"])
        self.assertEqual(
            x["source_destination_assignments"], y["source_destination_assignments"]
        )
        self.assertEqual(x["destination_fanout_histogram"], [0, 4, 0])
        self.assertEqual(y["destination_fanout_histogram"], [0, 0, 4])
        self.assertEqual(x["assignments_per_token_destination_pair"], 2)
        self.assertEqual(y["assignments_per_token_destination_pair"], 1)

    def test_old_and_partially_annotated_captures_do_not_guess_padding(self):
        records, routes = fixture()
        result = locality.summarize_locality(records, routes)
        self.assertEqual(result["row_classification"], "unknown")
        self.assertEqual(result["scopes"]["unknown"]["rows"], 8)
        records[0]["row_validity"] = {"status": "known", "valid_mask": [True] * 4}
        result = locality.summarize_locality(records, routes)
        self.assertEqual(result["row_classification"], "partial")
        self.assertEqual(result["scopes"]["valid"]["fully_masked_rows"], 1)
        self.assertEqual(result["scopes"]["padding"]["rows"], 0)
        self.assertEqual(result["scopes"]["unknown"]["rows"], 4)

    def test_reject_bad_validity_route_identity_and_counter_evidence(self):
        records, routes = fixture()
        mutations = [
            lambda r: r[0].update(
                row_validity={"status": "known", "valid_mask": [1] * 4}
            ),
            lambda r: r[0].update(
                row_validity={"status": "known", "valid_mask": [True]}
            ),
            lambda r: r[0].update(
                row_validity={"status": "unknown", "valid_mask": [True] * 4}
            ),
            lambda r: r[1].update(forward_id=99),
            lambda r: r[1].update(routing_overrides={"round_robin": True}),
            lambda r: r[1].update(
                expert_placement={"physical_to_logical": [0, 1, 2, 3]}
            ),
            lambda r: r[0].update(expert_recv_counts=[999, 0]),
            lambda r: r[0].pop("expert_recv_counts"),
            lambda r: r[0].update(source_expert_counts=[0] * 4),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(records)
            mutate(changed)
            with self.assertRaises(ValueError):
                locality.summarize_locality(changed, routes)
        for bad_row in ([0, 0], [-2, 1], [0.0, 1], [0, 4], [0]):
            changed = copy.deepcopy(routes)
            changed[0][0] = bad_row
            with self.assertRaises(ValueError):
                locality.summarize_locality(records, changed)

    def test_missing_native_counters_are_reported_unavailable(self):
        records, routes = fixture()
        for record in records:
            record.pop("expert_recv_counts")
        result = locality.summarize_locality(records, routes)
        self.assertEqual(
            result["validation"]["native_receive_counters"], "unavailable_in_capture"
        )

    def test_all_masked_routes_have_no_invented_locality_fraction(self):
        records, routes = fixture({0: [[-1, -1]], 1: [[-1, -1]]})
        result = locality.summarize_locality(records, routes)
        scope = result["scopes"]["all"]
        self.assertIsNone(scope["local_assignment_fraction"])
        self.assertIsNone(scope["max_destination_over_mean"])
        self.assertEqual(scope["destination_fanout_histogram"], [2, 0, 0])
        self.assertEqual(result["row_classification"], "unknown")

    @unittest.skipIf(torch is None, "CPU Torch required for trusted tensor inputs")
    def test_capture_to_json_markdown_and_csv_without_cuda(self):
        records, routes = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            for record in records:
                rank = record["ep_rank"]
                for field in ("inputs", "weights", "reference"):
                    payload = (
                        {"topk_idx": torch.tensor(routes[rank], dtype=torch.int32)}
                        if field == "inputs"
                        else torch.zeros(1)
                    )
                    path = root / f"rank-{rank}-{field}.pt"
                    record[field] = artifacts.save_tensors(path, payload)
                path = root / f"rank-{rank}.json"
                artifacts.write_json(path, record)
                manifests.append(
                    {"file": path.name, "sha256": artifacts.file_sha256(path)}
                )
            artifacts.write_json(
                root / "COMPLETE.json",
                {"schema": artifacts.CAPTURE_SCHEMA, "ranks": manifests},
            )
            output = root / "report.json"
            command = [
                sys.executable,
                str(ROOT / "tools/analyze_mega_moe_locality.py"),
                str(root),
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            report = json.loads(output.read_text())
            self.assertEqual(report["scopes"]["all"]["assignments"], 11)
            self.assertEqual(report["row_classification"], "unknown")
            self.assertIn(
                "C: source to destination", output.with_suffix(".md").read_text()
            )
            self.assertEqual(
                len(output.with_suffix(".csv").read_text().splitlines()), 17
            )
            self.assertEqual(
                len(output.with_suffix(".tokens.csv").read_text().splitlines()), 9
            )
            self.assertFalse(torch.cuda.is_initialized())
            retry = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn("FileExistsError", retry.stderr)


if __name__ == "__main__":
    unittest.main()
