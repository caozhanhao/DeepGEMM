"""CPU contract tests; no compiled DeepGEMM import and no GPU required."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


art = load("artifacts", ROOT / "deep_gemm/mega/diagnostics.py")
profile = load("profile_rank", ROOT / "tools/profile_mega_moe_rank.py")
try:
    import torch
except ImportError:
    torch = None


class ArtifactContracts(unittest.TestCase):
    def test_peer_failure_is_not_treated_as_a_complete_collective(self):
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            art.require_consensus([{"error": None}, {"error": "disk full"}], ())

    def test_forward_identity_distinguishes_same_shape_requests_and_chunks(self):
        batch = SimpleNamespace(
            batch_size=1,
            forward_mode="extend",
            rids=["a"],
            extend_seq_lens_cpu=[16],
            extend_prefix_lens_cpu=[32],
        )
        first = art.batch_identity(7, batch)
        batch.rids = ["b"]
        self.assertNotEqual(
            first["fingerprint"], art.batch_identity(7, batch)["fingerprint"]
        )
        batch.rids = ["a"]
        batch.extend_prefix_lens_cpu = [48]
        self.assertNotEqual(
            first["fingerprint"], art.batch_identity(7, batch)["fingerprint"]
        )
        with self.assertRaisesRegex(ValueError, "forward_id"):
            art.require_consensus(
                [{"forward_id": 7}, {"forward_id": 8}], ("forward_id",)
            )

    def test_missing_identity_and_device_metadata_fail_closed(self):
        batch = SimpleNamespace(batch_size=1, forward_mode="extend")
        with self.assertRaisesRegex(ValueError, "request IDs"):
            art.batch_identity(1, batch)
        batch.extend_seq_lens_cpu = SimpleNamespace(device=SimpleNamespace(type="cuda"))
        with self.assertRaisesRegex(ValueError, "already reside on CPU"):
            art.batch_identity(1, batch)

    def test_masked_routing_and_invalid_experts(self):
        self.assertEqual(
            art.source_counts([[0, 3], [3, -1], [-1, -1]], 4), ([1, 0, 0, 2], 2)
        )
        for routes in ([[4]], [[-2]], [[1, 1]]):
            with self.assertRaises(ValueError):
                art.source_counts(routes, 4)

    def test_load_and_tiling_do_not_assume_equal_rank_work(self):
        result = art.workload_summary([1, 2, 0, 9], 2, block_m=4)
        self.assertEqual(result["rank_assignments"], [3, 9])
        self.assertEqual(result["max_rank_over_mean"], 1.5)
        self.assertEqual(result["m_blocks_per_rank"], [2, 3])

    def _capture(self, folder):
        manifests = []
        for rank in range(2):
            record = {
                "schema": art.CAPTURE_SCHEMA,
                "status": "complete",
                "run_id": "fixture",
                "forward_id": 10,
                "batch_fingerprint": "same-batch",
                "layer_id": 3,
                "ep_size": 2,
                "ep_rank": rank,
                "input_m": 2,
                "native_m": 2,
                "output_shape": [2, 4],
                "buffer": {
                    "num_experts": 4,
                    "num_topk": 2,
                    "num_max_tokens_per_rank": 2,
                    "hidden": 4,
                    "mma_type": "fp4xfp4",
                    "num_shared_experts": 0,
                },
                "source_expert_counts": [1, 1, 1, 0],
                "source_masked_assignments": 1,
            }
            for key in ("inputs", "weights", "reference"):
                path = folder / f"r{rank}-{key}.pt"
                path.write_bytes(
                    b"opaque payload; no pickle deserialization in manifest validation"
                )
                record[key] = {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": art.file_sha256(path),
                }
            path = folder / f"rank-{rank:04d}.json"
            art.write_json(path, record)
            manifests.append({"file": path.name, "sha256": art.file_sha256(path)})
        art.write_json(
            folder / "COMPLETE.json", {"schema": art.CAPTURE_SCHEMA, "ranks": manifests}
        )

    def test_complete_capture_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self._capture(folder)
            self.assertEqual(len(art.validate_capture(folder)), 2)
            (folder / "r1-reference.pt").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                art.validate_capture(folder)

    def test_partial_and_mixed_forward_captures_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self._capture(folder)
            record = json.loads((folder / "rank-0001.json").read_text())
            record["forward_id"] = 11
            art.write_json(folder / "rank-0001.json", record)
            complete = json.loads((folder / "COMPLETE.json").read_text())
            complete["ranks"][1]["sha256"] = art.file_sha256(folder / "rank-0001.json")
            art.write_json(folder / "COMPLETE.json", complete)
            with self.assertRaisesRegex(ValueError, "forward_id"):
                art.validate_capture(folder)
            (folder / "COMPLETE.json").unlink()
            with self.assertRaises(FileNotFoundError):
                art.validate_capture(folder)

    def test_stats_merge_and_replayed_counters(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for rank in range(2):
                path = Path(temp) / f"stats-rank-{rank:04d}.json"
                doc = {
                    "schema": art.STATS_SCHEMA,
                    "status": "complete",
                    "run_id": "fixture",
                    "ep_size": 2,
                    "ep_rank": rank,
                    "config": {"num_forwards": 1, "layers": [3, 4]},
                    "calls": [
                        {
                            "forward_id": 10,
                            "layer_id": layer,
                            "batch_fingerprint": "same",
                            "native_m": 4,
                            "top_k": 2,
                            "num_experts": 4,
                            "hidden": 8,
                            "intermediate_hidden": 4,
                            "expert_recv_counts": [1, 2] if rank == 0 else [4, 5],
                            "native_ms": duration,
                        }
                        for layer, duration in ((3, 2.0), (4, 4.0))
                    ],
                }
                art.write_json(path, doc)
                paths.append(path)
            result = art.summarize_stats(paths)
            self.assertEqual(result["calls"][0]["rank_assignments"], [3, 9])
            self.assertEqual(result["calls"][0]["valid_routed_flops"], 6 * 8 * 4 * 12)
            self.assertEqual(result["representative_layers"]["slow"], 4)
            with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
                art.summarize_stats(paths[:1])
            doc["calls"][0]["expert_recv_counts"] = [100, 100]
            art.write_json(paths[1], doc)
            with self.assertRaisesRegex(ValueError, "exceed route slots"):
                art.summarize_stats(paths)
            doc["calls"][0]["forward_id"] += 1
            art.write_json(paths[1], doc)
            with self.assertRaisesRegex(ValueError, "same forwards"):
                art.summarize_stats(paths)

    def test_ncu_command_uses_all_ranks_and_no_destructive_overwrite(self):
        args = SimpleNamespace(
            ncu="ncu",
            host="head-node",
            port=29513,
            section=None,
            output=Path("results"),
            capture=Path("capture"),
            warmup=10,
            timeout_seconds=1800,
            deep_gemm_path=None,
            allow_native_change=False,
        )
        env = {"RANK": "3", "WORLD_SIZE": "8", "MASTER_PORT": "29500"}
        command = profile.build_command(args, env)
        self.assertEqual(
            command[command.index("--communicator-tcp-num-peers") + 1], "8"
        )
        self.assertIn("--lockstep-nvtx-include", command)
        self.assertNotIn("--force-overwrite", command)
        args.port = 29500
        with self.assertRaisesRegex(ValueError, "distinct"):
            profile.build_command(args, env)


@unittest.skipIf(torch is None, "Torch CPU is optional for tensor round-trip tests")
class TensorContracts(unittest.TestCase):
    def test_layout_dtype_and_padding_survive_serialization(self):
        tensor = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4).transpose(1, 2)
        sliced = torch.arange(30, dtype=torch.bfloat16).reshape(5, 6)[::2, ::2]
        original = {
            "weights": (tensor, sliced),
            "scales": torch.tensor([0.25, 0.5]),
            "args": {"activation_clamp": float("inf"), "fast_math": True},
        }
        copied = art.tensor_tree(original)
        self.assertEqual(copied["weights"][0].stride(), tensor.stride())
        self.assertEqual(copied["weights"][1].stride(), sliced.stride())
        torch.testing.assert_close(copied["weights"][1], sliced, rtol=0, atol=0)
        storage = copied["weights"][1].as_strided(
            (copied["weights"][1].untyped_storage().nbytes() // 2,), (1,)
        )
        self.assertEqual(
            storage[1].item(), 0
        )  # padding is initialized before torch.save
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.pt"
            art.save_tensors(path, copied)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(loaded["weights"][0].stride(), tensor.stride())
            torch.testing.assert_close(loaded["weights"][0], tensor, rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                art.save_tensors(path, copied)

    def test_capture_native_arguments_and_reference_round_trip(self):
        def native(
            y,
            l1_weights,
            l2_weights,
            sym_buffer,
            cumulative_local_expert_recv_stats=None,
            activation="swiglu",
            activation_clamp=None,
            fast_math=True,
            l1_alphas=None,
            l2_alphas=None,
            a2_scales=None,
            routed_scaling_factor=1.0,
        ):
            pass

        buffer = SimpleNamespace(
            num_experts=4,
            num_max_tokens_per_rank=2,
            num_topk=2,
            hidden=4,
            intermediate_hidden=2,
            num_shared_experts=0,
            mma_type="fp4xfp4",
            activation="swiglu",
            x=torch.zeros((2, 2), dtype=torch.int8),
            x_sf=torch.zeros((2, 1), dtype=torch.int32),
            topk_idx=torch.tensor([[0, 1], [3, -1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.2, 0.8], [1.0, 0.0]]),
        )
        metadata = {
            "run_id": "fixture",
            "forward_id": 8,
            "batch_fingerprint": "one",
            "layer_id": 3,
            "ep_rank": 0,
            "ep_size": 1,
            "input_m": 2,
            "native_m": 2,
        }
        y = torch.ones((2, 4), dtype=torch.bfloat16)
        weights = (
            torch.ones((4, 4, 2), dtype=torch.int8),
            torch.ones((4, 1, 4), dtype=torch.int32).transpose(1, 2),
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(
                art, "native_identity", return_value={"fingerprint": "fixture"}
            ),
        ):
            folder = Path(temp)
            record = art.capture_inputs(
                folder,
                package=None,
                native=native,
                y=y,
                l1_weights=weights,
                l2_weights=weights,
                buffer=buffer,
                kwargs={
                    "a2_scales": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                    "activation_clamp": None,
                },
                metadata=metadata,
            )
            manifest = art.capture_reference(folder, record, y)
            art.write_json(
                folder / "COMPLETE.json",
                {"schema": art.CAPTURE_SCHEMA, "ranks": [manifest]},
            )
            self.assertEqual(len(art.validate_capture(folder)), 1)
            inputs = torch.load(folder / record["inputs"]["file"], weights_only=True)
            self.assertIsNone(inputs["arguments"]["activation_clamp"])
            self.assertEqual(inputs["arguments"]["routed_scaling_factor"], 1.0)
            self.assertNotIn("cumulative_local_expert_recv_stats", inputs["arguments"])
            self.assertEqual(record["source_expert_counts"], [1, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
