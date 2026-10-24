"""Native BF16 compressed scoring versus the existing dense BF16 path.

Run: python tests/test_sm100_bf16_indexer.py
Requires a built checkout and an SM100-family CUDA device.
"""

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(
    torch is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10,
    "requires PyTorch CUDA and SM100-family hardware",
)
class TestSM100BF16Indexer(unittest.TestCase):
    def test_compressed_and_masked_rows(self):
        import deep_gemm

        torch.manual_seed(462)
        kv_tokens = 8192
        kv = torch.randn(kv_tokens, 128, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.rand(kv_tokens, device="cuda") + 0.25
        for rows in (1, 7, 65, 1025):
            q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
            weights = torch.randn(rows, 32, device="cuda").bfloat16()
            starts = (torch.arange(rows, device="cuda", dtype=torch.int32) * 13) % 1024
            ends = (
                starts + 1 + torch.arange(rows, device="cuda", dtype=torch.int32) * 97
            ).clamp(max=kv_tokens)
            ends[::5] = starts[::5]  # Empty valid intervals.
            dense = deep_gemm.fp8_mqa_logits_bf16(
                q, (kv, scale), weights, starts, ends, clean_logits=True
            )
            width = int((ends - starts).max().item()) or 1
            for mask_kind in ("none", "mixed", "empty"):
                with self.subTest(rows=rows, mask=mask_kind):
                    mask = None
                    if mask_kind != "none":
                        mask = torch.arange(rows, device="cuda") % 3 != 0
                        if mask_kind == "empty":
                            mask.zero_()
                    compressed = deep_gemm.fp8_mqa_logits_bf16(
                        q,
                        (kv, scale),
                        weights,
                        starts,
                        ends,
                        max_seqlen_k=width,
                        q_mask=mask,
                    )
                    self.assertEqual(compressed.dtype, torch.bfloat16)
                    self.assertEqual(
                        compressed.stride(0) * compressed.element_size() % 1024, 0
                    )
                    columns = torch.arange(width, device="cuda")[None, :]
                    valid = columns < (ends - starts)[:, None]
                    if mask is not None:
                        valid &= mask[:, None]
                    expected = dense.gather(
                        1, (starts[:, None] + columns).clamp(max=kv_tokens - 1).long()
                    )
                    self.assertTrue(torch.equal(compressed[valid], expected[valid]))
            torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
