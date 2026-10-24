# SM100 BF16 Indexer prefill

`fp8_mqa_logits_bf16(q, (kv, kv_scale), weights, ks, ke,
clean_logits=False, max_seqlen_k=width, q_mask=None)` returns BF16 logits.
Pass contiguous BF16 head weights to select the existing packed BF16 reduction.
Q/K remain ordinary E4M3 FP8, KV scales remain FP32, and dot products accumulate
in FP32. This API does not require an MXFP8 KV-cache format.

The optimized specialization applies to SM100-family, non-paged, compressed
prefill with 32 heads, head dimension 128, and `0 < width <= 262144`:

- Q block 4, KV tile 384, three math warpgroups, Q/KV/TMEM stages 3/3/3.
- Three 128-token TMA boxes per KV tile; one TMEM stage per math warpgroup.
- Register allocation accounts for all four producer/consumer warpgroups.
- Load and reduce 16 heads at a time, preserving head accumulation order.
- Hoist output row addresses and unroll the KV loop twice.

For compressed output, column `j` represents KV token `ks[row] + j`; valid
columns satisfy `j < ke[row] - ks[row]`. The caller must provide enough width.
Invalid columns and Q-masked rows remain uninitialized. A matching reader must
respect valid lengths and masks; it must not read them as ordinary scores.
TMEM release still occurs for an inactive final Q row.

Existing FP32, FP4, paged and long-context candidate APIs retain their dispatch.
BF16 logits should be consumed directly by a BF16-capable Top-K implementation;
materializing a full FP32 copy can erase the scoring benefit. BF16 scoring is
numerically approximate relative to FP32 head reduction.

Run `python tests/test_sm100_bf16_indexer.py` on SM100-family CUDA to compare
valid compressed results with the existing dense BF16 path, including tails,
non-unit scales, non-aligned starts, empty intervals and Q masks. This is a
numerical regression check, not a model-quality evaluation.

The native integration was compiled and tested on B300. Valid compressed logits
matched the existing dense BF16 path element for element in the regression test.
The paired serving integration was also benchmarked with direct BF16 Top-K;
throughput depends on workload and downstream selection. Model-quality
acceptance remains separate from these numerical and performance checks.

The native host allocates the required BF16 shared storage and omits the research
adapter's unused diagnostic reservation. The measured H32/D128 specialization
uses 202752 shared-memory bytes and 512 threads per CTA. Launch metadata reports
128 registers per thread; dynamic register allocation assigns 152 to math warps
and 56 to producer warps. Earlier adapter measurements should not be treated as
measurements of this native build.
