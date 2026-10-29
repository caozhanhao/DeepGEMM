# Real-input MegaMoE replay

These Python tools consume the `megamoe.capture.v1` format from the SGLang
Prefill diagnostic branch. They add no kernel changes. GPU execution and
profiler overhead must be accepted on the actual deployment before using the
results to choose an optimization.

## Inspect, then replay

Inspecting an artifact needs only Python; it does not import Torch or execute
the serialized tensors:

```bash
python tools/replay_mega_moe.py /scratch/megamoe/capture-01/forward-00000042-layer-041 --inspect
```

The complete EP group, matching forward/layer/request-chunk identity, payload
sizes and SHA256s are mandatory. Legacy signal-hook captures lack this contract
and are intentionally not treated as coordinated captures.

After the model capture is complete, release this task's model GPUs using its
saved owner/process identity and verify exclusive GPU availability. Run a
single-node eight-GPU replay, for example:

```bash
torchrun --standalone --nproc-per-node 8 tools/replay_mega_moe.py \
  /scratch/megamoe/capture-01/forward-00000042-layer-041 \
  --deep-gemm-path /opt/DeepGEMM/deep_gemm \
  --output /scratch/megamoe/replay-01 --warmup 10 --iterations 30
```

For two four-GPU nodes use matching `--nnodes 2 --nproc-per-node 4`, unique
`--node-rank`, and the same reachable `--master-addr/--master-port` on both nodes.
Every node must see the complete merged capture at its configured path. Keep
native builds and JIT caches node-local; do not share an actively written JIT
cache across workers/nodes. Set cache paths outside the captured model data.

Replay verifies each rank's own payloads, restores weight/scale strides and
native arguments without requantization, and reconstructs the same symmetric
buffer capacity. Source/binary changes fail by default; an intentional A/B must
use `--allow-native-change`, which is recorded. Runtime SM/TC/PDL settings are
restored from the capture. Per-rank native receive counts must exactly equal
the sum of the captured physical routes. Outputs are compared against the
captured routed output, **exactly by default**. Any relaxed `--rtol/--atol` is an
explicit numerical contract change and appears in the report. Do not relax a
tolerance simply to obtain a pass.

The normal run first checks correctness/counters, warms up, then records CUDA
events around native calls only. Input refill and the default EP barrier are
outside these events. No artificial GPU sleep is inserted. `--entry-mode free`
removes the explicit per-call barrier for a separate comparison; it does not
reproduce the model's original arrival schedule. This is repeated isolated
replay, with no L2 flush and no clock lock, not model TPS or a claim of production
cache equivalence. All individual durations and both capture/current native
identities are saved. Rank durations are not added together.

## Short trace and NCU

Use `--profile torch --iterations 3` in a separate replay invocation to save
Chrome/Perfetto trace JSONs, with shape/stack collection off. The output marks
all profiled timings as diagnostics, not baseline latency.

The NCU wrapper is launched **per rank**, with every EP rank under NCU. It prints
the command by default; `--execute` is required to start profiling:

```bash
torchrun --standalone --nproc-per-node 8 tools/profile_mega_moe_rank.py \
  /scratch/megamoe/capture-01/forward-00000042-layer-041 \
  --deep-gemm-path /opt/DeepGEMM/deep_gemm \
  --output /scratch/megamoe/ncu-duration-01 --host 127.0.0.1 --port 29513
```

For multi-node runs, set `--host` to a reachable address on the selected NCU
communicator server. Its port must differ from the torchrun rendezvous port.
Check that this NCU version supports the generated options and start with the
duration-only default. Add `--section SourceCounters` or a small supported
hardware section set only after the duration pass completes correctly. Check
available sections with the target installation's `ncu --list-sections`.

The wrapper uses application replay, TCP lockstep, and a `megamoe-replay/` NVTX
filter containing just the selected native call. Correctness/counter checks
and warmup are outside that filter. No auxiliary counters are passed to the
profiled native call, since replay may repeat side effects. Each process attempt
writes separate correctness/timing records; existing NCU reports are not
overwritten. The wrapper enables line information; use matching lineinfo/JIT
conditions for its unprofiled comparison and retain the actual reports.
These options follow NVIDIA's [mandatory concurrent kernel and NVTX guidance](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

Compare NCU durations with the independent unprofiled replay. A large ratio or
rank skew invalidates interpreting profiler time/utilization as normal execution.
Source/PC counters locate hot instructions and waits; they do not supply a full
G1/G2/dispatch/combine software timeline. Sparse internal timestamp probes are
a follow-up selected from the GPU evidence; this version does not add them.

## CPU checks

```bash
python tests/test_mega_moe_diagnostics_cpu.py -v
```

The manifest, consensus, route-count, statistics and NCU-command tests use the
standard library. With CPU Torch installed, two additional tests check tensor
stride/scale/padding serialization and complete capture/reference round trips.
These do not compile or execute a CUDA kernel.
