# Phase 2's recommended configuration, measured — superseded

This is what Phase 2 recommended on 2026-08-22. It is **not** what production runs. Phase 3 and
Phase 4 overturned the precision half of it: the KV cache is now `fp8_e4m3`, which selects the
FlashInfer backend, and `--kv-cache-memory-bytes` was removed so the pool is engine-sized at 0.92.
What the current configuration is, and the evidence for it, is `PHASE4_RESULTS.md`; the deployment
profile selected by `config/deployments/active` is the authority for what is actually running.

What survives here is the pool-sizing argument — why `27000000000` over-commits this device — which
is why no later phase raised the pool again.

```
--kv-cache-dtype auto              (BF16 KV)
--attention-backend                (unset -> FLASH_ATTN, FA2)
--kv-cache-memory-bytes 26000000000
--kv-offloading-size 32
  + the unchanged committed baseline (TP=2, 262144 context, prefix caching,
    max-num-batched-tokens 8192, tool/reasoning parsers)
```

The only change from what Phase 1 committed is `--kv-cache-memory-bytes`, **27000000000 ->
26000000000**.

## Why the byte count changes

Every boot in this study printed the engine's own accounting:

```
Free memory on device (43.83/44.39 GiB) on startup. Desired GPU memory utilization is (0.92, 40.84 GiB).
Actual usage is 14.93 GiB (weights + non-torch), 3.36 GiB peak activation, 1.13 GiB CUDAGraph.
Replace gpu_memory_utilization config with `--kv-cache-memory=22842080523` (21.27 GiB) to fit into
requested memory, or `--kv-cache-memory=26055386624` (24.27 GiB) to fully utilize gpu memory.
```

`26,055,386,624` is the point at which **100% of free device memory** is consumed. The shipped
`27,000,000,000` is 944 MB *past* that line — it over-commits the device by about 0.9 GiB against
the engine's own peak-activation estimate. It survived Phase 1 because BF16 never demanded the
peak and the transient at the same moment; it did **not** survive Phase 2, where a 956 MiB
`prompt_logprobs` logits allocation on the FP8 path found the missing GiB and killed the engine
with `EngineDeadError` (`~/vllm-study-results/PHASE2-INCIDENT-fp8pth-oom.md`). The mechanism is not FP8-specific.

`26,000,000,000` sits just under the engine's own anchor and restores roughly 1 GiB of reserve.

## What it costs, measured

| | shipped 27.0e9 | final 26.0e9 | default 0.92 |
|---|---|---|---|
| KV pool | 25.15 GiB | **24.21 GiB** | 22.55 GiB |
| GPU KV cache size | 807,188 tok | **777,975 tok** | 724,163 tok |
| max concurrency @262k | 3.08x | **2.97x** | 2.76x |

Load, final vs shipped (`load-vs-shipped.txt`), both BF16/FLASH_ATTN:

| metric | shipped | final | delta |
|---|---|---|---|
| aggregate output tok/s | 136.14 | **137.22** | +0.80% |
| aggregate total tok/s | 1639.3 | 1662.7 | +1.43% |
| long E2E p50 | 133.96 s | 131.48 s | -1.85% |
| long E2E p95 | 141.97 s | 141.70 s | -0.19% |
| long TPOT p50 | 39.4 ms | 40.2 ms | +2.17% (cross-boot noise ~2%) |
| long TTFT p95 | 29.38 s | 29.11 s | -0.91% |
| short TTFT p95 | 0.520 s | 0.525 s | +0.97% |
| short E2E p95 | 22.06 s | 22.14 s | +0.35% |
| max Running / Waiting | 8 / 3 | 8 / 3 | - |
| preemptions | 0 | 0 | - |

**The 3.6% capacity reduction buys back ~1 GiB of device reserve and costs nothing measurable on
the real workload.**

## Accuracy: bit-identical to the Phase-1 BF16 reference

| tier | vs cell 0 (`base-D`) |
|---|---|
| tier-1 short (n=40,429) | mean **0.000000**, argmax flip **0/40,429 = 0.0000 pp**, exact-zero **100.00%** |
| tier-3a reasoning trace (n=20,296) | mean **0.000000**, argmax flip **0/20,296 = 0.0000 pp**, exact-zero **100.00%** |
| tier-1 deep (n=384) | 13/384 = 3.39 pp, mean 0.022003, exact-zero 24.74% (this tier is inherently non-deterministic; the same-config floor measured on control-T is 0.26 pp / 0.008) |
| API conformance | **10/10 pass** |
| uncalibrated-scale warning | none |

This reproduces Phase 1's bisect finding from the other direction: `--kv-cache-memory-bytes` does
not enter the torch.compile hash, so changing it is numerically free.

## Envelope

Idle 0.047 cores, 42.7 GiB charged (anon 9.91 + shmem 32.25). Under the target load 3.2 cores,
`nr_throttled` 0, `memory.events` all zero. Under the L3 overload 3.04 cores, 42.4 GiB, still
zero memory events. **Inside the 8-core / 64 GB steady-state constraint with room to spare; the
16-core / 128 GB peak was never approached.**
