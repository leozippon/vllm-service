# vLLM trade study, phase 1 (cells 0/1/2) — measured results

Engine-side only. All runs on 2x L20 (GPUs 6,7), vLLM 0.27.1, TP=2, BF16 KV, direct to the
engine on loopback (the gateway is bypassed so the numbers are engine numbers).

Load generator: `agent_mix` — 4 concurrent long agent sessions x 4 rounds, round-1 prompt
30,000 tokens growing to ~60,000 by round 4 with verbatim re-send of the prior conversation
(so rounds 2-4 exercise prefix reuse), `reasoning_effort: xhigh`, min=max=3000 output tokens,
temperature 0; plus one 3-turn short chat launched every 15 s from t+20 s (256 output tokens).
16 long requests and 105-108 short requests per run.

## Configurations measured

| id | VLLM_ARGS delta vs the pre-study profile |
|---|---|
| cell 0 `base-D` | none (pre-study production configuration) |
| bisect A | `+ --kv-cache-memory-bytes 27000000000` |
| cell 1 `mem` | bisect A `+ --max-num-seqs 32 + --compilation-config '{"cudagraph_capture_sizes":[...17 sizes...]}'` |
| cell 2 `triton-bf16` | cell 1 `+ --attention-backend TRITON_ATTN` |

## Per-boot engine facts

| | cell 0 | bisect A | cell 1 | cell 2 |
|---|---|---|---|---|
| attention backend | FLASH_ATTN (FA2) | FLASH_ATTN | FLASH_ATTN | TRITON_ATTN |
| block size | 784 | 784 | 784 | 784 |
| KV pool | 22.55 GiB (profiled) | 27.0e9 B explicit | 27.0e9 B explicit | 27.0e9 B explicit |
| GPU KV cache size | 724,163 tok | 807,188 tok | 807,188 tok | 807,188 tok |
| max concurrency @262k | 2.76x | 3.08x | 3.08x | 3.08x |
| CUDA graph pool actual/est | 1.13 / 1.99 GiB | 1.18 GiB | 0.44 / 0.42 GiB | 0.44 GiB |
| capture count PIECEWISE/FULL | 51 / 35 | 51 / 35 | 17 / 12 | 17 / 12 |
| init engine (compilation) | 54.9 s (5.3 s) | 58.2 s (5.2 s) | 52.4 s (5.4 s) | 350.8 s (178.4 s, cache miss) |
| Started -> startup complete | 179 s | ~220 s | 180 s | 488 s (cache miss) |
| uncalibrated-scale warning | none | none | none | none |
| inference-time JIT warning | batch_memcpy_kernel | batch_memcpy_kernel | batch_memcpy_kernel | _w8a8_triton_block_scaled_mm |

Re-anchor: with the smaller capture list the engine's own advice moved from
`--kv-cache-memory=26051126784` (26.05e9, cell-0 boot) to `25269705995` (fit) /
**`28480849408`** (fully utilise) — 2.4e9 B/rank more than the design's pre-change estimate of
26048964096. Shipped value 27000000000 keeps ~0.7-0.9 GiB/rank of device margin, which is the
design's stated intent under the corrected numbers.

## Noise floors (measured, not assumed)

- Load, same config / same boot / different prefix salt: aggregate tok/s +/-0.08%,
  long E2E p50 +/-0.78%, long TPOT p50 +/-0.59%, long TTFT p50 +/-1.07%, long TTFT p95 +/-2.48%,
  short TTFT p50 +/-4.80%, short TTFT p95 +/-8.59%, mean cores +/-0.19%.
  Long TTFT p50 is additionally unstable (n=16, +25% between two cell-2 repeats) — do not
  read it as a precise quantity.
- Tier-1 teacher-forced logprobs (`prompt_logprobs`), same config: **exactly 0.000000** over
  40,429 + 20,296 positions, both same-boot and across two different boots. Zero noise floor.
- Tier-1 deep (single-position top-20 at 32k/128k/200k/235k), same config: 1.56 pp argmax flip,
  mean |d logprob| 0.021 — this tier is NOT deterministic (it relies on prefix-cache reuse), and
  the 200k/235k depths are systematically non-deterministic (0% exact-zero).

## Load results

| metric | cell 0 | bisect A | cell 1 (r1/r2) | cell 2 (r1/r2) |
|---|---|---|---|---|
| aggregate output tok/s | 136.18 | 136.06 | 133.25 / 133.14 | 133.78 / 133.90 |
| aggregate total tok/s | 1636.3 | 1637.9 | 1591.9 / 1593.2 | 1598.6 / 1600.8 |
| long E2E p50 / p95 (s) | 134.07 / 141.86 | 134.08 / 141.52 | 137.99 / 147.49 | 137.76 / 147.07 |
| long TTFT p50 / p95 (s) | 11.75 / 29.56 | 10.33 / 29.39 | 10.43 / 29.16 | 13.23 / 33.65 |
| long TPOT p50 / p95 (ms) | 39.3 / 42.7 | 40.2 / 42.8 | 41.2 / 45.4 | 40.2 / 43.3 |
| short TTFT p50 / p95 (s) | 0.437 / 0.519 | 0.434 / 0.521 | 0.430 / 0.636 | 0.423 / 0.634 |
| short E2E p50 / p95 (s) | 9.90 / 22.04 | 9.94 / 22.09 | 10.47 / 22.03 | 9.83 / 26.28 |
| short TPOT p50 (ms) | 37.3 | 37.4 | 39.7 | 37.1 |
| max Running / Waiting | 8 / 3 | 8 / 3 | 8 / 3 | 8 / 3 |
| max GPU KV usage | 40.7% | 36.6% | 36.6% | 36.1% |
| preemptions | 0 | 0 | 0 | 0 |
| prefix cache hit rate | 61.0% | 60.5% | 60.5% | 60.5% |
| external (offload) hit rate | 0.0% | 0.0% | 0.0% | 0.0% |
| offload CPU->GPU bytes | 0 | 0 | 0 | 0 |
| 429/5xx | none | none | none | none |

The KV offload tier read **zero bytes** in every run while writing ~36 GB GPU->CPU and holding
32.25 GiB of shmem — the load never evicted, so the second tier was never consulted.

## Resource envelope (cgroup counters, not ps sampling)

| run | mean cores | cpu core-s | throttled | memory.current | sampled max | anon | shmem | file |
|---|---|---|---|---|---|---|---|---|
| cell 0 | 3.23 | 1775 | 0 | 55.08 GiB | 55.09 | 10.04 | 32.26 | 44.79 |
| bisect A | 3.24 | 1783 | 0 | 50.67 | 50.68 | 10.03 | 32.26 | 40.40 |
| cell 1 | 3.23 / 3.24 | 1834 | 0 | 50.30 | 50.55 | 8.51 | 32.25 | 41.55 |
| cell 2 | 3.21 | 1814 | 0 | 41.59 | 46.27 | 8.84 | 32.25 | 32.50 |

Idle: 0.025-0.046 cores. `memory.events` all zero in every run (no MemoryHigh throttle, no OOM).
`nr_throttled` 0 against the 1600% CPUQuota throughout.
`shmem` is a subset of `file`; the genuinely charged working set is anon + shmem = 40.8-42.3 GiB,
the rest of `memory.current` is reclaimable page cache.

**Verdict: 3.2 cores and ~42 GiB charged (<=55 GiB including page cache) under the target load —
inside the 8-core / 64 GB steady-state constraint, with the 16-core / 128 GB peak never approached.**

## Accuracy

Reference: cell 0 (BF16 KV, FA2). Corpus sha 7039b2e1e3420d19 (real Python source, cached at
/Data2/lzp/vllm-eval/corpus/).

| candidate | tier-1 short mean/argmax-flip | tier-1 deep argmax-flip | trace tier mean/flip | needle |
|---|---|---|---|---|
| bisect A | **0.000000 / 0.0000 pp** | not run | not run | not run |
| final shipped | **0.000000 / 0.0000 pp** | not run | not run | not run |
| cell 1 | 0.567 / 2.78 pp | 11/384 = 2.86 pp | 0.031 / 1.89 pp | not run (design: tier 1 only) |
| cell 2 vs cell 0 | 0.570 / 2.69 pp | 15/384 = 3.91 pp | 0.032 / 2.00 pp | 20/20, 0 regressions |
| cell 2 vs cell 1 | 0.569 / 2.79 pp | 11/384 = 2.86 pp | - | - |

Reference needle grid: **100/100** (4 lengths x 5 depths x 5 codes).

Where cell 1's drift lives (vs cell 0, tier-1 short, banded by the reference logprob):

| ref logprob band | share | mean abs delta | argmax flip |
|---|---|---|---|
| > -0.1 | 12.4% | 0.032 | 0.42% |
| -0.1..-1 | 13.6% | 0.135 | 8.68% |
| -1..-3 | 13.6% | 0.273 | 3.91% |
| -3..-8 | 23.1% | 0.423 | 1.93% |
| < -8 | 37.3% | 1.100 | 1.53% |

Mean |delta p(chosen token)| = 0.0149.

## Tooling limit discovered (engine robustness defect)

`sampling_params.py:513` sets `skip_reading_prefix_cache` whenever `prompt_logprobs` is
requested, so the whole prompt is re-prefilled in `max_num_batched_tokens` chunks and
`gpu_model_runner._get_prompt_logprobs_dict` materialises `chunk_tokens x vocab` logits.
An ordinary authenticated request with an ~8k-token prompt and `prompt_logprobs=1` allocates
~3.6 GiB and **kills the engine with CUDA OOM (EngineDeadError)**; systemd restarted it.
Reproduced once, unintentionally, at the start of this study.
`eval_suite.py` now refuses prompts above `TF_MAX_PROMPT_TOKENS` client-side, and deep-context
teacher forcing uses single-position top-k scoring instead.

## Bisect: which of the three "adopt now" changes does what

`--kv-cache-memory-bytes` does not enter the torch.compile hash (5.2 s compile = AOT cache hit);
`--max-num-seqs` and `cudagraph_capture_sizes` do (186 s recompile on first boot).

| change | KV tokens | concurrency | tier-1 drift vs cell 0 | load vs cell 0 |
|---|---|---|---|---|
| `--kv-cache-memory-bytes` alone | 807,188 | 3.08x | **0.000000 / 0.0000 pp** | indistinguishable |
| `+ --max-num-seqs 32 + capture sizes` | 807,188 (no change) | 3.08x (no change) | 0.567 / 2.78 pp | -2.1% aggregate, +2.5% TPOT p50, +18% short TTFT p95 |

So the explicit byte count carries 100% of the capacity gain at zero numerical cost, and the
other two changes carry none of it while costing throughput and perturbing the numerics.

bisect A vs cell 1, measured: aggregate +2.11%, long E2E p95 -4.05%, long TPOT p50 -2.47%,
long TPOT p95 -5.75%, short TTFT p95 -18.15%, short TPOT p50 -5.80% — bisect A better on every
load metric.

## The gate: TRITON_ATTN vs FLASH_ATTN (cell 2 vs cell 1, identical KV pool and precision)

Both cells resolve block size 784 and GPU KV cache size 807,188 tokens, so this is a clean
kernel-only A/B. Two independent runs each.

| metric | r1 delta | r2 delta | noise floor | real? |
|---|---|---|---|---|
| aggregate output tok/s | **+0.40%** | **+0.57%** | +/-0.08% | yes, small gain |
| long TPOT p50 | **-2.45%** | **-4.52%** | +/-0.59% | yes, gain |
| long TPOT p95 | **-4.48%** | **-4.07%** | +/-0.35% | yes, gain |
| short TPOT p50 | -6.48% | -7.34% | +/-0.24% | yes, gain |
| short E2E p50 | -6.14% | -6.65% | +/-0.19% | yes, gain |
| long E2E p50 | -0.16% | -1.12% | +/-0.78% | no, within noise |
| long TTFT p95 | +15.40% | +13.46% | +/-2.48% | yes, ~14% regression |
| long TTFT p50 | +26.9% | +57.5% | unstable (n=16) | yes, direction only |
| short E2E p95 | +19.27% | +15.72% | +/-3.12% | yes, ~17% regression |
| short TTFT p95 | -0.27% | +5.38% | +/-8.59% | no, within noise |

Design gate rule: ">10% regression on long TPOT p50 or aggregate tok/s kills cells 4-6".
Both gate metrics **improved**. **PASS.**

Accuracy: 20/20 needle (0 regressions vs the 100/100 reference), and cell 2's tier-1 distance
from cell 1 (0.569 / 2.79 pp) is the same magnitude as cell 1's distance from cell 0
(0.567 / 2.78 pp) — i.e. swapping the attention kernel introduces no more numerical difference
than changing the CUDA-graph capture list does. No uncalibrated-scale warning.

Costs to carry into phase 2: long-prefill TTFT ~14% worse at p95; 8-minute cold start on a
compile-cache miss; a new inference-time Triton JIT (`_w8a8_triton_block_scaled_mm`).

## Shipped configuration

`+ --kv-cache-memory-bytes 27000000000` and nothing else. Verified on its own boot:
FLASH_ATTN, block 784, 807,188 KV tokens, 3.08x, NRestarts=0, active/running,
tier-1 accuracy **exactly 0.000000 drift / 0 argmax flips vs the BF16 reference**,
all 10 API conformance checks pass, load indistinguishable from the pre-study baseline
(aggregate -0.03%, long E2E p50 -0.08%, long TPOT p50 +0.19%, short TTFT p95 +0.28%).

A cross-boot repeat of this same configuration gives aggregate +0.06% and long TPOT p50 -1.90%,
which sets the cross-boot load noise floor (~2% on TPOT p50, ~0.1% on aggregate).

## Notes for phase 2 (cells 3-7)

1. The tier-1 reference at `cell0-base-D/accuracy.json` + `acctrace.json` is valid **only for
   configurations that share cell 0's torch.compile hash**. `--max-num-seqs`, `cudagraph_capture_sizes`,
   `--attention-backend` and `--kv-cache-dtype` all change that hash, and a recompile alone moves
   tier-1 by ~0.57 mean |d logprob| / ~2.8 pp argmax flip on the code corpus. Any FP8 candidate must
   be scored against a BF16 reference captured under the *same* non-precision settings, or that
   offset will be misread as precision loss. Capture a matched BF16 control per compile hash.
2. TRITON_ATTN cleared the gate, so `fp8_per_token_head` / `int8_per_token_head` (cells 4-5) remain
   live. Pair them WITHOUT `--max-num-seqs 32` / explicit capture sizes, which phase 1 measured as a
   net loss, and re-anchor `--kv-cache-memory-bytes` from that boot's own `gpu_worker.py:789` line
   (the block-size floor changes to 1552 for per-token-head dtypes, so the pool arithmetic changes).
3. `--kv-cache-memory-bytes` suppresses the memory-profiling log lines, so the re-anchor needs a
   probe boot without it (as done here).
4. The offload tier still reads zero bytes at this load; cell L3 (6 sessions x 120k) is still the
   only thing that would justify its 32.25 GiB of shmem.
5. Long TTFT p50 has n=16 and is unstable to +/-25%; use long TTFT p95 for prefill comparisons.
