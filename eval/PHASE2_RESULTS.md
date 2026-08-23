# vLLM trade study, phase 2 — KV dtype, attention backend, MTP, and the offload tier

Engine-side only, 2x L20 (GPUs 6,7), vLLM 0.27.1, TP=2, direct to the engine on loopback.
Load generator, corpus and accuracy tiers are Phase 1's, unchanged (`agent_mix`: 4 concurrent long
`xhigh` agent sessions x 4 rounds with verbatim prefix reuse, plus a short-chat stream; corpus sha
`7039b2e1e3420d19`). Per-cell detail is in the companion files; this is the decision document.

## The measurement rule Phase 1 handed over, and what it turned out to cost

Phase 1 established that a recompile alone moves tier-1 accuracy by ~0.57 mean |Δ logprob| and
~2.8 pp argmax-flip, so every quantised candidate had to be scored against a BF16 control at the
same compile hash. Phase 2 did that — and then measured the thing Phase 1 could not:

| comparison | tier-1 short mean | tier-1 short flip |
|---|---|---|
| **same config, same boot** (control-T repeat) | **0.000000** | **0.0000 pp** |
| recompile only (control-T vs cell 0) | 0.580577 | 2.7604 pp |
| recompile only (control-F vs cell 0) | 0.581863 | 2.8198 pp |
| recompile only (Phase 1, cell 1 vs cell 0) | 0.567 | 2.78 pp |

Four independent recompiles land at **~0.57-0.58 mean / ~2.8 pp**. The compile-cache directory
hash differs between every backend/dtype pair (`80ef9844c7` for TRITON+BF16 vs `848e7eccc9` for
TRITON+fp8pth), confirming `--kv-cache-dtype` enters the torch.compile hash, so **no quantised
candidate can get below that floor** no matter how good its quantiser is.

**Consequence, stated plainly: the decision rule's tier-1 thresholds (mean ≤ 0.15, flip ≤ 0.5 pp)
are below the recompile floor and cannot be applied as literal pass/fail tests.** What they can do
is discriminate: a candidate at ~0.58/2.8 pp is showing *no precision loss beyond the recompile*;
one materially above is. Every quantised candidate in this study came in at the floor. The deep,
trace and needle tiers remain interpretable on their own terms, and the same-config floor
(0.000000 short / 0.26 pp deep) is the yardstick for them.

## All cells

| cell | KV dtype | backend | block | KV tokens | conc @262k | aggregate tok/s | long TPOT p50 | short TTFT p95 | scale warning | conformance |
|---|---|---|---|---|---|---|---|---|---|---|
| cell 0 (Phase 1 baseline) | auto | FLASH_ATTN | 784 | 724,163 | 2.76x | 136.18 | 39.3 ms | 0.519 s | none | 10/10 |
| shipped (Phase 1) | auto | FLASH_ATTN | 784 | 807,188 | 3.08x | 136.14 | 39.4 ms | 0.520 s | none | 10/10 |
| **final (recommended)** | auto | FLASH_ATTN | 784 | **777,975** | **2.97x** | **137.22** | 40.2 ms | 0.525 s | none | **10/10** |
| Control-T | auto | TRITON_ATTN | 784 | 807,188 | 3.08x | 133.86 | 39.7 ms | 0.588 s | none | 10/10 |
| cell 4 `fp8pth` | fp8_per_token_head | TRITON_ATTN | 1552 | 1,403,593 | **5.35x** | 129.16 | 41.2 ms | **6.971 s** | none | 10/10 |
| cell 5 `int8pth` | int8_per_token_head | TRITON_ATTN | 1552 | 1,403,593 | **5.35x** | 127.46 | 41.8 ms | **6.064 s** | none | 10/10 |
| cell 6 `mtp2` | auto + MTP k=2 | FLASH_ATTN | 800 | 659,215 | 2.51x | 135.40 | 42.1 ms | 1.023 s | none | **8/10** |
| Control-F | auto | FLASHINFER | 784 | 693,413 | 2.65x | **138.45** | **38.6 ms** | **0.422 s** | none | 10/10 |
| cell 3 `fi-fp8` | fp8_e4m3 | FLASHINFER | 1568 | 1,358,930 | **5.18x** | **146.27** | **34.6 ms** | 0.934 s | **PRESENT** | 10/10 |

Noise floors (Phase 1, measured): aggregate ±0.08% same-boot / ~0.1% cross-boot; long TPOT p50
±0.59% same-boot / ~2% cross-boot; short TTFT p95 ±8.59%; long TTFT p50 unstable at n=16.

## Accuracy, every candidate against its matched control

| cell | matched control | tier-1 short | tier-1 deep (n=384) | trace (n=20,296) | needle |
|---|---|---|---|---|---|
| recompile floor | - | 0.58 / 2.8 pp | 3.1-3.4 pp / 0.026-0.031 | 0.032 / 2.00 pp | - |
| same-config floor | - | **0.000000 / 0.0000 pp** | **0.26 pp / 0.008** | - | - |
| final (recommended) | cell 0 | **0.000000 / 0.0000 pp** | 3.39 pp / 0.022 | **0.000000 / 0.0000 pp** | - |
| cell 4 `fp8pth` | control-T | 0.623 / 2.97 pp | 3.39 pp / 0.0367 (235k: **8.33 pp**) | 0.0338 / 1.97 pp | **20/20** |
| cell 5 `int8pth` | control-T | 0.601 / 3.01 pp | **2.08 pp** / 0.0299 (235k: 2.08 pp) | 0.0333 / 1.97 pp | not run |
| cell 6 `mtp2` | cell 0 | 0.575 / 2.71 pp | 3.39 pp / 0.0289 | **1.095 / 10.98 pp** (defect) | not run |
| Control-F | cell 0 | 0.582 / 2.82 pp | 3.12 pp / 0.0259 | 0.0323 / 1.90 pp | not run |
| cell 3 `fi-fp8` | control-F | 0.632 / 3.05 pp | **1.56 pp** / 0.0282 | 0.0346 / 2.08 pp | **20/20** |

Every quantised KV candidate sits at the recompile floor. The design's predicted accuracy ordering
(§4.5) is confirmed at the deep tier: `int8_per_token_head` (2.08 pp) beats `fp8_per_token_head`
(3.39 pp, and 8.33 pp at 235k specifically).

## Verdicts

**`fp8_per_token_head` (cell 4) — REJECT on performance, not precision.** Delivers exactly the
predicted 5.35x capacity with no calibration and no warning; the engine even logs *"Dynamic
per-token-head scales will be computed at runtime."* But it forces TRITON_ATTN, and TRITON_ATTN is
itself −1.67% aggregate and +14% prefill p95 versus FLASH_ATTN (see below). Net: −3.5% aggregate,
+4.0% long TPOT, and short TTFT p95 blows out from 0.59 s to **6.97 s**.

**`int8_per_token_head` (cell 5) — REJECT on performance.** The most accurate quantised option
measured, same capacity, but the slowest: −4.8% aggregate, +5.5% long TPOT, 6.06 s short TTFT p95.

**MTP (cell 6) — REJECT; the product decision never has to be made.** Measured −0.57% aggregate
and **+7.2% long TPOT p50** at concurrency 4-8, i.e. the opposite of the +33.8% recorded
historically at concurrency 3. Acceptance is healthy (72.8% overall, 1.456 accepted per draft,
81.5%/64.1% at depths 1/2) — speculation simply does not pay at batch 8. It also costs 9% capacity,
returns HTTP 400 for `min_p` and `logit_bias`, and exposed an unexplained defect (below). The
design said to surface the `min_p`/`logit_bias` break as a user decision; since there is no gain to
trade for it, there is nothing to ask.

**`fp8_e4m3` on FlashInfer (cell 3) — REJECT as written, flagged for deliberate reconsideration.**
The fastest and highest-capacity configuration measured (**+7.4% aggregate, −11.9% long TPOT,
5.18x vs 2.76x**), accuracy at the recompile floor, needle 20/20, conformance 10/10 — but it
carries the uncalibrated `k_scale = v_scale = 1.0` warning, which is an explicit hard line in the
decision rule, and short TTFT p95 rises 121%. The right next move is to calibrate and serialise
k/v scales for this checkpoint and re-run, not to waive the criterion.

**FLASHINFER with BF16 KV (Control-F) — the surprise, and the clearest next step.** Faster than
FLASH_ATTN on every metric (+1.67% aggregate, −1.9% long TPOT p50, −18.6% short TTFT p95) at the
recompile accuracy floor and 10/10 conformance, for a 4% capacity cost. One load run only.

## Two Phase-1 conclusions that measurement overturned

1. **The TRITON_ATTN gate does not reproduce.** Phase 1 passed cells 4-6 on the finding that
   TRITON_ATTN beat FLASH_ATTN on decode. That comparison was against cell 1, which carried the
   separately-rejected `--max-num-seqs 32`. Measured against the configuration actually shipped,
   TRITON_ATTN gives **no decode gain** (long TPOT p50 +0.67%, inside noise) and costs **−1.67%
   aggregate, +13.9% long TTFT p95, +18.9% short E2E p95**. The per-token-head branch was
   therefore paying a kernel tax the gate had hidden.
2. **`--kv-cache-memory-bytes 27000000000` over-commits the device.** It is 944 MB past the
   engine's own "fully utilize gpu memory" figure of 26,055,386,624 on this hardware. It survived
   Phase 1 at BF16 and killed the engine in Phase 2 (`EngineDeadError`, CUDA OOM on a 956 MiB
   allocation). The mechanism — a ~1 GiB transient landing when peak activation is live — is not
   FP8-specific.

## Two defects found

- **`/dev/shm` offload-region leak** (`PHASE2-INCIDENT-shm-leak.md`). vLLM 0.27.1 never unlinks
  `/dev/shm/vllm_offload_<engine_id>.mmap` (34.32 GB at `--kv-offloading-size 32`). Eight orphans
  had accumulated to 274.6 GB, filled a 252 GB `/dev/shm`, drove node-1 free memory to 2 GB and
  host-OOM-killed a boot. Reclaimed; the Phase-2 harness now sweeps before each start, but the
  repository's own `scripts/run-vllm` does not, so ordinary restarts still leak.
- **MTP + `prompt_logprobs` returns wrong logprobs for a subset of short prompts.** 6 of 30
  teacher-forced replay documents (198-466 prompt tokens) came back with 86-100% top-1
  disagreement and 9-12.5 nats mean divergence, while the other 24 sat at the recompile floor;
  same `prompt_ids_sha`, same scored length. Not established for ordinary generation, which showed
  no anomaly in the load run or the eight generation-path conformance checks.

## Recommended configuration

See `FINAL-CONFIG.md`. In short: keep BF16 KV and FLASH_ATTN, change
`--kv-cache-memory-bytes` from `27000000000` to **`26000000000`**, keep `--kv-offloading-size 32`
(cell L3 proved it reads). Verified numerically identical to the Phase-1 BF16 reference
(0.000000 drift, 0 flips on 40,429 + 20,296 positions), 10/10 conformance, and load
indistinguishable from the shipped config.

## What was not reached

- Depth-3 MTP acceptance (needs a `num_speculative_tokens: 3` boot).
- Needle grid for cell 5, cell 6 and Control-F.
- Cell 7 (`--long-prefill-token-threshold`) — the design called it optional and near-inert, and
  short TTFT p95 was not a problem on the recommended configuration (0.525 s).
- A same-boot repeat pair for Control-F and cell 3, which is what their promising numbers need
  before anyone acts on them.
- Free-generation reasoning correctness (tier 3b). The design already rated it a smoke test with
  ~20-30% power against a 10 pp regression; it was not run for any cell.
