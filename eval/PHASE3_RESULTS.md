# Phase 3 — replication of Control-F and cell 3, and the calibration question

Engine-side only, 2x L20 (GPUs 6,7), vLLM 0.27.1, TP=2, direct to the engine on loopback.
Load generator, corpus, and accuracy tiers are Phase 1/2's (`agent_mix`; corpus sha `7039b2e1e3420d19`).
Both FlashInfer cells run with **no** `--kv-cache-memory-bytes` (engine-sized at 0.92), exactly as
Phase 2 ran them. `known-good.conf` (the boot-failure restore target) was corrected from the
engine-killing `27000000000` to the production `26000000000` before the first boot; **no Phase-3 boot
used 27000000000**.

Part 1 (what `--calculate-kv-scales` does, and whether any calibrated FP8 KV route exists) is in
`PHASE3_CALIBRATION.md`.

## Noise floors, re-measured on this backend

Phase 1 measured its floors on FLASH_ATTN. On FLASHINFER they are **much tighter**, and one of Phase 2's
stated floors does not reproduce at all.

| quantity | Phase-1/2 floor | measured here |
|---|---|---|
| aggregate output tok/s | ±0.08% same-boot, ~0.1% cross-boot | **0.12% across 3 runs on 2 boots** |
| long TPOT p50 | ±0.59% same-boot, ~2% cross-boot | **0.20% across 3 runs on 2 boots** |
| short TTFT p95 | ±8.59% | **+20.9% between two same-boot runs** (see below) |
| tier-1 short, same config | 0.000000 / 0.0000 pp | **0.000000 / 0.0000 pp** (cross-boot) |
| tier-1 **deep**, same config | 0.26 pp / 0.008, "200k/235k systematically non-deterministic" | **0.00 pp / 0.000000, exact-zero 100% at all four depths** |
| tier-3a trace, same config | - | **0.000000 / 0.0000 pp** (cross-boot) |

Control-F replicated (Phase-2 boot, plus a same-boot pair on a fresh boot):

| run | aggregate out tok/s | long TPOT p50 | short TTFT p50 / p90 / p95 |
|---|---|---|---|
| control-F (Phase 2) | 138.447 | 38.572 ms | 0.369 / 0.410 / 0.422 s |
| control-F2 r1 | 138.311 | 38.547 ms | 0.368 / 0.416 / 0.421 s |
| control-F2 r2 (same boot) | 138.283 | 38.623 ms | 0.373 / 0.416 / **0.509** s |

**Control-F holds up exactly.** Aggregate and long TPOT p50 reproduce to ~0.1-0.2% across a boot
boundary, which is 10x tighter than the floor Phase 1 assumed.

**Short TTFT p95 does not hold up as a statistic.** Two runs on the *same boot* with the same
configuration gave 0.421 s and 0.509 s (+20.9%), while p50 (0.368 -> 0.373) and p90 (0.416 -> 0.416)
barely moved. At n=102 short requests, p95 is the 97th-largest sample and is set by two or three
requests that happened to land behind a chunked long prefill. **Any conclusion resting on a short
TTFT p95 comparison — including Phase 2's "+121%" rejection of cell 3 — has to be re-examined against
this floor.**

## The accuracy instrument is deterministic across boots

Same configuration, different boot (control-F vs control-F2), on the identical corpus:

- tier-1 short, n=40,429: mean |Δ logprob| **0.000000**, argmax-flip **0/40,429**, exact-zero 100.00%
- tier-1 deep, n=384 (32k/128k/200k/235k): **0/384 flips**, mean 0.000000, exact-zero 100.00% at *every*
  depth including 200k and 235k
- tier-3a trace, n=20,296: mean **0.000000**, flip **0/20,296**, exact-zero 100.00%

and control-F2 vs cell 0 reproduces Phase 2's recompile floor to six decimals (0.581863 mean /
2.8198 pp short; 12/384 = 3.12 pp deep; mean 0.025940).

This matters more than it looks. Phase 2 reported a 0.26 pp same-config floor on the deep tier and
concluded that tier could not resolve small differences. On FLASHINFER the floor is **zero**, so cell 3's
deep-tier distance from its matched control is entirely attributable to the FP8 KV plus the recompile —
there is no nondeterminism to hide behind.

## Matched control, full battery (control-F2, one boot)

- needle grid, **N=100** (4 lengths x 5 depths x 5 codes): **100/100**, 25/25 at each of
  32k / 128k / 200k / 250k. Wall 4,012 s.
- deep-weighted teacher forcing, **N=381** at 128k/200k/250k x 127 continuation positions (new tier,
  added this phase): captured clean, 127 positions at every depth.
- API conformance: **10/10**.
- boot facts: FLASHINFER, `prefill=bf16 decode=bf16 decode_backend=flashinfer-native
  kv_cache_dtype=bf16 arch=sm89`, block 784, KV pool 21.59 GiB, 693,413 tokens, 2.65x at 262,144,
  no uncalibrated-scale warning. Reproduces Phase 2's Control-F boot exactly.

## Candidate, replicated (cell 3 = `--kv-cache-dtype fp8_e4m3`, FLASHINFER)

Boot facts reproduce Phase 2 exactly: block **1568**, KV pool 21.59 GiB, **1,358,930 tokens**,
**5.18x** at 262,144, both scale warnings present.

New detail from the kernel-resolution line, which changes how one of the two warnings should be read:

```
FlashInfer resolved query dtypes: prefill=torch.bfloat16, decode=torch.bfloat16,
decode_backend=flashinfer-native, kv_cache_dtype=torch.float8_e4m3fn, arch=sm89
```

**The query is never quantised** — only the KV cache is FP8, and FlashInfer dequantises it inside the
kernel. So the second warning ("Checkpoint does not provide a q scaling factor ... This only matters for
FP8 Attention backends") is *provably inert* on this configuration. Only the `k_scale = v_scale = 1.0`
warning describes anything that actually happens.

| run | aggregate out tok/s | long TPOT p50 | long E2E p50 | short TTFT p50 |
|---|---|---|---|---|
| cell 3 (Phase 2) | 146.269 | 34.616 ms | 118.31 s | 0.299 s |
| cell3-b r1 | 146.369 | 34.505 ms | 117.77 s | 0.296 s |
| cell3-b r2 (same boot) | 146.284 | 34.553 ms | 118.22 s | 0.296 s |

Aggregate spread **0.07%**, long TPOT p50 spread **0.32%**. The candidate replicates as tightly as the
control. Every headline Phase-2 number for cell 3 stands.

## The short-TTFT question, measured properly

Pooled over the replicate runs (n=279-306 short requests per configuration rather than ~100):

| statistic | production FA2/BF16 | Control-F FI/BF16 | cell 3 FI/FP8 |
|---|---|---|---|
| short TTFT mean | 0.526 s | 0.489 s | 0.597 s |
| p50 | 0.429 | 0.370 | **0.298** |
| p75 | 0.472 | 0.391 | 0.538 |
| p90 | 0.495 | 0.415 | 0.569 |
| p95 | 0.525 | 0.432 | 0.770 |
| p99 | 5.615 | 5.557 | 7.301 |

cell 3 versus the **production** configuration, in absolute seconds:
p50 **-0.132 s**, p75 +0.066 s, p90 +0.074 s, p95 +0.245 s, mean **+0.070 s**.

So FP8 gives the *fastest median* first token of anything measured and a heavier upper-middle tail. The
crossover sits between p50 and p75. Phase 2's headline "+121%" was a single-run p95 ratio against
Control-F; the same-boot p95 floor is ±20-58%, so that number was never separable from noise at the
precision it was quoted.

**And the first token is the wrong thing to measure.** A short chat is 256 output tokens, so what the
user waits for is the whole reply:

| | production | Control-F | cell 3 |
|---|---|---|---|
| short-chat E2E mean | 10.977 s | 10.637 s | **9.808 s** |
| short-chat E2E p50 | 9.956 s | 9.731 s | **8.654 s** |
| short-chat decode | 37.72 ms/tok | 37.08 | **32.59** |

FP8 decodes short chats **5.13 ms/token faster**, which over a 256-token reply saves **1.31 s** — about
19x the +0.070 s mean first-token cost. **Short chats finish 1.30 s sooner at the median under FP8, not
later.** The interactive class does not pay for this change; it benefits from it.

## What FP8 buys the dominant workload (pooled, n=48 long requests per configuration)

| | production FA2/BF16 | Control-F FI/BF16 | cell 3 FI/FP8 |
|---|---|---|---|
| aggregate output tok/s | 137.22 | 138.35 | **146.31** |
| long TPOT p50 | 40.24 ms | 38.57 | **34.52** |
| long TPOT p95 | 43.07 ms | 42.10 | **38.55** |
| long E2E p50 | 131.48 s | 130.43 | **118.22** |
| long TTFT p95 | 29.11 s | 30.84 | 31.40 |
| max GPU KV usage under load | 37.2% | 42.2% | **23.9%** |
| full-length (262,144) concurrency | 2.97x | 2.65x | **5.18x** |

cell 3 versus production, in terms the workload feels:

- **aggregate throughput +6.62%**
- **long TPOT p50 -14.21% (-5.72 ms/token)** — a 3,000-token `xhigh` reasoning turn decodes **17.2 s faster**
- **long-request E2E p50 -13.3 s per turn**, i.e. **-53 s over a 4-round agent session**
- **full-length concurrency 2.97x -> 5.18x (+74%)**, and KV pressure under the same load falls from 37% to 24%
- the one long-side regression: **long TTFT p95 +2.29 s (+7.9%)** — real, but set against -13.3 s of E2E

Zero preemptions, zero 429/5xx, zero offload CPU->GPU reads in every run. Envelope unchanged:
3.24 cores under load, `nr_throttled` 0, `memory.events` all zero, inside 8 cores / 64 GB.

## The full accuracy battery on cell 3, with honest power

All scored against the **matched control at the same KV pool and backend** (control-F2), not against the
FLASH_ATTN production baseline. The same-config floor for every tier below is **exactly zero**
(measured cross-boot, both on BF16 and on FP8), so nothing here is nondeterminism.

| tier | N | cell 3 vs matched control | BF16 recompile-only control | what this N can detect |
|---|---|---|---|---|
| tier-1 short teacher-forced | 40,429 positions | mean 0.631846, argmax-flip **3.0473 pp** | 0.581863 / 2.8198 pp | ±0.17 pp on the flip rate (95% CI), but positions within a document are correlated, so the effective N is lower than 40k |
| tier-1 deep (32k/128k/200k/235k) | 384 | **6/384 = 1.56 pp**, mean 0.028267; per depth **0.00 / 0.00 / 3.12 / 3.12 pp** | 12/384 = 3.12 pp, mean 0.025940 | ±1.24 pp; resolves a gross divergence and a mean-logprob shift, not a 1 pp change |
| tier-1 deep-weighted (128k/200k/250k x 127) | 381 | **9/381 = 2.36 pp**, mean 0.022918; per depth **2.36 / 3.94 / 0.79 pp** | not captured on this tier (see gap below) | ±1.52 pp |
| tier-3a reasoning-trace replay | 20,296 positions | mean 0.034584, flip **2.0792 pp** | 0.032323 / 1.8969 pp | ±0.20 pp; the FP8 increment over recompile is +0.18 pp, i.e. *at* the resolution limit |
| tier-2 needle grid | **100** (4 lengths x 5 depths x 5 codes) | **100/100**, 25/25 at each length including 250k, **0 regressions** vs a 100/100 matched control | - | 0/100 puts a 95% upper bound of **3%** on the true failure rate; a 1-2 pp drop is not resolvable |
| API conformance | 10 checks | **10/10** | 10/10 | - |

Two things this battery establishes that Phase 2 could not:

1. **There is no saturation signature.** If `k_scale = 1.0` were clipping KV activations above e4m3's
   448, the damage would grow with context length. It does not: on the deep-weighted tier the **deepest**
   bucket is the **cleanest** (250k = 0.79 pp against 200k = 3.94 pp), mean |Δ logprob| is flat across
   depth (0.0253 / 0.0214 / 0.0220), and the needle grid is perfect at 250k. Combined across both deep
   tiers, 765 scored deep positions and 50 deep retrieval items show no depth trend.
2. **FP8 KV moves the model less than recompiling it does.** On the deep tier cell 3 sits at 1.56 pp
   against its matched control while a BF16-to-BF16 recompile sits at 3.12 pp. On the trace tier the FP8
   increment over a recompile is +0.18 pp against a ±0.20 pp resolution limit.

**What the battery does not cover, stated plainly:**
- Every tier is prefill-plus-one-token or a short retrieval answer. **Nothing here measures accuracy after
  long decode residency** — KV written at token 1 of a 3,000-token `xhigh` trace and read at token 3,000.
  That is the one place FP8 KV could still degrade unobserved, and this study cannot exclude it.
- Free-generation reasoning correctness (tier 3b) was **not run**, in this phase or in Phase 2. The design
  rates it ~20-30% power against a 10 pp regression, i.e. a smoke test.
- The literal decision-rule threshold of "≤0.5 pp deep-context argmax flip" is **below the drift a bare
  recompile produces** and cannot be met by any configuration, quantised or not.

## Cell 3-cal — `--calculate-kv-scales`, measured

Booted as `--kv-cache-dtype fp8_e4m3 --calculate-kv-scales`. The engine's own log, in order:

```
non-default args: {... 'kv_cache_dtype': 'fp8_e4m3', 'calculate_kv_scales': True, ...}
WARNING [cache.py:277]  The `--calculate-kv-scales` option is deprecated and will be removed in v0.19.
                        The scales will be loaded from the model checkpoint if available, otherwise
                        they default to 1.0.
WARNING [config.py:455] Disabling calculate_kv_scales for hybrid model '/Data/public/Qwen3.8-27B-FP8'.
                        Hybrid models with recurrent layers (GDN, Mamba, SSM) produce unreliable KV
                        cache scales during the calibration pass because recurrent state is
                        uninitialized. Using default scale of 1.0 instead.
WARNING [kv_cache.py:134] Checkpoint does not provide a q scaling factor. Setting it to k_scale.
WARNING [kv_cache.py:151] Using KV cache scaling factor 1.0 for fp8_e4m3.
```

Boot facts are identical to cell 3: block 1568, KV pool 21.59 GiB, 1,358,930 tokens, 5.18x.

**Answers to the three questions asked of this flag:**
- *Does it apply to this architecture's 16 full-attention layers?* **No.** It is switched off before any
  layer is constructed, by the hybrid-model hook, because of the 48 GDN layers.
- *Does it remove the `k_scale = v_scale = 1.0` warning?* **No.** The warning is unchanged.
- *What does it cost in throughput and latency?* **Nothing, and it gains nothing** — the resulting engine
  configuration is byte-for-byte cell 3. No load run was spent on it; measuring a no-op twice would be
  dishonest padding, not evidence.

The trap worth recording for anyone who revisits this: on a *non-hybrid* model the flag would have taken
a single-batch amax from the first real forward pass, disabled itself, and — because
`kv_cache.py:96-150` only emits the warning on the `not calculate_kv_scales` branch — **suppressed the
warning without improving accuracy at all**. Anyone who reaches for this flag to "clear the warning" is
buying silence, not calibration.

## Replication table (every load run in this phase, plus Phase 2's)

| run | aggregate tok/s | long TPOT p50 | long E2E p50 | short TTFT p50 / p90 / p95 |
|---|---|---|---|---|
| Control-F boot 1 (Phase 2) | 138.45 | 38.57 | 130.38 | 0.369 / 0.410 / 0.422 |
| Control-F2 boot 2 run 1 | 138.31 | 38.55 | 131.03 | 0.368 / 0.416 / 0.421 |
| Control-F2 boot 2 run 2 | 138.28 | 38.62 | 131.17 | 0.373 / 0.416 / 0.509 |
| Control-F3 boot 3 | 138.44 | 38.57 | 130.52 | 0.370 / 0.414 / 0.437 |
| cell 3 boot 1 (Phase 2) | 146.27 | 34.62 | 118.31 | 0.299 / 0.581 / 0.933 |
| cell3-b boot 2 run 1 | 146.37 | 34.51 | 117.77 | 0.296 / 0.561 / 0.604 |
| cell3-b boot 2 run 2 | 146.28 | 34.55 | 118.22 | 0.296 / 0.579 / 0.957 |
| cell3-c boot 3 | 146.29 | 34.57 | 118.35 | 0.297 / 0.560 / 0.656 |

Within-configuration spread: aggregate **0.12% / 0.07%**, long TPOT p50 **0.20% / 0.32%**,
short TTFT p50 **1.2% / 0.02%**, short TTFT p90 **1.5% / 3.3%**, short TTFT **p95 20.7% / 58.5%**.

So the FP8-vs-BF16 separations are 30-50x their noise floors on aggregate throughput and long TPOT,
clearly real on short TTFT p50 (better) and p90 (worse), and **not measurable at all on short TTFT p95** —
the statistic Phase 2 rejected the candidate on.

## The e5m2 differential probe is impossible on this checkpoint

Planned as a direct test of the saturation hypothesis (e5m2's max is 57,344, so it cannot clip where e4m3
clips at 448). The engine refuses the configuration outright:

```
ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints.
```

`model_executor/layers/attention/attention.py:206-214` — e5m2 KV is only permitted when the checkpoint's
quantisation is a `CompressedTensorsKVCacheMethod` carrying a `kv_cache_scheme`. This is a vendor block-FP8
checkpoint, so the route is closed. The boot failed, the harness captured the error, restored the
known-good profile and re-verified health automatically (`RESTORED_HEALTHY`); the service was never left
down. **The saturation question therefore rests on the depth-resolved evidence above, which is what it
has.**

## Deep-tier drift is a kernel effect, not a precision effect

The single cleanest comparison in this phase. Deep-weighted tier, n=381, all >=128k, both scored against
the **production FLASH_ATTN BF16 engine** as the common reference:

| candidate | KV precision | argmax flip vs production | mean abs delta | per depth 128k / 200k / 250k |
|---|---|---|---|---|
| Control-F2 (FLASHINFER) | **BF16 — no precision change at all** | **8/381 = 2.10 pp** | 0.025459 | 1.57 / 3.94 / 0.79 pp |
| cell3-b (FLASHINFER) | **FP8 e4m3** | **3/381 = 0.79 pp** | 0.020170 | - |

**The FP8 configuration is closer to the production engine than the BF16 one is.** Swapping the attention
kernel moves the deep-context distribution more than quantising the KV cache to FP8 does, and both show
the same per-depth pattern with the 200k bucket most affected and 250k least. Whatever is moving these
tokens is the kernel and compile path, not e4m3.

One more correction to the record: the deep tier's non-determinism that Phase 1 and 2 reported is a
**FLASH_ATTN** property. On the restored production engine the deep tier is only 10.68% exact-zero against
its own reference, while on FLASHINFER it is 100.00% exact-zero across boots on both BF16 and FP8.

## Scoring cell 3 against the design's decision rule, with what changed

| criterion | Phase 2 | Phase 3 |
|---|---|---|
| no uncalibrated-scale warning | FAIL | **FAIL, and now known to be permanently unfixable here** — see below |
| deep-context argmax flip <= 0.5 pp | "not measurable" | **0.79 pp vs production** (n=381 deep); a BF16-only kernel swap scores 2.10 pp, so no configuration can meet the letter of this rule |
| mean abs delta logprob <= 0.15 | deep PASS | **PASS** (deep 0.020-0.028) |
| needle >= 96/100, no systematic deep failure | 20/20 | **PASS — 100/100, 0 regressions, 25/25 at 250k** |
| trace agreement in the same band | PASS | **PASS** (+0.18 pp over recompile, at the ±0.20 pp resolution limit) |
| long TPOT p50 no worse | PASS | **PASS by -14.2%** |
| short TTFT p95 no worse by >10% | FAIL (+121%) | **the statistic is unmeasurable** (within-config spread 21-58%); at p90 the regression is real and small (+0.074 s vs production), and short-chat **end-to-end is 1.30 s better** |
| API conformance | 10/10 | **10/10** |

The warning criterion cannot be satisfied on this build and this checkpoint by any means available:
`--calculate-kv-scales` is disabled for hybrid models, the checkpoint ships no k/v scales, no
FlashInfer-compatible dynamically-scaled FP8 dtype exists, and e5m2 is refused on fp8 checkpoints.
The only question left is whether to waive it on evidence. What the evidence says:

- the warning describes an **identity** scale, not an unfitted one; e4m3 keeps ~3 mantissa bits at every
  exponent, so scale 1.0 costs range headroom, not relative precision
- half of the warning pair is **inert**: the query is never quantised on this path
  (`prefill=bf16, decode=bf16`), so the q-scale message does not describe anything that happens
- the failure it guards against — clipping above 448 — **has no signature**: 765 scored deep positions
  with no depth trend and the deepest bucket cleanest, and 50 deep retrieval items at 200k/250k all correct
- FP8 KV moves the deep-context distribution **less** than swapping the attention kernel at identical
  BF16 precision does

## Recommendation: this is a judgement call, and here are the two options

**Phase 4 settled it.** Option B was adopted on 2026-08-23 on the paired evidence in `PHASE4_RESULTS.md`,
which measured the two things this section says do not exist. "What is running now" below is Phase 3's
state, not today's.

I am **not adopting FP8 unilaterally**, and I have left production on BF16. But the reason has moved.
Phase 2 framed this as a workload-weighting trade — throughput for the agents versus latency for the
chats. **Measurement has settled that half: the chats win too** (1.30 s faster at the median, because
5.13 ms/token of decode more than repays 0.07 s of first token). What remains is a pure residual-risk
judgement, and that belongs to the user.

**Option A — keep BF16 (what is running now).** No change, no risk, the configuration whose numerics are
bit-identical to the reference. Costs: 2.97x full-length concurrency, 40.24 ms long TPOT, 137.2 tok/s.

**Option B — adopt FP8 KV.** One line in the deployment profile, `--kv-cache-dtype auto` ->
`--kv-cache-dtype fp8_e4m3` (which selects FLASHINFER; the kernel resolves from the warm local cache, no
JIT, no network). Reversible in one restart. Measured, replicated over three boots:

| | gain |
|---|---|
| full-length 262,144 concurrency | **2.97x -> 5.18x (+74%)** — the concurrency >= 4 requirement becomes reachable at full context |
| aggregate throughput | **+6.6%** |
| long `xhigh` turn decode | **-14.2% TPOT, i.e. 17.2 s off a 3,000-token reasoning turn** |
| long-request end to end | **-13.3 s per turn, -53 s per 4-round session** |
| short 3-turn chat | **-1.30 s per turn end to end** |
| KV pressure under the same load | 37% -> 24% |
| costs | long TTFT p95 +2.29 s; short TTFT p90 +0.074 s; a boot-log warning that cannot be cleared |

**What would turn Option B into a clear "adopt", and does not exist yet:** a free-generation reasoning
correctness comparison at meaningful N (never run, in any phase), and a probe of accuracy after long
decode residency (KV written at token 1 of a 3,000-token trace and read at token 3,000). Every tier in
this study is prefill-plus-one-token or a short retrieval answer. If the user's tolerance for an
unquantified residual is low, Option A is the right answer and costs them the numbers in the table above.

**If Option B is chosen**, ship exactly the measured configuration: add `--kv-cache-dtype fp8_e4m3` and
**remove** `--kv-cache-memory-bytes` (every cell-3 boot was engine-sized at 0.92 -> 21.59 GiB). Do not
also raise the pool to the FlashInfer "fully utilize" anchor of 24,247,707,136 — that was never measured,
and Phase 2 already showed what over-committing this device does.
