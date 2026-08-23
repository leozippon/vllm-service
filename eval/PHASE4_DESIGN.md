# Phase 4 — free-generation reasoning and long-decode residency: pre-registered design

Written and frozen **before** any comparison run. Engine-side only (port 8010, the gateway is
bypassed), 2x L20 (GPUs 6,7), vLLM 0.27.1, TP=2, `reasoning_effort: xhigh` throughout.

Phase 3 settled throughput, latency, capacity and prefill-side accuracy. Two things it could not
see, for either configuration:

1. free-generation reasoning correctness at meaningful N;
2. accuracy after long decode residency — KV written at token 1 of a long trace and read at token N.

Every Phase 1-3 accuracy tier is prefill-plus-one-token (teacher forcing) or a short retrieval
answer. Teacher forcing pins the trajectory, so it structurally cannot see the one thing that
distinguishes free generation: an error at token *i* changes token *i+1* and compounds. This phase
measures exactly that.

## 1. Reference-answered data available offline (no network egress)

Everything below is present on local disk and readable with locally installed packages.

| dataset | location | items | gold answer | usable |
|---|---|---|---|---|
| MMLU-Pro, test split | `routing/src/dataset/MMLU-Pro` (arrow) | **12,032** | letter, 10 options | **yes** |
| MMLU-Pro, validation split | same | 70 | letter + CoT | yes (not used) |
| MATH-500 | `routing/src/dataset/MATH-500/test.jsonl` | **500** (level 3-5: 367) | `answer` field | **yes**, graded with the locally installed `math_verify` |
| GSM8K test | `routing/src/dataset/GSM8K/gsm8k_test.parquet` | **1,319** | `#### n` | yes (not used: at ceiling for this model) |
| GSM8K train | same directory | 7,473 | `#### n` | yes (not used) |
| LiveCodeBench `code_generation_lite` | `routing/src/dataset/code_generation_lite/*.jsonl` | 400 (v1 file; v2-v5 larger) | zlib+base64 private tests | **problems yes, grading no** — needs an execution sandbox that does not exist here; used as *ungraded* long code prompts for the divergence probe |
| HumanEval | — | — | — | **absent**: no data file on disk, `human_eval` package not installed anywhere |
| SWE-bench Lite | referenced by a demo config | 3-row test fixture only | — | **absent**: the referenced dataset directory does not exist |

So the usable graded pool is large (≈13,800 items) and the binding constraint is decode time, not
data. Sample sizes below are chosen against a wall-clock budget, not against data scarcity.

## 2. Configurations (all other profile settings identical)

| tag | `--kv-cache-dtype` | KV pool | attention backend | what it is |
|---|---|---|---|---|
| **A1**, **A2** | `auto` (BF16) | `--kv-cache-memory-bytes 26000000000` | FLASH_ATTN (default) | current production, booted twice |
| **C** | `auto` (BF16) | engine-sized at 0.92 | `--attention-backend FLASHINFER` | matched-kernel control |
| **B** | `fp8_e4m3` | engine-sized at 0.92 | FLASHINFER (forced by the dtype) | the FP8 candidate |

C is included because Phase 3 found the kernel swap moves deep tokens **more** than the quantisation
does (2.10 pp vs 0.79 pp against production). Without C, any A-vs-B difference cannot be attributed
to precision rather than to the kernel. A1 and A2 are the same configuration on two different boots:
they are the **null arm** that measures how much of any A-vs-B difference is simply free generation
being chaotic.

Cell order: A1 -> C -> B -> A2. The last cell restores production and supplies the cross-boot floor.

## 3. Instruments

### 3.1 Free-generation reasoning correctness (paired)

- Items: **402** = 252 MMLU-Pro (stratified over all 14 categories) + 150 MATH-500 at level >= 3.
  Fixed sample, fixed order, identical in every cell.
- Sampling: `temperature 0` (greedy), `top_p 1.0`, `top_k -1`, `seed 12345`, `max_tokens 8192`,
  `reasoning_effort xhigh`. Concurrency 32, identical submission order in every cell.
- **Pre-registered scoring rule**: the answer is the last `ANSWER:` line. MMLU-Pro is scored by exact
  letter; MATH-500 by `math_verify` symbolic equivalence against the gold answer. A reply that does
  not emit a parseable answer within 8,192 tokens is **incorrect**. (Measured pilot truncation rate
  on MMLU-Pro: 22.5%. This rule is applied identically to every configuration and matches what a
  user with a token budget actually experiences.)
- **Test**: McNemar's exact (binomial) two-sided test on the discordant pairs, per configuration
  pair. Reported with the full 2x2 table, not just the p-value.

### 3.2 Greedy divergence (high sensitivity, no reference answers needed)

- 24 prompts (8 MMLU-Pro, 8 MATH-500, 8 LiveCodeBench code-generation), `max_tokens 1536`,
  **concurrency 1** so that batch composition is fixed and the engine is deterministic. The same
  prompts in the same order in every cell.
- Token sequences are captured exactly (`logprobs: true` returns every generated token).
- Reported: fraction of prompt pairs that are token-identical; the distribution of the **first
  divergence position**; and the **per-token divergence hazard by decode-position bucket** — among
  pairs still identical entering a bucket, the rate of first divergence per 1,000 surviving tokens.
- The hazard curve is a residency instrument in its own right: if FP8 error accumulated as KV built
  up through the generation, the hazard would **rise** with decode position. A flat or falling
  hazard is evidence against accumulation.

### 3.3 Long-decode residency (constructed, verifiable, matched short/long)

One task in two lengths that differ **only** in how long the model must generate:

- `ledgerS`: 16 items x **15** steps of exact modular arithmetic (M = 1,000,003).
- `ledgerL`: 16 items x **50** steps, same generator, same per-step difficulty.

Each reply ends with a `RECALL = ...` line asking the model to read back values **it emitted itself**
earlier in the same generation (3 targets for S, 5 for L). Those values exist only in decode-written
KV, so RECALL is a direct probe of long-range attention over KV the decode phase wrote.

Endpoints, all paired per (item, step):
- **local step correctness** — is step *k* consistent with the value the model itself reported at
  step *k-1*? This is immune to error cascade, so it isolates degradation from propagation.
- **global step correctness** — is step *k* the true value?
- **RECALL correctness** — does the recalled value match what the model emitted at that line?

**The hypothesis test is the length x configuration interaction**: if quality degrades with decode
residency, the B-minus-A penalty must be larger on `ledgerL` than on `ledgerS`. A penalty of equal
size on both is not a residency effect.

### 3.4 Residency, second reading, from the correctness set at no extra cost

Correctness items are binned by the generation length observed on the reference run, and the A-vs-B
accuracy delta is reported per bin. If residency degrades quality, the delta must grow with length.

## 4. Statistical power — stated before the results

For a paired binary comparison, the McNemar statistic depends on the discordant count, not on N
alone. With N items and a discordance rate psi, the minimum detectable accuracy difference at 80%
power and two-sided alpha = 0.05 is approximately `MDE = 2.80 * sqrt(psi / N)`.

Free greedy generation is chaotic: two engines that differ by a rounding step take different
reasoning paths, so psi is not zero even between two runs of the *same* configuration. **The A1-vs-A2
arm measures psi empirically**; the table below spans the plausible range.

| instrument | N | psi = 0.05 | psi = 0.10 | psi = 0.15 | psi = 0.20 |
|---|---|---|---|---|---|
| correctness, all | 402 | 3.1 pp | 4.4 pp | 5.4 pp | 6.3 pp |
| correctness, MMLU-Pro only | 252 | 3.9 pp | 5.6 pp | 6.8 pp | 7.9 pp |
| correctness, MATH-500 only | 150 | 5.1 pp | 7.2 pp | 8.9 pp | 10.2 pp |
| ledger local step, L | ~800 | 2.2 pp | 3.1 pp | 3.8 pp | 4.4 pp |
| ledger RECALL, L | 80 | 7.0 pp | 9.9 pp | 12.1 pp | 14.0 pp |

For comparison, comparing two **independent** accuracy rates at N = 402 each and p ~ 0.72 detects
only **8.8 pp**. The paired design is 1.6x to 2.8x more sensitive at the same cost, which is why it
is used.

**So, plainly: this phase can detect a reasoning regression of roughly 3-6 percentage points on the
402-item free-generation set, and roughly 2-4 pp on the 800 long-variant ledger steps. It cannot resolve a 1 pp
regression.** The divergence instrument is far more sensitive than either but measures *difference*,
not *damage*: it detects that two configurations produce different text long before that difference
becomes a wrong answer.

## 5. What this design still cannot do

- It cannot prove equivalence. A null result bounds the regression at the MDE above; it does not
  show zero.
- Correctness at N = 402 cannot resolve a sub-3 pp change, and RECALL at N = 80 cannot resolve less
  than about 7 pp.
- The graded suites are knowledge/maths/multiple-choice reasoning. Agentic multi-round coding — the
  user's actual dominant workload — has **no gradeable reference data on this host**, so it enters
  only through the ungraded divergence probe.
- Generation-length binning is observational, not randomised: long-generation items are also harder
  items, so a length trend confounds difficulty with residency. The matched short/long ledger exists
  precisely to break that confound.
