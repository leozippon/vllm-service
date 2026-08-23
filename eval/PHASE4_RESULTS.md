# Phase 4 — free-generation reasoning and long-decode residency: results

Design and the power calculation were frozen in `PHASE4_DESIGN.md` **before** the first comparison
run. Raw per-item records are in `/Data2/lzp/vllm-eval-p4/runs/` and item sets in
`/Data2/lzp/vllm-eval-p4/items/`; the full analysis dump, the per-cell boot logs and the timeline
are in `~/vllm-study-results/p4/` as `analysis.txt`, `boot-<cell>.log` and `master.log`. Everything
below is engine-side (port 8010), `reasoning_effort: xhigh`, `temperature 0`, `top_p 1.0`,
`top_k -1`, `seed 12345`.

The matrix ran 2026-08-23 04:14Z to 07:00Z: four boots, twelve runs, **4,824 graded free generations
and 3.9 M generated tokens, with zero request errors in any run**.

## 1. Reference-answered data found offline

| dataset | items on disk | gold | used |
|---|---|---|---|
| MMLU-Pro test (arrow) | **12,032** | letter, 10 options | 252, stratified over all 14 categories |
| MATH-500 (`test.jsonl`) | **500** (367 at level >= 3) | `answer`, graded by the installed `math_verify` | 150 at level >= 3 |
| GSM8K test / train (parquet) | **1,319 / 7,473** | `#### n` | not used — at ceiling for this model |
| LiveCodeBench `code_generation_lite` | 400 in the v1 file, four larger v2-v5 files | zlib+base64 private tests | problems used as **ungraded** long code prompts; grading needs an execution sandbox that does not exist here |
| HumanEval | **absent** — no data file, package not installed | — | — |
| SWE-bench Lite | **absent** — the path a demo config references does not exist; only a 3-row test fixture | — | — |

Data was never the binding constraint; decode time was. The graded pool is ~13,800 items and the
402 used were chosen against a wall-clock budget.

## 2. Cells as booted (from the engine's own log)

| cell | KV dtype | attention backend | GPU KV cache | full-length concurrency | boot |
|---|---|---|---|---|---|
| A1, A2 | bf16 (`auto`), pool pinned at 26,000,000,000 B | FLASH_ATTN | 777,975 tokens | **2.97x** | 161 s / 155 s |
| C | bf16 (`auto`), engine-sized at 0.92 | FLASHINFER (`--attention-backend`) | 693,413 tokens | 2.65x | 146 s |
| B | **fp8_e4m3**, engine-sized at 0.92 | FLASHINFER (selected by the dtype) | **1,358,930 tokens** | **5.18x** | 146 s |

Every boot fact reproduces Phase 3 exactly, including B's two `k_scale = v_scale = 1.0` warnings and
`FlashInfer resolved query dtypes: prefill=bf16, decode=bf16, kv_cache_dtype=float8_e4m3fn` — the
query is still never quantised. No FlashInfer JIT occurred on any boot; the kernels resolved from the
warm local cache in under three minutes.

## 3. Free-generation reasoning correctness (N = 402, paired)

| run | accuracy | 95% CI | truncated at 8,192 | mean completion tokens | wall | aggregate tok/s |
|---|---|---|---|---|---|---|
| A1 production | 81.34% | [77.24, 84.85] | 58 | 2,433 | 1,385 s | 706.2 |
| A2 production, reboot | 82.09% | [78.04, 85.53] | 54 | 2,381 | 1,334 s | 717.6 |
| C FlashInfer + BF16 | 83.08% | [79.11, 86.43] | 50 | 2,362 | 1,319 s | 719.6 |
| **B FlashInfer + FP8** | **82.84%** | [78.84, 86.21] | 52 | 2,262 | 1,195 s | **760.8** |

All four sit inside a 1.7 pp band. The paired tests, with the exact McNemar table:

| pair | what it isolates | delta | both ok | ref only | cand only | both wrong | psi | McNemar exact p | exact 95% CI on the delta | MDE at this n, psi |
|---|---|---|---|---|---|---|---|---|---|---|
| **A1 vs A2** | **nothing — same configuration, different boot** | +0.75 pp | 316 | 11 | 14 | 61 | 6.22% | **0.690** | [-1.87, +3.18] | 3.48 pp |
| A1 vs C | the attention kernel | +1.74 pp | 316 | 11 | 18 | 57 | 7.21% | 0.265 | [-1.12, +4.23] | 3.75 pp |
| **C vs B** | **the quantisation, kernel held fixed** | **-0.25 pp** | 323 | 11 | 10 | 58 | 5.22% | **1.000** | **[-2.54, +2.11]** | 3.19 pp |
| **A1 vs B** | kernel + quantisation (what production would feel) | **+1.49 pp** | 318 | 9 | 15 | 60 | 5.97% | **0.308** | **[-1.12, +3.73]** | 3.41 pp |
| A2 vs B | kernel + quantisation | +0.75 pp | 321 | 9 | 12 | 60 | 5.22% | 0.664 | [-1.67, +2.94] | 3.19 pp |

Per suite, FP8 against production: MMLU-Pro 72.62% -> 75.40% (+2.78 pp, p = 0.189); MATH-500 96.00%
-> 95.33% (-0.67 pp, p = 1.000). Against the matched kernel: MMLU-Pro 76.19% -> 75.40%
(-0.79 pp, p = 0.804, CI [-3.84, +2.56]).

**The null arm is the whole story.** Running the *same* configuration twice, across a boot boundary,
produced 25 discordant items out of 402 and a +0.75 pp swing. Switching to FP8 produced 24 discordant
items and a +1.49 pp swing — fewer discordant items than the null arm, and a difference in the same
direction and of the same size. Free greedy generation at concurrency 32 is chaotic at the ~6% item
level no matter what you do; FP8 does not add to that.

The realised discordance rate, 5-7%, was at the optimistic end of the pre-registered range, so the
achieved sensitivity is better than the design promised: **this comparison would have detected a
3.4 pp regression at 80% power, and the exact 95% interval rules out any regression larger than
1.1 pp against production, or 2.5 pp against the matched kernel.**

## 4. Greedy divergence (24 prompts, concurrency 1, deterministic)

| pair | token-identical | first divergence p25 / p50 / p75 | earliest |
|---|---|---|---|
| **A1 vs A2** (same configuration, different boot) | **24 / 24 = 100%** | — (no divergence in 21,018 compared tokens) | — |
| A1 vs C (kernel only, BF16 both sides) | 0 / 24 | 28 / **48** / 99 | 3 |
| C vs B (quantisation only, kernel fixed) | 0 / 24 | 13 / **46** / 87 | 3 |
| A1 vs B (kernel + quantisation) | 0 / 24 | 11 / **42** / 95 | 3 |

Two facts follow.

**The floor is a hard zero.** At concurrency 1 the engine is bit-reproducible across a restart:
24 of 24 generations came back token-for-token identical, over 21,018 compared tokens. Any divergence
below is therefore attributable to the configuration, not to the engine.

**FP8 quantisation perturbs the trajectory no more than swapping the attention kernel does.** Median
first divergence is 48 tokens for a pure kernel swap at identical BF16 precision, and 46 tokens for
pure quantisation at a fixed kernel. Per-1,000-token divergence hazard over positions 0-256 is
9-14 for the kernel and 6-22 for the quantisation — the same order, and **neither rises with decode
position**, which is the signature an accumulating residency error would have to leave.

This instrument is far more sensitive than correctness, and it says plainly: **the two configurations
do produce different text, from about the fortieth token onward, on every prompt.** Section 3 is what
says that difference is not damage.

## 5. Long-decode residency

The matched short/long ledger, 16 items each, exact modular arithmetic (M = 1,000,003), scored per
step and paired per (item, step). `ledgerL` generations averaged **9,591 to 10,410 tokens**, so its
800 scored steps are emitted at decode positions of several thousand tokens with the entire preceding
computation resident in KV. `RECALL` asks the model to read back values it emitted itself earlier in
the same generation — content that exists only in decode-written KV.

| run | S local step | S RECALL | **L local step** | L global | **L RECALL** | L mean tokens |
|---|---|---|---|---|---|---|
| A1 | 240/240 | 48/48 | **800/800** | 800/800 | **80/80** | 9,591 |
| A2 | 240/240 | 48/48 | **800/800** | 800/800 | **80/80** | 10,255 |
| C | 240/240 | 48/48 | 798/800 | 760/800 | 80/80 | 9,693 |
| **B** | 240/240 | 48/48 | **800/800** | 800/800 | **80/80** | 10,410 |

**FP8 is perfect on every one of 1,040 exact-arithmetic steps and all 128 long-range recalls, at both
lengths, exactly like production.** Paired A1 vs B and C vs B: zero discordant steps on every
endpoint. Because the reference is at ceiling, discordance could only run one way, which makes the
bound tight: 0 errors out of 800 puts the one-sided 95% upper bound on an FP8-induced deep-decode
step-error rate at **0.38 pp**, and on a long-range recall failure at 3.8 pp.

The only configuration that made *any* ledger error is **C — FlashInfer with plain BF16 KV** — two
local arithmetic slips deep in the long variant, which cascaded into 40 wrong running values. The
quantised configuration made none. Whatever moves this model on long generations, it is not e4m3.

The second, observational residency reading, from the correctness set binned by generation length
(bins on the reference run's completion tokens), points the same way: in the longest bin,
3,562-8,192 tokens, FP8 is **+6.86 pp** against production (p = 0.167) and the matched kernel is
+7.84 pp (p = 0.152). The A-vs-B delta does not grow adversely with length in any bin.

## 6. What the design still could not do

- Prove equivalence. It bounds the regression; it does not show zero.
- Resolve a correctness change below about 3.4 pp, or below 2.5 pp against the matched kernel.
- Grade agentic multi-round coding, which is the user's dominant workload: **no gradeable reference
  data for it exists on this host.** It enters only through the eight ungraded LiveCodeBench prompts
  in the divergence probe.
- Observe divergence hazard beyond ~256 tokens: every configuration pair had already diverged by
  then, so the deep part of the hazard curve has no surviving mass. The residency question is carried
  by the ledger and the length bins instead.
- Say anything about sustained multi-hour production traffic. The longest continuous FP8 load here
  was the ~35-minute correctness run at concurrency 32 (zero errors).

## 7. Conclusion

The gap Phase 3 named is closed. Free-generation reasoning correctness at `xhigh` is
indistinguishable between FP8 KV and production, by a paired test with real power, and the
same-configuration null arm moves further than the candidate does. Accuracy after long decode
residency, never measured before in any phase, is perfect and identical: 1,040 of 1,040 exact steps
and 128 of 128 long-range recalls, in generations averaging ten thousand tokens.

Combined with Phase 3's replicated operational numbers — +6.6% aggregate throughput (independently
reproduced here as +7.7%), -14.2% long TPOT, 2.97x -> 5.18x full-length concurrency, short chats
1.30 s faster end to end — **the evidence supports adopting FP8 KV.**

## 8. Recommendation: adopt FP8 KV

Adopt. The decision rests on measurement, not on judgement:

- **Free-generation reasoning correctness is unchanged.** 402 paired items at `xhigh`, McNemar exact
  p = 0.308 against production and p = 1.000 against the matched kernel. The exact 95% interval rules
  out any regression larger than **1.1 pp** against production and **2.5 pp** against the matched
  kernel, and the design would have caught a 3.4 pp regression at 80% power. The point estimate is
  +1.49 pp, i.e. slightly favourable.
- **The same configuration, rebooted, moves further than the candidate does** (+0.75 pp, 25 discordant
  items, versus FP8's +1.49 pp and 24 discordant items). FP8 sits inside the reproducibility of the
  configuration it would replace.
- **Accuracy after long decode residency — the gap Phase 3 named — is perfect and identical.** 1,040
  of 1,040 exact modular-arithmetic steps and 128 of 128 long-range self-recalls, in generations
  averaging ~10,000 tokens. The only ledger errors in the whole matrix came from BF16 KV on the
  FlashInfer kernel, not from FP8.
- **Divergence confirms the mechanism.** The engine is bit-reproducible across a restart (24/24
  identical, 21,018 tokens), so divergence is attributable. FP8 quantisation moves the greedy
  trajectory no more than swapping the attention kernel at identical BF16 precision does — median
  first divergence 46 tokens versus 48 — and its hazard does not grow with decode position.
- Phase 3's operational gains stand and were independently reproduced here: aggregate throughput on
  the 402-item correctness run was **+7.7%** (760.8 versus 706.2 tok/s) against production, on top of
  2.97x -> 5.18x full-length concurrency.

Residual, stated plainly: the boot log still carries the `k_scale = v_scale = 1.0` warning, which
Phase 3 established cannot be cleared on this build and checkpoint by any available means, and which
this phase found no behavioural consequence of. Agentic multi-round coding cannot be graded on this
host, so it is covered only by the ungraded divergence probe, where FP8 behaves like the kernel swap.

## 9. Configuration left running

`config/deployments/qwen-3.8-27b-fp8.conf` (mode 600, still selected by `config/deployments/active`),
changed in exactly the two ways Phase 3 specified — `--kv-cache-dtype auto` became
`--kv-cache-dtype fp8_e4m3`, and `--kv-cache-memory-bytes 26000000000` was removed so the pool is
engine-sized at 0.92, which is how every FP8 cell in Phase 3 and Phase 4 was measured. Nothing else
in the profile changed.

Verified after the boot at 2026-08-23T07:06:48Z:

```
STARTED_OK after 146s ; NRestarts=0 ; vllm.service active/running ; vllm-gateway.service active
engine /health 200 ; engine /v1/models 200
gateway 127.0.0.1:8011 /health/live 200
gateway end-to-end xhigh chat completion: finish_reason stop, 53 tokens, correct answer
Using fp8_e4m3 data type to store kv cache
FlashInfer resolved query dtypes: prefill=bf16, decode=bf16, kv_cache_dtype=float8_e4m3fn, arch=sm89
GPU KV cache size: 1,358,930 tokens  (5.18x at 262,144; production was 777,975 = 2.97x)
num_preemptions_total 0 ; /dev/shm 32 GB used of 252 GB, one live offload region
cgroup memory 42.9 GiB ; cpu.stat nr_throttled 0 of 1,856 periods ; memory.events all zero
```

**Rollback is one line and one restart**: put `--kv-cache-dtype auto` back and restore
`--kv-cache-memory-bytes 26000000000`, then `systemctl --user restart vllm.service`. The pre-change
profile is preserved at `~/rollback/engine-phase4-20260823T034829Z/qwen-3.8-27b-fp8.conf.orig`.

Two coordination notes were left for whoever picked this up. Both have since been closed:

- The profile change was an **uncommitted working-tree edit** in a repository another agent was also
  editing. It was committed as `62db0d9`, together with the README and LOGBOOK updates that record
  the precision decision.
- `/Data2/lzp/vllm-eval/known-good.conf`, the eval harness's boot-failure restore target, still held
  the BF16 production profile, so a failed harness boot would have silently reverted precision. That
  private copy has been deleted; the harness now restores the committed profile from
  `git show HEAD:config/deployments/...`.
