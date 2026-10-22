# MEASUREMENT

Measuring the engine is how any change to its configuration is judged, and the harness in `eval/` does most of that measuring. This document is the standing rule for doing it truthfully: what has to be held constant, which numbers are statistics, what each accuracy tier can and cannot resolve, and what is still unmeasured. The measured results themselves are the phase documents alongside the harness; operating procedure is in `README.md`, and dated history in `LOGBOOK.md`.

## The instrument

`eval/eval_suite.py` measures one engine configuration at a time.

| command | what it produces |
| --- | --- |
| `bootfacts` | per-boot engine facts from the journal: attention backend, block size, GPU KV tokens, concurrency at full context, compile time, scale warnings |
| `agent-mix` | the load run: 4 concurrent long `xhigh` agent sessions x 4 rounds, a 30k-token first round growing to ~60k by verbatim re-send of the prior conversation, plus a 3-turn short chat launched every 15 s — 16 long and roughly 100-108 short requests |
| `accuracy` | the accuracy tiers `tf_short`, `tf_deep`, `needle`, `traces`, `trace_replay` |
| `conformance` | ten declarative API-semantics checks |
| `acc-score`, `load-compare` | scoring of a candidate capture against a reference capture |

Long prompts are cut from a corpus built deterministically from the runtime's own package sources and cached on disk, identified by a sha that every capture records. Scoring refuses outright to compare captures built on different corpora, and skips any document whose prompt token ids differ. Every constructed prompt is checked against the server's reported `prompt_tokens`, so a tokenizer that has drifted from the API side fails the run instead of quietly shifting the numbers. Results are written atomically and the tool refuses to overwrite one.

Rebuilding the corpus changes its sha and invalidates every existing reference, so a venv change or a corpus edit ends the comparability of everything captured before it. Treat the corpus as frozen for the life of a study.

## Measure the engine directly

The harness talks to the engine on loopback (port 8010 on this deployment), never through the gateway. There are two reasons for that, and both matter.

Load through the gateway measures the gateway as much as the engine. Its admission limits — thirty-two requests in flight and thirty-two more waiting by default — are host-local policy that `.env` can change without the engine moving at all, and every request pays a proxy hop on the way through. A capture taken through it is not comparable with one taken beside it.

Accuracy needs `prompt_logprobs`, and the gateway refuses `prompt_logprobs` and `echo` with a 400 because one such request can kill the engine. Requesting prompt logprobs sets `skip_reading_prefix_cache`, so the whole prompt is re-prefilled in `max-num-batched-tokens` chunks and the model runner materialises a `chunk x vocab` logits tensor. An ordinary authenticated request with an ~8k-token prompt allocates about 3.6 GiB that way and takes the engine down with `EngineDeadError`; this happened once, unintentionally, at the start of the study, and the same allocation later found the missing gigabyte on a KV pool that had been over-committed by 944 MB past the engine's own anchor.

So teacher forcing is capped client-side at 1,200 prompt tokens and fails fast above it, and deep-context accuracy uses single-position top-k scoring instead of prompt logprobs. That cap is a safety property, not a tuning knob. Do not route the harness through the gateway to get past the refusal, and do not relax the refusal to let the harness through.

## Control the compile hash

`--kv-cache-dtype`, `--attention-backend`, `--max-num-seqs` and `cudagraph_capture_sizes` all enter the torch.compile hash. `--kv-cache-memory-bytes` does not.

A recompile on its own moves teacher-forced accuracy by roughly 0.57-0.58 mean absolute logprob delta and 2.8 pp of argmax flip on the code corpus. Four independent recompiles landed there, and the compile-cache directory hash differs between every backend and dtype pair. A candidate scored against a control captured at a different compile hash therefore reports a compiler offset as precision loss.

Every candidate is scored against a control captured at the same compile hash on the same corpus. Because changing the KV dtype changes the hash, a quantised candidate always needs its own BF16 control at the same backend and pool — the production baseline will not do.

Two consequences follow. First, no quantised candidate can score below the recompile floor, so decision thresholds set below it (mean ≤ 0.15, flip ≤ 0.5 pp) cannot be applied as pass/fail. They discriminate — a candidate at the floor shows no precision loss beyond the recompile, one materially above it does — but they do not gate. Second, the mistake this rule exists to prevent has already happened once: Phase 1 passed its attention-backend gate against a control that also carried a separately-rejected `--max-num-seqs 32`, and against the configuration actually shipped that backend gave no decode gain and cost 1.7% of aggregate throughput. An entire branch of the study was paying a kernel tax the contaminated control had hidden.

## Measure the floor on the backend under test

Same configuration, different boot, on FlashInfer: mean 0.000000 and zero argmax flips on the short tier (40,429 positions), the deep tier (384 positions, 100% exact-zero at every depth including 235k) and the trace tier (20,296 positions). The instrument contributes no noise of its own there.

Phase 1 and 2 reported a 0.26 pp same-config floor on the deep tier and concluded the tier could not resolve small differences. That floor is a FLASH_ATTN property, not a property of the instrument: on the production FLASH_ATTN engine the deep tier is only 10.68% exact-zero against its own reference, while on FlashInfer it is 100.00% exact-zero across boots on both BF16 and FP8. Quoting 0.26 pp as "the instrument's floor" would have written off a tier that is exact on another backend.

Capture the same-config floor on the backend under test before scoring any candidate on it. Anything at or below that floor is not a finding.

## Which load numbers are statistics

One load run yields 16 long and roughly 100 short requests, so tail statistics on the short stream are thin. Measured spreads within a single configuration:

| quantity | spread within one configuration | usable |
| --- | --- | --- |
| aggregate output tok/s | ±0.08% same boot; 0.12% over three runs on two boots | yes — real separations ran 30-50x this |
| long TPOT p50 | ±0.59% same boot and ~2% cross-boot on FLASH_ATTN; 0.20-0.32% on FlashInfer | yes |
| long E2E p50 | ±0.78% | yes |
| long TTFT p95 | ±2.48% | yes — this is the prefill comparison |
| long TTFT p50 | +25% between two repeats at n=16 | direction only |
| short TTFT p50 / p90 | 0.02-1.2% / 1.5-3.3% | yes |
| short TTFT p95 | +21% between two runs on one boot, up to 58% within a configuration | no |
| short E2E p95 | ±3.12% | with care |
| mean cores, from cgroup counters | ±0.19% | yes |

Short TTFT p95 is the 97th-largest of about 102 samples and is set by the two or three requests that happened to land behind a chunked long prefill. It moved 21% between two runs of one configuration on one boot while p50 and p90 barely moved. Phase 2 rejected a candidate on a single-run p95 ratio quoted as "+121%"; pooled over replicate runs at n=279-306, the same candidate has the fastest median first token of anything measured, a small real regression at p90 of +0.074 s, and finishes a short chat 1.30 s sooner end to end because it decodes 5.13 ms/token faster. The rejection was never separable from noise at the precision it was quoted.

Run each configuration at least twice on one boot with different prefix salts and once more on a fresh boot. Pool the samples before quoting a tail, report p50 and p90 with the distribution rather than a lone p95, and compare what a class of request costs end to end rather than what its first token costs.

## What each accuracy tier can detect

| tier | N per capture | resolution | what it reads as |
| --- | --- | --- | --- |
| tier-1 short teacher-forced | 40,429 positions, 40 documents of 1,000 tokens | ±0.17 pp on the flip rate, though positions within a document are correlated, so the effective N is lower | the sensitive discriminator, read against the recompile floor |
| tier-1 deep | 384, four depths x 96 single-position top-20 scorings | ±1.24 pp | gross divergence and mean-logprob shifts at depth; not a 1 pp change |
| tier-1 deep-weighted | 381, three depths from 128k up x 127 positions | ±1.52 pp | the same, weighted to the depths that matter |
| tier-3a reasoning-trace replay | 20,296 positions over 30 `xhigh` traces | ±0.20 pp | agreement on the model's own reasoning text |
| tier-2 needle grid | 100 items, four lengths x five depths x five codes | 0/100 puts a 95% upper bound of 3% on the true failure rate | retrieval correctness against ground truth; a 1-2 pp drop is not resolvable |
| API conformance | ten checks | pass/fail | vendor-equivalent semantics, not a statistic |

Do not report a difference smaller than its tier's half-width as a finding. The FP8 increment over a recompile on the trace tier was +0.18 pp against a ±0.20 pp resolution limit — that is at the limit, not measured.

Among these tiers, only the needle grid and the conformance checks score against ground truth. Every teacher-forced tier measures agreement with a reference engine, so it can say that a change moved the model and cannot say the result is worse — which is why the free-generation correctness tier below had to exist.

## What is measured beyond the tiers, and what is still not

Every tier above is prefill plus one token or a short retrieval answer, so until Phase 4 nothing in this harness scored free generation at all — the load run generates freely, but only times it. Phase 4 closed both gaps, with instruments that are not part of `eval/eval_suite.py`: they live with their per-item records in `/Data2/lzp/vllm-eval-p4/`, and only their design and results are in the repository, as `eval/PHASE4_DESIGN.md` and `eval/PHASE4_RESULTS.md`. Reproducing either tier means using those scripts, and the design was frozen before the first comparison run, which is what lets its power calculation be read as a promise rather than a rationalisation.

**Free-generation reasoning correctness** is now a real instrument. 402 paired items — 252 MMLU-Pro stratified over all fourteen categories, 150 MATH-500 at level 3 or above — greedy, seed-fixed, the same prompts submitted in the same order in every cell, each configuration's answers graded against the gold answer and the two configurations compared by paired McNemar exact. Pairing is what makes it affordable: at the discordance actually observed, 5-7%, it resolves a 3.4 pp regression at 80% power, where two independent rates at the same N would need 8.8 pp. It is the only tier here that both scores against ground truth and exercises free generation, and it needs its own null arm — the same configuration rebooted against itself — because greedy generation at concurrency 32 is chaotic at the ~6% item level no matter what changes.

**Accuracy after long decode residency** is measured by a matched short/long arithmetic ledger. The long variant generates around ten thousand tokens per item and then asks the model to read back values it emitted itself earlier in the same generation — content that exists only in decode-written KV. Steps are scored and paired per (item, step), and the hypothesis is the length by configuration interaction, so a penalty of the same size at both lengths is not a residency effect. At 800 long steps a clean run bounds a deep-decode step-error rate at 0.38 pp one-sided; long-range recall, at 80 items, bounds only at 3.8 pp.

Neither tier can show equivalence — a null result bounds a regression at that resolution and no further — and three things remain genuinely unmeasured.

**Agentic multi-round coding**, which is the dominant real workload, has no gradeable reference data on this host. HumanEval is absent, the SWE-bench path a demo config names does not exist, and LiveCodeBench ships its tests in a form that needs an execution sandbox this host does not have. It enters only through the ungraded greedy-divergence probe, which can say that two configurations produce different text and cannot say which text is better.

**Any change smaller than about 3 pp** in free-generation correctness, or about 2.5 pp against a matched attention kernel. The divergence probe is far more sensitive, but it measures difference rather than damage.

**Sustained production traffic.** The longest continuous load ever run on the current FP8 configuration was the roughly 35-minute correctness sweep at concurrency 32. Multi-hour behaviour under real traffic has never been measured on any configuration.

The CPU KV offload tier no longer exists. It was measured under real multi-hour agent traffic on 2026-08-28, returned 4.59% of the lookups that reached it against 48x write amplification, and was removed; `LOGBOOK.md` carries the evidence.

## Reproducing a cell

1. Write the candidate as its own deployment — with its own model profile when it changes engine arguments — and boot it by setting `VLLM_DEPLOYMENT`, rather than editing the shipped files in place. The study harness does edit in place: it regenerates the shipped configuration as the committed baseline plus the cell's diff, and after a failed boot restores it with `git show HEAD:<file>`. It was written when the engine arguments lived in `config/deployments/<id>.conf`; they now live in the model profile under `config/profiles/`, so point it there before using it again. That replaced a private restore copy which had silently gone stale — by the end it carried a KV pool value already proven to kill the engine, and a precision the service no longer ran — but the safety comes with a dependency: a restore is only ever as good as `HEAD`. The repository must be a real checkout with its history present, and the profile committed at `HEAD` must be one that boots. So commit a profile change before running a cell, never during one, and never leave the working tree holding the only copy of a good profile.
2. Restart `vllm.service` alone and wait for readiness. A compile-cache miss adds several minutes on top of the normal load.
3. Record the boot with `bootfacts` and read it: backend, block size, GPU KV tokens, concurrency at full context, scale warnings, compile time. A cell whose boot facts do not match its intent is not that cell.
4. Capture the matched control at this compile hash if one does not already exist. This is the step that gets skipped.
5. Measure the envelope and the load: a 60-second idle cgroup sample, then two load runs on this boot with different salts, and a third on a fresh boot.
6. Capture accuracy — the short and deep teacher-forced tiers, then the trace replay against the reference traces, then the needle grid — and the conformance checks. Each writes its own file and the tool refuses to overwrite.
7. Score against the matched control, never against a baseline at a different compile hash.
8. Restore the shipped profile and confirm health before leaving.

The captures those comparisons are scored against are data and live outside the repository, under `~/vllm-study-results/<cell>/`; the tier-1 and trace references are the `cell0-base-D` capture. Production has since moved off the BF16 configuration that produced its reasoning traces, so those traces can no longer be regenerated, which is why they alone are committed, as `eval/reference/bf16-reference-traces.json`. The tier-1 logprob captures are not backed up anywhere in this repository.

A cell is expensive but bounded. Measured on this hardware: a boot plus one load run takes about 12 minutes, and one cell carrying two load runs and every accuracy tier took 1 hour 48 minutes, of which the needle grid alone was 67. Spend that on questions the boot facts cannot already answer — a flag the engine disables for this architecture produces a byte-identical engine, and measuring a no-op twice is padding, not evidence.
