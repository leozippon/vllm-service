# Phase 3, part 1 — can FP8 KV be calibrated on this build? (source-verified)

vLLM 0.27.1, `Qwen3_5ForConditionalGeneration` (48 GDN `linear_attention` + 16 `full_attention`),
venv `/Data/public/.venvs/vllm-runtime-py311`.

## `--calculate-kv-scales` is inert on this model

The flag exists and parses (`engine/arg_utils.py:1218`, bare boolean; `EngineArgs.calculate_kv_scales`
resolves to `True`), and it is marked deprecated at `config/cache.py:273-283`
("deprecated and will be removed in v0.19").

It is then unconditionally turned off for this architecture:

- `config/vllm.py:981` (`VllmConfig.__post_init__`) calls `try_verify_and_update_config()`.
- `config/vllm.py:2086-2087`: `if self.model_config.is_hybrid: HybridAttentionMambaModelConfig.verify_and_update_config(self)`.
- `model_executor/models/config.py:450-463`: sets `cache_config.calculate_kv_scales = False` and logs
  *"Disabling calculate_kv_scales for hybrid model ... recurrent state is uninitialized. Using default
  scale of 1.0 instead."* (upstream issue 37554).

`ModelConfig.is_hybrid` (`config/model.py:1764-1772`) is True here on both of its conditions:
the registry entry reports `is_hybrid=True` for `Qwen3_5ForConditionalGeneration` (verified by
inspecting `_ModelInfo`), and `text_config.layer_types` is `Counter({'linear_attention': 48,
'full_attention': 16})`, i.e. not all `"attention"`.

`try_verify_and_update_config` is the only caller of the hybrid hook, it is invoked unconditionally from
`VllmConfig.__post_init__`, and its only early exit is the `config_updated` re-entry guard (which is set
by that same function). `ModelConfig._try_verify_and_update_model_config` is a different, narrower hook
(`verify_and_update_model_config`) and does not reach the hybrid path.

**Consequence: `fp8_e4m3 + --calculate-kv-scales` is byte-for-byte the same engine configuration as
cell 3, and reproduces the `k_scale = v_scale = 1.0` warning exactly.** (Confirmed by boot, below.)

## What it would have done, had it run

`model_executor/layers/attention/attention.py:584-594` — `calc_kv_scales` takes the **amax of the first
real forward pass only**:

```
self._k_scale.copy_(torch.abs(key).max() / self.k_range)
self._v_scale.copy_(torch.abs(value).max() / self.v_range)
...
self.calculate_kv_scales = False        # once, then never again
```

`v1/worker/gpu_model_runner.py:4405-4411` forces `CUDAGraphMode.NONE` for that one step. So this is not
calibration in the usual sense: it is a single-batch amax from whatever request happens to arrive first,
and it is not serialised anywhere. A first request that is short and unrepresentative would set a scale
that makes *later* traffic saturate — arguably worse than the identity scale. It is deprecated for good
reason, and disabled here for a better one.

## The warning itself, and what a scale of 1.0 actually means

`model_executor/layers/quantization/kv_cache.py:96-150`. With a quantised KV dtype and
`calculate_kv_scales == False`, the checkpoint scales are read; both are absent (`< 0.0`), so
`k_scale = v_scale = 1.0` and the warning fires. Note the warning is emitted **only on this branch** —
had `calculate_kv_scales` survived, the whole block would have been skipped and the warning would have
disappeared without any scale being any better. The warning is a proxy for "no checkpoint scales", not a
measurement of harm.

A scale of 1.0 with a *floating-point* 8-bit format is an identity scale, not an unfitted one. e4m3 keeps
~3 mantissa bits at every exponent, so relative precision is scale-invariant inside its normal range.
Measured on this build's torch:

| input | e4m3 result |
|---|---|
| 448 / 449 / 1e3 / 1e5 | 448 (saturating clamp, no inf, no NaN) |
| 0.02 | 0.01953 |
| 0.001 | 0.001953 (subnormal) |
| 1e-4, 1e-6 | 0.0 (flush to zero) |

So the two real failure modes at scale 1.0 are **silent clipping of |k| or |v| above 448** and
**flush-to-zero below ~0.00195**. Both are the failure the decision rule's "no uncalibrated-scale
warning" line exists to prevent. Neither is directly observable from outside the engine on this build,
so they are probed indirectly (needle grid at the deepest lengths, and an `fp8_e5m2` differential —
e5m2 has max 57,344 and cannot clip, at the cost of one mantissa bit).

## The checkpoint ships no KV scales

`model.safetensors.index.json`, 1,606 tensors: **zero** matches for `k_scale`, `v_scale`, `q_scale`,
`prob_scale`, `kv_scale`. The only scale tensors are `weight_scale_inv` (block-FP8 *weight* scales) on
`self_attn.{q,k,v,o}_proj`, `linear_attn.*` and `mlp.*`, plus the `mtp.*` copies.
`config.json.quantization_config` is `{activation_scheme: dynamic, fmt: e4m3, quant_method: fp8,
weight_block_size: [128,128], modules_to_not_convert: [...vision tower...]}` — **no `kv_cache_scheme`**,
so the llm-compressor "honour the checkpoint's FP8 KV scheme" path
(`attention.py:283-290`) never activates either.

## No other calibrated or dynamically-scaled FP8 KV route exists on the FlashInfer path

Per-backend `supported_kv_cache_dtypes` in this build:

| backend | accepted KV dtypes |
|---|---|
| FLASHINFER | auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2, nvfp4 |
| TRITON_ATTN | auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2, int4/int8/fp8_per_token_head |
| FLASH_ATTN | auto, float16, bfloat16, fp8, fp8_e4m3 — but `flash_attn_supports_kv_cache_dtype` (`fa_utils.py:296-315`) requires FA3+SM90 or FA4+SM100, so **no FP8 KV on SM 8.9** |
| TURBOQUANT | turboquant_* only |

- `fp8_per_token_head` — the only dynamically-scaled FP8 KV in the build — is **not** in FlashInfer's list.
  It exists on TRITON_ATTN only, which Phase 2 measured at 6.97 s short TTFT p95. There is no way to
  combine dynamic per-token-head scaling with the FlashInfer kernel.
- `nvfp4` on FlashInfer requires `is_device_capability_family(100)` (Blackwell) plus trtllm attention →
  unavailable on SM 8.9.
- `fp8_inc` is Intel Gaudi/HPU; `fp8_ds_mla` is DeepSeek-V3.2 MLA (`kv_lora_rank == 512`). Neither applies.
- `--kv-cache-dtype-skip-layers` exists (`attention.py:298-320`) and could hold chosen full-attention
  layers at BF16, but it mitigates rather than calibrates, and costs capacity.

## Offline calibration: what it would take (assessed, not attempted)

Loading side is the easy half. `AutoWeightsLoader` (which `Qwen3_5ForConditionalGeneration` uses) does
**not** call `maybe_remap_kv_scale_name`, so the `.k_scale -> .attn.k_scale` remapping never runs; the
tensors would have to be named exactly as the registered parameters,
`model.language_model.layers.<N>.self_attn.attn.{k_scale,v_scale}` (16 full-attention layers, scalar
each; optionally `q_scale`, `prob_scale`). Given correct names, `kv_cache.py:103-140` picks them up with
no code change.

Producing the values is the hard half, and it is not feasible in this environment:
- `llmcompressor` / `modelopt` are **not installed** (only `compressed_tensors`, the loader side), and
  Compute has no route to huggingface.co. (Corrected 2026-08-23: PyPI *is* reachable, so the
  packages could in principle be installed; the remaining obstacles below still stand.)
- A bespoke calibration pass would need the 27.8 B model resident with forward hooks on the 16
  full-attention layers over a representative long-context corpus — i.e. a single-process (TP=1) run that
  does not fit the production service shape, plus GPU time on the two GPUs this service owns.
- The result must then be written into a 30 GB vendor checkpoint (new shard + index edit), which forks it
  from the vendor artefact the repository deliberately runs unmodified.
- Upstream's own position is that this is unreliable for hybrid recurrent models (issue 37554); that
  objection is about the *runtime* pass, but any calibration set for a 262k-context GDN hybrid has the
  same "which regime do you calibrate on" problem.

**Verdict: not worth a future session unless the measured evidence says the identity scale is hurting.**
The rest of this phase measures whether it is.
