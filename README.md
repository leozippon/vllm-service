# vLLM Service

This repository serves models with vLLM in two ways from one set of model profiles: a single shared deployment behind an authenticated OpenAI-compatible gateway, run by systemd, and instances started on demand by a caller who has claimed the GPUs. Both go through the same launcher. It deliberately does not route requests among models.

## Architecture

```text
local client -> gateway (127.0.0.1:8011) -> vLLM engine (127.0.0.1:8010)
```

`vllm.service` starts the deployment named by `config/deployments/active`, and `vllm-gateway.service` runs the gateway in front of it. The deployment names a model profile and declares the port the engine binds; the gateway answers on the next port and proxies to it, so the pair moves together and is stated in one place. Every engine and the gateway run on `.venv/bin/python` inside this repository; git ignores that environment and no checkout reproduces it, because it carries locally tuned kernel files that are not package content.

Everything under `/v1` is forwarded byte for byte, with two deliberate exceptions. The client's `Authorization` header is replaced with the gateway's own upstream credential — both are the same key, and the substitution is what stops a client's header from reaching the engine. And one `reasoning_effort` value is renamed when the active deployment configures an alias, so a model whose chat template spells its top reasoning tier differently still serves an unmodified OpenAI client. A successful stream holds its concurrency slot until the stream ends, the client disconnects, or it fails.

The gateway's limits are a guard rail against a runaway client rather than a throughput regulator: the engine's own scheduler decides throughput, and it is prefix-cache and KV aware where the gateway is blind. Thirty-two requests may be in flight and thirty-two more may wait, for up to five minutes, before the gateway answers with an OpenAI-style 429 carrying `Retry-After`. A non-streaming generation emits nothing until it is complete, so a separate read timeout of one hour bounds the whole generation rather than an idle gap; exceeding it is reported as a 504, and only a genuinely unreachable upstream gives a 502. All of these limits can be overridden in `.env`.

## Context, precision and capacity

The deployment serves the checkpoint's native 262,144-token context. No rope scaling is involved — the full window is what the model was trained for, and preserving it is the one property no tuning decision may trade away.

What that costs in practice is KV cache. On the current configuration the engine reports 1,358,930 GPU KV tokens, which it also states as 5.18x concurrency at 262,144 tokens per request: five simultaneous sessions fit at the maximum context length, and far more at realistic ones, because a session only pays for the tokens it is actually holding. Plan capacity from those two numbers, and re-read them after any change — they move whenever the KV dtype, the pool size, or the model does.

```console
journalctl --user -u vllm.service --no-pager | grep -E 'GPU KV cache size|Maximum concurrency' | tail -2
```

Weights are the vendor's serialized block-FP8 checkpoint, loaded exactly as shipped with no re-quantisation. The KV cache is FP8 `e4m3`, which selects the FlashInfer attention backend. It was adopted on measurement rather than on the memory saving: free-generation reasoning correctness at the model's top reasoning tier, and accuracy after long decode residency, were both compared against the BF16 configuration it replaced, and neither found a regression — the same configuration rebooted against itself moved further than the change did. The boot log carries an uncalibrated `k_scale = v_scale = 1.0` warning that cannot be cleared on this build and checkpoint; it describes an identity scale rather than a missing one, and no behavioural consequence of it was found. The study behind the decision is in `eval/`, and `LOGBOOK.md` carries the exact rollback — it is a line of the model profile and one restart.

## Configuration boundaries

| Source | Responsibility |
| --- | --- |
| `config/profiles/<name>.conf` | One model: checkpoint, served name, task (`generate` or `embed`), default tensor-parallel size and context length, default vLLM arguments, release date, knowledge cutoff, and a one-line note |
| `config/deployments/active` | The single active deployment ID; a `VLLM_DEPLOYMENT` value in the environment overrides it without editing the file |
| Selected `config/deployments/<id>.conf` | The shared deployment: its profile, engine port, GPU set, CPU core budget, the launcher options it adds to the profile's defaults, and the gateway's reasoning-effort alias |
| `scripts/run-vllm` | The launcher, and the one place that fixes the runtime environment, loopback binding, GPU visibility and NUMA placement |
| `.env` | Local secrets and host facts — API key, download credentials and mirror, GPU allowlist — plus optional gateway, readiness and watchdog overrides; never commit it |
| `systemd/` | Stable unit names, restart and start-limit policy, generic cgroup limits, repository paths, and service topology |
| `README.md` | Current architecture, installation, operation, client configuration and validation |
| `LOGBOOK.md` | Dated history: what was tried, what was rejected and why, evidence, decisions and rollbacks |
| `MEASUREMENT.md` | Standing rules for measuring the engine truthfully, and the limits of what those measurements can show |
| `eval/` | The trade-study harness and the study decision documents; results and corpus stay outside the repository |

Create local configuration from the sanitized template:

```console
cp .env.example .env
chmod 600 .env
```

Fill `VLLM_API_KEY` and any required download credentials. Process environment values take precedence over values read from `.env`. Profiles and deployments are sourced as shell code and must be treated as trusted repository configuration. A deployment changes its profile only through launcher options, so every engine argument has one source: the profile's default, or the deployment's stated override of it.

Compute has no direct route to `huggingface.co`, so a checkpoint is fetched through the mirror named by `HF_ENDPOINT` with the token from the same file. Export the file first, then download to the path the profile will name; weights stay on shared storage outside the repository.

```console
set -a; source .env; set +a
hf download organization/model-name --local-dir /path/to/checkpoint --token "$HF_TOKEN"
```

Two placement facts are deliberately kept out of the configuration. The NUMA node and the CPU list are derived from the GPU set at every start, through the host's own topology, and logged as part of the boot record, so no file carries one machine's core numbering to another. The deployment states a core budget and takes the first cores of its GPUs' node, because how much of a shared node a long-running service may take is policy rather than topology; setting both `VLLM_NUMA_NODES` and `VLLM_CPU_AFFINITY` in the deployment bypasses the derivation, which is the answer for a host where those cores are not yours to take. An instance binds to its GPUs' whole node. Either way the GPUs of one start must share a NUMA node. Separately, `VLLM_HOST_GPUS` in `.env`, when set, states which GPUs the machine has given this service, and any start asking for another GPU fails instead of taking a GPU that belongs to another tenant.

Ports follow one convention. The deployment declares the engine port, the gateway answers on the next one, and both health probes derive from it, so a deployment needs exactly one port number. Space deployments ten apart so adding one never collides with an existing pair, and check what is already bound before choosing — this host is shared, and other tenants hold scattered ports through the 8000s and 8100s:

```console
ss -ltn
```

## Install the units

The checked-in units assume this repository remains at `/Data/lzp/vllm-service`. Link them into the user systemd directory so repository updates remain authoritative:

```console
mkdir -p ~/.config/systemd/user
for unit in vllm.service vllm-gateway.service vllm-watchdog.service vllm-watchdog.timer; do
    ln -sfn "/Data/lzp/vllm-service/systemd/$unit" "$HOME/.config/systemd/user/$unit"
done
systemctl --user daemon-reload
systemctl --user enable vllm.service vllm-gateway.service vllm-watchdog.timer
```

## Operating the service

```console
systemctl --user start vllm.service vllm-gateway.service vllm-watchdog.timer
systemctl --user status vllm.service vllm-gateway.service vllm-watchdog.timer
journalctl --user -u vllm.service -u vllm-gateway.service -f
systemctl --user restart vllm.service
systemctl --user restart vllm-gateway.service
```

Restarting the engine blocks for the whole model load, which on this deployment takes three to four minutes with a warm compile cache and longer without one. That is deliberate: a readiness gate runs as `ExecStartPost` and holds the unit in `activating` until the engine answers its health endpoint and is confirmed to reject an unauthenticated request, so nothing downstream ever sees `active` on a server that cannot serve. A load that never finishes fails the start after twenty minutes rather than hanging forever.

Both long-running units have a start limiter, because a five-second restart delay spaces attempts too widely for systemd's own default limiter to trip: five failed starts within an hour for the engine, or within five minutes for the gateway, put the unit into `failed` instead of restarting it forever. Recovery from `failed` is explicit.

```console
systemctl --user reset-failed vllm.service
systemctl --user start vllm.service
```

Each engine start sweeps leaked CPU KV offload regions out of `/dev/shm`. vLLM unlinks such a region only on a clean shutdown path that a terminated engine never reaches, so a deployment that enables CPU KV offloading leaks one region the size of its offload cache on every ordinary restart; eight of them once filled `/dev/shm`, and the host killed a boot as a result. The active deployment enables no offloading and creates none, but the sweep still runs at every start, because a leaked region outlives the configuration that made it — and this host is shared with tenants whose own regions we can neither remove nor rely on. A region is removed only when it is provably this service's and provably held open by no live process — anything unprovable is left alone, and a failed sweep never keeps the engine from serving.

The one thing needing occasional attention is the compiled-kernel cache under `~/.cache/vllm`. The KV dtype, the attention backend, `--max-num-seqs`, the CUDA-graph capture list and the KV offloading size all enter the compile hash, so ordinary operation reuses a single entry indefinitely and the cache does not grow; a tuning sweep, by contrast, leaves an entry of a few hundred megabytes behind for every configuration it booted. Prune it after a tuning campaign: find the entry the running engine loaded, keep that one, and remove its siblings.

```console
journalctl --user -u vllm.service --no-pager | grep -o 'torch_aot_compile/[0-9a-f]*' | tail -1
du -sh ~/.cache/vllm/torch_compile_cache/torch_aot_compile/*
```

Removing the wrong entry costs a recompile on the next start, not correctness.

## Clients and health checks

Configure local OpenAI-compatible clients with:

- base URL `http://127.0.0.1:8011/v1`;
- the same Bearer key as `VLLM_API_KEY`;
- the served model name of the active deployment's profile.

Two API behaviours are worth knowing before a client runs into them. `reasoning_effort: "high"` is accepted: this model's chat template names its top tier `xhigh` and rejects OpenAI's spelling, so the gateway renames it for this deployment, and every other value is forwarded untouched. And `prompt_logprobs`, together with the `echo` that implies it, is refused with a 400 rather than forwarded. Serving one such request makes the engine skip the prefix cache, re-prefill the whole prompt and materialise a chunk-by-vocabulary logits tensor for it, which exhausts device memory and terminates the engine — that has already happened once. Measurement work that genuinely needs prompt logprobs addresses the engine directly on its own port.

Health endpoints have distinct semantics:

| Endpoint | Meaning |
| --- | --- |
| Gateway `/health/live` | Gateway process liveness; independent of backend loading |
| Gateway `/health/ready` and `/health` | 200 while backend health is 200, otherwise 503 |
| Backend `/health` | Direct vLLM health |

The watchdog runs every minute and probes backend health but gateway liveness, so a loading backend never recycles a healthy gateway. Three consecutive failures restart the unit concerned, and a unit that is `activating` is skipped entirely, which is what keeps a cold load from being interrupted by its own watchdog. Any failed probe is reported as a failure even when the restart it triggered succeeded — a restart is an intervention, not health. Its URLs, unit names and thresholds can be overridden through `.env`; the defaults are listed in `.env.example`.

```console
curl -fsS http://127.0.0.1:8011/health/live
curl -fsS http://127.0.0.1:8011/health/ready
curl -fsS http://127.0.0.1:8010/health
set -a; source .env; set +a
curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8011/v1/models
```

## Instances

An instance is one engine started on demand, in the foreground, on GPUs the caller has already claimed. It has no gateway, unit or watchdog; the caller owns its lifetime and its logs.

```console
bash scripts/run-vllm --profile <name> --gpus <i,j,...> --port <port> \
    [--tensor-parallel-size N] [--max-model-len N] [--gpu-memory-utilization F] \
    [--extra-arg <token>]...
```

- The server binds `127.0.0.1:<port>`. The GPU indices are the host's, as `nvidia-smi` numbers them, and become `CUDA_VISIBLE_DEVICES`; name one GPU per tensor-parallel rank.
- Tensor-parallel size and context length default to the profile's. Each `--extra-arg` appends one token to vLLM's arguments after the profile's own, so it can override a profile default: `--extra-arg --max-num-seqs --extra-arg 64`, or `--extra-arg=--max-num-seqs=64`. Options the launcher sets itself — model, served name, host, port, runner, the three options above, and the API key — are refused there.
- `VLLM_API_KEY` in the environment, when set, is the key the server requires. It never appears on a command line, because other users can read those; without it the server is open to every user of the host, and its log says so.
- The launcher replaces itself with the server, so signals reach vLLM directly, and everything is logged to stdout and stderr for the caller to redirect.
- The instance is ready once `GET /health` returns 200 and `GET /v1/models` lists the profile's served name. A `generate` profile answers `/v1/chat/completions` with thinking off unless a request sets `"chat_template_kwargs": {"enable_thinking": true}`; an `embed` profile answers `/v1/embeddings`.

`bash scripts/list-profiles` prints the profiles as a JSON array, one object each with `name`, `served_model_name`, `task`, `model_path`, `weights_bytes` (the checkpoint's safetensors on disk), the default `tensor_parallel_size` and `max_model_len`, `release_date`, `knowledge_cutoff` (null where the model's documentation states none) and a one-line `note`. It is the place to look up what is available; a profile that does not load fails the whole listing.

## Add a model or switch the deployment

1. Copy an existing profile to `config/profiles/<name>.conf` and replace every model-specific value: checkpoint, served name, task, default tensor-parallel size and context length, the vLLM arguments (parsers, KV cache, batching, prefix cache), the release date and knowledge cutoff from the model's own documentation (an empty cutoff when it states none), and the note.
2. To serve it as the shared deployment, write `config/deployments/<new-id>.conf` naming the profile, with its own port pair following the convention above, its GPU set and core budget, any launcher options it adds to the profile's defaults, and the reasoning-effort alias the new model's chat template needs — or none, which forwards `reasoning_effort` untouched.
3. Run the validation below.
4. Replace the single line in `config/deployments/active` with `<new-id>`; the selector omits `.conf`.
5. During an approved maintenance window, restart `vllm.service` and `vllm-gateway.service`, then wait for readiness. Update clients to the new served model name if it changed.

A selector change takes effect for each service when that service next starts. The stable unit and gateway names do not change, and there is still only one active deployment. To boot a candidate deployment without touching the selector — for a trade-study run, say — set `VLLM_DEPLOYMENT` in the environment instead. A profile change reaches the deployment at its next restart too, so a change meant only for instances belongs in the instance's options, not in the profile.

## Validation and measurement

These checks do not start, stop, or restart a service:

```console
bash -n scripts/* config/profiles/*.conf config/deployments/*.conf
systemd-analyze --user verify systemd/*.service systemd/*.timer
bash scripts/list-profiles >/dev/null
VLLM_API_KEY=test-key .venv/bin/python -c "import gateway"
.venv/bin/python -m py_compile gateway.py eval/eval_suite.py tests/*.py
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Expect the whole suite to run — 60 tests at the time of writing — and report `OK`, because a discovery pattern that silently matches nothing reports `OK` too. The tests run every entry script as a real process against a synthetic repository and never start an engine or touch a GPU.

None of this says whether a change is a good idea. Anything that alters engine behaviour — the KV dtype, the attention backend, the pool size, the scheduler flags — has to be measured, and `MEASUREMENT.md` is the standing rule for doing that truthfully: what must be held constant, which numbers are statistics and which are not, what each accuracy tier can resolve, and what none of them measure. The measured results themselves are the phase documents in `eval/`.
