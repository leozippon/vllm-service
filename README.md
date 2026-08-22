# vLLM Service

This repository runs one replaceable vLLM deployment behind an authenticated OpenAI-compatible gateway. It deliberately does not route requests among models or run templated model instances.

## Architecture

The default request flow is:

```text
local client -> gateway (127.0.0.1:8011) -> active vLLM deployment (127.0.0.1:8010)
```

`vllm.service` launches the deployment selected by `config/deployments/active`, and `vllm-gateway.service` runs with the Python executable from that deployment's trusted profile. The gateway forwards every `/v1` request to one upstream; `VLLM_GATEWAY_UPSTREAM_URL` can replace its default loopback URL. The gateway authenticates clients and upstream requests with the same `VLLM_API_KEY`, preserves OpenAI-compatible response and streaming behavior, and holds each concurrency slot until its stream ends, disconnects, or fails.

The gateway defaults to four in-flight requests, eight additional waiters, and a 30-second slot timeout. A full queue or timeout returns an OpenAI-style 429 response. These limits remain configurable through the variables documented in `.env.example`.

## Configuration boundaries

| Source | Responsibility |
| --- | --- |
| `config/deployments/active` | The single active deployment ID |
| Selected `config/deployments/<id>.conf` | Trusted shell configuration for the model, Python runtime, vLLM arguments and runtime environment, GPUs, and CPU/NUMA placement |
| `.env` | Local secrets, mirror settings, and optional gateway/watchdog overrides; never commit it |
| `systemd/` | Stable unit names, restart policy, generic cgroup limits, repository paths, and service topology |
| `README.md` | Current installation and operation procedures |
| `LOGBOOK.md` | Append-oriented history, evidence, decisions, and rollbacks |

Create local configuration from the sanitized template:

```console
cp .env.example .env
chmod 600 .env
```

Fill `VLLM_API_KEY` and any required download credentials. Process environment values take precedence over values read from `.env`. Deployment profiles are sourced as shell code and must be treated as trusted repository configuration.

## Add or switch a deployment

1. Copy an existing profile to `config/deployments/<new-id>.conf`.
2. Replace every model- and hardware-specific value in the new profile, including its Python executable, checkpoint, served name, runtime environment, device and NUMA placement, parser flags, context and KV settings, speculative decoding, batching, prefix cache, and offload arguments.
3. Run the syntax and focused validation below.
4. Replace the single line in `config/deployments/active` with `<new-id>`; the selector omits `.conf`.
5. During an approved maintenance window, restart `vllm.service` and `vllm-gateway.service`, then wait for readiness. Update clients to the new served model name if it changed.

A selector change takes effect for each service when that service starts again. The stable unit and gateway names do not change, and there is still only one active deployment.

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

Start or control the stack explicitly:

```console
systemctl --user start vllm.service vllm-gateway.service vllm-watchdog.timer
systemctl --user status vllm.service vllm-gateway.service vllm-watchdog.timer
journalctl --user -u vllm.service -u vllm-gateway.service -f
systemctl --user restart vllm.service
systemctl --user restart vllm-gateway.service
```

The backend unit intentionally does not glob-delete `/dev/shm/vllm_offload_*.mmap`: those names are not scoped to one possible instance, so global cleanup could corrupt a concurrent test. If stale files require manual cleanup, first identify their owner and ensure no relevant vLLM process is using them.

## Client and health checks

Configure local OpenAI-compatible clients with:

- base URL `http://127.0.0.1:8011/v1`;
- the same Bearer key as `VLLM_API_KEY`;
- the served model name from the active deployment profile.

Health endpoints have distinct semantics:

| Endpoint | Meaning |
| --- | --- |
| Gateway `/health/live` | Gateway process liveness; independent of backend loading |
| Gateway `/health/ready` and `/health` | Backend health is currently 200, otherwise 503 |
| Backend `/health` | Direct vLLM health |

The watchdog probes backend health but gateway liveness. Its URLs, unit names, and consecutive-failure thresholds can be overridden through `.env`; defaults are listed in `.env.example`. A failed restart remains a watchdog failure and does not reset the counter.

```console
curl -fsS http://127.0.0.1:8011/health/live
curl -fsS http://127.0.0.1:8011/health/ready
curl -fsS http://127.0.0.1:8010/health
set -a; source .env; set +a
curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8011/v1/models
```

## Focused validation

These checks do not start, stop, or restart a service:

```console
bash -n scripts/run-vllm scripts/run-gateway scripts/health-watchdog config/deployments/*.conf
systemd-analyze --user verify systemd/vllm.service systemd/vllm-gateway.service systemd/vllm-watchdog.service systemd/vllm-watchdog.timer
active=$(<config/deployments/active)
source "config/deployments/$active.conf"
VLLM_API_KEY=test-key "$VLLM_PYTHON" -c "import gateway"
"$VLLM_PYTHON" -m py_compile gateway.py tests/test_gateway.py tests/test_health_watchdog.py
"$VLLM_PYTHON" -m unittest discover -s tests -p 'test_*.py'
```
