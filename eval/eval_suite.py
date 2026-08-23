#!/usr/bin/env python3
"""Capture and compare local vLLM chat runs (greedy, chosen-token logprobs, needle, perf).

Talks only to loopback vLLM. Reads VLLM_API_KEY from the environment or
/Data/lzp/vllm-service/.env and never prints it. Refuses to overwrite results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8010"
ENV_FILE = Path("/Data/lzp/vllm-service/.env")
SEED = 42
GREEDY_MAX_TOKENS = 64
NEEDLE_MAX_TOKENS = 32
PERF_MAX_TOKENS = 128
LOGPROB_MAX_TOKENS = 1
READ_TIMEOUT = 3600.0
IDLE_WAIT_S = 30.0
TARGET_TOLERANCE = 32

# vLLM 0.27.1 ChatCompletionRequest: chat_template_kwargs.enable_thinking
# is passed into the Qwen chat template (empty <think></think> when false).
THINKING_OFF = {"enable_thinking": False}

SHORT_CASES = [
    ("zh_fact", "中国的首都是哪个城市？只回答城市名。"),
    ("en_fact", "What is the capital of France? Reply with the city name only."),
    ("zh_instr", "只输出两个汉字：春天"),
    ("en_instr", "Reply with exactly these three characters: XYZ"),
    ("code_write", "Write a Python function named add that returns a+b. Output only the function, no markdown."),
    ("code_copy", "Repeat the following code exactly, with no extra text:\ndef ping():\n    return 42"),
    ("en_copy", "Repeat the following sentence exactly, with no extra text:\nSphinx of black quartz, judge my vow."),
    ("zh_copy", "请原样复述下面这句话，不要添加任何其他内容：\n明月几时有，把酒问青天。"),
]

LOGPROB_USER = "Complete the description. Do not add extra words."
LOGPROB_CONT = (
    "The copper kettle sat on the oak table beside three yellow lemons, "
    "and a blue notebook listed every river in order."
)

NEEDLE_CASES = [
    {"name": "21k", "target": 21000, "pos": 0.15, "secret": "NEVA-21K-ORCHID-9173"},
    {"name": "128k", "target": 128000, "pos": 0.50, "secret": "NEVA-128K-MAPLE-4821"},
    {"name": "240k", "target": 240000, "pos": 0.85, "secret": "NEVA-240K-QUARTZ-6058"},
]

PERF_CELLS = [
    ("short_c1", 256, 1),
    ("short_c4", 256, 4),
    ("long21k_c1", 21000, 1),
    ("long21k_c4", 21000, 4),
]

PERF_QUESTION = (
    "\nWrite the word ping repeatedly on one line until the output token limit. "
    "Do not stop early and do not write anything else."
)

FILLER_CANDIDATES = (" ping", " z", " A", " x", "的")
DELTA_TEXT_KEYS = ("content", "reasoning", "reasoning_content")
BEARER_RE = re.compile(r"(?i)bearer\s+\S+")

# Prometheus names from vllm/v1/metrics/loggers.py (0.27.1). Counters are
# scraped with a _total suffix; histograms expose _sum/_count.
METRIC_ALIASES = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv": ("vllm:kv_cache_usage_perc",),
    "prompt_tokens": ("vllm:prompt_tokens_total", "vllm:prompt_tokens"),
    "generation_tokens": ("vllm:generation_tokens_total", "vllm:generation_tokens"),
    "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    "queue_sum": ("vllm:request_queue_time_seconds_sum",),
    "queue_count": ("vllm:request_queue_time_seconds_count",),
    "ttft_sum": ("vllm:time_to_first_token_seconds_sum",),
    "ttft_count": ("vllm:time_to_first_token_seconds_count",),
    "e2e_sum": ("vllm:e2e_request_latency_seconds_sum",),
    "e2e_count": ("vllm:e2e_request_latency_seconds_count",),
    "itl_sum": ("vllm:inter_token_latency_seconds_sum",),
    "itl_count": ("vllm:inter_token_latency_seconds_count",),
}


class EvalError(Exception):
    pass


def redact(text: str) -> str:
    return BEARER_RE.sub("Bearer <redacted>", text)


def load_api_key() -> str:
    direct = os.environ.get("VLLM_API_KEY", "").strip()
    if direct:
        return direct
    if not ENV_FILE.is_file():
        raise EvalError("VLLM_API_KEY is not set and .env is missing")
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if key == "VLLM_API_KEY":
            got = value.strip().strip("'\"")
            if got:
                return got
    raise EvalError("VLLM_API_KEY is not set")


def atomic_write_json(path: Path, obj: object) -> None:
    if path.exists():
        raise EvalError(f"refusing to overwrite existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def parse_prom(text: str) -> tuple[dict[str, float], dict[str, float]]:
    sums: dict[str, float] = {}
    maxes: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if "{" in line:
                name, rest = line.split("{", 1)
                _, rhs = rest.split("}", 1)
                val_s = rhs.strip().split()[0]
            else:
                name, val_s = line.split()[:2]
        except (ValueError, IndexError):
            continue
        try:
            val = float(val_s)
        except ValueError:
            continue
        if not math.isfinite(val):
            continue
        name = name.strip()
        sums[name] = sums.get(name, 0.0) + val
        maxes[name] = val if name not in maxes else max(maxes[name], val)
    return sums, maxes


def _pick(table: dict[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in table:
            return table[name]
    return None


def metric_bundle(sums: dict[str, float], maxes: dict[str, float]) -> dict:
    out = {}
    for key, names in METRIC_ALIASES.items():
        table = maxes if key == "kv" else sums
        out[key] = _pick(table, names)
    return out


def metrics_delta(before: dict, after: dict) -> dict:
    out = {}
    for key in (
        "prompt_tokens",
        "generation_tokens",
        "preemptions",
        "queue_sum",
        "queue_count",
        "ttft_sum",
        "ttft_count",
        "e2e_sum",
        "e2e_count",
        "itl_sum",
        "itl_count",
    ):
        a, b = before.get(key), after.get(key)
        out[key] = None if a is None or b is None else b - a
    return out


def delta_text(delta: object) -> str:
    """First visible stream text: DeltaMessage.content/reasoning, plus extra reasoning_content."""
    if not isinstance(delta, dict):
        return ""
    parts = []
    for key in DELTA_TEXT_KEYS:
        val = delta.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    return "".join(parts)


def usage_dict(usage: object) -> dict:
    if not isinstance(usage, dict):
        raise EvalError("response missing usage object")
    try:
        prompt = int(usage["prompt_tokens"])
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalError("usage missing prompt_tokens") from exc
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def check_prompt_tokens(constructed: int, usage: dict) -> None:
    prompt = usage["prompt_tokens"]
    if abs(prompt - constructed) > TARGET_TOLERANCE:
        raise EvalError(
            f"server prompt_tokens={prompt} vs constructed={constructed} "
            f"(tolerance {TARGET_TOLERANCE})"
        )


def client_tpot_s(e2e_s: float, ttft_s: float, completion_tokens: int) -> float:
    """Mean client inter-token time: (e2e - ttft) / (completion_tokens - 1)."""
    if completion_tokens <= 1:
        raise EvalError(
            f"completion_tokens={completion_tokens}<=1; cannot compute client TPOT/ITL"
        )
    if e2e_s < ttft_s:
        raise EvalError("e2e < ttft; cannot compute client TPOT/ITL")
    return (e2e_s - ttft_s) / (completion_tokens - 1)


def check_perf_generation(usage: dict, finish: str | None) -> None:
    completion = usage["completion_tokens"]
    if completion != PERF_MAX_TOKENS:
        raise EvalError(
            f"perf completion_tokens={completion} != {PERF_MAX_TOKENS}"
        )
    if finish != "length":
        raise EvalError(f"perf finish_reason={finish!r} is not 'length'")


def _logprob_float(entry: object) -> float:
    if isinstance(entry, dict):
        if "logprob" not in entry:
            raise EvalError("prompt_logprobs value missing logprob")
        return float(entry["logprob"])
    if isinstance(entry, (int, float)) and math.isfinite(float(entry)):
        return float(entry)
    raise EvalError(f"unexpected prompt_logprobs value type {type(entry).__name__}")


def chosen_logprobs(prompt_logprobs: object, token_ids: list[int], start: int) -> list[dict]:
    """Bind chosen-token logprobs using server prompt_token_ids (not rank==1).

    vLLM ChatCompletionResponse.prompt_logprobs is list[dict[int, Logprob] | None].
    Position 0 is None. JSON object keys are strings. Logprob has logprob/rank/decoded_token.
    """
    if not isinstance(prompt_logprobs, list):
        raise EvalError("prompt_logprobs missing or not a list")
    if len(prompt_logprobs) != len(token_ids):
        raise EvalError(
            f"prompt_logprobs length {len(prompt_logprobs)} != prompt_token_ids {len(token_ids)}"
        )
    if start <= 0 or start >= len(token_ids):
        raise EvalError(f"continuation start {start} is not inside the prompt")
    out = []
    for offset, token_id in enumerate(token_ids[start:]):
        pos = start + offset
        slot = prompt_logprobs[pos]
        if slot is None:
            raise EvalError(f"prompt_logprobs[{pos}] is null (cannot score continuation)")
        if not isinstance(slot, dict) or not slot:
            raise EvalError(f"prompt_logprobs[{pos}] is not a token dict")
        entry = None
        for key in (token_id, str(token_id)):
            if key in slot:
                entry = slot[key]
                break
        if entry is None:
            raise EvalError(f"chosen token id {token_id} missing at prompt_logprobs[{pos}]")
        rec = {"index": pos, "token_id": token_id, "logprob": _logprob_float(entry)}
        if isinstance(entry, dict):
            if entry.get("rank") is not None:
                rec["rank"] = entry["rank"]
            if entry.get("decoded_token") is not None:
                rec["decoded_token"] = entry["decoded_token"]
        out.append(rec)
    if not out:
        raise EvalError("no continuation logprobs extracted")
    return out


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)


def chat_raw(tok, messages, *, add_generation_prompt: bool, continue_final_message: bool = False) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    if continue_final_message:
        kwargs["continue_final_message"] = True
        kwargs["add_generation_prompt"] = False
    return tok.apply_chat_template(messages, **kwargs)


def chat_ids(tok, messages, **kwargs) -> tuple[list[int], str]:
    raw = chat_raw(tok, messages, **kwargs)
    return tok.encode(raw, add_special_tokens=False), raw


def find_filler_unit(tok) -> tuple[str, int]:
    for unit in FILLER_CANDIDATES:
        ids = tok.encode(unit, add_special_tokens=False)
        if not ids:
            continue
        if tok.encode(unit * 8, add_special_tokens=False) == ids * 8:
            return unit, len(ids)
    raise EvalError("no linearly repeatable 1-token filler unit")


def build_user_text(
    tok,
    target: int,
    *,
    extra_prefix: str = "",
    needle: str = "",
    question: str = "\nReply with the single word OK.",
    pos: float = 0.0,
) -> tuple[str, int, dict]:
    unit, unit_n = find_filler_unit(tok)

    def assemble(left: int, right: int) -> str:
        return f"{extra_prefix}{unit * left}{needle}{unit * right}{question}"

    def ntokens(text: str) -> int:
        ids, _ = chat_ids(tok, [{"role": "user", "content": text}], add_generation_prompt=True)
        return len(ids)

    base = ntokens(assemble(0, 0))
    need = target - base
    if need < unit_n:
        raise EvalError(f"target {target} smaller than template overhead {base}")
    n_units = need // unit_n
    left = int(round(n_units * pos))
    left = max(0, min(n_units, left))
    right = n_units - left
    text = assemble(left, right)
    n = ntokens(text)
    for _ in range(8):
        if abs(n - target) <= TARGET_TOLERANCE:
            break
        diff_units = int(round((target - n) / unit_n))
        if diff_units == 0:
            diff_units = 1 if n < target else -1
        if right + diff_units >= 0:
            right += diff_units
        elif left + diff_units >= 0:
            left += diff_units
        else:
            break
        text = assemble(left, right)
        n = ntokens(text)
    if abs(n - target) > TARGET_TOLERANCE:
        raise EvalError(f"constructed {n} tokens, target {target}")
    return text, n, {
        "unit": unit,
        "unit_tokens": unit_n,
        "left_units": left,
        "right_units": right,
        "overhead_tokens": base,
        "needle_pos_frac": pos,
        "constructed_tokens": n,
        "target_tokens": target,
    }


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if not ys:
        raise EvalError("percentile of empty list")
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ys[int(k)]
    return ys[lo] * (hi - k) + ys[hi] * (k - lo)


def pct_diff(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None or base == 0:
        return None
    return (cand - base) / abs(base) * 100.0


def auth_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def http_error(resp: httpx.Response) -> EvalError:
    body = redact(resp.text[:500])
    path = resp.request.url.path
    return EvalError(f"HTTP {resp.status_code} {resp.request.method} {path}: {body}")


def fetch_metrics(client: httpx.Client) -> dict:
    resp = client.get("/metrics")
    if resp.status_code != 200:
        raise http_error(resp)
    return metric_bundle(*parse_prom(resp.text))


def wait_idle(client: httpx.Client, timeout: float = IDLE_WAIT_S) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fetch_metrics(client)
        running, waiting = last.get("running"), last.get("waiting")
        if running is None or waiting is None:
            raise EvalError("metrics missing vllm:num_requests_running/waiting; cannot confirm idle")
        if running == 0 and waiting == 0:
            return last
        time.sleep(0.5)
    raise EvalError(f"server not idle after {timeout:.0f}s: running={last.get('running')} waiting={last.get('waiting')}")


def confirm_ready(client: httpx.Client) -> tuple[str, list[str], dict]:
    health = client.get("/health")
    if health.status_code != 200:
        raise http_error(health)
    models = client.get("/v1/models")
    if models.status_code != 200:
        raise http_error(models)
    try:
        payload = models.json()
        data = payload["data"]
        ids = [item["id"] for item in data]
    except Exception as exc:
        raise EvalError("/v1/models returned an unexpected body") from exc
    if not ids:
        raise EvalError("/v1/models returned no models")
    idle = wait_idle(client)
    return ids[0], ids, idle


def chat_body(model: str, messages: list, extra: dict) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": SEED,
        "chat_template_kwargs": dict(THINKING_OFF),
        "n": 1,
    }
    body.update(extra)
    return body


def chat_nonstream(client: httpx.Client, body: dict) -> dict:
    resp = client.post("/v1/chat/completions", json=body)
    if resp.status_code != 200:
        raise http_error(resp)
    try:
        return resp.json()
    except Exception as exc:
        raise EvalError("non-JSON chat completion response") from exc


def message_fields(payload: dict) -> tuple[str, str | None, str | None, dict]:
    try:
        choice = payload["choices"][0]
        msg = choice["message"]
    except Exception as exc:
        raise EvalError("chat response missing choices[0].message") from exc
    content = msg.get("content") if isinstance(msg, dict) else None
    reasoning = None
    if isinstance(msg, dict):
        reasoning = msg.get("reasoning")
        if reasoning is None:
            reasoning = msg.get("reasoning_content")
    finish = choice.get("finish_reason")
    return (content or "") if isinstance(content, str) or content is None else str(content), (
        reasoning if isinstance(reasoning, str) else None
    ), finish, usage_dict(payload.get("usage"))


def chat_stream(client: httpx.Client, body: dict) -> dict:
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    start = time.perf_counter()
    ttft = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish = None
    usage = None
    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        if resp.status_code != 200:
            resp.read()
            raise http_error(resp)
        for line in resp.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                raise EvalError("invalid SSE JSON chunk") from exc
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            text = delta_text(delta)
            if text and ttft is None:
                ttft = time.perf_counter() - start
            if isinstance(delta, dict):
                if isinstance(delta.get("content"), str) and delta["content"]:
                    content_parts.append(delta["content"])
                for key in ("reasoning", "reasoning_content"):
                    val = delta.get(key)
                    if isinstance(val, str) and val:
                        reasoning_parts.append(val)
    e2e = time.perf_counter() - start
    if ttft is None:
        raise EvalError("stream had no content/reasoning/reasoning_content delta")
    if e2e <= 0:
        raise EvalError("non-positive e2e duration")
    used = usage_dict(usage)
    completion = used["completion_tokens"]
    tpot = client_tpot_s(e2e, ttft, completion)
    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts) or None,
        "finish_reason": finish,
        "usage": used,
        "ttft_s": ttft,
        "e2e_s": e2e,
        "output_tok_s": completion / e2e,  # full-request rate, not decode
        "client_tpot_s": tpot,
    }


def sample_peaks(client: httpx.Client, stop: threading.Event, box: dict) -> None:
    while True:
        try:
            snap = fetch_metrics(client)
        except Exception as exc:
            box["error"] = str(exc)
            return
        for key in ("running", "waiting", "kv"):
            val = snap.get(key)
            if val is None:
                continue
            cur = float(val)
            prev = box.get(key)
            box[key] = cur if prev is None else max(float(prev), cur)
        if stop.wait(0.05):
            return


def run_perf_cell(client: httpx.Client, model: str, tok, target: int, conc: int) -> dict:
    wait_idle(client)
    jobs = []
    for _ in range(conc):
        salt = uuid.uuid4().hex
        text, constructed, meta = build_user_text(
            tok,
            target,
            extra_prefix=f"[{salt}]\nRead the document.\n",
            question=PERF_QUESTION,
        )
        jobs.append((text, constructed, meta))
    stop = threading.Event()
    start_gate = threading.Barrier(conc)
    peaks = {"running": None, "waiting": None, "kv": None}
    results: list[dict | None] = [None] * conc
    errors: list[str | None] = [None] * conc
    headers = dict(client.headers)
    timeout = client.timeout
    base_url = str(client.base_url)

    def work(i: int) -> None:
        text, constructed, meta = jobs[i]
        try:
            body = chat_body(
                model,
                [{"role": "user", "content": text}],
                {"min_tokens": PERF_MAX_TOKENS, "max_tokens": PERF_MAX_TOKENS},
            )
            with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as inner:
                start_gate.wait()
                rec = chat_stream(inner, body)
            check_prompt_tokens(constructed, rec["usage"])
            check_perf_generation(rec["usage"], rec.get("finish_reason"))
            rec["constructed_tokens"] = constructed
            rec["construct"] = {k: meta[k] for k in ("target_tokens", "constructed_tokens", "unit")}
            rec.pop("content", None)
            rec.pop("reasoning", None)
            results[i] = rec
        except Exception as exc:
            try:
                start_gate.abort()
            except Exception:
                pass
            errors[i] = str(exc) or type(exc).__name__

    with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as mclient:
        before = fetch_metrics(mclient)
        poller = threading.Thread(target=sample_peaks, args=(mclient, stop, peaks), daemon=True)
        poller.start()
        threads = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
        wall0 = time.perf_counter()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall = time.perf_counter() - wall0
        stop.set()
        poller.join()
        after = fetch_metrics(mclient)
    if peaks.get("error"):
        raise EvalError(f"metrics poll failed: {peaks['error']}")
    failed = [err for err in errors if err]
    if failed:
        raise EvalError(f"perf conc={conc} target={target} failed: {failed[0]}")
    rows = [row for row in results if row is not None]
    completion_sum = sum(row["usage"]["completion_tokens"] for row in rows)
    dlt = metrics_delta(before, after)
    itl_sum, itl_n = dlt.get("itl_sum"), dlt.get("itl_count")
    server_itl = (
        itl_sum / itl_n
        if itl_sum is not None and itl_n is not None and itl_n > 0
        else None
    )
    return {
        "target_input_tokens": target,
        "concurrency": conc,
        "wall_s": wall,
        "requests": rows,
        "aggregate": {
            "min_ttft_s": min(row["ttft_s"] for row in rows),
            "max_e2e_s": max(row["e2e_s"] for row in rows),
            "mean_client_tpot_s": sum(row["client_tpot_s"] for row in rows) / len(rows),
            "mean_output_tok_s": sum(row["output_tok_s"] for row in rows) / len(rows),
            "completion_tokens": completion_sum,
            "wall_output_tok_s": completion_sum / wall if wall > 0 else None,
            "server_itl_s": server_itl,
        },
        "metrics_delta": dlt,
        "peaks": {
            "running": peaks.get("running"),
            "waiting": peaks.get("waiting"),
            "kv_cache_usage_perc": peaks.get("kv"),
        },
    }


def extract_continuation(tok, prompt_token_ids: list[int]) -> int:
    messages = [
        {"role": "user", "content": LOGPROB_USER},
        {"role": "assistant", "content": LOGPROB_CONT},
    ]
    local_ids, raw = chat_ids(
        tok,
        messages,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    if local_ids != prompt_token_ids:
        raise EvalError(
            "prompt_token_ids do not match local chat template encoding; "
            "chosen-token logprobs would not be comparable across configs"
        )
    if not raw.endswith(LOGPROB_CONT):
        raise EvalError("rendered prompt does not end with the fixed continuation")
    prefix_ids = tok.encode(raw[: -len(LOGPROB_CONT)], add_special_tokens=False)
    if prompt_token_ids[: len(prefix_ids)] != prefix_ids:
        raise EvalError("BPE merge across continuation boundary; refusing unstable alignment")
    return len(prefix_ids)


def cmd_capture(args: argparse.Namespace) -> int:
    out_path = Path(args.output)
    if out_path.exists():
        print(f"refusing to overwrite existing result: {out_path}", file=sys.stderr)
        return 1
    errors: list[str] = []
    result = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "name": args.name,
        "served_model": None,
        "params": {
            "base_url": args.base_url,
            "temperature": 0,
            "seed": SEED,
            "enable_thinking": False,
            "greedy_max_tokens": GREEDY_MAX_TOKENS,
            "needle_max_tokens": NEEDLE_MAX_TOKENS,
            "perf_max_tokens": PERF_MAX_TOKENS,
            "prompt_logprobs": 1,
            "return_token_ids": True,
            "skip_long": bool(args.skip_long),
            "skip_performance": bool(args.skip_performance),
            "short_case_ids": [cid for cid, _ in SHORT_CASES],
            "needle_cases": [
                {k: c[k] for k in ("name", "target", "pos", "secret")} for c in NEEDLE_CASES
            ],
            "perf_cells": [
                {"id": cid, "target_input_tokens": tgt, "concurrency": conc}
                for cid, tgt, conc in PERF_CELLS
            ],
        },
        "ok": False,
        "errors": errors,
        "greedy": [],
        "prompt_logprobs": None,
        "needles": [],
        "performance": None,
    }

    def fail(msg: str) -> None:
        errors.append(msg)
        print(msg, file=sys.stderr, flush=True)

    try:
        key = load_api_key()
        tok = load_tokenizer(args.tokenizer_path)
    except Exception as exc:
        fail(str(exc))
        try:
            atomic_write_json(out_path, result)
        except EvalError as write_exc:
            print(str(write_exc), file=sys.stderr)
        return 1

    timeout = httpx.Timeout(connect=10.0, read=READ_TIMEOUT, write=120.0, pool=30.0)
    try:
        with httpx.Client(base_url=args.base_url, headers=auth_headers(key), timeout=timeout) as client:
            try:
                served, model_ids, idle = confirm_ready(client)
            except Exception as exc:
                fail(str(exc))
                atomic_write_json(out_path, result)
                return 1
            result["served_model"] = served
            result["params"]["model_ids"] = model_ids
            result["idle"] = {
                "running": idle.get("running"),
                "waiting": idle.get("waiting"),
                "kv_cache_usage_perc": idle.get("kv"),
            }

            for cid, prompt in SHORT_CASES:
                print(f"greedy {cid}", file=sys.stderr, flush=True)
                try:
                    payload = chat_nonstream(
                        client,
                        chat_body(
                            served,
                            [{"role": "user", "content": prompt}],
                            {"max_tokens": GREEDY_MAX_TOKENS},
                        ),
                    )
                    content, reasoning, finish, usage = message_fields(payload)
                    result["greedy"].append(
                        {
                            "id": cid,
                            "prompt": prompt,
                            "content": content,
                            "reasoning": reasoning,
                            "finish_reason": finish,
                            "usage": usage,
                        }
                    )
                except Exception as exc:
                    fail(f"greedy {cid}: {exc}")
                    result["greedy"].append({"id": cid, "prompt": prompt, "error": str(exc)})

            print("prompt_logprobs", file=sys.stderr, flush=True)
            try:
                wait_idle(client)
                payload = chat_nonstream(
                    client,
                    chat_body(
                        served,
                        [
                            {"role": "user", "content": LOGPROB_USER},
                            {"role": "assistant", "content": LOGPROB_CONT},
                        ],
                        {
                            "max_tokens": LOGPROB_MAX_TOKENS,
                            "prompt_logprobs": 1,
                            "return_token_ids": True,
                            "add_generation_prompt": False,
                            "continue_final_message": True,
                            "stream": False,
                        },
                    ),
                )
                token_ids = payload.get("prompt_token_ids")
                logprobs = payload.get("prompt_logprobs")
                if not isinstance(token_ids, list) or not token_ids:
                    raise EvalError(
                        "prompt_token_ids missing; set return_token_ids=true on vLLM 0.27.1 "
                        "chat completions or chosen-token logprobs cannot be bound"
                    )
                token_ids = [int(t) for t in token_ids]
                start = extract_continuation(tok, token_ids)
                chosen = chosen_logprobs(logprobs, token_ids, start)
                result["prompt_logprobs"] = {
                    "user": LOGPROB_USER,
                    "continuation": LOGPROB_CONT,
                    "prompt_token_count": len(token_ids),
                    "continuation_start": start,
                    "tokens": chosen,
                }
            except Exception as exc:
                fail(f"prompt_logprobs: {exc}")
                result["prompt_logprobs"] = {"error": str(exc)}

            greedy_failed = any(
                isinstance(row, dict) and "error" in row for row in result["greedy"]
            )
            pl = result["prompt_logprobs"]
            logprob_failed = (
                not isinstance(pl, dict) or "error" in pl or "tokens" not in pl
            )
            if greedy_failed or logprob_failed:
                atomic_write_json(out_path, result)
                return 1

            if not args.skip_long:
                for case in NEEDLE_CASES:
                    print(f"needle {case['name']}", file=sys.stderr, flush=True)
                    rec = {
                        "name": case["name"],
                        "target": case["target"],
                        "pos": case["pos"],
                        "secret": case["secret"],
                    }
                    try:
                        wait_idle(client)
                        needle = f"\n[SECRET] The secret code is {case['secret']}.\n"
                        text, constructed, meta = build_user_text(
                            tok,
                            case["target"],
                            extra_prefix="Read the document below. A secret code appears exactly once.\n",
                            needle=needle,
                            question="\nWhat is the secret code? Reply with only the secret code.",
                            pos=case["pos"],
                        )
                        rec["constructed_tokens"] = constructed
                        rec["construct"] = meta
                        payload = chat_nonstream(
                            client,
                            chat_body(
                                served,
                                [{"role": "user", "content": text}],
                                {"max_tokens": NEEDLE_MAX_TOKENS},
                            ),
                        )
                        content, reasoning, finish, usage = message_fields(payload)
                        check_prompt_tokens(constructed, usage)
                        answer = content or ""
                        hay = " ".join(p for p in (content, reasoning) if p)
                        rec.update(
                            {
                                "prompt_tokens": usage["prompt_tokens"],
                                "answer": answer,
                                "reasoning": reasoning,
                                "finish_reason": finish,
                                "usage": usage,
                                "hit": case["secret"] in hay,
                            }
                        )
                    except Exception as exc:
                        fail(f"needle {case['name']}: {exc}")
                        rec["error"] = str(exc)
                    result["needles"].append(rec)
            else:
                result["needles"] = [{"skipped": True}]

            if not args.skip_performance:
                print("performance warmup", file=sys.stderr, flush=True)
                try:
                    wait_idle(client)
                    warmup_targets = (256, 2048)
                    for target in warmup_targets:
                        salt = uuid.uuid4().hex
                        text, constructed, _ = build_user_text(
                            tok,
                            target,
                            extra_prefix=f"[{salt}]\nWarm up decode kernels.\n",
                            question=PERF_QUESTION,
                        )
                        payload = chat_nonstream(
                            client,
                            chat_body(
                                served,
                                [{"role": "user", "content": text}],
                                {
                                    "min_tokens": PERF_MAX_TOKENS,
                                    "max_tokens": PERF_MAX_TOKENS,
                                },
                            ),
                        )
                        _, _, finish, usage = message_fields(payload)
                        check_prompt_tokens(constructed, usage)
                        check_perf_generation(usage, finish)
                    wait_idle(client)
                    perf = {}
                    for cid, tgt, conc in PERF_CELLS:
                        print(f"perf {cid}", file=sys.stderr, flush=True)
                        try:
                            perf[cid] = run_perf_cell(client, served, tok, tgt, conc)
                        except Exception as exc:
                            fail(f"perf {cid}: {exc}")
                            perf[cid] = {"error": str(exc)}
                    result["performance"] = {
                        "warmup_input_tokens": list(warmup_targets),
                        "warmup_output_tokens": PERF_MAX_TOKENS,
                        "cells": perf,
                    }
                except Exception as exc:
                    fail(f"performance: {exc}")
                    result["performance"] = {"error": str(exc)}
            else:
                result["performance"] = {"skipped": True}
    except Exception as exc:
        fail(str(exc))

    result["ok"] = not errors
    try:
        atomic_write_json(out_path, result)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0 if result["ok"] else 1


def _load_capture(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvalError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvalError(f"{path} is not a capture object")
    return data


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        base = _load_capture(Path(args.base))
        cand = _load_capture(Path(args.candidate))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    lines = [
        f"base: {base.get('name')}  model={base.get('served_model')}  ok={base.get('ok')}",
        f"cand: {cand.get('name')}  model={cand.get('served_model')}  ok={cand.get('ok')}",
        "",
    ]
    exit_code = 0
    if not base.get("ok") or not cand.get("ok"):
        exit_code = 1
        lines.append("note: at least one capture has ok=false")

    bg = {row.get("id"): row for row in base.get("greedy") or [] if isinstance(row, dict)}
    cg = {row.get("id"): row for row in cand.get("greedy") or [] if isinstance(row, dict)}
    ids = [cid for cid, _ in SHORT_CASES]
    if any(i not in bg or i not in cg for i in ids):
        lines.append("greedy: missing cases; cannot compute exact-match rate")
        exit_code = 1
    else:
        matches = 0
        mismatches = []
        for cid in ids:
            b, c = bg[cid], cg[cid]
            if "error" in b or "error" in c:
                mismatches.append(cid)
                exit_code = 1
                continue
            if b.get("content") == c.get("content"):
                matches += 1
            else:
                mismatches.append(cid)
        lines.append(f"greedy exact-match: {matches}/{len(ids)} ({100.0 * matches / len(ids):.1f}%)")
        if mismatches:
            lines.append("  mismatches: " + ", ".join(mismatches))

    bp, cp = base.get("prompt_logprobs"), cand.get("prompt_logprobs")
    if not isinstance(bp, dict) or not isinstance(cp, dict) or "tokens" not in bp or "tokens" not in cp:
        lines.append("chosen-token logprob: missing; cannot compare")
        exit_code = 1
    else:
        bt, ct = bp["tokens"], cp["tokens"]
        if len(bt) != len(ct):
            lines.append(f"chosen-token logprob: length mismatch {len(bt)} vs {len(ct)}")
            exit_code = 1
        else:
            diffs = []
            for i, (a, b) in enumerate(zip(bt, ct)):
                if a.get("token_id") != b.get("token_id"):
                    lines.append(f"chosen-token logprob: token_id mismatch at {i}")
                    exit_code = 1
                    diffs = None
                    break
                diffs.append(abs(float(a["logprob"]) - float(b["logprob"])))
            if diffs:
                mean = sum(diffs) / len(diffs)
                p95 = percentile(diffs, 95)
                lines.append(
                    f"chosen-token logprob |Δ|: mean={mean:.6f} p95={p95:.6f} n={len(diffs)}"
                )

    def needle_map(capture: dict) -> dict:
        out = {}
        for row in capture.get("needles") or []:
            if isinstance(row, dict) and row.get("name"):
                out[row["name"]] = row
        return out

    bn, cn = needle_map(base), needle_map(cand)
    if base.get("params", {}).get("skip_long") or cand.get("params", {}).get("skip_long"):
        lines.append("needles: skipped in at least one capture")
    elif not bn or not cn:
        lines.append("needles: missing")
        exit_code = 1
    else:
        lines.append("needles:")
        for case in NEEDLE_CASES:
            name = case["name"]
            br, cr = bn.get(name, {}), cn.get(name, {})
            if "error" in br or "error" in cr or "hit" not in br or "hit" not in cr:
                lines.append(f"  {name}: incomparable")
                exit_code = 1
            else:
                lines.append(
                    f"  {name}: base_hit={br['hit']} cand_hit={cr['hit']} "
                    f"base_prompt_tokens={br.get('prompt_tokens')} "
                    f"cand_prompt_tokens={cr.get('prompt_tokens')}"
                )

    def cells(capture: dict) -> dict:
        perf = capture.get("performance") or {}
        return perf.get("cells") or {}

    bc, cc = cells(base), cells(cand)
    if base.get("params", {}).get("skip_performance") or cand.get("params", {}).get("skip_performance"):
        lines.append("performance: skipped in at least one capture")
    elif not bc or not cc:
        lines.append("performance: missing")
        exit_code = 1
    else:
        lines.append("performance (cand vs base, percent = (cand-base)/|base|):")
        for cid, _, _ in PERF_CELLS:
            br, cr = bc.get(cid) or {}, cc.get(cid) or {}
            if "error" in br or "error" in cr or "aggregate" not in br or "aggregate" not in cr:
                lines.append(f"  {cid}: incomparable")
                exit_code = 1
                continue
            ba, ca = br["aggregate"], cr["aggregate"]
            mapping = (
                ("min_ttft_s", ba.get("min_ttft_s"), ca.get("min_ttft_s")),
                ("max_e2e_s", ba.get("max_e2e_s"), ca.get("max_e2e_s")),
                ("mean_client_tpot_s", ba.get("mean_client_tpot_s"), ca.get("mean_client_tpot_s")),
                ("server_itl_s", ba.get("server_itl_s"), ca.get("server_itl_s")),
                ("wall_output_tok_s", ba.get("wall_output_tok_s"), ca.get("wall_output_tok_s")),
            )
            parts = []
            for label, bv, cv in mapping:
                d = pct_diff(bv, cv)
                ds = "n/a" if d is None else f"{d:+.2f}%"
                parts.append(f"{label} base={bv} cand={cv} {ds}")
            lines.append(f"  {cid}: " + "; ".join(parts))

    lines.append("")
    lines.append("No absolute quality judgement is implied.")
    print("\n".join(lines))
    return exit_code


def cmd_self_check(args: argparse.Namespace) -> int:
    # parse_prom
    sample = """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m",engine="0"} 0.0
vllm:num_requests_waiting{model_name="m",engine="0"} 0.0
vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.25
vllm:prompt_tokens_total{model_name="m",engine="0"} 10
vllm:generation_tokens_total{model_name="m",engine="0"} 3
vllm:num_preemptions_total{model_name="m",engine="0"} 1
vllm:time_to_first_token_seconds_sum{model_name="m",engine="0"} 1.5
vllm:time_to_first_token_seconds_count{model_name="m",engine="0"} 3
vllm:inter_token_latency_seconds_sum{model_name="m",engine="0"} 0.4
vllm:inter_token_latency_seconds_count{model_name="m",engine="0"} 8
"""
    bundle = metric_bundle(*parse_prom(sample))
    assert bundle["running"] == 0.0 and bundle["prompt_tokens"] == 10
    assert bundle["kv"] == 0.25 and bundle["ttft_count"] == 3
    assert bundle["itl_sum"] == 0.4 and bundle["itl_count"] == 8
    dlt = metrics_delta(bundle, {**bundle, "itl_sum": 0.8, "itl_count": 16.0})
    assert dlt["itl_sum"] == 0.4 and dlt["itl_count"] == 8.0

    assert abs(client_tpot_s(1.0, 0.2, 5) - 0.2) < 1e-12
    try:
        client_tpot_s(1.0, 0.2, 1)
        raise AssertionError("tpot with completion_tokens<=1 must fail")
    except EvalError:
        pass
    check_prompt_tokens(100, {"prompt_tokens": 120})
    try:
        check_prompt_tokens(100, {"prompt_tokens": 140})
        raise AssertionError("prompt token drift must fail")
    except EvalError:
        pass
    check_perf_generation({"completion_tokens": PERF_MAX_TOKENS}, "length")
    try:
        check_perf_generation({"completion_tokens": 3}, "length")
        raise AssertionError("short perf completion must fail")
    except EvalError:
        pass
    try:
        check_perf_generation({"completion_tokens": PERF_MAX_TOKENS}, "stop")
        raise AssertionError("perf stop finish must fail")
    except EvalError:
        pass

    # stream delta fields
    assert delta_text({"content": "a"}) == "a"
    assert delta_text({"reasoning": "r"}) == "r"
    assert delta_text({"reasoning_content": "x"}) == "x"
    assert delta_text({"role": "assistant"}) == ""

    # chosen-token logprobs (vLLM JSON shape: null first, string keys, Logprob dict)
    token_ids = [10, 11, 12, 13]
    pl = [
        None,
        {"11": {"logprob": -0.2, "rank": 1, "decoded_token": "a"}},
        {"12": {"logprob": -1.5, "rank": 3, "decoded_token": "b"}},
        {"13": {"logprob": -0.01, "rank": 1, "decoded_token": "c"}},
    ]
    got = chosen_logprobs(pl, token_ids, 2)
    assert [g["logprob"] for g in got] == [-1.5, -0.01]
    assert [g["logprob"] for g in chosen_logprobs(pl, token_ids, 1)] == [-0.2, -1.5, -0.01]
    try:
        chosen_logprobs([None, {"99": {"logprob": -1.0, "rank": 1}}], [1, 2], 1)
        raise AssertionError("missing chosen token must fail")
    except EvalError:
        pass

    assert percentile([1.0], 95) == 1.0
    assert percentile([0.0, 1.0], 0) == 0.0
    assert percentile([0.0, 1.0], 100) == 1.0

    with tempfile.TemporaryDirectory(prefix="vllm-eval-selfcheck-") as td:
        tmpdir = Path(td)
        path = tmpdir / "out.json"
        atomic_write_json(path, {"ok": True})
        try:
            atomic_write_json(path, {"ok": False})
            raise AssertionError("overwrite must fail")
        except EvalError:
            pass
        _self_check_rest(args, tmpdir)
    print("self-check ok", flush=True)
    return 0


def _self_check_rest(args: argparse.Namespace, tmpdir: Path) -> None:
    if args.tokenizer_path:
        tok = load_tokenizer(args.tokenizer_path)
        unit, n = find_filler_unit(tok)
        assert n == 1, unit
        for target, pos in ((256, 0.0), (21000, 0.15), (128000, 0.50), (240000, 0.85)):
            text, got_n, meta = build_user_text(
                tok,
                target,
                extra_prefix="Read the document below. A secret code appears exactly once.\n",
                needle="\n[SECRET] The secret code is NEVA-SELF-CHECK.\n",
                question="\nWhat is the secret code? Reply with only the secret code.",
                pos=pos,
            )
            assert abs(got_n - target) <= TARGET_TOLERANCE, (target, got_n, meta)
            assert "NEVA-SELF-CHECK" in text
            print(f"construct {target}: got {got_n} unit={unit!r} pos={pos}", flush=True)
        start_ids, raw = chat_ids(
            tok,
            [
                {"role": "user", "content": LOGPROB_USER},
                {"role": "assistant", "content": LOGPROB_CONT},
            ],
            add_generation_prompt=False,
            continue_final_message=True,
        )
        assert raw.endswith(LOGPROB_CONT)
        prefix = tok.encode(raw[: -len(LOGPROB_CONT)], add_special_tokens=False)
        assert start_ids[: len(prefix)] == prefix
        print(f"continuation tokens: {len(start_ids) - len(prefix)}", flush=True)

    def fake_cell(scale: float) -> dict:
        return {
            "aggregate": {
                "min_ttft_s": 0.10 * scale,
                "max_e2e_s": 1.00 * scale,
                "mean_client_tpot_s": 0.010 * scale,
                "mean_output_tok_s": 80.0 / scale,
                "wall_output_tok_s": 100.0 / scale,
                "server_itl_s": 0.009 * scale,
            }
        }

    base = {
        "name": "base",
        "served_model": "m",
        "ok": True,
        "params": {"skip_long": True, "skip_performance": False},
        "greedy": [{"id": cid, "content": "x"} for cid, _ in SHORT_CASES],
        "prompt_logprobs": {
            "tokens": [{"token_id": 1, "logprob": -0.2}, {"token_id": 2, "logprob": -0.4}]
        },
        "needles": [],
        "performance": {"cells": {cid: fake_cell(1.0) for cid, _, _ in PERF_CELLS}},
    }
    cand = json.loads(json.dumps(base))
    cand["name"] = "cand"
    cand["greedy"][0]["content"] = "y"
    cand["prompt_logprobs"]["tokens"][0]["logprob"] = -0.25
    cand["performance"] = {"cells": {cid: fake_cell(1.1) for cid, _, _ in PERF_CELLS}}
    bp, cp = tmpdir / "base.json", tmpdir / "cand.json"
    bp.write_text(json.dumps(base), encoding="utf-8")
    cp.write_text(json.dumps(cand), encoding="utf-8")
    ns = argparse.Namespace(base=str(bp), candidate=str(cp))
    rc = cmd_compare(ns)
    if rc != 0:
        raise AssertionError("synthetic compare should exit 0")


# ---------------------------------------------------------------------------
# Trade-study extensions (phase 1): agent_mix load generator, accuracy tiers,
# reference scoring, API-conformance, cgroup resource envelope, boot facts.
#
# Hard limit discovered on this host (vLLM 0.27.1, 2x L20, TP=2):
#   sampling_params.py:513 sets skip_reading_prefix_cache when prompt_logprobs
#   is requested, so the whole prompt is re-prefilled in max_num_batched_tokens
#   chunks and gpu_model_runner._get_prompt_logprobs_dict materialises
#   chunk_tokens x vocab logits.  At 8k prompt tokens that allocation is
#   ~3.6 GiB and the engine dies with CUDA OOM (EngineDeadError).  Teacher
#   forcing with prompt_logprobs is therefore only safe on SHORT prompts;
#   TF_MAX_PROMPT_TOKENS is enforced client-side so the tool fails fast
#   instead of killing the server.  Deep-context teacher forcing uses
#   single-position top-k scoring instead (tf_deep below).
# ---------------------------------------------------------------------------

TF_MAX_PROMPT_TOKENS = 1200
TF_SHORT_DOCS = 40
TF_SHORT_DOC_TOKENS = 1000
TF_DEEP_DEPTHS = (32_000, 128_000, 200_000, 235_000)
TF_DEEP_POSITIONS = 96
TF_DEEP_TOPK = 20
TF_CONT_TOKENS = 128

NEEDLE_GRID_LENGTHS = (32_000, 128_000, 200_000, 250_000)
NEEDLE_GRID_DEPTHS = (0.05, 0.25, 0.50, 0.75, 0.95)
NEEDLE_GRID_CODES = (
    "NEVA-ALPHA-7731", "NEVA-BRAVO-2984", "NEVA-CIRRUS-5106",
    "NEVA-DELTA-8452", "NEVA-ECHO-3617",
    # Slots 5-9 exist so a deep-only grid can be widened without changing the
    # first five, which earlier captures were scored on.
    "NEVA-FOXTROT-9245", "NEVA-GOLF-1873", "NEVA-HOTEL-6034",
    "NEVA-INDIA-4529", "NEVA-JULIET-7168",
)
NEEDLE_DEFAULT_SLOTS = 5

CORPUS_PATH = Path("/Data2/lzp/vllm-eval/corpus/corpus-v1.txt")
CORPUS_IDS_PATH = Path("/Data2/lzp/vllm-eval/corpus/corpus-v1.ids.json")
CORPUS_SOURCE_ROOT = Path(
    "/Data/lzp/vllm-service/.venv/lib/python3.11/site-packages/vllm"
)
CORPUS_MIN_CHARS = 4_800_000

CGROUP_DIR = Path(
    "/sys/fs/cgroup/user.slice/user-1003.slice/user@1003.service"
    "/app.slice/vllm.service"
)

# agent_mix scenario (design 7.1).
AGENT_SESSIONS = 4
AGENT_ROUNDS = 4
AGENT_ROUND1_TOKENS = 30_000
AGENT_TOOL_TOKENS = 7_000
AGENT_OUTPUT_TOKENS = 3_000
AGENT_EFFORT = "xhigh"
SHORT_CHAT_START_S = 20.0
SHORT_CHAT_PERIOD_S = 15.0
SHORT_CHAT_TURN_TOKENS = (200, 400, 700)
SHORT_CHAT_OUTPUT_TOKENS = 256

REASON_TASKS = [
    ("r01", "A bag has 3 red, 5 blue and 7 green marbles. Two are drawn without replacement. What is the probability both are green? Give the reduced fraction."),
    ("r02", "Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals and returns them sorted. Output only the code."),
    ("r03", "A train leaves at 09:15 travelling 72 km/h and another leaves the same station at 09:45 travelling 96 km/h on the same track. When does the second catch the first?"),
    ("r04", "Explain why binary search requires a sorted array, then give the loop invariant that proves its correctness."),
    ("r05", "Find all integer solutions of x^2 - 5y^2 = 1 with 0 < x < 100."),
    ("r06", "Given a singly linked list, write Python to detect a cycle and return the node where the cycle begins. Output only the code."),
    ("r07", "A cube is painted on all faces then cut into 27 equal cubes. How many small cubes have exactly two painted faces?"),
    ("r08", "Debug: `def f(xs): return sum(xs)/len(xs)` fails on an empty list. Rewrite it defensively and explain the choice of behaviour."),
    ("r09", "What is the time complexity of building a heap from n elements by repeated sift-down, and why is it O(n) rather than O(n log n)?"),
    ("r10", "Solve for x: log_2(x) + log_2(x - 6) = 4."),
    ("r11", "Write a SQL query returning, per department, the employee with the highest salary, breaking ties by earliest hire date."),
    ("r12", "Three switches downstairs control three bulbs upstairs. You may go upstairs once. How do you determine which switch controls which bulb?"),
    ("r13", "Implement a Python LRU cache with O(1) get and put using a dict and a doubly linked list. Output only the code."),
    ("r14", "Prove that the sum of the first n odd positive integers equals n^2."),
    ("r15", "A hash table with 1000 slots holds 700 keys under separate chaining. What is the expected number of probes for an unsuccessful search?"),
    ("r16", "Given `git rebase -i HEAD~5`, explain what happens if you mark the second commit `squash` and the fourth `drop`."),
    ("r17", "Compute the determinant of [[2,1,3],[0,4,1],[5,2,0]] showing the cofactor expansion."),
    ("r18", "Write a Python generator that yields the Collatz sequence from n and stops at 1. Then state the largest value reached for n=27."),
    ("r19", "Why does floating-point 0.1 + 0.2 != 0.3? Give the exact binary64 values involved."),
    ("r20", "Design the table schema for a many-to-many relation between users and roles with an effective date range, and state the primary key."),
    ("r21", "A rope burns unevenly in exactly 60 minutes. With two such ropes, measure 45 minutes."),
    ("r22", "In Python, explain the difference between `__new__` and `__init__` and when overriding `__new__` is required."),
    ("r23", "Find the remainder when 7^100 is divided by 13."),
    ("r24", "Write a function that returns the longest common subsequence of two strings using O(min(m,n)) extra space. Output only the code."),
    ("r25", "Given a 5x5 grid, how many monotone lattice paths from the bottom-left to the top-right move only right or up?"),
    ("r26", "Explain the difference between a process and a thread with respect to address space, scheduling and failure isolation."),
    ("r27", "Solve the recurrence T(n) = 2T(n/2) + n log n using the Akra-Bazzi or master theorem, stating which applies."),
    ("r28", "Write a Python context manager that times its block and logs the elapsed milliseconds. Output only the code."),
    ("r29", "Two players alternately remove 1, 2 or 3 stones from a pile of 21; the player taking the last stone wins. Who wins with optimal play, and what is the strategy?"),
    ("r30", "Explain what a memory barrier is and why double-checked locking is broken without one."),
]

TRACE_MAX_TOKENS = 1200
TRACE_REPLAY_TOKENS = 900


def build_corpus() -> str:
    """Deterministic offline text corpus (real Python source), cached on disk."""
    if CORPUS_PATH.is_file():
        return CORPUS_PATH.read_text(encoding="utf-8")
    parts: list[str] = []
    total = 0
    for path in sorted(CORPUS_SOURCE_ROOT.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if len(text) < 3000:
            continue
        parts.append(f"# file: {path.relative_to(CORPUS_SOURCE_ROOT)}\n{text}\n")
        total += len(parts[-1])
        if total >= CORPUS_MIN_CHARS:
            break
    if total < CORPUS_MIN_CHARS:
        raise EvalError(f"corpus source yielded only {total} chars")
    body = "".join(parts)
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CORPUS_PATH.with_suffix(".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, CORPUS_PATH)
    return body


_CORPUS_IDS: list[int] | None = None


def corpus_ids(tok) -> list[int]:
    global _CORPUS_IDS
    if _CORPUS_IDS is not None:
        return _CORPUS_IDS
    text = build_corpus()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if CORPUS_IDS_PATH.is_file():
        cached = json.loads(CORPUS_IDS_PATH.read_text(encoding="utf-8"))
        if cached.get("sha256") == digest:
            _CORPUS_IDS = cached["ids"]
            return _CORPUS_IDS
    ids = tok.encode(text, add_special_tokens=False)
    tmp = CORPUS_IDS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"sha256": digest, "ids": ids}), encoding="utf-8")
    os.replace(tmp, CORPUS_IDS_PATH)
    _CORPUS_IDS = ids
    return ids


def corpus_text(tok, offset: int, count: int) -> str:
    ids = corpus_ids(tok)
    if offset + count > len(ids):
        raise EvalError(f"corpus too small for offset {offset} + {count} (have {len(ids)})")
    return tok.decode(ids[offset : offset + count])


def corpus_sha() -> str:
    return hashlib.sha256(build_corpus().encode("utf-8")).hexdigest()[:16]


# --- cgroup resource envelope ------------------------------------------------

def _read_kv(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            bits = line.split()
            if len(bits) == 2:
                try:
                    out[bits[0]] = int(bits[1])
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def cgroup_snapshot() -> dict:
    stat = _read_kv(CGROUP_DIR / "cpu.stat")
    mstat = _read_kv(CGROUP_DIR / "memory.stat")
    return {
        "wall_ns": time.monotonic_ns(),
        "cpu_usage_usec": stat.get("usage_usec"),
        "nr_throttled": stat.get("nr_throttled"),
        "nr_periods": stat.get("nr_periods"),
        "throttled_usec": stat.get("throttled_usec"),
        "memory_current": _read_int(CGROUP_DIR / "memory.current"),
        "memory_peak": _read_int(CGROUP_DIR / "memory.peak"),
        "memory_events": _read_kv(CGROUP_DIR / "memory.events"),
        "memory_anon": mstat.get("anon"),
        "memory_file": mstat.get("file"),
        "memory_shmem": mstat.get("shmem"),
        "memory_slab": mstat.get("slab"),
        "memory_swap_current": _read_int(CGROUP_DIR / "memory.swap.current"),
    }


def cgroup_reset_peak() -> bool:
    """Try to reset memory.peak; report whether it actually reset.

    On this kernel the write succeeds but the historical peak is retained, so
    memory.peak is peak-since-service-start and the per-run peak has to come
    from sampling memory.current.  Report the truth rather than assume."""
    path = CGROUP_DIR / "memory.peak"
    before = _read_int(path)
    try:
        path.write_text("0")
    except OSError:
        return False
    after = _read_int(path)
    return before is not None and after is not None and after < before


def cgroup_envelope(before: dict, after: dict, mem_samples: list[int]) -> dict:
    wall_s = (after["wall_ns"] - before["wall_ns"]) / 1e9
    cpu_usec = None
    if before.get("cpu_usage_usec") is not None and after.get("cpu_usage_usec") is not None:
        cpu_usec = after["cpu_usage_usec"] - before["cpu_usage_usec"]
    cores = (cpu_usec / 1e6 / wall_s) if (cpu_usec is not None and wall_s > 0) else None
    gib = 1024.0 ** 3
    samples = [m for m in mem_samples if m is not None]
    return {
        "wall_s": wall_s,
        "cpu_core_seconds": None if cpu_usec is None else cpu_usec / 1e6,
        "mean_cores": cores,
        "nr_throttled_delta": (
            None
            if before.get("nr_throttled") is None or after.get("nr_throttled") is None
            else after["nr_throttled"] - before["nr_throttled"]
        ),
        "nr_periods_delta": (
            None
            if before.get("nr_periods") is None or after.get("nr_periods") is None
            else after["nr_periods"] - before["nr_periods"]
        ),
        "memory_current_start_gib": (before.get("memory_current") or 0) / gib,
        "memory_current_end_gib": (after.get("memory_current") or 0) / gib,
        "memory_peak_gib": (after.get("memory_peak") or 0) / gib,
        "memory_sampled_max_gib": (max(samples) / gib) if samples else None,
        "memory_anon_end_gib": (after.get("memory_anon") or 0) / gib,
        "memory_shmem_end_gib": (after.get("memory_shmem") or 0) / gib,
        "memory_events_before": before.get("memory_events"),
        "memory_events_after": after.get("memory_events"),
        "memory_swap_current": after.get("memory_swap_current"),
        "before": before,
        "after": after,
    }


# --- boot facts --------------------------------------------------------------

BOOT_PATTERNS = (
    ("attention_backend", re.compile(r"Using (\S+) attention backend")),
    ("block_size", re.compile(r"Setting attention block size to (\d+) tokens")),
    ("kv_cache_memory_gib", re.compile(r"Available KV cache memory: ([\d.]+) GiB")),
    ("kv_cache_tokens", re.compile(r"GPU KV cache size: ([\d,]+) tokens")),
    ("max_concurrency", re.compile(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x")),
    ("graph_estimated_gib", re.compile(r"Estimated CUDA graph memory: ([\d.]+) GiB")),
    ("graph_actual_gib", re.compile(r"CUDA graph pool memory: ([\d.]+) GiB \(actual\), ([\d.]+) GiB \(estimated\)")),
    ("capture_profile", re.compile(r"Profiling CUDA graph memory: PIECEWISE=(\d+) \(largest=(\d+)\), FULL=(\d+) \(largest=(\d+)\)")),
    ("init_engine_s", re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s \(compilation: ([\d.]+) s\)")),
    ("kv_memory_advice", re.compile(r"--kv-cache-memory=(\d+)")),
)


def collect_boot_facts(since: str) -> dict:
    import subprocess

    cmd = [
        "journalctl", "--user", "-u", "vllm.service",
        "--since", since, "--no-pager", "-o", "cat",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    text = proc.stdout
    facts: dict[str, object] = {"since": since}
    for key, pat in BOOT_PATTERNS:
        hits = pat.findall(text)
        if not hits:
            continue
        if key == "graph_actual_gib":
            facts["graph_actual_gib"] = float(hits[-1][0])
            facts["graph_estimated_at_capture_gib"] = float(hits[-1][1])
        elif key == "capture_profile":
            facts["capture_piecewise_count"] = int(hits[-1][0])
            facts["capture_piecewise_largest"] = int(hits[-1][1])
            facts["capture_full_count"] = int(hits[-1][2])
            facts["capture_full_largest"] = int(hits[-1][3])
        elif key == "init_engine_s":
            facts["init_engine_s"] = float(hits[-1][0])
            facts["compilation_s"] = float(hits[-1][1])
        elif key == "kv_memory_advice":
            facts["kv_memory_advice_bytes"] = sorted({int(h) for h in hits})
        elif key in ("block_size", "kv_cache_tokens"):
            facts[key] = int(str(hits[-1]).replace(",", ""))
        elif key in ("kv_cache_memory_gib", "max_concurrency", "graph_estimated_gib"):
            facts[key] = float(hits[-1])
        else:
            facts[key] = hits[-1]
    warn = re.findall(r"(WARNING.*(?:scale|accuracy|calibrat)[^\n]*)", text)
    facts["accuracy_warnings"] = sorted({w.strip()[:220] for w in warn})[:12]
    jit = re.findall(r"(JIT[^\n]{0,160})", text)
    facts["jit_warnings"] = sorted({j.strip()[:160] for j in jit})[:12]
    cap = re.findall(r"'cudagraph_capture_sizes': \[([^\]]*)\]", text)
    if cap:
        facts["cudagraph_capture_sizes"] = [
            int(x) for x in cap[-1].split(",") if x.strip()
        ]
        facts["cudagraph_capture_count"] = len(facts["cudagraph_capture_sizes"])
    started = re.findall(r"Application startup complete", text)
    facts["startup_complete_seen"] = len(started)
    return facts


def cmd_bootfacts(args: argparse.Namespace) -> int:
    facts = collect_boot_facts(args.since)
    if args.output:
        atomic_write_json(Path(args.output), facts)
    print(json.dumps(facts, ensure_ascii=False, indent=2))
    return 0


# --- teacher-forced scoring helpers ------------------------------------------

def prompt_top1_id(slot: object) -> int | None:
    if not isinstance(slot, dict):
        return None
    best_id, best_lp = None, None
    for key, entry in slot.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("rank") == 1:
            return int(key)
        lp = entry.get("logprob")
        if lp is None:
            continue
        if best_lp is None or float(lp) > best_lp:
            best_lp, best_id = float(lp), int(key)
    return best_id


def tf_score_prompt(client: httpx.Client, model: str, messages: list, extra: dict) -> dict:
    body = chat_body(model, messages, dict(extra))
    body.update(
        {
            "max_tokens": LOGPROB_MAX_TOKENS,
            "prompt_logprobs": 1,
            "return_token_ids": True,
            "stream": False,
        }
    )
    payload = chat_nonstream(client, body)
    ids = payload.get("prompt_token_ids")
    pl = payload.get("prompt_logprobs")
    if not isinstance(ids, list) or not ids:
        raise EvalError("prompt_token_ids missing (return_token_ids)")
    ids = [int(t) for t in ids]
    if len(ids) > TF_MAX_PROMPT_TOKENS:
        raise EvalError(
            f"prompt {len(ids)} tokens exceeds TF_MAX_PROMPT_TOKENS={TF_MAX_PROMPT_TOKENS}"
        )
    chosen = chosen_logprobs(pl, ids, 1)
    if not isinstance(pl, list):
        raise EvalError("prompt_logprobs missing")
    for rec in chosen:
        rec["top1_id"] = prompt_top1_id(pl[rec["index"]])
        rec.pop("decoded_token", None)
    return {
        "prompt_tokens": len(ids),
        "prompt_ids_sha": hashlib.sha256(
            ",".join(str(i) for i in ids).encode()
        ).hexdigest()[:16],
        "positions": chosen,
    }


def run_tf_short(client: httpx.Client, model: str, tok) -> dict:
    docs = []
    for i in range(TF_SHORT_DOCS):
        offset = 500 + i * 2971
        text = corpus_text(tok, offset, TF_SHORT_DOC_TOKENS)
        rec = {"id": f"tf{i:02d}", "corpus_offset": offset}
        try:
            wait_idle(client)
            scored = tf_score_prompt(
                client,
                model,
                [{"role": "user", "content": text}],
                {},
            )
            rec.update(scored)
        except Exception as exc:
            rec["error"] = str(exc)
        docs.append(rec)
    n = sum(len(d.get("positions") or []) for d in docs)
    return {"docs": docs, "scored_positions": n}


def run_tf_deep(
    client: httpx.Client,
    model: str,
    tok,
    depths: tuple[int, ...] = TF_DEEP_DEPTHS,
    positions: int = TF_DEEP_POSITIONS,
) -> dict:
    """Deep-context teacher forcing.

    prompt_logprobs cannot be used past ~1.2k tokens on this host (see the note
    at the top of this section), so depth is probed one position at a time:
    the document is followed by an assistant continuation truncated at k tokens
    and the next-token top-k distribution is read from `logprobs`.  Ascending k
    means every request but the first re-uses the prefix cache, so the whole
    sweep costs roughly one full prefill per depth plus a short tail per
    position.
    """
    ids = corpus_ids(tok)
    cont_off = 900_000 if len(ids) > 900_000 + TF_CONT_TOKENS else len(ids) - TF_CONT_TOKENS - 10
    cont_ids = ids[cont_off : cont_off + TF_CONT_TOKENS]
    # Prefer truncation points whose decoded prefix re-encodes exactly, so the
    # prompt really ends on the intended token boundary; top up with the
    # remaining points if the corpus slice has too few.  Scoring never depends
    # on target_id, only on reference-vs-candidate agreement at the same
    # prompt, so a non-round-tripping k is still a valid paired sample.
    stable, other = [], []
    for k in range(1, TF_CONT_TOKENS):
        prefix = cont_ids[:k]
        if tok.encode(tok.decode(prefix), add_special_tokens=False) == prefix:
            stable.append(k)
        else:
            other.append(k)
    valid = sorted((stable + other)[:positions])
    if len(valid) < 8:
        raise EvalError("not enough continuation truncation points for tf_deep")

    base_url = str(client.base_url)
    headers = dict(client.headers)
    timeout = client.timeout
    results: dict[int, dict] = {}
    errors: dict[int, str] = {}

    def sweep(depth: int) -> None:
        try:
            doc = corpus_text(tok, 0, depth)
            out = {"depth_tokens": depth, "positions": []}
            with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as inner:
                for k in valid:
                    body = chat_body(
                        model,
                        [
                            {"role": "user", "content": doc + "\n\nContinue the passage verbatim."},
                            {"role": "assistant", "content": tok.decode(cont_ids[:k])},
                        ],
                        {
                            "max_tokens": 1,
                            "logprobs": True,
                            "top_logprobs": TF_DEEP_TOPK,
                            "add_generation_prompt": False,
                            "continue_final_message": True,
                            "stream": False,
                        },
                    )
                    payload = chat_nonstream(inner, body)
                    choice = payload["choices"][0]
                    entries = (choice.get("logprobs") or {}).get("content") or []
                    if not entries:
                        raise EvalError("no logprobs returned for tf_deep position")
                    entry = entries[0]
                    # The OpenAI logprobs schema carries the token STRING, not
                    # an id, so the token text is the identity key here.
                    top = [
                        {"token": t.get("token"), "logprob": float(t["logprob"])}
                        for t in (entry.get("top_logprobs") or [])
                    ]
                    out["positions"].append(
                        {
                            "k": k,
                            "argmax_token": entry.get("token"),
                            "argmax_logprob": float(entry["logprob"]),
                            "top": top,
                            "prompt_tokens": payload["usage"]["prompt_tokens"],
                        }
                    )
            results[depth] = out
        except Exception as exc:
            errors[depth] = str(exc)

    wait_idle(client, timeout=120.0)
    threads = [threading.Thread(target=sweep, args=(d,)) for d in depths]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return {
        "continuation_source_offset": cont_off,
        "positions_per_depth": len(valid),
        "depths": [
            results.get(d, {"depth_tokens": d, "error": errors.get(d, "missing")})
            for d in depths
        ],
    }


def run_needle_grid(
    client: httpx.Client,
    model: str,
    tok,
    slots: int = NEEDLE_DEFAULT_SLOTS,
    lengths: tuple[int, ...] = NEEDLE_GRID_LENGTHS,
) -> dict:
    ids = corpus_ids(tok)
    items = []
    for length in lengths:
        for depth in NEEDLE_GRID_DEPTHS:
            for ci, code in enumerate(NEEDLE_GRID_CODES[:slots]):
                items.append((length, depth, ci, code))
    rows = []
    for length, depth, ci, code in items:
        pos = int(round(length * depth))
        needle = f"\n[SECRET] The secret code for slot {ci} is {code}.\n"
        head = tok.decode(ids[:pos]) if pos > 0 else ""
        tail = tok.decode(ids[pos:length])
        text = (
            "Read the document below. A secret code appears exactly once.\n"
            + head + needle + tail
            + "\nWhat is the secret code? Reply with only the secret code."
        )
        rec = {"length": length, "depth": depth, "slot": ci, "code": code}
        try:
            payload = chat_nonstream(
                client,
                chat_body(
                    model,
                    [{"role": "user", "content": text}],
                    {"max_tokens": NEEDLE_MAX_TOKENS},
                ),
            )
            content, reasoning, finish, usage = message_fields(payload)
            hay = " ".join(p for p in (content, reasoning) if p)
            rec.update(
                {
                    "prompt_tokens": usage["prompt_tokens"],
                    "answer": (content or "")[:120],
                    "finish_reason": finish,
                    "hit": code in hay,
                }
            )
        except Exception as exc:
            rec["error"] = str(exc)
            rec["hit"] = False
        rows.append(rec)
    hits = sum(1 for r in rows if r.get("hit"))
    return {
        "items": rows,
        "n": len(rows),
        "hits": hits,
        "slots": slots,
        "lengths": list(lengths),
    }


TRACE_WORKERS = 4


def run_traces(client: httpx.Client, model: str) -> dict:
    """Generate the reference xhigh reasoning traces (run once, on the BF16 cell).

    Independent tasks, so they are issued a few at a time; this is a corpus
    build, not a timing measurement, and concurrency does not affect the text
    (temperature 0, fixed seed, no cross-request state)."""
    base_url = str(client.base_url)
    headers = dict(client.headers)
    timeout = client.timeout
    rows: list[dict | None] = [None] * len(REASON_TASKS)
    sem = threading.Semaphore(TRACE_WORKERS)

    def work(idx: int, tid: str, prompt: str) -> None:
        rec = {"id": tid, "prompt": prompt}
        with sem:
            try:
                with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as inner:
                    payload = chat_nonstream(
                        inner,
                        {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0,
                            "seed": SEED,
                            "n": 1,
                            "max_tokens": TRACE_MAX_TOKENS,
                            "reasoning_effort": AGENT_EFFORT,
                        },
                    )
                content, reasoning, finish, usage = message_fields(payload)
                rec.update(
                    {
                        "content": content,
                        "reasoning": reasoning,
                        "finish_reason": finish,
                        "usage": usage,
                    }
                )
            except Exception as exc:
                rec["error"] = str(exc)
        rows[idx] = rec

    threads = [
        threading.Thread(target=work, args=(i, tid, prompt))
        for i, (tid, prompt) in enumerate(REASON_TASKS)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return {
        "tasks": [r for r in rows if r is not None],
        "reasoning_effort": AGENT_EFFORT,
        "max_tokens": TRACE_MAX_TOKENS,
    }


def run_trace_replay(client: httpx.Client, model: str, tok, traces: dict) -> dict:
    rows = []
    for row in traces.get("tasks") or []:
        if row.get("error"):
            continue
        trace = (row.get("reasoning") or "") + "\n" + (row.get("content") or "")
        trace = trace.strip()
        if not trace:
            continue
        prompt = row["prompt"]
        head = len(tok.encode(prompt, add_special_tokens=False))
        budget = max(64, TRACE_REPLAY_TOKENS - head)
        tids = tok.encode(trace, add_special_tokens=False)[:budget]
        if len(tids) < 32:
            continue
        rec = {"id": row["id"], "replay_tokens": len(tids)}
        try:
            wait_idle(client)
            scored = tf_score_prompt(
                client,
                model,
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": tok.decode(tids)},
                ],
                {"add_generation_prompt": False, "continue_final_message": True},
            )
            rec.update(scored)
        except Exception as exc:
            rec["error"] = str(exc)
        rows.append(rec)
    n = sum(len(r.get("positions") or []) for r in rows)
    return {"docs": rows, "scored_positions": n}


def cmd_accuracy(args: argparse.Namespace) -> int:
    out_path = Path(args.output)
    if out_path.exists():
        print(f"refusing to overwrite existing result: {out_path}", file=sys.stderr)
        return 1
    tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    tf_deep_depths = (
        tuple(int(x) for x in args.tf_deep_depths.split(",") if x.strip())
        if getattr(args, "tf_deep_depths", None)
        else TF_DEEP_DEPTHS
    )
    needle_lengths = (
        tuple(int(x) for x in args.needle_lengths.split(",") if x.strip())
        if getattr(args, "needle_lengths", None)
        else NEEDLE_GRID_LENGTHS
    )
    if args.needle_slots > len(NEEDLE_GRID_CODES):
        raise EvalError(
            f"--needle-slots {args.needle_slots} exceeds the "
            f"{len(NEEDLE_GRID_CODES)} distinct needle codes defined"
        )
    errors: list[str] = []
    result = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "name": args.name,
        "kind": "accuracy",
        "tiers": sorted(tiers),
        "corpus_sha": corpus_sha(),
        "params": {
            "tf_short_docs": TF_SHORT_DOCS,
            "tf_short_doc_tokens": TF_SHORT_DOC_TOKENS,
            "tf_deep_depths": list(tf_deep_depths),
            "tf_deep_positions": args.tf_deep_positions,
            "tf_deep_topk": TF_DEEP_TOPK,
            "needle_lengths": list(needle_lengths),
            "needle_depths": list(NEEDLE_GRID_DEPTHS),
            "needle_codes": args.needle_slots,
            "tf_max_prompt_tokens": TF_MAX_PROMPT_TOKENS,
        },
        "errors": errors,
        "ok": False,
    }
    key = load_api_key()
    tok = load_tokenizer(args.tokenizer_path)
    timeout = httpx.Timeout(connect=10.0, read=READ_TIMEOUT, write=300.0, pool=60.0)
    traces = None
    if args.traces:
        raw = json.loads(Path(args.traces).read_text(encoding="utf-8"))
        # a traces capture stores the corpus under the "traces" tier key
        traces = raw.get("traces") if isinstance(raw.get("traces"), dict) else raw
        if not (traces or {}).get("tasks"):
            raise EvalError(f"{args.traces} holds no reasoning traces")
    with httpx.Client(base_url=args.base_url, headers=auth_headers(key), timeout=timeout) as client:
        served, model_ids, _ = confirm_ready(client)
        result["served_model"] = served
        for tier, fn in (
            ("tf_short", lambda: run_tf_short(client, served, tok)),
            (
                "tf_deep",
                lambda: run_tf_deep(
                    client, served, tok, tf_deep_depths, args.tf_deep_positions
                ),
            ),
            (
                "needle",
                lambda: run_needle_grid(
                    client, served, tok, args.needle_slots, needle_lengths
                ),
            ),
            ("traces", lambda: run_traces(client, served)),
            ("trace_replay", lambda: run_trace_replay(client, served, tok, traces or {})),
        ):
            if tier not in tiers:
                continue
            if tier == "trace_replay" and not traces:
                errors.append("trace_replay requested without --traces")
                continue
            print(f"tier {tier}", file=sys.stderr, flush=True)
            t0 = time.perf_counter()
            try:
                result[tier] = fn()
                result[tier]["wall_s"] = time.perf_counter() - t0
            except Exception as exc:
                errors.append(f"{tier}: {exc}")
                result[tier] = {"error": str(exc), "wall_s": time.perf_counter() - t0}
    result["ok"] = not errors
    atomic_write_json(out_path, result)
    print(f"wrote {out_path} ok={result['ok']}")
    return 0 if result["ok"] else 1


# --- accuracy scoring --------------------------------------------------------

def _tf_pairs(ref: dict, cand: dict, tier: str) -> tuple[list[float], int, int, list[str]]:
    diffs: list[float] = []
    flips = 0
    notes: list[str] = []
    rdocs = {d.get("id"): d for d in (ref.get(tier) or {}).get("docs") or []}
    cdocs = {d.get("id"): d for d in (cand.get(tier) or {}).get("docs") or []}
    for did, rd in rdocs.items():
        cd = cdocs.get(did)
        if cd is None or rd.get("error") or cd.get("error"):
            notes.append(f"{did}: incomparable")
            continue
        if rd.get("prompt_ids_sha") != cd.get("prompt_ids_sha"):
            notes.append(f"{did}: prompt token ids differ; skipped")
            continue
        rp, cp = rd.get("positions") or [], cd.get("positions") or []
        if len(rp) != len(cp):
            notes.append(f"{did}: position count {len(rp)} vs {len(cp)}")
            continue
        for a, b in zip(rp, cp):
            if a["token_id"] != b["token_id"]:
                notes.append(f"{did}: token id mismatch at {a['index']}")
                break
            diffs.append(abs(float(a["logprob"]) - float(b["logprob"])))
            if a.get("top1_id") != b.get("top1_id"):
                flips += 1
    return diffs, flips, len(diffs), notes


def _fmt_dist(diffs: list[float]) -> str:
    if not diffs:
        return "n=0"
    return (
        f"n={len(diffs)} mean={sum(diffs)/len(diffs):.6f} "
        f"p50={percentile(diffs,50):.6f} p95={percentile(diffs,95):.6f} "
        f"p99={percentile(diffs,99):.6f} max={max(diffs):.6f} "
        f"exact0={sum(1 for d in diffs if d==0.0)/len(diffs)*100:.2f}%"
    )


def _wilson_hw(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    z = 1.96
    return z * math.sqrt(max(p * (1 - p), 1e-12) / n) * 100.0


def cmd_acc_score(args: argparse.Namespace) -> int:
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    cand = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    lines = [
        f"reference: {ref.get('name')}  candidate: {cand.get('name')}",
        f"corpus: ref={ref.get('corpus_sha')} cand={cand.get('corpus_sha')}",
        "",
    ]
    code = 0
    if ref.get("corpus_sha") != cand.get("corpus_sha"):
        lines.append("FATAL: corpus differs; scores are not comparable")
        code = 1

    for tier, label in (("tf_short", "tier-1 short teacher-forced"),
                        ("trace_replay", "tier-3a reasoning-trace teacher-forced")):
        if tier not in ref or tier not in cand:
            continue
        diffs, flips, n, notes = _tf_pairs(ref, cand, tier)
        if n:
            fr = flips / n
            lines.append(f"{label}: |d logprob| {_fmt_dist(diffs)}")
            lines.append(
                f"{label}: argmax-flip {flips}/{n} = {fr*100:.4f} pp "
                f"(95% CI half-width +/-{_wilson_hw(fr, n):.4f} pp)"
            )
        else:
            lines.append(f"{label}: no comparable positions")
            code = 1
        for note in notes[:5]:
            lines.append(f"  note: {note}")

    if "tf_deep" in ref and "tf_deep" in cand:
        lines.append("tier-1 deep teacher-forced (per depth):")
        rd = {d.get("depth_tokens"): d for d in (ref["tf_deep"].get("depths") or [])}
        cd = {d.get("depth_tokens"): d for d in (cand["tf_deep"].get("depths") or [])}
        all_diffs: list[float] = []
        all_flip = all_n = 0
        for depth in sorted(rd):
            r, c = rd[depth], cd.get(depth)
            if not c or r.get("error") or c.get("error"):
                lines.append(f"  {depth}: incomparable ({r.get('error') or (c or {}).get('error')})")
                code = 1
                continue
            rp = {p["k"]: p for p in r.get("positions") or []}
            cp = {p["k"]: p for p in c.get("positions") or []}
            diffs, flips, n, oob = [], 0, 0, 0
            for k in sorted(set(rp) & set(cp)):
                a, b = rp[k], cp[k]
                if a.get("prompt_tokens") != b.get("prompt_tokens"):
                    continue
                n += 1
                if a.get("argmax_token") != b.get("argmax_token"):
                    flips += 1
                tok_s = a.get("argmax_token")
                bl = {t["token"]: t["logprob"] for t in b.get("top") or []}
                if tok_s in bl:
                    diffs.append(abs(float(a["argmax_logprob"]) - float(bl[tok_s])))
                else:
                    oob += 1
            all_diffs.extend(diffs)
            all_flip += flips
            all_n += n
            fr = flips / n if n else float("nan")
            lines.append(
                f"  {depth:>7}: n={n} argmax-flip={flips} ({fr*100:.2f} pp) "
                f"out-of-top{TF_DEEP_TOPK}={oob} |d logprob| {_fmt_dist(diffs)}"
            )
        if all_n:
            fr = all_flip / all_n
            lines.append(
                f"  ALL DEPTHS: argmax-flip {all_flip}/{all_n} = {fr*100:.2f} pp "
                f"(95% CI +/-{_wilson_hw(fr, all_n):.2f} pp); |d logprob| {_fmt_dist(all_diffs)}"
            )

    if "needle" in ref and "needle" in cand:
        rn = ref["needle"]
        cn = cand["needle"]
        lines.append(
            f"tier-2 needle grid: reference {rn.get('hits')}/{rn.get('n')}  "
            f"candidate {cn.get('hits')}/{cn.get('n')}"
        )
        rmap = {(i["length"], i["depth"], i["slot"]): i for i in rn.get("items") or []}
        regressions = []
        for item in cn.get("items") or []:
            k = (item["length"], item["depth"], item["slot"])
            r = rmap.get(k)
            if r is not None and r.get("hit") and not item.get("hit"):
                regressions.append(f"{k[0]}@{k[1]:.2f}#{k[2]}")
        lines.append(
            f"  regressions vs reference: {len(regressions)}"
            + ((" -> " + ", ".join(regressions[:12])) if regressions else "")
        )
        lines.append(
            "  power: N=100, a drop to <=96% is detectable at alpha~0.05; "
            "a 1-2 pp drop is NOT resolvable."
        )

    lines.append("")
    lines.append(
        "Power note: tier-1 short/trace tiers carry the statistical power "
        "(tens of thousands of paired positions); tier-1 deep has ~hundreds of "
        "positions and detects gross divergence plus mean logprob shift only."
    )
    print("\n".join(lines))
    return code


# --- agent_mix load generator ------------------------------------------------

def _percentiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    return {
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "max": max(xs),
        "min": min(xs),
        "mean": sum(xs) / len(xs),
        "n": len(xs),
    }


def run_agent_mix(client: httpx.Client, model: str, tok, *, sessions: int,
                  round1_tokens: int, rounds: int, out_tokens: int,
                  salt: str = "r1") -> dict:
    base_url = str(client.base_url)
    headers = dict(client.headers)
    timeout = client.timeout
    ids = corpus_ids(tok)
    long_rows: list[dict] = []
    short_rows: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()
    done = threading.Event()
    peaks = {"running": None, "waiting": None, "kv": None}
    mem_samples: list[int] = []
    stop = threading.Event()

    def mem_poller() -> None:
        while not stop.wait(0.5):
            v = _read_int(CGROUP_DIR / "memory.current")
            if v is not None:
                mem_samples.append(v)

    def long_session(sid: int) -> None:
        # The salt makes each session's prefix unique.  Repeat runs on the same
        # boot must pass a different --salt or they would simply replay the
        # previous run's prefix cache and measure nothing.
        head = f"[session-{salt}-{sid}-{sid*7919:07d}]\n"
        doc = corpus_text(tok, 300_000 + sid * 70_000, round1_tokens)
        messages = [{"role": "user", "content": head + doc + "\n\nAnalyse this code base and plan the next change."}]
        try:
            with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as inner:
                for rnd in range(1, rounds + 1):
                    body = chat_body(
                        model,
                        list(messages),
                        {
                            "min_tokens": out_tokens,
                            "max_tokens": out_tokens,
                            "reasoning_effort": AGENT_EFFORT,
                        },
                    )
                    body.pop("chat_template_kwargs", None)
                    rec = chat_stream(inner, body)
                    with lock:
                        long_rows.append(
                            {
                                "session": sid,
                                "round": rnd,
                                "ttft_s": rec["ttft_s"],
                                "e2e_s": rec["e2e_s"],
                                "client_tpot_s": rec["client_tpot_s"],
                                "usage": rec["usage"],
                                "finish_reason": rec["finish_reason"],
                            }
                        )
                    # Re-send the whole assistant turn verbatim, the way a real
                    # agent client does: that is what creates the prefix reuse.
                    assistant = (rec.get("content") or "").strip() or (
                        rec.get("reasoning") or ""
                    ).strip()
                    messages.append({"role": "assistant", "content": assistant or "(continuing)"})
                    if rnd < rounds:
                        tool = corpus_text(
                            tok, 550_000 + sid * 20_000 + rnd * 8_000, AGENT_TOOL_TOKENS
                        )
                        messages.append(
                            {"role": "user", "content": f"[tool result round {rnd}]\n" + tool + "\nContinue."}
                        )
        except Exception as exc:
            with lock:
                errors.append(f"long session {sid}: {exc}")

    def short_chat(cid: int) -> None:
        try:
            with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as inner:
                messages: list[dict] = []
                for ti, ntok in enumerate(SHORT_CHAT_TURN_TOKENS):
                    text = corpus_text(tok, 700_000 + cid * 3_000 + ti * 800, ntok)
                    messages.append({"role": "user", "content": f"[chat {salt} {cid} turn {ti}]\n" + text + "\nSummarise in one sentence."})
                    body = chat_body(
                        model,
                        list(messages),
                        {
                            "min_tokens": SHORT_CHAT_OUTPUT_TOKENS,
                            "max_tokens": SHORT_CHAT_OUTPUT_TOKENS,
                        },
                    )
                    # default reasoning effort, per the design: do not force
                    # enable_thinking off for the latency-sensitive class
                    body.pop("chat_template_kwargs", None)
                    rec = chat_stream(inner, body)
                    with lock:
                        short_rows.append(
                            {
                                "chat": cid,
                                "turn": ti,
                                "ttft_s": rec["ttft_s"],
                                "e2e_s": rec["e2e_s"],
                                "client_tpot_s": rec["client_tpot_s"],
                                "usage": rec["usage"],
                            }
                        )
                    reply = (rec.get("content") or "").strip() or (rec.get("reasoning") or "ok").strip()
                    messages.append({"role": "assistant", "content": reply[-1500:]})
        except Exception as exc:
            with lock:
                errors.append(f"short chat {cid}: {exc}")

    def short_driver() -> None:
        cid = 0
        if done.wait(SHORT_CHAT_START_S):
            return
        threads = []
        while not done.is_set():
            th = threading.Thread(target=short_chat, args=(cid,))
            th.start()
            threads.append(th)
            cid += 1
            if done.wait(SHORT_CHAT_PERIOD_S):
                break
        for th in threads:
            th.join()

    wait_idle(client, timeout=120.0)
    peak_reset = cgroup_reset_peak()
    with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as mclient:
        before_metrics = fetch_metrics(mclient)
        before_cg = cgroup_snapshot()
        poller = threading.Thread(target=sample_peaks, args=(mclient, stop, peaks), daemon=True)
        mempoll = threading.Thread(target=mem_poller, daemon=True)
        poller.start()
        mempoll.start()
        wall0 = time.perf_counter()
        sess_threads = [threading.Thread(target=long_session, args=(i,)) for i in range(sessions)]
        driver = threading.Thread(target=short_driver)
        for th in sess_threads:
            th.start()
        driver.start()
        for th in sess_threads:
            th.join()
        done.set()
        driver.join()
        wall = time.perf_counter() - wall0
        stop.set()
        poller.join()
        mempoll.join(timeout=2.0)
        after_cg = cgroup_snapshot()
        after_metrics = fetch_metrics(mclient)

    dlt = metrics_delta(before_metrics, after_metrics)
    gen = sum(r["usage"]["completion_tokens"] for r in long_rows) + sum(
        r["usage"]["completion_tokens"] for r in short_rows
    )
    prm = sum(r["usage"]["prompt_tokens"] for r in long_rows) + sum(
        r["usage"]["prompt_tokens"] for r in short_rows
    )
    return {
        "scenario": {
            "sessions": sessions,
            "rounds": rounds,
            "round1_tokens": round1_tokens,
            "output_tokens": out_tokens,
            "reasoning_effort": AGENT_EFFORT,
            "salt": salt,
            "short_chat_period_s": SHORT_CHAT_PERIOD_S,
            "short_chat_turn_tokens": list(SHORT_CHAT_TURN_TOKENS),
            "short_chat_output_tokens": SHORT_CHAT_OUTPUT_TOKENS,
        },
        "wall_s": wall,
        "errors": errors,
        "long_requests": long_rows,
        "short_requests": short_rows,
        "summary": {
            "long_e2e_s": _percentiles([r["e2e_s"] for r in long_rows]),
            "long_ttft_s": _percentiles([r["ttft_s"] for r in long_rows]),
            "long_tpot_s": _percentiles([r["client_tpot_s"] for r in long_rows]),
            "short_e2e_s": _percentiles([r["e2e_s"] for r in short_rows]),
            "short_ttft_s": _percentiles([r["ttft_s"] for r in short_rows]),
            "short_tpot_s": _percentiles([r["client_tpot_s"] for r in short_rows]),
            "generated_tokens": gen,
            "prompt_tokens": prm,
            "aggregate_output_tok_s": gen / wall if wall > 0 else None,
            "aggregate_total_tok_s": (gen + prm) / wall if wall > 0 else None,
            "long_requests_n": len(long_rows),
            "short_requests_n": len(short_rows),
        },
        "peaks": {
            "running": peaks.get("running"),
            "waiting": peaks.get("waiting"),
            "kv_cache_usage_perc": peaks.get("kv"),
        },
        "metrics_delta": dlt,
        "resource": dict(
            cgroup_envelope(before_cg, after_cg, mem_samples),
            peak_reset_supported=peak_reset,
        ),
    }


EXTRA_COUNTERS = (
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def scrape_extras(client: httpx.Client) -> dict:
    """Cumulative counters that metric_bundle does not carry, plus offload bytes."""
    resp = client.get("/metrics")
    if resp.status_code != 200:
        raise http_error(resp)
    sums, _maxes = parse_prom(resp.text)
    out = {k: sums.get(k) for k in EXTRA_COUNTERS}
    for line in resp.text.splitlines():
        if line.startswith("vllm:kv_offload_total_bytes_total"):
            try:
                labels = line.split("{", 1)[1].split("}", 1)[0]
                direction = "GPU_to_CPU" if "GPU_to_CPU" in labels else "CPU_to_GPU"
                out[f"offload_{direction}"] = float(line.split()[-1])
            except (IndexError, ValueError):
                continue
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            # Accepted-token counter per draft position; the label set also
            # carries engine/model labels, so match on position= only.
            try:
                labels = line.split("{", 1)[1].split("}", 1)[0]
                match = re.search(r'position="(\d+)"', labels)
                if match is None:
                    continue
                key = f"spec_accepted_pos_{int(match.group(1))}"
                out[key] = out.get(key, 0.0) + float(line.split()[-1])
            except (IndexError, ValueError):
                continue
    return out


def extras_delta(before: dict, after: dict) -> dict:
    out = {}
    for key in set(before) | set(after):
        a, b = before.get(key), after.get(key)
        out[key] = None if a is None or b is None else b - a
    q = out.get("vllm:prefix_cache_queries_total")
    h = out.get("vllm:prefix_cache_hits_total")
    out["prefix_cache_hit_rate_pct"] = (100.0 * h / q) if q else None
    eq = out.get("vllm:external_prefix_cache_queries_total")
    eh = out.get("vllm:external_prefix_cache_hits_total")
    out["external_prefix_cache_hit_rate_pct"] = (100.0 * eh / eq) if eq else None
    drafts = out.get("vllm:spec_decode_num_drafts_total")
    draft_toks = out.get("vllm:spec_decode_num_draft_tokens_total")
    accepted = out.get("vllm:spec_decode_num_accepted_tokens_total")
    # Overall acceptance = accepted draft tokens / drafted tokens.
    out["spec_acceptance_rate_pct"] = (
        (100.0 * accepted / draft_toks) if draft_toks else None
    )
    # Mean accepted tokens per draft step (0 means every draft was rejected).
    out["spec_accepted_per_draft"] = (accepted / drafts) if drafts else None
    # Per-position acceptance: share of drafts whose position-i token was kept.
    # Position i is only reached when every earlier position was accepted, so
    # this is the cumulative (not conditional) acceptance at depth i+1.
    for key in sorted(k for k in out if k.startswith("spec_accepted_pos_")):
        pos_accepted = out.get(key)
        out[f"{key}_rate_pct"] = (
            (100.0 * pos_accepted / drafts) if drafts else None
        )
    return out


def cmd_agent_mix(args: argparse.Namespace) -> int:
    out_path = Path(args.output)
    if out_path.exists():
        print(f"refusing to overwrite existing result: {out_path}", file=sys.stderr)
        return 1
    key = load_api_key()
    tok = load_tokenizer(args.tokenizer_path)
    timeout = httpx.Timeout(connect=10.0, read=READ_TIMEOUT, write=300.0, pool=60.0)
    result = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "name": args.name,
        "kind": "agent_mix",
        "corpus_sha": corpus_sha(),
    }
    with httpx.Client(base_url=args.base_url, headers=auth_headers(key), timeout=timeout) as client:
        served, _ids, _idle = confirm_ready(client)
        result["served_model"] = served
        # warmup so decode kernels and any first-request JIT are not measured
        if not args.no_warmup:
            print("warmup", file=sys.stderr, flush=True)
            for tgt in (256, 4096):
                text = corpus_text(tok, 950_000 + tgt, tgt)
                chat_nonstream(
                    client,
                    chat_body(
                        served,
                        [{"role": "user", "content": f"[warmup {uuid.uuid4().hex}]\n" + text}],
                        {"min_tokens": 64, "max_tokens": 64},
                    ),
                )
            wait_idle(client, timeout=120.0)
        print("agent_mix", file=sys.stderr, flush=True)
        extras_before = scrape_extras(client)
        result["run"] = run_agent_mix(
            client,
            served,
            tok,
            sessions=args.sessions,
            round1_tokens=args.round1_tokens,
            rounds=args.rounds,
            out_tokens=args.output_tokens,
            salt=args.salt,
        )
        try:
            extras_after = scrape_extras(client)
            result["run"]["counters_before"] = extras_before
            result["run"]["counters_after"] = extras_after
            result["run"]["counters_delta"] = extras_delta(extras_before, extras_after)
        except Exception as exc:
            result["run"]["counters_error"] = str(exc)
    result["ok"] = not result["run"].get("errors")
    atomic_write_json(out_path, result)
    s = result["run"]["summary"]
    print(
        f"wrote {out_path} ok={result['ok']} wall={result['run']['wall_s']:.1f}s "
        f"agg_out={s['aggregate_output_tok_s']:.1f} tok/s "
        f"long_tpot_p50={s['long_tpot_s'].get('p50')} "
        f"short_ttft_p95={s['short_ttft_s'].get('p95')}"
    )
    return 0 if result["ok"] else 1


# --- API semantics conformance ----------------------------------------------

def cmd_conformance(args: argparse.Namespace) -> int:
    out_path = Path(args.output)
    if out_path.exists():
        print(f"refusing to overwrite existing result: {out_path}", file=sys.stderr)
        return 1
    key = load_api_key()
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=120.0, pool=30.0)
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail[:300]})

    with httpx.Client(base_url=args.base_url, headers=auth_headers(key), timeout=timeout) as client:
        served, _ids, _ = confirm_ready(client)
        prompt = "List three prime numbers greater than 100."
        base = {
            "model": served, "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0, "top_p": 1.0, "seed": SEED, "max_tokens": 40, "n": 1,
        }
        try:
            a = chat_nonstream(client, dict(base, min_p=0.0))
            b = chat_nonstream(client, dict(base, min_p=0.95))
            ta = a["choices"][0]["message"].get("content")
            tb = b["choices"][0]["message"].get("content")
            add("min_p accepted", True, f"min_p=0 and min_p=0.95 both 200; differ={ta != tb}")
        except Exception as exc:
            add("min_p accepted", False, str(exc))
        try:
            r = chat_nonstream(client, dict(base, logit_bias={"9707": -100}))
            add("logit_bias accepted", True, "200")
        except Exception as exc:
            add("logit_bias accepted", False, str(exc))
        for eff in ("xhigh", "medium", "low"):
            try:
                r = chat_nonstream(
                    client,
                    {"model": served, "messages": [{"role": "user", "content": "What is 2+2?"}],
                     "max_tokens": 64, "temperature": 0, "seed": SEED, "reasoning_effort": eff},
                )
                content, reasoning, _f, usage = message_fields(r)
                add(f"reasoning_effort {eff}", True,
                    f"completion={usage['completion_tokens']} reasoning_len={len(reasoning or '')}")
            except Exception as exc:
                add(f"reasoning_effort {eff}", False, str(exc))
        try:
            r = client.post("/v1/chat/completions", json={
                "model": served, "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8, "reasoning_effort": "high"})
            add("reasoning_effort high rejected (known vendor gap)", r.status_code == 400,
                f"status={r.status_code}")
        except Exception as exc:
            add("reasoning_effort high rejected (known vendor gap)", False, str(exc))
        try:
            r = chat_nonstream(client, {
                "model": served,
                "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
                "max_tokens": 200, "temperature": 0, "seed": SEED,
                "tools": [{"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                   "required": ["city"]}}}],
                "tool_choice": "auto",
            })
            msg = r["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            add("tool calling", bool(calls),
                f"tool_calls={len(calls)} name={calls[0]['function']['name'] if calls else None}")
        except Exception as exc:
            add("tool calling", False, str(exc))
        try:
            rec = chat_stream(client, {
                "model": served, "messages": [{"role": "user", "content": "Count to five."}],
                "max_tokens": 64, "temperature": 0, "seed": SEED, "reasoning_effort": "low"})
            add("streaming", rec["usage"]["completion_tokens"] > 0,
                f"ttft={rec['ttft_s']:.3f}s tokens={rec['usage']['completion_tokens']}")
        except Exception as exc:
            add("streaming", False, str(exc))
        try:
            rec = chat_stream(client, {
                "model": served, "messages": [{"role": "user", "content": "Count to five."}],
                "max_tokens": 64, "temperature": 0, "seed": SEED, "stream_interval": 4})
            add("stream_interval", True, f"tokens={rec['usage']['completion_tokens']}")
        except Exception as exc:
            add("stream_interval", False, str(exc))
        try:
            r = chat_nonstream(client, {
                "model": served, "messages": [{"role": "user", "content": "Think briefly then say DONE."}],
                "max_tokens": 300, "temperature": 0, "seed": SEED, "reasoning_effort": "low"})
            _c, reasoning, _f, _u = message_fields(r)
            add("reasoning parser splits reasoning field", reasoning is not None,
                f"reasoning_present={reasoning is not None}")
        except Exception as exc:
            add("reasoning parser splits reasoning field", False, str(exc))

    result = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "name": args.name,
        "kind": "conformance",
        "checks": checks,
        "ok": all(c["ok"] for c in checks),
    }
    atomic_write_json(out_path, result)
    for c in checks:
        print(f"  [{'ok ' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
    return 0 if result["ok"] else 1


# --- load comparison ---------------------------------------------------------

LOAD_METRICS = (
    ("aggregate_output_tok_s", "higher"),
    ("aggregate_total_tok_s", "higher"),
)


def cmd_load_compare(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    cand = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    b, c = base["run"], cand["run"]
    bs, cs = b["summary"], c["summary"]
    lines = [f"base: {base.get('name')}   cand: {cand.get('name')}", ""]
    rows = [
        ("wall_s", b["wall_s"], c["wall_s"], "lower"),
        ("aggregate_output_tok_s", bs["aggregate_output_tok_s"], cs["aggregate_output_tok_s"], "higher"),
        ("aggregate_total_tok_s", bs["aggregate_total_tok_s"], cs["aggregate_total_tok_s"], "higher"),
        ("long_e2e_p50", bs["long_e2e_s"].get("p50"), cs["long_e2e_s"].get("p50"), "lower"),
        ("long_e2e_p95", bs["long_e2e_s"].get("p95"), cs["long_e2e_s"].get("p95"), "lower"),
        ("long_ttft_p50", bs["long_ttft_s"].get("p50"), cs["long_ttft_s"].get("p50"), "lower"),
        ("long_ttft_p95", bs["long_ttft_s"].get("p95"), cs["long_ttft_s"].get("p95"), "lower"),
        ("long_tpot_p50", bs["long_tpot_s"].get("p50"), cs["long_tpot_s"].get("p50"), "lower"),
        ("long_tpot_p95", bs["long_tpot_s"].get("p95"), cs["long_tpot_s"].get("p95"), "lower"),
        ("short_ttft_p50", bs["short_ttft_s"].get("p50"), cs["short_ttft_s"].get("p50"), "lower"),
        ("short_ttft_p95", bs["short_ttft_s"].get("p95"), cs["short_ttft_s"].get("p95"), "lower"),
        ("short_e2e_p50", bs["short_e2e_s"].get("p50"), cs["short_e2e_s"].get("p50"), "lower"),
        ("short_e2e_p95", bs["short_e2e_s"].get("p95"), cs["short_e2e_s"].get("p95"), "lower"),
        ("short_tpot_p50", bs["short_tpot_s"].get("p50"), cs["short_tpot_s"].get("p50"), "lower"),
        ("max_running", b["peaks"].get("running"), c["peaks"].get("running"), "info"),
        ("max_waiting", b["peaks"].get("waiting"), c["peaks"].get("waiting"), "info"),
        ("max_kv_usage_perc", b["peaks"].get("kv_cache_usage_perc"), c["peaks"].get("kv_cache_usage_perc"), "info"),
        ("preemptions", (b["metrics_delta"] or {}).get("preemptions"), (c["metrics_delta"] or {}).get("preemptions"), "info"),
        ("mean_cores", b["resource"].get("mean_cores"), c["resource"].get("mean_cores"), "info"),
        ("memory_peak_gib", b["resource"].get("memory_peak_gib"), c["resource"].get("memory_peak_gib"), "info"),
        ("short_requests_n", bs.get("short_requests_n"), cs.get("short_requests_n"), "info"),
    ]
    lines.append(f"{'metric':<26} {'base':>14} {'cand':>14} {'delta':>10}  dir")
    for name, bv, cv, direction in rows:
        d = pct_diff(bv, cv)
        ds = "n/a" if d is None else f"{d:+.2f}%"
        bstr = f"{bv:.4f}" if isinstance(bv, float) else str(bv)
        cstr = f"{cv:.4f}" if isinstance(cv, float) else str(cv)
        lines.append(f"{name:<26} {bstr:>14} {cstr:>14} {ds:>10}  {direction}")
    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal local vLLM eval: capture greedy/logprob/needle/perf JSON, "
            "then compare two captures. Does not overwrite results or print secrets."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser(
        "capture",
        help="run the eval against 127.0.0.1:8010 and atomically write JSON",
    )
    cap.add_argument("--name", required=True, help="label stored in the result (e.g. per-tensor-fp8)")
    cap.add_argument("--output", required=True, help="JSON path; must not already exist")
    cap.add_argument("--tokenizer-path", required=True, help="local tokenizer dir used to size prompts")
    cap.add_argument("--skip-long", action="store_true", help="skip 21k/128k/240k needle retrieval")
    cap.add_argument("--skip-performance", action="store_true", help="skip warmup and the 2x2 perf grid")
    cap.add_argument("--base-url", default=BASE_URL, help=f"vLLM base URL (default {BASE_URL})")
    cap.set_defaults(func=cmd_capture)

    cmp_ = sub.add_parser("compare", help="print greedy/logprob/needle/perf diffs; no quality verdict")
    cmp_.add_argument("base", help="baseline capture JSON")
    cmp_.add_argument("candidate", help="candidate capture JSON")
    cmp_.set_defaults(func=cmd_compare)

    sc = sub.add_parser("self-check", help="offline construction/CLI checks; does not call the server")
    sc.add_argument("--tokenizer-path", default=None)
    sc.set_defaults(func=cmd_self_check)

    acc = sub.add_parser(
        "accuracy",
        help="accuracy tiers: tf_short, tf_deep, needle, traces, trace_replay",
    )
    acc.add_argument("--name", required=True)
    acc.add_argument("--output", required=True)
    acc.add_argument("--tokenizer-path", required=True)
    acc.add_argument(
        "--tiers",
        default="tf_short,tf_deep,needle",
        help="comma list of tf_short,tf_deep,needle,traces,trace_replay",
    )
    acc.add_argument("--traces", default=None, help="reference capture holding xhigh traces")
    acc.add_argument(
        "--needle-slots",
        type=int,
        default=NEEDLE_DEFAULT_SLOTS,
        help="needle codes per (length, depth) cell; a subset still scores against a wider reference",
    )
    acc.add_argument(
        "--tf-deep-depths",
        default=None,
        help="comma list of context depths for the deep teacher-forced tier",
    )
    acc.add_argument(
        "--tf-deep-positions",
        type=int,
        default=TF_DEEP_POSITIONS,
        help=f"continuation positions scored per depth (max {TF_CONT_TOKENS - 1})",
    )
    acc.add_argument(
        "--needle-lengths",
        default=None,
        help="comma list of context lengths for the needle grid; default is the full grid",
    )
    acc.add_argument("--base-url", default=BASE_URL)
    acc.set_defaults(func=cmd_accuracy)

    sco = sub.add_parser("acc-score", help="score an accuracy capture against the reference")
    sco.add_argument("--reference", required=True)
    sco.add_argument("--candidate", required=True)
    sco.set_defaults(func=cmd_acc_score)

    am = sub.add_parser("agent-mix", help="concurrent long xhigh agent sessions + short chats")
    am.add_argument("--name", required=True)
    am.add_argument("--output", required=True)
    am.add_argument("--tokenizer-path", required=True)
    am.add_argument("--sessions", type=int, default=AGENT_SESSIONS)
    am.add_argument("--rounds", type=int, default=AGENT_ROUNDS)
    am.add_argument("--round1-tokens", type=int, default=AGENT_ROUND1_TOKENS)
    am.add_argument("--output-tokens", type=int, default=AGENT_OUTPUT_TOKENS)
    am.add_argument("--no-warmup", action="store_true")
    am.add_argument("--salt", default="r1", help="prefix salt; change it for a repeat run on the same boot")
    am.add_argument("--base-url", default=BASE_URL)
    am.set_defaults(func=cmd_agent_mix)

    lc = sub.add_parser("load-compare", help="compare two agent-mix runs")
    lc.add_argument("--base", required=True)
    lc.add_argument("--candidate", required=True)
    lc.set_defaults(func=cmd_load_compare)

    cf = sub.add_parser("conformance", help="OpenAI API semantics conformance checks")
    cf.add_argument("--name", required=True)
    cf.add_argument("--output", required=True)
    cf.add_argument("--base-url", default=BASE_URL)
    cf.set_defaults(func=cmd_conformance)

    bf = sub.add_parser("bootfacts", help="extract per-boot engine facts from the journal")
    bf.add_argument("--since", required=True, help="journalctl --since value")
    bf.add_argument("--output", default=None)
    bf.set_defaults(func=cmd_bootfacts)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except EvalError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
