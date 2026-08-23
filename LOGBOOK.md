# LOGBOOK

此文件是面向维护者的 append-oriented 历史证据，按本地日期追加事件，不改写旧快照。
当前操作事实以 README 与 systemd user unit（`vllm.service`、`vllm-gateway.service`、`vllm-watchdog.timer`）为准；历史条目里的配置不得当成“现在仍是如此”。
禁止写入密钥、token、生成文本或大段评测全文。数字与路径指向证据文件，不复制其内容。

## 2026-08-21

### 初始审计（accepted）

- 动机：在改并发、健康检查、配置源与权重之前，先看清仓库、运行时和文档各自负责什么。
- 变更：只读审计；未改权重、unit 或推理参数。
- 证据/结果：Python 环境在 `/Data/public/.venvs/vllm-qwen38`；vLLM 在 GPU 6–7、端口 8010，served name `qwen-3.8-27b-fp8`；gateway 8011；watchdog 每分钟探测。
- 决定/回滚：继续补齐 gateway 并发保护、真实 429、live/ready、`.env` 单一来源与可复现评测。无生产回滚。

### Gateway 并发保护、429 与 stream 槽位（accepted）

- 动机：上游并发不能无界；队满或等不到槽位必须失败得真实，不能假装成功。
- 变更：成功的流式响应把并发槽位持有到流结束、客户端断开或出错；队满或等待超时返回 OpenAI 风格 429，并带 `Retry-After`；默认 cap 4 / queue 8 / slot timeout 30s。
- 证据/结果：`gateway.py`；默认值见 README 与 `.env.example`。
- 决定/回滚：接受为默认。同日 cap 8 试验最终回滚，见后文。

### Health live/ready 与 watchdog（accepted）

- 动机：vLLM 冷加载可达数分钟，不能把“gateway 进程活着但上游未就绪”当成 gateway 故障而重启。
- 变更：`/health/live` 表示进程存活，恒 200；`/health/ready` 在上游 health 为 200 时就绪，否则 503，且不含上游错误细节；`/health` 与 ready 相同，保留兼容。watchdog 每分钟探 8010 `/health`（连续 15 次失败重启 vLLM）与 8011 `/health/live`（连续 3 次失败重启 gateway）。
- 证据/结果：`health_watchdog.sh`、`vllm-watchdog.timer`。
- 决定/回滚：接受。

### 三关键测试（accepted）

- 动机：锁住必须始终成立的槽位持有、队满/超时 429、以及 live/ready 语义。
- 变更：`test_gateway.py` 保留三项：`test_stream_holds_slot_until_done`、`test_queue_full_and_timeout_return_429`、`test_liveness_and_readiness`。
- 证据/结果：三项通过。
- 决定/回滚：只保留这三项，不扩大量 mock。

### `.env` 单一来源与 hf-mirror（accepted）

- 动机：计算节点没有到 `huggingface.co` 的直连；本地密钥只能有一份来源。
- 变更：忽略的 `.env`（mode 600）持有 `VLLM_API_KEY`、`HF_TOKEN`、`HF_ENDPOINT`；进程环境变量覆盖文件值；`.env.example` 是非秘密模板，镜像为 `https://hf-mirror.com`。
- 证据/结果：README「Local configuration」、AGENTS/CLAUDE「Network And Downloads」。本日志不记录任何密钥值。
- 决定/回滚：接受；`.env` 永不提交。

### 官方 FP8 下载与校验（accepted）

- 动机：用厂商序列化 block FP8（128×128）替代当时的 online per-tensor，并在切换前完成校验。
- 变更：官方权重落到 `/Data/public/Qwen3.8-27B-FP8`；unit 不传 `--quantization`。
- 证据/结果：权重目录含 `crc32.txt`、`safetensors-md5sum.txt`；下载走 `.env` 中的镜像与 token。校验全文不写入本日志。
- 决定/回滚：权重保留；是否切生产取决于随后的三方评测。

### eval_suite 建立及 cn→cr 修复（accepted）

- 动机：对 greedy、chosen-token logprob、needle、perf 做可复现 capture/compare，且拒绝覆盖结果、不打印密钥。
- 变更：建立 `/Data2/lzp/vllm-eval/eval_suite.py`，并完成 cn→cr 修复。
- 证据/结果：后续 capture 均由该工具写入 `/Data2/lzp/vllm-eval/results/`，见文末文件表。
- 决定/回滚：接受为工作树外评测工具；本仓库不内嵌评测生成物。

### Online per-tensor / 官方 block FP8 / BF16 三方评测（accepted 官方 FP8）

- 动机：精度以 BF16 为基准；性能在切换前对比当时的 online per-tensor 与官方 block FP8。
- 变更：依次 capture，不把生成文本写入本日志。
- 证据/结果：`online-per-tensor-fp8.json`（约 03:21）、`official-block-fp8.json`（约 03:47）、`bf16-anchor.json`（约 04:15）。官方 FP8 greedy 8/8，21k/128k/240k needle 全部通过，相对 BF16 的 mean abs logprob drift 0.117；online per-tensor drift 7.360。切换前官方相对 online 的吞吐：short c1 +16%，short c4 −32%，21k c1 +2%，21k c4 −4%。
- 决定/回滚：接受官方序列化 block FP8；拒绝 online per-tensor 作为生产权重。

### 生产切换官方 FP8（accepted）

- 动机：官方权重在 greedy/needle 上与 BF16 对齐，logprob drift 可接受；online per-tensor 的 drift 不可接受。
- 变更：`vllm.service` 的 `--model-path` 指向 `/Data/public/Qwen3.8-27B-FP8`，不传 `--quantization`；文档同步为官方 checkpoint。
- 证据/结果：三方评测见上条；README/AGENTS/CLAUDE 于当日记录官方路径。
- 决定/回滚：生产切到官方 FP8，未回滚。

### CPU / memory / NUMA / offload 32GiB（accepted）

- 动机：给 vLLM 明确的主机资源包络，并把 KV offload 定在 32GiB，而不是更大的 64GiB 猜测值。
- 变更：unit 设 `CPUQuota=1600%`、`CPUAffinity`/`AllowedCPUs=48-63`、`NUMAPolicy=bind`、`NUMAMask=1`、`MemoryHigh=112G`、`MemoryMax=128G`、`MemorySwapMax=0`，以及 `--kv-offloading-size 32`。
- 证据/结果：`official-block-fp8-resource32.json`（约 06:31）。GPU KV 约 1,222,850 tokens / 4.66 个满长上下文；capture 时 memory current/peak 约 54.8/55.6GiB；offload mmap 31.958GiB 且在 NUMA node 1 上 100%；CPU throttle 0。
- 决定/回滚：接受该包络与 32GiB offload。

### vLLM 升级审计（rejected 升级）

- 动机：确认 0.27.1 之后是否有值得立刻升级的稳定收益。
- 变更：只读审计；未升级 venv。
- 证据/结果：已安装 `vllm==0.27.1`（tag commit `6e448d0e` / `g6e448d0ea`）即当时 latest stable，无升级收益。
- 决定/回滚：拒绝升级，留在 0.27.1。

### L20 Triton kernel 调优（accepted）

- 动机：生产 L20 上这五个 w8a8 block-FP8 shape 没有设备 JSON，loader 对所有 M 走 v0.27.1 默认配置。
- 变更：在授权借用的空闲 GPU 3 上调优（生产 GPU 6–7 保持在线）。decode M 搜索 1/2/4/8/16/32；large M 64..8192 填相同 fallback。五个 `(N,K)` 为 `(8192,5120)`、`(7168,5120)`、`(17408,5120)`、`(5120,3072)`、`(5120,8704)`。候选 JSON 安装进 venv 的 vLLM configs 目录。
- 证据/结果：上游脚本钉在 tag `v0.27.1` commit `6e448d0e`。候选、runs、logs 保留在 `/Data2/lzp/vllm-tuning/`。安装回滚清单 `/Data2/lzp/vllm-tuning/install-backup/20260821-084146/`。08:48 曾因 `snapshot_limits` 的 `printf ---` 被当成选项而误判回滚，生产已恢复；08:54 确认 tuned boot 已加载五份 JSON。capture：`official-block-fp8-kernel-tuned.json`（约 09:06）；相对调优前 short c1 −2.90%，short c4 +13.65%，21k c1 +2.97%，21k c4 +0.33%；quality exact。
- 决定/回滚：保留 tuned JSON。升级或重建 venv 会覆盖这些文件，应按该 manifest 恢复或重调，不要当作 venv 的持久内容。

### maxseq=8（rejected / rolled back）

- 动机：降低 CUDA graph 占用，换更多 KV。
- 变更：试验 `max-num-seqs=8`，其余保持当时生产。
- 证据/结果：`official-block-fp8-kernel-maxseq8.json`（约 09:28）。graph −0.63GiB，KV +10.54% 到 5.16× 满长，但 short c4 −14.29%。
- 决定/回滚：拒绝该默认；回滚 `max-num-seqs` 默认 256。结果文件保留。

### batch=16384（rejected / rolled back）

- 动机：加大 `--max-num-batched-tokens` 看长上下文吞吐。
- 变更：试验 16384，对照生产 8192。
- 证据/结果：`official-block-fp8-kernel-batch16384.json`（约 10:02）。KV 4.54×，启动约 470s，short c4 −19.64%，无 long 吞吐收益。
- 决定/回滚：拒绝；回滚 8192。结果文件保留。

### gateway cap=8（rejected / rolled back 到 cap 4）

- 动机：短请求在 cap 4 下吃不满 GPU；试验全局 cap 8 是否可作默认。
- 变更：对照默认 cap 4 / queue 8 / timeout 30s 与 cap 8（其余不变），concurrency 8 走 gateway。
- 证据/结果：`gateway-cap4-c8.json`（约 10:16）、`gateway-cap8-c8.json`（约 10:24）。cap 4 short c8 = 247.89 tok/s；cap 8 short c8 = 477.03（+92.4%）。cap 4 的 long c8 在 30s slot timeout 返回真实 429；cap 8 long c8 = 22.02 tok/s，max E2E 46.44s，vLLM waiting 6 / queue_sum 118s，无 preempt/throttle。GPU KV 只有约 4.66 个满长上下文，全局 cap 8 不能保证 8 个 262K。
- 决定/回滚：因满长容量原则拒绝把 cap 8 作为默认，回滚 cap 4。cap 8 只可作为未来短请求专用 profile，不是默认。结果文件保留。

### 最终生产状态（accepted）

- 动机：把当日仍成立的运行时写清，并与历史试验分开。
- 变更：无新试验；默认保持如下。当前操作事实在 README 与 unit，不在本条的历史数字里。
- 证据/结果：官方 `/Data/public/Qwen3.8-27B-FP8` serialized block 128×128；unit 不传 `--quantization`；TP2，maxlen 262144，FP8 KV，prefix cache，MTP 3，batch 8192，maxseq 默认 256，offload 32GiB。gateway 默认 cap 4 / queue 8 / slot timeout 30s。`CPUQuota=1600%`，affinity/Allowed 48–63，NUMA bind 1，MemoryHigh 112G，MemoryMax 128G，Swap 0。运行时使用 tuned L20 JSON。GPU KV 约 1,222,850 tokens / 4.66 满长上下文。
- 决定/回滚：接受上述为当日结束时的生产状态。

### 统一日志入口（accepted）

- 动机：开发、调优、实验、运维、升级、故障、回滚和验证需要一份统一日志，文件名是 `LOGBOOK.md`，不是 `optimization-log.md`。
- 变更：新建本文件；README 增加入口；AGENTS 与 CLAUDE 的 Environment 补充 16 核 / 128G / NUMA 1 / offload 32GiB，并指向本文件。二者保持字节一致。
- 证据/结果：仓库内文档；不修改 `gateway.py`、测试、health、`.env`、`.env.example`、pyright 或 systemd unit。
- 决定/回滚：接受。之后只追加，不重写既有事件。

## 关键结果路径

评测工具：`/Data2/lzp/vllm-eval/eval_suite.py`。
结果目录：`/Data2/lzp/vllm-eval/results/`。

| 文件 | 用途 |
| --- | --- |
| `online-per-tensor-fp8.json` | 切换前 online per-tensor FP8 的精度与性能 |
| `official-block-fp8.json` | 官方 serialized block FP8 的精度与性能（资源包络与 kernel 调优之前） |
| `bf16-anchor.json` | BF16 精度基准 |
| `official-block-fp8-resource32.json` | 16 核 / 128G / NUMA 1 / offload 32GiB 之后的 capture |
| `official-block-fp8-kernel-tuned.json` | 安装 L20 tuned JSON 之后的 quality/perf |
| `official-block-fp8-kernel-maxseq8.json` | 被拒绝的 maxseq=8 |
| `official-block-fp8-kernel-batch16384.json` | 被拒绝的 batch=16384 |
| `gateway-cap4-c8.json` | 默认 cap 4 下 concurrency 8（long 为真实 429） |
| `gateway-cap8-c8.json` | 被拒绝作为默认的 cap 8 |

Kernel 产物（工作树外，保留）：候选 JSON `/Data2/lzp/vllm-tuning/candidate/`；runs `/Data2/lzp/vllm-tuning/runs/`；logs `/Data2/lzp/vllm-tuning/logs/`；安装清单 `/Data2/lzp/vllm-tuning/install-backup/20260821-084146/`。

## 2026-08-21（质量优先最终部署）

### 仓库与服务架构收敛（accepted）

- 架构：当日重构已把单一 active deployment selector 与 trusted deployment profile 设为运行时、checkpoint、served name、推理参数及 CPU/GPU/NUMA 放置的权威来源；gateway 与 watchdog 保持稳定服务入口。`~/.config/systemd/user/` 下的 `vllm.service`、`vllm-gateway.service`、`vllm-watchdog.service`、`vllm-watchdog.timer` 均已核实链接到仓库 `systemd/` 下同名权威 unit。
- 边界：本次最终切换只改 `config/deployments/qwen-3.8-27b-fp8.conf` 与本日志；未改 README、AGENTS、gateway、unit、selector、权重、served name、并发、batch、offload 或其他参数。

### A/B/C/D/A2 受控证据与决定（accepted D）

- 工作负载：每个 variant 先发 3 个并发约 32K prompt、`reasoning_effort=xhigh`、强制 4,096-token decode 的长请求，进入 steady decode 后再发 6 个短请求；五次均为 9/9 成功，错误/429/preemption 均为 0。

| Variant | KV / MTP | aggregate tok/s | long E2E p50 (s) | long TPOT p50 (ms) | short TTFT p50/p95 (s) | short E2E p50/p95 (s) | offload (GB) | 262K GPU-KV 并发 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | FP8 / MTP3 | 120.874 | 116.472 | 23.550 | 0.664/1.197 | 8.679/11.667 | 6.19 | 4.66 |
| B | FP8 / off | 98.341 | 148.240 | 31.062 | 0.654/1.120 | 11.820/15.842 | 6.16 | 5.18 |
| C | BF16 (`auto`) / MTP3 | 83.122 | 173.456 | 36.600 | 0.829/1.327 | 13.129/18.093 | 10.40 | 2.41 |
| D | BF16 (`auto`) / off | 90.338 | 161.159 | 34.142 | 0.721/1.198 | 13.092/17.460 | 10.07 | 2.76 |
| A2 | FP8 / MTP3 | 123.390 | 115.078 | 22.875 | 0.664/1.235 | 8.547/11.616 | 6.19 | 4.66 |

- 精度/API：A/B 的 FP8 KV 启动日志显示 checkpoint 没有已校准的 FP8 attention q/prob scale，实际使用 1.0 并明确警告可能影响精度；C/D 没有这些警告。A/C 的非零 `min_p` 与 `logit_bias` 因 MTP 返回 400；B/D 均返回 200，强制 token 与固定 seed 的 `min_p` 差异也证明参数实际生效。
- Quality guard：A/B/C/D 都通过 128K、235K retrieval 和 3/3 个 reference-validated reasoning/code 检查。该小型确定性套件只是回归 guard，没有统计功效，不能证明任一 variant 的质量优越性；A2 只用于 A 重复性能测量。
- 决定：显式质量、原生 262,144 context 与 vendor-equivalent API semantics 优先于吞吐/并发，因此永久部署 D：`--kv-cache-dtype auto` 且禁用 MTP。相对 A/A2，实测 aggregate throughput 低约 25.3%/26.8%；GPU KV 从约 1,222,850 tokens / 4.66 个满长上下文降到 724,163 / 2.76；该负载 offload traffic 从约 6.19GB 增至 10.07GB。这是已接受的容量/性能代价。
- 精确回滚：在同一权威 profile 中把 `--kv-cache-dtype auto` 恢复为 `--kv-cache-dtype fp8`，并在 `--enable-prefix-caching` 后恢复 `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`，其余字节保持不变，再按 owner-scoped mmap 流程仅重启 `vllm.service`。
- 持久证据：`/Data2/lzp/vllm-eval/results/20260821-134205-variant-d-quality-first/`；只保留最终 summaries、A/B/C/D startup/API/quality JSON、原配置恢复健康状态、永久部署验证和一份汇总报告，没有 raw Prometheus、脚本、失败尝试、Authorization header 或真实凭据。

### 永久 D 部署验证（accepted）

- 安全重启：停机前从 `vllm.service` cgroup 精确记录其打开的 `/dev/shm/vllm_offload_2f1f7f6e-8d9c-4b6e-b1ce-65e2192d1b91.mmap`（inode `27:18513595`）；stop 后该文件已自行消失，因此未删除任何 mmap，更未使用全局 glob。只重启 backend，126 秒恢复健康；gateway 与 watchdog timer 的 invocation ID 未变化且持续 active。
- 有效状态：进程参数为 max context 262144、TP2、显式 `--kv-cache-dtype auto`，没有 `--speculative-config`；启动配置为 `dtype=torch.bfloat16`、`kv_cache_dtype=auto`、`speculative_config=None`，即 BF16 compute/KV。容量为 22.55GiB / 724,163 GPU KV tokens / 2.76 个 262K context。启动日志无 uncalibrated FP8-KV q/prob scale 警告，五个 tuned L20 W8A8 block-FP8 weight kernel JSON 均加载。
- Live/API：backend health、gateway live/ready、带认证 models 均为 200 且只返回 `qwen-3.8-27b-fp8`，未认证 models 为 401；一次 `xhigh` 最高推理请求为 200、有 reasoning 且答案校验通过；`logit_bias`、`min_p=0`、非零 `min_p=0.95` 均为 200，强制 token 命中且两个 `min_p` token 序列不同。
- 仓库验证：Bash syntax、四个 systemd unit verify、gateway import、Python compile、4 项 unittest 均通过；Python primary LSP 为 0 finding。未改的 `gateway.py` 仍有 3 个 auxiliary 静态规则提示（两个 unchecked-throwing-call、一个 env secret-name heuristic），不属于本次 profile 决策范围；未据此改变 gateway 行为。
- 剩余限制：quality guard 不具统计功效；D 的满长并发和吞吐较低、offload 较高，容量规划必须按 2.76 个满长 GPU-KV context 而非原 4.66 执行。

### 测试布局整理（accepted）

- 变更：两项测试移入 `tests/`，watchdog 测试按新位置解析仓库根目录；README 改用根目录下的标准 unittest discovery。`LOGBOOK.md` 继续作为根目录唯一的追加式运维、决策与回滚记录。
- 决定：`gateway.py` 保留在根目录，继续作为唯一运行时模块和明确的 ASGI 入口；只有出现第二个内聚运行时模块、真实需要 package 时才迁移。`pyrightconfig.json` 保留在根目录，以维持 Pyright 的项目配置自动发现，不引入显式 `-p` 依赖。
- 证据/结果：Python primary LSP 3 文件 0 finding；gateway 仓库根目录 import、Python compile 与标准 discovery 的 4 项 unittest 均通过；backend health、gateway live/ready、认证 models 均为 200，相关服务与 watchdog timer 保持 active，未重启服务。
- 回滚：将两项测试移回根目录，恢复 watchdog 相对路径与 README 验证命令；本次无运行时或服务配置需要回滚。

## 2026-08-21（运行时 venv 命名迁移）

### canonical venv rename（accepted）

- 动机/变更：把带模型代号的 `/Data/public/.venvs/vllm-qwen38` 同文件系统原子改名为通用的 `/Data/public/.venvs/vllm-runtime-py311`；修复 `pyvenv.cfg`、56 个 bin shebang、3 个 activation script 与 vLLM console-script RECORD，并删除旧 bytecode；活动引用只改 deployment profile 的 `VLLM_PYTHON`/runtime `PATH`、Pyright venv 名和三份 tuning script 默认路径。
- 不变项：GPU 6–7、TP2、BF16 compute / `auto` KV、MTP off、262144 context、batch 8192、prefix caching、32 GiB offload、gateway cap 4 / queue 8 / timeout 30s，以及 model、parser、unit、launcher、selector、gateway、测试和 `.env` 均未改变。
- 身份/启动证据：package inventory SHA-256 `dafd8afe010be286c2f377e453dc20b053edfc83e22a95669cb918a16b1fe081`；`vllm==0.27.1` commit `g6e448d0ea`，7,098 条 RECORD 结构与既有 supplied hashes 保持验证；五份 mode 0644 tuned L20 JSON 的既有 SHA-256 保持不变并由新路径启动日志全部加载。installed CUDA source 动态导出的必需扩展为 `vllm._C_stable_libtorch`，旧/新路径均成功导入，`vllm._custom_ops` 也成功。
- 运行/API/测试：backend 约 258 秒恢复 health；TP0/TP1 分别在物理 GPU 6/7；backend health、gateway live/ready、认证 models 为 200，未认证 models 为 401；一次 `reasoning_effort=xhigh` 返回非空 reasoning（681 chars）且答案 `42` 校验通过；repository unittest 4/4 通过。
- 回滚：bundle 为 `/Data2/lzp/vllm-runtime-py311-rollback-20260822T042507Z/`；失败时先精确记录当前 `vllm.service` mmap，再在 attached shell 执行该目录的 `rollback.sh EXACT_FAILED_SERVICE_MMAP`，其归档会恢复旧目录、venv metadata/bin/RECORD 与全部 maintained files，并等待旧 stack healthy。

## 2026-08-22

### 评测工具入库（accepted，修订「不内嵌」决定）

- 动机：`eval_suite.py` 是 LOGBOOK 全部调优决定与 Phase 1–3 研究的唯一可复现基础，却没有版本控制，所在文件系统已用到 86%，丢失即失去在产配置的审计链。
- 变更：把 `/Data2/lzp/vllm-eval/eval_suite.py` 逐字节导入 `eval/eval_suite.py`，不重构、不做模型无关化（模型无关化留到待定研究结束之后，以免改动 corpus 或 prompt 使既有 reference 失效）。把只存在于 home 目录的研究决定文档一并导入 `eval/`，按仓库命名约定改为 UPPER_SNAKE_CASE：`PHASE1_RESULTS.md`、`PHASE2_RESULTS.md`、`PHASE3_CALIBRATION.md`、`PHASE3_RESULTS.md`、`FINAL_CONFIG.md`。`.gitignore` 增加 `eval/results/` 与 `eval/corpus/`。未改 `scripts/`、`systemd/`、`config/`、`gateway.py`、`tests/`、`.env.example`，未重启任何服务。
- 证据/结果：六个文件 `cmp` 全部与来源逐字节一致，`eval_suite.py` SHA-256 `5d79086cbc32f75040c6075e502416d404680047d4708ea4b46cabe913cce139`；用部署 profile 的 Python 3.11.15 `py_compile` 通过，仓库副本 `self-check` 离线自检通过；`unittest discover` 36 项通过。
- 决定/回滚：修订 2026-08-21「接受为工作树外评测工具；本仓库不内嵌评测生成物」——生成物（results、corpus）仍留在树外并按路径引用，工具本身是代码，入库。`/Data2/lzp/vllm-eval/` 的工作副本不动，仍是研究实际运行位置，导入时与仓库副本字节一致；何时改用仓库副本另行决定。原始 study 目录下的逐 cell 工作笔记与两份 incident 记录未入库，其结论已在上述 phase 文档中。
