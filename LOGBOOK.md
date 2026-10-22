# LOGBOOK

此文件是面向维护者的历史记录：按本地日期追加事件，写清做了什么、为什么、证据在哪里、是否回滚。旧条目是当时的快照，不改写，也不得当成“现在仍是如此”——当前事实见下一节。

条目格式为 动机 / 变更 / 证据·结果 / 决定·回滚。被拒绝的方案连同其理由与被接受的方案同等重要：它们挡住后来的会话去重跑同一条死路。

禁止写入密钥、token、生成文本或大段评测全文。数字与路径指向证据文件，不复制其内容。

## 当前状态与权威来源

- 模型事实——checkpoint、served name、任务（generate / embed）、默认 TP 与上下文长度、默认 vLLM 参数、发布日期与知识截止——以 `config/profiles/<name>.conf` 为准。共享部署的事实——引用哪个 profile、engine 端口、GPU 集合、CPU 核预算、它在 profile 默认值之上叠加的启动选项、gateway 的 reasoning-effort alias——以 `config/deployments/active` 选中的 deployment 为准。Python 运行时（仓库内 `.venv/bin/python`）、运行环境变量与 `127.0.0.1` 绑定只由 `scripts/run-vllm` 给出。
- CPU 与 NUMA 绑定不写在任何配置里。`scripts/resolve-placement` 每次启动从 GPU 集合经主机 sysfs 推导——共享部署取所在节点的前 `VLLM_CPU_CORES` 个核，按需实例取整个节点——`run-vllm` 把结果 numactl 到进程上并写进启动日志。systemd unit 不含任何 CPU/NUMA 指令。
- unit 名称、重启与启动限流策略、cgroup 限额以 `systemd/` 下的 unit 为准。架构、安装、运维、客户端配置与验证以 `README.md` 为准；测量规则以 `MEASUREMENT.md` 为准；研究证据以 `eval/` 下的 phase 文档为准。
- 在产容量与精度以引擎自己的启动日志为准，不以本文件的历史数字为准。最近一次影响它们的变更是 2026-08-23「Phase 4 与采用 FP8 KV cache」，当时实测 1,358,930 GPU KV tokens、262,144 满长上下文 5.18×。2026-10-22 把配置拆成 profile 与 deployment，部署的有效参数逐项不变，但部署本身未在新布局下重启过（见当日条目）。

## 证据与产物路径

| 位置 | 内容 |
| --- | --- |
| `eval/eval_suite.py` | 评测工具；与研究实际运行位置 `/Data2/lzp/vllm-eval/eval_suite.py` 逐字节一致 |
| `eval/` 下的 `PHASE1_RESULTS.md`、`PHASE2_RESULTS.md`、`PHASE2_RECOMMENDATION.md`、`PHASE3_CALIBRATION.md`、`PHASE3_RESULTS.md`、`PHASE4_DESIGN.md`、`PHASE4_RESULTS.md` | Phase 1–4 的研究设计与决定文档；「当前是什么配置、为什么」只由 `PHASE4_RESULTS.md` 回答 |
| `eval/reference/bf16-reference-traces.json` | trace-replay tier 的一次性 BF16 参考轨迹 |
| `MEASUREMENT.md` | 测量规则与各 tier 的分辨力边界 |
| `/Data2/lzp/vllm-eval/results/` | 2026-08-21 的 capture，见下表 |
| `~/vllm-study-results/` | Phase 1–3 的逐 cell capture 与两份 incident 记录 |
| `~/vllm-study-results/p4/` | Phase 4 的逐 cell 启动日志、分析 dump 与时间线（设计与结果已入库 `eval/`）；逐题原始记录与其工具在 `/Data2/lzp/vllm-eval-p4/` |
| `/Data2/lzp/vllm-tuning/` | L20 kernel 调优候选、runs、logs；安装清单 `install-backup/20260821-084146/` |
| `/Data2/lzp/vllm-runtime-py311-rollback-20260822T042507Z/` | venv 改名的回滚 bundle；其还原目标 `/Data/public/.venvs/vllm-runtime-py311` 已于 2026-08-23 迁入仓库 `.venv`，该 bundle 此后只作历史记录，不可直接执行 |
| `~/rollback/engine-phase4-20260823T034829Z/` | 切换 FP8 KV 之前的 profile 副本 |
| `/Data2/lzp/vllm-eval/results/20260828-workload-offload-sizing/` | 2026-08-28 offload 定尺寸的变更前后 Prometheus 快照、分析脚本、patch、verify runbook 与探针日志 |
| `~/rollback/kv-offload-64gib-20260828T105500Z/` | offload 32 → 64 GiB 之前的 profile 副本 |
| `~/rollback/venv-move-20260823T173204Z/` | venv 迁入仓库 `.venv` 的迁移前记录与回滚脚本 |
| `/Data2/lzp/vllm-eval/results/20261022-profiles-instances/` | 2026-10-22 单卡实例验证的引擎日志（27B 的失败与成功两次、8B embedding、9B）、批量探针脚本与启动计时 |

`/Data2/lzp/vllm-eval/results/` 下 2026-08-21 的逐次 capture：

| 文件 | 用途 |
| --- | --- |
| `online-per-tensor-fp8.json` | 切换前 online per-tensor FP8 的精度与性能 |
| `official-block-fp8.json` | 官方 serialized block FP8 的精度与性能（资源包络与 kernel 调优之前） |
| `bf16-anchor.json` | BF16 精度基准 |
| `official-block-fp8-resource32.json` | 16 核 / 128G / NUMA 1 / offload 32GiB 之后的 capture |
| `official-block-fp8-kernel-tuned.json` | 安装 L20 tuned JSON 之后的 quality/perf |
| `official-block-fp8-kernel-maxseq8.json` | 被拒绝的 maxseq=8 |
| `official-block-fp8-kernel-batch16384.json` | 被拒绝的 batch=16384 |
| `gateway-cap4-c8.json` | 当时默认 cap 4 下的 concurrency 8（long 为真实 429） |
| `gateway-cap8-c8.json` | 被拒绝作为默认的 cap 8 |
| `20260821-134205-variant-d-quality-first/` | A/B/C/D/A2 五变体对比与永久 D 部署验证 |

## 2026-08-21

### 初始审计（accepted）

- 动机：在改并发、健康检查、配置源与权重之前，先看清仓库、运行时和文档各自负责什么。
- 变更：只读审计；未改权重、unit 或推理参数。
- 证据/结果：Python 环境在 `/Data/public/.venvs/vllm-qwen38`；vLLM 在 GPU 6–7、端口 8010，served name `qwen-3.8-27b-fp8`；gateway 8011；watchdog 每分钟探测。
- 决定/回滚：继续补齐 gateway 并发保护、真实 429、live/ready、`.env` 单一来源与可复现评测。无生产回滚。

### Gateway 并发保护、429 与 stream 槽位（accepted）

- 动机：上游并发不能无界；队满或等不到槽位必须失败得真实，不能假装成功。
- 变更：成功的流式响应把并发槽位持有到流结束、客户端断开或出错；队满或等待超时返回 OpenAI 风格 429，并带 `Retry-After`；当日默认 cap 4 / queue 8 / slot timeout 30s。
- 证据/结果：当日版本的 `gateway.py`。
- 决定/回滚：接受为当日默认。同日 cap 8 试验最终回滚；这组默认值本身在当日稍晚的 gateway 加固里被重新参数化，见后文。

### Health live/ready 与 watchdog（accepted）

- 动机：vLLM 冷加载可达数分钟，不能把“gateway 进程活着但上游未就绪”当成 gateway 故障而重启。
- 变更：`/health/live` 表示进程存活，恒 200；`/health/ready` 在上游 health 为 200 时就绪，否则 503，且不含上游错误细节；`/health` 与 ready 相同，保留兼容。watchdog 每分钟探 8010 `/health`（当日阈值：连续 15 次失败重启 vLLM）与 8011 `/health/live`（连续 3 次失败重启 gateway）。
- 证据/结果：`scripts/health-watchdog`、`systemd/vllm-watchdog.timer`。
- 决定/回滚：接受。该 15 次阈值在 2026-08-22 引入真正的加载状态后降为 3。

### 三关键测试（accepted）

- 动机：锁住必须始终成立的槽位持有、队满/超时 429、以及 live/ready 语义。
- 变更：`test_gateway.py` 保留三项：`test_stream_holds_slot_until_done`、`test_queue_full_and_timeout_return_429`、`test_liveness_and_readiness`。
- 证据/结果：三项通过。
- 决定/回滚：当日只保留这三项，不扩大量 mock。套件此后随每次行为变更增长，当前数量以 `unittest discover` 的实际输出为准。

### `.env` 单一来源与 hf-mirror（accepted）

- 动机：计算节点没有到 `huggingface.co` 的直连；本地密钥只能有一份来源。
- 变更：忽略的 `.env`（mode 600）持有 `VLLM_API_KEY`、`HF_TOKEN`、`HF_ENDPOINT`；进程环境变量覆盖文件值；`.env.example` 是非秘密模板，镜像为 `https://hf-mirror.com`。
- 证据/结果：README 的配置边界一节（下载步骤已于 2026-08-23 由指令文件迁入该节）。本日志不记录任何密钥值。
- 决定/回滚：接受；`.env` 永不提交。

### 官方 FP8 下载与校验（accepted）

- 动机：用厂商序列化 block FP8（128×128）替代当时的 online per-tensor，并在切换前完成校验。
- 变更：官方权重落到 `/Data/public/Qwen3.8-27B-FP8`；不传 `--quantization`。
- 证据/结果：权重目录含 `crc32.txt`、`safetensors-md5sum.txt`；下载走 `.env` 中的镜像与 token。校验全文不写入本日志。
- 决定/回滚：权重保留；是否切生产取决于随后的三方评测。

### eval_suite 建立及 cn→cr 修复（accepted）

- 动机：对 greedy、chosen-token logprob、needle、perf 做可复现 capture/compare，且拒绝覆盖结果、不打印密钥。
- 变更：建立 `/Data2/lzp/vllm-eval/eval_suite.py`，并完成 cn→cr 修复。
- 证据/结果：后续 capture 均由该工具写入 `/Data2/lzp/vllm-eval/results/`。
- 决定/回滚：当日接受为工作树外评测工具；本仓库不内嵌评测生成物。该决定在 2026-08-22 被部分修订（工具入库，生成物仍在树外）。

### Online per-tensor / 官方 block FP8 / BF16 三方评测（accepted 官方 FP8）

- 动机：精度以 BF16 为基准；性能在切换前对比当时的 online per-tensor 与官方 block FP8。
- 变更：依次 capture，不把生成文本写入本日志。
- 证据/结果：`online-per-tensor-fp8.json`（约 03:21）、`official-block-fp8.json`（约 03:47）、`bf16-anchor.json`（约 04:15）。官方 FP8 greedy 8/8，21k/128k/240k needle 全部通过，相对 BF16 的 mean abs logprob drift 0.117；online per-tensor drift 7.360。切换前官方相对 online 的吞吐：short c1 +16%，short c4 −32%，21k c1 +2%，21k c4 −4%。
- 决定/回滚：接受官方序列化 block FP8；拒绝 online per-tensor 作为生产权重。

### 生产切换官方 FP8（accepted）

- 动机：官方权重在 greedy/needle 上与 BF16 对齐，logprob drift 可接受；online per-tensor 的 drift 不可接受。
- 变更：启动参数的 `--model` 指向 `/Data/public/Qwen3.8-27B-FP8`（由 `scripts/run-vllm` 传出），不传 `--quantization`；文档同步为官方 checkpoint。
- 证据/结果：三方评测见上条。
- 决定/回滚：生产切到官方 FP8，未回滚。

### CPU / memory / NUMA / offload 32GiB（accepted）

- 动机：给 vLLM 明确的主机资源包络，并把 KV offload 定在 32GiB，而不是更大的 64GiB 猜测值。
- 变更：unit 设 `CPUQuota=1600%`、`MemoryHigh=112G`、`MemoryMax=128G`、`MemorySwapMax=0`；CPU 与 NUMA 绑定不在 unit 里，由启动脚本经 numactl 施加（当时为 `--physcpubind=48-63 --membind=1`）；vLLM 参数加 `--kv-offloading-size 32`。
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
- 证据/结果：`gateway-cap4-c8.json`（约 10:16）、`gateway-cap8-c8.json`（约 10:24）。cap 4 short c8 = 247.89 tok/s；cap 8 short c8 = 477.03（+92.4%）。cap 4 的 long c8 在 30s slot timeout 返回真实 429；cap 8 long c8 = 22.02 tok/s，max E2E 46.44s，vLLM waiting 6 / queue_sum 118s，无 preempt/throttle。当时 GPU KV 只有约 4.66 个满长上下文，全局 cap 8 不能保证 8 个 262K。
- 决定/回滚：当日按满长容量原则拒绝把 cap 8 作为默认，回滚 cap 4。结果文件保留。这条推理的前提在当日稍晚被换掉：gateway 被重新定位为护栏而非调节器，容量保证交回引擎调度器，见「gateway 加固与版本控制入库」。

### 当日状态快照（已被同日后续条目取代）

- 动机：把当时仍成立的运行时写清，并与历史试验分开。
- 变更：无新试验。
- 证据/结果：官方 `/Data/public/Qwen3.8-27B-FP8` serialized block 128×128，不传 `--quantization`；TP2，maxlen 262144，FP8 KV，prefix cache，MTP 3，batch 8192，maxseq 默认 256，offload 32GiB。gateway 默认 cap 4 / queue 8 / slot timeout 30s。`CPUQuota=1600%`、`MemoryHigh=112G`、`MemoryMax=128G`、Swap 0，CPU/NUMA 由 numactl 绑到 48–63 / node 1。运行时使用 tuned L20 JSON。GPU KV 约 1,222,850 tokens / 4.66 满长上下文。
- 决定/回滚：这只是当日中段的快照。其中的 FP8 KV 与 MTP 在同日「A/B/C/D/A2 受控证据与决定」被换成 BF16 KV + MTP off，gateway 默认值在同日的 gateway 加固里被重新参数化，FP8 KV 又在 2026-08-23 以完全不同的证据重新采用。当前状态一律不从本条读取。

### 统一日志入口（accepted）

- 动机：开发、调优、实验、运维、升级、故障、回滚和验证需要一份统一日志，文件名是 `LOGBOOK.md`，不是 `optimization-log.md`。
- 变更：新建本文件；README 增加入口；`AGENTS.md` 与 `CLAUDE.md` 指向本文件。
- 证据/结果：仓库内文档；不修改 `gateway.py`、测试、health、`.env`、`.env.example`、pyright 或 systemd unit。
- 决定/回滚：接受。

### 仓库与服务架构收敛（accepted）

- 架构：当日重构已把单一 active deployment selector 与 trusted deployment profile 设为运行时、checkpoint、served name、推理参数与设备放置的权威来源；gateway 与 watchdog 保持稳定服务入口。`~/.config/systemd/user/` 下的四个 unit 均已核实链接到仓库 `systemd/` 下同名权威 unit。
- 边界：本次最终切换只改 `config/deployments/qwen-3.8-27b-fp8.conf` 与本日志；未改 README、AGENTS、gateway、unit、selector、权重、served name、并发、batch、offload 或其他参数。
- 后续：其中的 CPU/NUMA 放置在 2026-08-23 改为由 GPU 集合推导，不再写在 profile 里。

### A/B/C/D/A2 受控证据与决定（accepted D）

- 工作负载：每个 variant 先发 3 个并发约 32K prompt、`reasoning_effort=xhigh`、强制 4,096-token decode 的长请求，进入 steady decode 后再发 6 个短请求；五次均为 9/9 成功，错误/429/preemption 均为 0。

| Variant | KV / MTP | aggregate tok/s | long E2E p50 (s) | long TPOT p50 (ms) | short TTFT p50/p95 (s) | short E2E p50/p95 (s) | offload (GB) | 262K GPU-KV 并发 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | FP8 / MTP3 | 120.874 | 116.472 | 23.550 | 0.664/1.197 | 8.679/11.667 | 6.19 | 4.66 |
| B | FP8 / off | 98.341 | 148.240 | 31.062 | 0.654/1.120 | 11.820/15.842 | 6.16 | 5.18 |
| C | BF16 (`auto`) / MTP3 | 83.122 | 173.456 | 36.600 | 0.829/1.327 | 13.129/18.093 | 10.40 | 2.41 |
| D | BF16 (`auto`) / off | 90.338 | 161.159 | 34.142 | 0.721/1.198 | 13.092/17.460 | 10.07 | 2.76 |
| A2 | FP8 / MTP3 | 123.390 | 115.078 | 22.875 | 0.664/1.235 | 8.547/11.616 | 6.19 | 4.66 |

- 精度/API：A/B 的 FP8 KV 启动日志显示 checkpoint 没有已校准的 FP8 attention q/prob scale，实际使用 1.0 并明确警告可能影响精度；C/D 没有这些警告。A/C 的非零 `min_p` 与 `logit_bias` 因 MTP 返回 400；B/D 均返回 200，强制 token 与固定 seed 下两个 `min_p` 的 token 序列不同，证明参数确实生效。
- Quality guard：A/B/C/D 都通过 128K、235K retrieval 和 3/3 个 reference-validated reasoning/code 检查。该小型确定性套件只是回归 guard，没有统计功效，不能证明任一 variant 的质量优越性；A2 只用于 A 重复性能测量。
- 决定：显式质量、原生 262,144 context 与 vendor-equivalent API semantics 优先于吞吐/并发，因此部署 D：`--kv-cache-dtype auto` 且禁用 MTP。相对 A/A2，实测 aggregate throughput 低约 25.3%/26.8%；GPU KV 从约 1,222,850 tokens / 4.66 个满长上下文降到 724,163 / 2.76；该负载 offload traffic 从约 6.19GB 增至 10.07GB。这是已接受的容量/性能代价。
- 持久证据：`/Data2/lzp/vllm-eval/results/20260821-134205-variant-d-quality-first/`；只保留最终 summaries、A/B/C/D startup/API/quality JSON、原配置恢复健康状态、部署验证和一份汇总报告，没有 raw Prometheus、脚本、失败尝试、Authorization header 或真实凭据。
- 后续：这条决定在 2026-08-23 被 Phase 4 推翻。当时唯一的 quality guard 自己承认没有统计功效，而 Phase 4 用有功效的配对检验重新度量了同一取舍。

### 永久 D 部署验证（accepted）

- 安全重启：停机前从 `vllm.service` cgroup 精确记录其打开的 `/dev/shm/vllm_offload_2f1f7f6e-8d9c-4b6e-b1ce-65e2192d1b91.mmap`（inode `27:18513595`）；stop 后该文件已自行消失，因此未删除任何 mmap，更未使用全局 glob。只重启 backend，126 秒恢复健康；gateway 与 watchdog timer 的 invocation ID 未变化且持续 active。
- 有效状态：进程参数为 max context 262144、TP2、显式 `--kv-cache-dtype auto`，没有 `--speculative-config`；启动配置为 `dtype=torch.bfloat16`、`kv_cache_dtype=auto`、`speculative_config=None`。容量为 22.55GiB / 724,163 GPU KV tokens / 2.76 个 262K context。启动日志无 uncalibrated FP8-KV q/prob scale 警告，五个 tuned L20 W8A8 block-FP8 weight kernel JSON 均加载。
- Live/API：backend health、gateway live/ready、带认证 models 均为 200 且只返回 `qwen-3.8-27b-fp8`，未认证 models 为 401；一次 `xhigh` 最高推理请求为 200、有 reasoning 且答案校验通过；`logit_bias`、`min_p=0`、非零 `min_p=0.95` 均为 200，强制 token 命中且两个 `min_p` token 序列不同。
- 仓库验证：Bash syntax、四个 systemd unit verify、gateway import、Python compile、当时的 4 项 unittest 均通过；Python primary LSP 为 0 finding。
- 剩余限制：quality guard 不具统计功效；D 的满长并发和吞吐较低、offload 较高，当时的容量规划按 2.76 个满长 GPU-KV context 执行。

### 测试布局整理（accepted）

- 变更：两项测试移入 `tests/`，watchdog 测试按新位置解析仓库根目录；README 改用根目录下的标准 unittest discovery。
- 决定：`gateway.py` 保留在根目录，继续作为唯一运行时模块和明确的 ASGI 入口；只有出现第二个内聚运行时模块、真实需要 package 时才迁移。`pyrightconfig.json` 保留在根目录，以维持 Pyright 的项目配置自动发现。
- 证据/结果：Python primary LSP 3 文件 0 finding；gateway 仓库根目录 import、Python compile 与标准 discovery 的 4 项 unittest 均通过；backend health、gateway live/ready、认证 models 均为 200，相关服务与 watchdog timer 保持 active，未重启服务。
- 回滚：将两项测试移回根目录，恢复 watchdog 相对路径与 README 验证命令；本次无运行时或服务配置需要回滚。

### canonical venv rename（accepted）

- 动机/变更：把带模型代号的 `/Data/public/.venvs/vllm-qwen38` 同文件系统原子改名为通用的 `/Data/public/.venvs/vllm-runtime-py311`；修复 `pyvenv.cfg`、56 个 bin shebang、3 个 activation script 与 vLLM console-script RECORD，并删除旧 bytecode；活动引用只改 deployment profile 的 `VLLM_PYTHON`/runtime `PATH`、Pyright venv 名和三份 tuning script 默认路径。
- 不变项：GPU 6–7、TP2、BF16 compute / `auto` KV、MTP off、262144 context、batch 8192、prefix caching、32 GiB offload、gateway 当时的并发默认值，以及 model、parser、unit、launcher、selector、gateway、测试和 `.env` 均未改变。
- 身份/启动证据：package inventory SHA-256 `dafd8afe010be286c2f377e453dc20b053edfc83e22a95669cb918a16b1fe081`；`vllm==0.27.1` commit `g6e448d0ea`，7,098 条 RECORD 结构与既有 supplied hashes 保持验证；五份 mode 0644 tuned L20 JSON 的既有 SHA-256 保持不变并由新路径启动日志全部加载；installed CUDA source 动态导出的必需扩展 `vllm._C_stable_libtorch` 与 `vllm._custom_ops` 在新旧路径均成功导入。
- 运行/API/测试：backend 约 258 秒恢复 health；TP0/TP1 分别在物理 GPU 6/7；backend health、gateway live/ready、认证 models 为 200，未认证 models 为 401；一次 `reasoning_effort=xhigh` 返回非空 reasoning（681 chars）且答案校验通过；repository unittest 4/4 通过。
- 回滚：bundle 为 `/Data2/lzp/vllm-runtime-py311-rollback-20260822T042507Z/`；失败时先精确记录当前 `vllm.service` mmap，再在 attached shell 执行该目录的 `rollback.sh EXACT_FAILED_SERVICE_MMAP`。

### gateway 加固与版本控制入库（accepted）

- 动机：本仓库此前不是 git 仓库，没有任何文件有历史，AGENTS/CLAUDE 的「Git and Delivery」整节不可执行。同时 gateway 上有几处“报告得不真实”的行为需要一并修掉再入库。
- 变更（先加固后入库，一并进入初始 commit）：并发准入被重新参数化为护栏而不是主要调节器——默认 cap 4 / queue 8 / 30s 改为 in-flight 32 / queue 32 / 等待 300s，理由是引擎调度器才是 prefix-cache 与 KV 感知的一方，gateway 对此完全盲。超时分类拆开，上游读超时返回 504 而不是伪装成 502（`httpx.ReadTimeout` 是 `RequestError` 的子类，先前被误分类）；health 探测另给一个短超时，避免上游卡死时 `/health/ready` 挂满读超时；SSE 流中途失败发出 OpenAI 形状的错误帧而不再静默停住（非流式没有安全的带内错误帧，改为让传输层报告截断）；header 以原始字节转发；`reasoning_effort` 的 `high` 归一化。非流式生成没有中途字节，读超时因此界定整次生成，默认 3600s。
- 变更：初始 commit 把工作中的部署整体纳入版本控制——gateway、deployment profile 与 selector、入口脚本、四个 systemd user unit 与运维文档；`.env`、Python 缓存与任务产物由 `.gitignore` 排除，`.env.example` 留作非秘密模板。
- 证据/结果：`tests/test_gateway_socket.py` 让至少一项测试真正走 socket 上的 HTTP，先前 504/502 与流中途失败这两个缺陷正好活在那个空隙里。
- 决定/回滚：接受。此后每次行为变更都以独立 commit 落库，本日志与 commit body 互为索引。

## 2026-08-22

### Phase 1：显式 KV 池；拒绝 max-num-seqs 32（accepted / rejected）

- 动机：内存 profiler 按 CUDA graph 的**估计值** 1.99 GiB/rank 扣减，而实际只用 0.42 GiB，这段差额被扣住不给 KV 池。
- 变更：用显式 `--kv-cache-memory-bytes 27000000000` 取代隐式的 gpu-memory-utilization 比例。取值刻意落在引擎自己给出的变更后锚点 28,480,849,408 之下，保留约 1.5 GiB 余量——池子过大不是在启动时失败，而是在 CUDA graph capture 时失败。
- 证据/结果：GPU KV 724,163 → 807,188 tokens，262,144 满长并发 2.76× → 3.08×。精度影响恰好为零：对 BF16 参考的 teacher-forced logprob drift 0.000000，40,429 个位置 0 次 argmax flip，而该仪器的噪声底由同 boot 与跨 boot 重复实测为零。aggregate 吞吐与基线相差 0.03%，10 项 API conformance 全过。稳态 3.2 核、约 41 GiB charged working set，CPU throttle 与 cgroup memory event 均为零。`eval/PHASE1_RESULTS.md`。
- 同期拒绝：`--max-num-seqs 32` 配显式 `cudagraph_capture_sizes` 列表。它不带来任何额外容量（KV token 与并发一字未变），却花掉 2.1% aggregate 吞吐和 18% 的短 prompt TTFT p95，并且改变 torch.compile hash——单是一次重编译就让 tier-1 精度移动约 0.57 mean / 2.8 pp，这本身会污染后续所有比较。
- 决定/回滚：接受显式字节数，拒绝 maxseq/capture-list 组合。回滚即删除该行。

### Phase 2：KV 池回退；拒绝三条精度分支（accepted / rejected）

- 事故与动机：`27000000000` 比引擎自己的“完全利用显存”锚点 26,055,386,624 高出 944 MB，超额占用设备。它在自己那次启动上活了下来，却在后来一次启动上以 `EngineDeadError` 杀死引擎——一次约 956 MiB 的分配正好撞上 peak activation。关键教训是失败点不在启动：过大的 KV 池在 CUDA graph capture 阶段才失败，所以一次“看起来成功”的启动完全不能证明这个值是安全的。
- 变更：降到 `26000000000`，取回约 1 GiB 设备余量。
- 证据/结果：容量 777,975 KV tokens / 2.97× 满长并发（不安全值为 807,188 / 3.08×）；负载差异不可测（aggregate +0.80%，long E2E p50 −1.85%，其余在噪声内）；对 BF16 参考的 drift 仍是 0.000000 / 0 flips，conformance 10/10。`eval/PHASE2_RESULTS.md`、`eval/PHASE2_RECOMMENDATION.md`（当时名为 `FINAL_CONFIG.md`）。
- 拒绝 `fp8_per_token_head`（TRITON_ATTN）：它兑现了预测的 5.35× 容量，动态 per-token-head 定标完全不需要校准，也不产生任何 scale 警告——按精度它是好的。但它强制 TRITON_ATTN，而短 prompt TTFT p95 从 0.59 s 炸到 **6.97 s**，aggregate −3.5%、long TPOT +4.0%。**按性能拒绝，不是按精度拒绝**。
- 拒绝 `int8_per_token_head`：deep tier 上是所有量化选项里最准的（2.08 pp），容量与上者相同，但最慢：aggregate −4.8%、long TPOT +5.5%、短 TTFT p95 6.06 s。
- 拒绝 MTP（k=2）：在目标并发 4–8 上实测 aggregate −0.57%、long TPOT p50 **+7.2%**，与历史上并发 3 时记录的 +33.8% **符号相反**。接受率本身是健康的（整体 72.8%，每 draft 1.456 个 token，深度 1/2 为 81.5%/64.1%）——speculation 在 batch 8 上就是不划算。它另外吃掉 9% 容量，`min_p` 与 `logit_bias` 返回 400（conformance 8/10），还暴露一个未解释的缺陷：MTP + `prompt_logprobs` 在 30 篇 teacher-forced 重放文档里有 6 篇（198–466 prompt tokens）出现 86–100% 的 top-1 不一致与 9–12.5 nats 的散度，其余 24 篇正常。设计原本要求把 `min_p`/`logit_bias` 的破坏作为一个取舍抛给用户；既然没有任何收益可换，就没有需要问的问题。
- 一条被测量推翻的 Phase 1 结论：TRITON_ATTN 的门禁是对着一个被污染的对照打的——那个对照同时带着已被单独拒绝的 `--max-num-seqs 32`。对着真正上线的配置重测，TRITON_ATTN 没有 decode 收益（long TPOT p50 +0.67%，在噪声内），还要花 1.67% aggregate、13.9% long TTFT p95、18.9% 短 E2E p95。整条 per-token-head 分支一直在付一笔门禁替它掩盖掉的 kernel 税。
- 决定/回滚：回到 `26000000000`；三条分支均不上线，测量数据保留以免重跑。

### `/dev/shm` KV offload 区域泄漏与回收（accepted）

- 事故：vLLM 只在 `SharedOffloadRegion.cleanup()` 里 unlink `/dev/shm/vllm_offload_<engine-id>.mmap`，被 SIGTERM 的引擎永远走不到那里，因此每次重启都泄漏一个 34.32 GB 的区域。累计 8 个孤儿达到 **274.6 GB**，填满 252 GB 的 `/dev/shm`，把 NUMA node 1 的可用内存压到 2 GB，并导致一次启动被主机 OOM 杀掉。
- 变更：`run-vllm` 在分配新区域之前清扫孤儿。只有同时被证明“属于我们”（普通文件、名字匹配 `vllm_offload_*.mmap`、属主是本进程有效 UID；vLLM 以 0600 创建区域，`/dev/shm` 是 sticky 的，内核本身也拦着）和“无人使用”（读 `/proc/<pid>/fd` 与 `/proc/<pid>/maps`，没有任何活进程持有其描述符或映射）才会删除。证明不了的一律原样留下，回收失败绝不阻止引擎服务。
- 证据/结果：对着在跑的引擎验证——它自己的区域被正确留下，引擎保持健康。测试用真实文件与真实持有进程驱动真实脚本，覆盖描述符与仅映射两条存活路径。
- 决定/回滚：接受。此前的做法是人工精确记录 mmap 再重启，现在这件事在每次启动自动完成。

### 入口脚本前言合并；`--api-key` 实现后被推翻（accepted / reverted）

- 动机：`run-vllm` 与 `run-gateway` 各带约 20 行几乎相同的前言，而这个分歧已经造成了缺陷：只有 `run-gateway` 在 source profile 之前清掉继承值，所以一个继承来的 `VLLM_SERVED_MODEL_NAME` 能悄悄满足 `run-vllm` 的“必须在 profile 里”检查。
- 变更：两者共用 `scripts/_profile.sh`，由它解析 active profile 并清掉 profile 拥有的每一个名字。`VLLM_GPUS` 过去被校验却从不使用，现在成为 `CUDA_VISIBLE_DEVICES` 的来源（`VLLM_RUNTIME_ENV` 里显式给出的仍然优先，因此当日行为逐字节不变）。host check 的注释不再声称 `free -h` 会失败。
- **实现后被测量推翻**：审计给出的“一行高价值修复”是把 `VLLM_API_KEY` 作为 `--api-key` 传给 vLLM，理由是当前只靠 vLLM 的环境变量默认值鉴权。照做之后测量发现，这台 47 租户主机上 `/proc/<pid>/cmdline` 全局可读，而且 vLLM 会把解析后的参数回显进 journal——这个 flag 会把生产凭据公开给机器上每一个用户。因此**实现、测量、回退**：改为由一个测试钉住密钥永不出现在命令行，并由就绪门禁在每次启动断言未认证请求确实被拒绝。审计提出的风险是真的，它给的修复不是。
- 证据/结果：重启 `vllm.service` 验证，引擎按真实 profile 起来，未认证 `/v1/models` 401、带凭据 200。新测试把两个脚本当作真实进程对着一次性目录树运行。

### 让 `vllm.service` 有真正的加载状态，让 watchdog 说实话（accepted）

- 动机：systemd 里根本没有“引擎正在加载”这个状态。`Type=simple` 让 unit 从多分钟模型加载的第一秒起就是 active，`TimeoutStartSec` 因此是惰性的，watchdog 的 `activating` 跳过分支永远不可达。watchdog 只能靠 15 个周期的阈值撑住冷启动；而因为 `systemctl restart` 对一个正在加载的 unit 总是成功，它每分钟重置自己的计数器并退出 0——**一个永远不服务的系统每分钟报告一次成功**，直接违反 Failure Principle。
- 变更：`scripts/wait-ready` 作为 `ExecStartPost` 阻塞到服务器应答 `/health`，unit 因此在 activating/start-post 里停留整个加载过程，上界是 `TimeoutStartSec=20min`。它每轮检查 `$MAINPID`（systemd 在主进程死后仍停在 start-post，没有这个检查，一次崩掉的加载会把自己的重启拖满整个启动超时），并断言未认证的 `/v1/models` 被拒绝——这正是让 `run-vllm` 那个 `VLLM_API_KEY` 检查有意义、又不必把密钥放到全局可读命令行上的办法。两个长驻 unit 加 `StartLimitIntervalSec`/`StartLimitBurst` 让限流器真正可达：`RestartSec=5s` 的间隔对管理器默认的 10s/5 太宽，致命故障会永远重启而不进入 failed。恢复动作是 `systemctl --user reset-failed` 后 `start`。watchdog 阈值 15 → 3（现在只需要给运行中的引擎去抖），防御性读取计数器，用 `--no-block` 重启以免 10 分钟的 stop 拖住 gateway 探测，并对任何失败探测返回非零——包括那些重启成功的：重启是干预，不是健康。
- 证据/结果：实测一次冷加载中 unit 在 activating/start-post 停留 2m26s，只有引擎真的服务时才进入 active；同一窗口内手工运行 watchdog 打印 `backend skip (activating)`、以 0 退出且计数器未动；就绪门禁对健康引擎 35 ms 退出 0。`.env.example` 同步为实际默认值。

### 拒绝 prompt-logprob 请求（accepted）

- 事故：一次普通的、已认证的请求带上 `prompt_logprobs`，让引擎跳过 prefix cache、按 8192 token 分块重新 prefill 整个 prompt，并物化一个 8192 × 248,320 的 logits 张量；设备内存耗尽，引擎被终止，systemd 重启了服务。这发生在研究开始时，是一次意外，不是攻击。
- 变更：gateway 以 OpenAI 形状的 400 拒绝，说明拒绝了什么、为什么；不改写任何响应体，且拒绝发生在取得槽位之前。`echo` 一并覆盖，因为 vLLM 由它推导 prompt logprobs：chat completions 的 `top_logprobs` 默认 0 而不是 null，因此总会请求；completions 的 `logprobs` 默认 null，只有客户端设了才请求——后一种情况仍是 200，`echo` 在那里无害。原始字节预过滤让普通流量不必为一次 JSON 往返买单。
- 决定/回滚：这是真实的能力削减，如实记录：gateway 上没有 prompt logprobs。需要它的测量工作直接访问引擎自己的 loopback 端口，harness 也据此把 teacher forcing 的 prompt 长度在客户端封顶。

### Phase 3：复现、校准问题，与被推翻的短 TTFT p95 结论（accepted 结论，未改生产）

- 动机：Phase 2 把 FlashInfer + `fp8_e4m3` 标为测过的最快、容量最高的配置，却按“不得有未校准 scale 警告”这条硬线、以及 short TTFT p95 +121% 拒绝了它。两个理由都需要复核。
- 复现：候选与对照在三次启动上复现到 aggregate 0.07–0.12%、long TPOT p50 0.20–0.32%，比 Phase 1 假定的噪声底紧十倍。
- **short TTFT p95 不是统计量**：同一启动、同一配置的两次运行相差 20.9%，配置内跨度 21–58%。n≈102 时 p95 是第 97 大的样本，由碰巧排在一次分块长 prefill 后面的两三个请求决定。Phase 2 那个“+121%”从来不曾脱离噪声。把重复运行汇总（n=279–306）之后，FP8 反而拥有所有配置里**最快的中位首 token**（p50 −0.132 s），代价是 p90 +0.074 s；而短对话真正等的是整条回复：FP8 每 token 快 5.13 ms，256 token 的回复端到端**快 1.30 s**，约为首 token 代价的 19 倍。
- **深上下文漂移是 kernel 效应而不是精度效应**：以生产 FLASH_ATTN 引擎为共同参照，BF16 的 FlashInfer 对照是 2.10 pp argmax flip，FP8 是 **0.79 pp**——换 attention kernel 比把 KV 量化到 FP8 移动得更多。needle grid 100/100（含 250k 处 25/25），且没有随深度增长的饱和特征：最深的一档反而最干净。
- 另一处记录更正：Phase 1/2 报告的 deep tier 非确定性是 **FLASH_ATTN 的属性**，不是仪器的属性。在 FlashInfer 上该 tier 跨启动 100% 精确为零。把 0.26 pp 当作“仪器噪声底”会白白报废一个在另一个 backend 上完全精确的 tier。
- 拒绝 `--calculate-kv-scales`（无效）：对本架构，它在任何层被构造之前就被 hybrid 模型钩子关掉（48 个 GDN 层），产生的引擎配置与不加它逐字节相同，警告一字未变。更值得记的陷阱是：在**非** hybrid 模型上，它会取第一次真实前向的单批 amax、随即自我关闭，而因为该警告只在 `not calculate_kv_scales` 分支发出，它会在完全没有改善精度的情况下**让警告消失**——伸手去拿这个 flag 是买静音，不是买校准。没有为它花一次负载运行：把一个 no-op 测两遍是充数，不是证据。
- 不可能项 `fp8_e5m2` 差分探针：引擎直接拒绝该配置（`fp8_e5m2 kv-cache is not supported with fp8 checkpoints`），e5m2 KV 只在 checkpoint 带 `kv_cache_scheme` 时允许，本厂商 block-FP8 checkpoint 没有。启动失败被 harness 捕获，自动恢复 known-good profile 并复验健康，服务未曾中断。
- 评估后不做的离线校准：`llmcompressor`/`modelopt` 未安装且无网络出口；自制校准需要 27.8 B 模型常驻并在 16 个 full-attention 层挂 forward hook，跑不进生产服务形状；结果还要写回 30 GB 厂商 checkpoint，把仓库刻意原样运行的厂商产物 fork 掉。上游对 hybrid recurrent 模型的这套做法本身也持否定态度。
- 决定：不单方面采用 FP8，生产留在 BF16，把判断交给用户；但把 Phase 2 的两条拒绝理由分别改判为“该统计量不可测量”和“可在证据上豁免”，并明确指出还缺两件事：自由生成的推理正确性，以及长 decode 驻留后的精度。`eval/PHASE3_CALIBRATION.md`、`eval/PHASE3_RESULTS.md`。

### 评测工具入库（accepted，修订「不内嵌」决定）

- 动机：`eval_suite.py` 是 LOGBOOK 全部调优决定与 Phase 1–3 研究的唯一可复现基础，却没有版本控制，所在文件系统已用到 86%，丢失即失去在产配置的审计链。
- 变更：把 `/Data2/lzp/vllm-eval/eval_suite.py` 逐字节导入 `eval/eval_suite.py`，不重构、不做模型无关化（模型无关化留到待定研究结束之后，以免改动 corpus 或 prompt 使既有 reference 失效）。把只存在于 home 目录的研究决定文档一并导入 `eval/`，按仓库命名约定改为 UPPER_SNAKE_CASE。`.gitignore` 增加 `eval/results/` 与 `eval/corpus/`。未改 `scripts/`、`systemd/`、`config/`、`gateway.py`、`tests/`、`.env.example`，未重启任何服务。
- 证据/结果：六个文件 `cmp` 全部与来源逐字节一致，`eval_suite.py` SHA-256 `5d79086cbc32f75040c6075e502416d404680047d4708ea4b46cabe913cce139`；用部署 profile 的 Python 3.11.15 `py_compile` 通过，仓库副本 `self-check` 离线自检通过；`unittest discover` 36 项通过。
- 决定/回滚：修订 2026-08-21「接受为工作树外评测工具；本仓库不内嵌评测生成物」——生成物（results、corpus）仍留在树外并按路径引用，工具本身是代码，入库。`/Data2/lzp/vllm-eval/` 的工作副本不动，仍是研究实际运行位置。原始 study 目录下的逐 cell 工作笔记与两份 incident 记录未入库，其结论已在 phase 文档中。

### 测量规则成文（accepted）

- 动机：Phase 1–3 产生的“怎样才算诚实地测量这个服务”的知识没有任何持久去处：匹配 compile hash 的对照要求、每个指标实测的噪声底、哪些数根本不是统计量、精度底属于哪个 backend、那次要掉一条引擎命的 `prompt_logprobs` 危险、每个 tier 能分辨什么，以及没有任何 phase 关闭的两个缺口。写下来，正是为了让已经犯过的那两次昂贵错误——对着被污染的对照打门禁、用单次 p95 做拒绝——不再重演。
- 变更：新增 `MEASUREMENT.md`。phase 文档仍是证据，本日志仍是历史，README 只得到一个指针，不复述任何数字。
- 决定/回滚：接受为常设规则。纯文档变更，未改脚本、unit、profile 或 gateway，未重启服务。

### 保留 BF16 参考推理轨迹（accepted）

- 动机：trace-replay 精度 tier 靠一次性在 BF16 参考配置上生成的推理轨迹给候选打分。harness 本身就把它们标为“只生成一次”，它们与某个 stack 状态绑定：venv、vLLM build 或生产精度一变，重新生成就会得到不同的参考，并终结与既有分数的可比性。唯一副本原本在即将被清理的研究输出目录里。
- 变更：91 KB，作为 fixture 入 `eval/reference/`。这是未来测量的参考**输入**而不是测量结果，因此不推翻“评测生成物不入库”的决定。
- 证据/结果：sha256 `4d6c774bb1312338d10f38b03e948e76bef632a155fc0551dd29deec849fc746`。

## 2026-08-23

### Phase 4 与采用 FP8 KV cache（accepted）

- 动机：Phase 3 之后剩下的是一个纯残余风险判断，而它依赖的两件事——自由生成的推理正确性、长 decode 驻留之后的精度——在此前任何 phase 都没有被测过。用户的第一优先级是不让推理能力退化，这种量级的变更不能建立在冒烟测试上。
- 变更：`--kv-cache-dtype auto` → `--kv-cache-dtype fp8_e4m3`（该 dtype 直接选中 FlashInfer backend），并**移除** `--kv-cache-memory-bytes`：Phase 3 与 Phase 4 里每一次 FP8 启动都是引擎按 0.92 自行定尺寸的，把 FP8 与更大的显式池配对从未被测量，不能未经测试就上线。profile 其余部分逐字节未变。
- 容量与性能：GPU KV 777,975 → **1,358,930** tokens，262,144 满长并发 2.97× → **5.18×**，“并发 ≥ 4 且保持原生上下文”这条要求由此在满长上就成立；aggregate 吞吐 +7.7%（Phase 3 独立测得 +6.6%），long TPOT −14.2%（一次 3,000 token 的 `xhigh` 推理轮次快 17.2 s），长请求端到端每轮 −13.3 s、四轮 agent 会话 −53 s，短对话每轮端到端 −1.30 s，同负载下 KV 压力 37% → 24%。唯一的长边回归是 long TTFT p95 +2.29 s。
- 自由生成推理正确性，N=402 配对：252 条 MMLU-Pro 分层覆盖全部 14 个 category，150 条 level ≥ 3 的 MATH-500；greedy、固定 seed、相同 prompt 与顺序、`reasoning_effort: xhigh`，配对 McNemar 精确检验。达成灵敏度为 80% power 下可检出 **3.4 pp**。生产 81.34%，FP8 82.84%，p = 0.308，95% CI [−1.12, +3.73] pp；对着同 kernel 的 BF16 对照则是 −0.25 pp，p = 1.000。
- **零假设臂是这一节的关键**：把生产配置对着它自己重启一次再跑，402 题里 25 题不一致、摆动 +0.75 pp；换成 FP8 是 24 题不一致、+1.49 pp——比零假设臂**更少**的不一致项，方向与幅度相同。并发 32 下的自由 greedy 生成本来就在约 6% 的题目层面上是混沌的，FP8 没有往上加。
- 长 decode 驻留（此前从未测过）：每题生成约 9,591–10,410 token 的账本任务，逐步精确模算并按 (题, 步) 配对。FP8 在全部 **1,040** 个步骤和全部 **128** 次长程自回忆（内容只存在于 decode 写入的 KV 里）上全对，与生产完全一致，把 FP8 引起的深 decode 步骤错误率单侧 95% 上界压到 **0.38 pp**。整个矩阵里唯一出过账本错误的配置是 **FlashInfer + 普通 BF16 KV**，不是 FP8。
- greedy 分歧：并发 1 下引擎跨重启逐 token 可复现（24/24 相同，21,018 个比较 token 零分歧），因此任何分歧都可归因。FP8 对生成轨迹的扰动不比在相同 BF16 精度下换 attention kernel 更大（首次分歧中位 46 vs 48 token），且其风险不随 decode 位置上升——那正是“累积驻留误差”必须留下的特征。
- 上线后核验：146 s 恢复健康，NRestarts=0；engine `/health` 与 `/v1/models` 200，gateway live 200，一次经 gateway 的 `xhigh` 端到端补全 finish_reason stop 且答案正确；`num_preemptions_total` 0；`/dev/shm` 252 GB 中用 32 GB、一个活的 offload 区域；cgroup 42.9 GiB，`nr_throttled` 0/1,856 periods，`memory.events` 全零。
- 残余与未覆盖，如实记录：未校准的 `k_scale = v_scale = 1.0` 警告仍在，Phase 3 已确认在本 build 与本 checkpoint 上无法用任何可用手段清除——`--calculate-kv-scales` 对 hybrid 模型被强制关闭，checkpoint 不带 KV scale 张量，FlashInfer 路径上不存在任何动态定标的 FP8 dtype，e5m2 对 fp8 checkpoint 被拒绝。在浮点格式上 scale 1.0 是**恒等**定标，代价是动态范围余量而不是相对精度，而且 query 从未被量化（`prefill=bf16, decode=bf16`）；本次未发现它的任何行为后果。未覆盖：agentic 多轮编码（本机没有可评分的参考数据，只经由未评分的分歧探针间接覆盖）、任何小于约 3 pp 的变化、以及持续数小时的生产流量（本次最长的连续 FP8 负载是并发 32 下约 35 分钟的正确性运行，零错误）。
- 证据：`~/vllm-study-results/p4/`，设计与功效计算在第一次比较运行**之前**冻结。四次启动、十二次运行、4,824 次评分自由生成、390 万生成 token，零请求错误。
- 回滚：一行加一次重启——把 `--kv-cache-dtype auto` 放回并恢复 `--kv-cache-memory-bytes 26000000000`，然后重启 `vllm.service`。变更前的 profile 副本在 `~/rollback/engine-phase4-20260823T034829Z/`。

### `VLLM_DEPLOYMENT` 覆盖 active selector（accepted）

- 动机：研究要启动候选 profile，此前只能覆写生产 selector 或覆写生产 profile 本身。
- 变更：selector 仍是默认值，环境里的 `VLLM_DEPLOYMENT` 优先。该名字刻意不在共享前言的 unset 列表里——与 profile 自己拥有的名字不同，它按设计就是从环境来的。
- 证据/结果：今天行为逐字节不变，因为没有任何地方设置它。它同时是将来做 per-instance unit 的挂点：所有入口脚本都经同一段前言解析 profile，因此不需要再改其中任何一个。

### 端口在 profile 里声明一次（accepted）

- 动机：8010/8011 这一对被写在 profile、`run-gateway`、`wait-ready`、`health-watchdog` 和 `.env.example` 五处。一个不声明自己端口的 profile 无法被复制到 8010 已被占用的机器上——本机就有其他租户占着 8000、8001、8020、8030、8040、8050、8060——而 `.env` 覆盖是按 unit 加载的，能扩展到“另一台机器”，扩展不到“第二个实例”。
- 变更：profile 声明 `VLLM_PORT`；引擎绑定它，gateway 在下一个端口应答并代理到它，两个健康探测都从它派生。`--port` 离开 `VLLM_ARGS`，五处默认值收敛到一处，在共享前言里校验一次。`wait-ready` 与 `health-watchdog` 也改为 source 那段前言，因此一个不可读的 profile 会让启动失败、让 watchdog 这一轮判为不健康——这是诚实的结果：那时没有任何东西能说出这个部署应该在哪里应答。
- 证据/结果：整组一次落地是刻意的——在仍有消费者硬编码 8010 的情况下先从 `VLLM_ARGS` 移除 `--port`，会在完整模型加载之后才失败，而门禁在轮询一个没人绑定的端口。变更成为 load-bearing 之前先验证：`run-vllm` 与 `run-gateway` 用生产 profile 的值经记录器 dry-run，产出 `--port 8010` / `--port 8011`、upstream `http://127.0.0.1:8010`、numactl 绑定 48-63；真实的 `wait-ready` 与 `health-watchdog` 对着在跑的服务运行，都派生出 8010/8011 并报告健康。43 项测试通过。

### CPU 与 NUMA 放置由 GPU 集合推导（accepted）

- 动机：`VLLM_CPU_AFFINITY` 与 `VLLM_NUMA_NODES` 是按本机拓扑手算的，也是 deployment profile 里唯一一处换台机器就会**静默出错**的东西：错误的 node 会在这个部署的整个生命周期里付跨 socket 内存带宽的代价，而没有任何东西会报告它。
- 变更：GPU 集合是操作者唯一需要陈述的放置事实，node 与 CPU 池现在都由它经 sysfs 推导，得到的绑定写进启动记录。核数仍显式保留为 `VLLM_CPU_CORES`——那是策略（这个部署在共享主机上可以占用该 node 多少），不是拓扑，推导它等于猜。同时设置 `VLLM_NUMA_NODES` 与 `VLLM_CPU_AFFINITY` 仍可绕过推导，这是给“该 node 的前 N 个 CPU 不归我们”的主机准备的答案。profile 也因此丢掉了 `CUDA_VISIBLE_DEVICES` 条目，它只是复述了 `run-vllm` 已经导出的 `VLLM_GPUS`。
- 不可信的拓扑一律失败而不猜：GPU 跨两个 NUMA node、设备报告 `numa_node = -1`、node 的 CPU 数少于请求、GPU 索引越界，以及 `CUDA_DEVICE_ORDER` 不是 `PCI_BUS_ID`（那时索引根本不指这些设备）。
- 证据/结果：验收条件被精确满足——本机 GPU 6,7 解析为 node 1 与 48-63，与它替换掉的值逐字节相同。对着真实 sysfs 验证，并用生产 profile 经记录器 dry-run `run-vllm`，其 `Cpus_allowed_list` 为 48-63。失败路径也对着真实拓扑走了一遍：GPU 3,4 跨 node 0 与 1，GPU 8 不存在，200 核放不进 node 1。

### 拒绝主机未授予的 GPU（accepted）

- 动机：这台机器八块 GPU 里有六块跑着其他租户 25–33 GiB 的负载，而在此之前，唯一挡住本服务碰它们的东西是一个 profile 里写对了的常量。把 CPU 与 NUMA 放置也从同一个常量推导出来之后，一个笔误的代价变大了：错的 GPU 集合现在还会连带放错 CPU 与内存。
- 变更：`.env` 里的 `VLLM_HOST_GPUS` 陈述这台机器给了本服务哪些 GPU，`run-vllm` 在任何东西对该集合动手之前拒绝集合之外的任何 GPU。不设置即不限制，那是自有整机时正确的默认。没有启发式，没有误报；按整数索引成员判断，因此允许列表 10,11 不会放行 GPU 1。
- 证据/结果：用生产 profile 加本机允许列表 dry-run 验证：6,7 解析为 node 1 / 48-63 并继续；一个要 0,1 的 profile 以 `GPU 0 is not in VLLM_HOST_GPUS (6,7)` 失败。

### reasoning-effort alias 随 deployment 配置（accepted）

- 动机：`gateway.py` 把 `reasoning_effort` 的 `high` 映射到 `xhigh`，是一个编译进共享代码的常量。那其实是关于**某一个 checkpoint 的 chat template** 的事实：对一个接受 `high` 的模型——大多数模型都接受——gateway 会把一个合法请求改写成模板会拒绝的请求，也就是说 README 自己的「新增或切换部署」流程带着一个静默的行为回归出厂。
- 变更：改为 `VLLM_GATEWAY_REASONING_EFFORT_ALIAS`，放在 deployment profile 而不是 `.env`——它是被服务模型的属性，不是机器的属性。`run-gateway` 把它交给 gateway，共享前言先清掉它，因此一个对此只字未提的 profile 不会从残留环境里继承上一个模型的拼法。未设置即原样转发；格式非法在 import 时失败，而不是静默地不做别名。让普通流量免付 JSON 往返的原始字节预过滤保留，现在针对配置值。
- 证据/结果：这是共享代码里最后一个模型专属常量。用生产 profile dry-run `run-gateway` 验证 alias 以 `high:xhigh` 到达，而省略它的 profile 即使环境里有残留值也保持未设置。49 项测试通过。

### README 与 LOGBOOK 重写（accepted）

- 动机：`README.md` 被指定为架构、安装、运维、客户端配置与验证的权威，却缺了上下文长度、KV 容量与精度决定，而且其中若干条陈述已被最近十五个 commit 作废；`LOGBOOK.md` 完全没有 2026-08-22 与 2026-08-23 两天，四个 phase 的取舍研究与那十五个 commit 全部未记录，同时一个已被推翻的「最终生产状态」标题坐在目录里，一份证据索引夹在两段历史之间。
- 变更（README）：重新组织为 架构 → 上下文/精度/容量 → 配置边界 → 安装 → 运维 → 客户端与健康检查 → 新增或切换部署 → 验证与测量。补上原生 262,144 上下文、当前 1,358,930 GPU KV tokens / 5.18× 满长并发、FP8 e4m3 精度决定与其指针；补上重启会阻塞整个模型加载、启动限流与 `reset-failed` 恢复、gateway 对 `prompt_logprobs`/`echo` 的 400、`/dev/shm` 自动清扫、以及编译缓存的清理流程。更正 gateway 并发默认值、`.env` 与 profile 各自的职责、端口约定、以及验证命令中缺失的脚本与测试文件。研究数字与逐次实验一律不进 README。
- 变更（LOGBOOK）：重组为 当前状态与权威来源 → 证据与产物路径 → 按日期的历史；三个同为 2026-08-21 的顶层小节合并为一个；补齐 2026-08-22 与 2026-08-23；修正九项已核实的文档缺陷——权威指向、被取代的「最终生产状态」标题、把 CPU/NUMA 放置归给 unit、`--model-path` 的错误 flag 名、`health_watchdog.sh` 的旧文件名、指向 README 不存在小节的引用、关于 AGENTS/CLAUDE 的失实断言、测试数量的自相矛盾，以及夹在文中的证据索引。
- 证据/结果：README 自己的验证块逐字运行通过，49 项测试 OK；两份文档的内部引用全部落在真实存在的文件与小节上；文中每个数字都对着 profile、commit body、phase 文档或在跑的引擎核对过。未改 `scripts/`、`systemd/`、`config/`、`gateway.py`、`tests/`、`eval/`、`MEASUREMENT.md`，未重启任何服务。
- 遗留：
  - Phase 4 的决定文档只在树外的 `~/vllm-study-results/p4/`，而在产精度正是靠它决定的；`eval/` 目前只有 Phase 1–3。按 Phase 1–3 的先例，它应当同样入库。（已关闭，见下条）
  - `eval/FINAL_CONFIG.md` 的标题是「推荐的最终配置」，内容是 BF16 KV 加 `--kv-cache-memory-bytes 26000000000`。那是 Phase 2 的产物，已被 Phase 4 取代，但从标题读不出来。（已关闭，见下条）
  - `MEASUREMENT.md`「What no phase measured」称自由生成推理正确性“在任何 phase 都没有跑过”，Phase 4 已关闭该缺口，该节需要更新。（已关闭，见下条）
  - harness 的启动失败恢复目标已从会漂移的私有副本改为从 git HEAD 读取已提交的 profile，但 `MEASUREMENT.md` 的「Reproducing a cell」仍按旧做法叮嘱刷新私有副本。（已关闭，见下条）
  - `CLAUDE.md`/`AGENTS.md` 的 Environment 仍称 profile 是 CPU/NUMA 放置的权威来源，且未提及 `.env` 里的 `VLLM_HOST_GPUS`。**（仍开放，由用户单独处理）**

### 关闭 Phase 4 与测量文档的遗留项（accepted）

- 动机：上一条留下的五项遗留里有四项会让读者读到已经不成立的事实——决定在产精度的 Phase 4 文档不在树内、`eval/FINAL_CONFIG.md` 的标题自称是最终配置、`MEASUREMENT.md` 说自由生成正确性从未跑过、以及它仍按一份已被删除的私有副本描述 harness 的失败恢复路径。
- 变更（`eval/`）：`PHASE4_DESIGN.md` 与 `PHASE4_RESULTS.md` 由 `~/vllm-study-results/p4/` 入库。设计逐字节原样——它在第一次比较运行之前冻结，这一性质本身是证据的一部分。结果只动两处：指向设计文档与 p4 产物的路径改为在 `eval/` 里能解析的形式，以及当时留下的两条交接注记标为已关闭（profile 变更已作为 `62db0d9` 提交；harness 的私有恢复副本已删除）。`FINAL_CONFIG.md` 更名为 `PHASE2_RECOMMENDATION.md` 并在开头写明被取代——不新写一份「当前配置」文档：在产配置的权威是 profile，叙述在 `README.md`，日期与回滚在本文件，理由与证据在 `PHASE4_RESULTS.md`，`eval/` 里再放第五份只会以同样的方式过时。Phase 3 的建议小节加一行前指，因为它写着「现在跑的是 BF16」；四处失效的文档间引用一并修好。
- 变更（`MEASUREMENT.md`）：「What no phase measured」重写，写清两个 tier 现在是什么、能分辨到什么程度（正确性在实测 5–7% 不一致率下 80% power 可检出 3.4 pp；长驻留 800 步全对把深 decode 步骤错误率单侧压到 0.38 pp），以及仍然没测的三件事（agentic 多轮编码本机无可评分数据、小于约 3 pp 的变化、持续数小时的生产流量），并如实写明这两个 tier 的工具**不在** `eval/eval_suite.py` 里，而在 `/Data2/lzp/vllm-eval-p4/`。「Reproducing a cell」第 1 步改写为现状：harness 用 `git show HEAD:` 恢复已提交的 profile，因此新增了对 git 历史与一个能启动的 `HEAD` profile 的依赖。另修两处已失效的陈述：gateway 的准入上限不是 4/8 而是默认 32/32，以及推理轨迹参考已作为 fixture 入库，不再「仓库里没有任何备份」。
- 证据/结果：数字逐条核对——在跑的引擎自报 `fp8_e4m3`、FlashInfer、GPU KV 1,358,930 tokens、262,144 满长 5.18×；`gateway.py` 的默认值为 32/32；`/Data2/lzp/vllm-eval/known-good.conf` 已不存在，`restore_profile.sh` 从 `HEAD` 读取；`eval/eval_suite.py` 与研究运行位置仍逐字节一致。README 的验证块逐字运行通过，49 项测试 OK；被改文档的引用全部落在真实存在的文件上。未改 `README.md`、`scripts/`、`systemd/`、`config/`、`gateway.py`、`tests/`，未重启任何服务。
- 未做/仍开放：Phase 4 的逐题原始记录与打分脚本仍在树外（体量与「评测生成物不入库」的既有决定），`MEASUREMENT.md` 已点明这一依赖；`CLAUDE.md`/`AGENTS.md` 的那一项由用户单独处理。

### venv 迁入仓库 `.venv`（accepted）

- 动机：解释器与库此前在 `/Data/public/.venvs/vllm-runtime-py311`。其父目录 `/Data/public` 在这台 47 租户主机上是 `root:root` 0777 且没有 sticky bit，任何用户都可以改名或替换 `.venvs` 这个目录项，从而替换本服务实际执行的解释器。这不是整洁问题而是可篡改面。新位置 `/Data/lzp/vllm-service/.venv` 的父链是 `/Data/lzp` 0700 与仓库 0775，都属 lzp。
- 变更（迁移本身）：同文件系统（`/dev/sdc`）原子改名，耗时 4 ms，7.8 GiB 未发生任何复制。改名之前先把 `.venv/` 写进 `.gitignore`，使它在任何时刻都不可能被 stage。venv 内部按迁移前扫描出的 60 个文本文件逐一重写绝对路径：`pyvenv.cfg` 的 `command`、`bin/` 下 56 个 shebang、3 个 activation script；另按新内容重算 42 份 dist-info 中的 56 条 `RECORD` 哈希与大小；删除 1,244 个 `__pycache__` 下共 8,394 个 `.pyc`（全部内嵌旧路径），交由首次导入重建。
- 变更（引用）：deployment profile 的 `VLLM_PYTHON` 与 runtime `PATH`；`pyrightconfig.json` 改为约定式相对布局 `venvPath: "."` / `venv: ".venv"`，顺带去掉一处硬编码的主机绝对路径；`eval/eval_suite.py` 的 `CORPUS_SOURCE_ROOT`，并以 `cmp` 确认与 `/Data2/lzp/vllm-eval/eval_suite.py` 仍逐字节一致；树外 harness 的 8 个脚本；`CLAUDE.md` 与 `AGENTS.md` 的 Runtime Environment 段（两份仍只差自称文件名那一行）；`README.md` 架构段补一句，说明该解释器在仓库内、git 忽略、且 checkout 不可复现。`eval/PHASE3_CALIBRATION.md` 与既有 capture 里的旧路径是带日期的历史记录，不改写；上一条 venv 改名的 bundle 在证据表里标注为不可直接执行。
- 不变项：model、checkpoint、served name、端口、GPU 6–7、TP2、`fp8_e4m3` KV、262144 context、batch 8192、prefix caching、32 GiB offload、reasoning-effort alias、gateway、systemd unit、selector 与 `.env` 都没有改变；Python 3.11.15 / vLLM 0.27.1 / torch 2.13.0+cu130 与迁移前相同。
- 证据（身份）：五份 tuned L20 block-FP8 JSON 的 SHA-256 与迁移前记录逐条相同，且新路径的启动日志在 TP0 与 TP1 上各自加载了全部五份；`vllm._C_stable_libtorch` 与 `vllm._custom_ops` 导入成功；`hf`、`pip`、`vllm`、`uvicorn` 四个 console script 与 activation script 均可执行；venv 内已无任何文本文件指向旧路径；`bin/` 条目的 RECORD 失配从迁移前既有的 57 条降到 2 条，剩下两条是早于本次的 `bin/__pycache__` 缺失项。
- 证据（运行）：引擎 4 分 30 秒恢复 health，略长于常态，与清空 bytecode 后的重编译相符；两个 unit `active` 且 `NRestarts=0`。容量与迁移前逐位相同：GPU KV 1,358,930 tokens、262,144 满长 5.18×、`fp8_e4m3`、FlashInfer。backend `/health`、gateway `/health/live`、`/health/ready`、`/health` 与认证 models 均 200，未认证的 models 与 chat completions 均 401；经 gateway 的一次 `reasoning_effort=high` 返回 259 字符 reasoning 且答案正确，同一取值直连引擎返回 400，说明 alias 仍在生效。README 的验证块逐字运行通过，49 项测试 OK；`git status` 只含本次的 8 个文件，`.venv` 下任何路径都无法被 track，`.git` 仍为 1.8 MiB。
- 附带发现：Coder 镜像的同步层按名字排除 `.venv/`，但并不遵循 `.gitignore`——实测同样被 git 忽略的 `.venv-synctest/` 被同步，而 `.venv/` 没有。因此 `.venv` 这个名字本身是必要条件，换成任何其它名字都会把 7.8 GiB 拖进镜像同步。
- 决定/回滚：接受。`/Data/public/.venvs` 清空后已删除。回滚 bundle 为 `~/rollback/venv-move-20260823T173204Z/`，其 `rollback.sh` 执行反向原子改名、还原 venv 内路径、revert 本次 commit 并重启服务。

## 2026-08-28

### CPU KV offload 32 → 64 GiB（accepted）

- 动机：消费方的四个 agent 实验是串行的长上下文会话链，上下文 50–180k 并在会话中增长，thinking on，约 78% wall-clock 在 decode。主要尾部风险是一次未命中的约 98k prefill：TTFT 约 37 s，期间其余每条流从约 30 tok/s 掉到约 0.63 tok/s。CPU offload tier 本应把被逐出的前缀救回来，实测却几乎从不命中。
- 诊断：本模型是 hybrid，64 层里只有 16 层 full attention，每 token KV = 16 × 2 × 4 kv head × 256 dim × 1 B = 32,768 B（跨两个 TP rank 合计），block = 1568 token = 51.38 MB；由 9.108 TB / 177,387 block = 51,347,278 B/block 独立验证。于是 32 GiB tier 只有 668 block，而 GPU pool 是 867 block / 1,358,930 token——**L2 比 L1 还小**。被逐出的前缀在 CPU 里的历史深度比它刚失去的更浅，0.11% 的外部命中率是算术结果，不是缺陷。
- 变更：`--kv-offloading-size 32` → `64`。profile 其余部分与 unit 均未改动。
- 证据（变更前；661 s 窗口 + 5 天生命期）：GPU prefix 命中 96.95% / 91.94%；外部命中窗口内 0/122,193，生命期 158,368/140,815,061 = 0.11%；store 9.108 TB 对 load 5.805 GB = 1569:1，占用 3,081 s 传输时间；`num_preemptions_total` 0，offload 分配失败 0。TTFT p50 0.90 s、p95 13.6 s、p99 34.7 s。prefill 累计 57,936 s = 运行时长的 13.4%，其中 54% 落在占 6% 的 >10 s 请求上；计算 >20k token 的 1,839 个请求（7.6%）承担了全部 140.8 M 计算 token 的约 65%。
- 尺寸依据：64 GiB = 1,337 block = 2.10 M token = GPU pool 的 1.54×，按实测 0.383 block/s 的写入率驻留约 58 分钟，覆盖父上下文等待子任务的时间尺度。恢复实测 16.4 GB/s = 499,084 token/s，重算约 2,430 token/s，快 205×。更大的 tier 不会增加写入：`v1/kv_offload/cpu/manager.py` 的 `prepare_store` 过滤掉已驻留的 key，写流量只会持平或下降。
- 为什么不是 96/128/192 GiB：`/dev/shm` 共 252 GiB，其中另一租户泄漏了四个 34.32 GB 的 offload 区域（合计 137 GB），不属于我们也无法回收；回收自己那个之后可用约 125 GiB。64 GiB 用后尚余 61 GiB 且 cgroup 限额不必改；96 GiB 只余 29 GiB 并需要抬高 MemoryHigh/MemoryMax；2 小时驻留需要约 128 GiB，在本机拿不到。
- 精度与长上下文不变：offload 路径是纯字节拷贝——`v1/kv_offload/cpu/gpu_worker.py` 断言 GPU 与 CPU 张量都是 `torch.int8` 的 page 视图，按指针/长度描述符搬运，没有 dtype 转换或重量化，恢复出来的 block 与被逐出的逐位相同。KV dtype、量化、定标、context 长度、sampler 与 speculative decoding 均未改动。上线后引擎自报容量与变更前逐位相同：GPU KV 1,358,930 tokens、262,144 满长 5.18×、`fp8_e4m3`。
- 上线后核验：重启 613 s（11:01:14 发起 → 11:11:27 active），`init engine` 457.69 s，其中 `torch.compile` 166.53 s。这是一次冷编译：`~/.cache/vllm/torch_compile_cache` 在本次之前已被清空，而 `kv_offloading_size` 本身也进入 cache config hash，因此该目录无论如何都要重建。旧 32 GiB 区域由 `run-vllm` 回收，新区域 68,695,359,488 B（63.98 GiB），`/dev/shm` 192 GiB 已用、余 61 GiB。verify 全部通过：两个 unit active，engine `/health` 与 gateway `/health/live`、`/health/ready` 均 200，未认证的 engine `/v1/models` 401，gateway 列出模型，`kv_offloading_size=64.0`，一次经 gateway 的 `xhigh` 端到端补全 finish_reason stop 且答案正确，`num_preemptions_total` 0，`memory.events` 全零。cgroup 92.56 GiB，其中不可回收的是 anon 10.22 + shmem 64.87 = 75.09 GiB，其余约 17 GiB 是可回收 page cache，对 MemoryMax 128 GiB 仍余 52.9 GiB。49 项测试 OK。
- 并发探针（引擎空载，61,577 token 共享前缀，added 1/4/8）：per-stream decode 40.13 / 34.84 / 30.58 tok/s，aggregate 38.8 / 132.6 / 224.0 tok/s，TTFT p50 0.38 / 0.75 / 1.48 s，`waiting_max` 0，preemption 0。这组数不能与变更前那条 `t ≈ 24.1 + 2.44·c` ms 的拟合直接比较：拟合是在四个实验在跑、上下文 50–180k 的混合负载下测的，本次是空载均匀前缀。它只说明 decode 伸缩健康、无回归。
- 未被本次验证的部分，如实记录：prefill-interference 探针给出 cold TTFT 35.67 s、期间每流 0.610 tok/s，对比变更前约 37 s / 约 0.63 tok/s **基本没变，而这是预期的**——探针的冷 prompt 是新生成的唯一文本，任何 tier 都不可能命中它；本次变更针对的是**重复** prefill，不是首次 prefill。真正的验收指标是外部 prefix 命中率，而重启后 tier 是空的（当前 235,149 次查询 0 命中），需要 30–60 分钟真实实验流量才能读数。若在 64 GiB 下它仍低于约 10%，正确的下一步是**移除**这一层——它目前每 5 天花掉 3,081 s 传输时间只换到 0.11% 命中——而不是继续加大。
- 未改动，及其理由：`--max-num-batched-tokens` 保持 8192，把它降到 4096 只是把停顿切碎而不是缩短它，且 16384 使 short c4 −19.64% 已有记录，与本次同窗口改动还会破坏归因。0.27.1 里 `long_prefill_token_threshold` 确实存在并按请求逐 step 截断 token（`v1/core/sched/scheduler.py:521`、`:899`），是停顿仍在时的下一个候选，需单独 A/B；`max_num_partial_prefills` 与 `max_long_partial_prefills` 在本 build 中不存在。`--max-num-seqs` 已两次被拒（8 与 32）且进入编译 hash；gateway 32/32 是失控保护而非节流器，cap 8 已被拒；`--gpu-memory-utilization` 0.92 不动；MTP 不在本次范围。
- 附带发现：「证据与产物路径」表里的几个历史目录在磁盘上已不存在——`~/vllm-study-results/` 缺失，`/Data2/lzp/vllm-eval/` 在本次之前也不存在，其 `results/` 由本次重建。本条不改写那张表的历史行，只记录这一事实。
- 决定/回滚：接受。回滚是一行加一次重启——把 `--kv-offloading-size` 改回 32 并重启 `vllm.service`；变更前的 profile 副本在 `~/rollback/kv-offload-64gib-20260828T105500Z/`，SHA-256 `f618004e…3198f`。因为编译缓存目录已被清掉且该参数进入 hash，回滚那次重启同样是冷编译，约 10 分钟。
- 证据：`/Data2/lzp/vllm-eval/results/20260828-workload-offload-sizing/`；探针 JSON 在消费方 `logs/analysis/concurrency/bench_20260828T181748Z.json`。

### 64 GiB offload tier 的中期观测：verdict 推迟（no change）

- 观测：18:11:27Z 重启之后，新一轮四实验 18:28Z 启动，读数取到 19:24Z。本条只记录观测——未重启、未改 profile、未改 unit。
- 数字：`vllm:external_prefix_cache_hits_total` = **0**，对 `vllm:external_prefix_cache_queries_total` = 1,374,055（协调方在 18:30–19:16Z 窗口内读到 0/1,031,151，自重启累计 0/1,361,645）。`vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` = 12,507,936 / 13,881,991 = **90.10%**。`vllm:kv_offload_store_bytes_total` = 108.90 GB，`vllm:kv_offload_store_size_count` = 664，`vllm:kv_offload_cpu_allocation_size_sum` = **2,121** block，`vllm:kv_offload_lookup_sync_delay_seconds_count` = 402 且 `vllm:kv_offload_load_size_count` 无样本（一次 load 都没发生过）。`vllm:kv_cache_usage_perc` 瞬时 0.251（协调方窗口内约 0.055）。`vllm:num_preemptions_total` 0，`vllm:kv_offload_allocation_failure_total` 无样本。
- 为什么 0% 还不是结论：tier **不是空的**——2,121 次 block 分配打进 1,337 个槽位，已经绕过 1.59 圈，写入路径完全正常。原因在读侧：`vllm:external_prefix_cache_queries_total` 与 `vllm:prompt_tokens_by_source_total{source="local_compute"}` 逐位相等（都是 1,374,055），即这一层只会被问到 GPU pool 未命中的那些 token。而这一轮上下文是全新的，1,358,930 token 的 GPU 池远未填满，还没有发生过 LRU 逐出，因此落到这一层的 402 次 lookup 全是**首次计算**的新内容（平均尾巴约 3.5k token），任何 tier 都不可能命中它们。本次变更针对的是重复 prefill，而这一轮到目前为止还没有产生过重复 prefill。
- 本次观测新暴露的风险，如实记录：实测写入率 0.565 block/s（自本轮启动）与 0.485 block/s（自重启），对应 tier 完整翻转 **39–46 分钟**，短于变更时按 0.383 block/s 估出的约 58 分钟——新一轮正在建上下文，写得比稳态更快。若父上下文等待子任务的间隔超过约 40 分钟，即使 64 GiB 也可能在被问到之前就把该前缀丢掉。下次复核要连这一点一起判，而不是只看命中率。
- 复核计划（预登记规则的执行条件）：等 GPU 侧真正发生逐出之后再判——即缓存的 distinct token 逼近 1,358,930、各实验上下文长到 100k+，本轮约 4–6 小时后。届时采样 `vllm:external_prefix_cache_hits_total`、`vllm:external_prefix_cache_queries_total`、`vllm:prefix_cache_hits_total`、`vllm:prefix_cache_queries_total`，加上 `vllm:kv_cache_usage_perc` 与 `vllm:kv_offload_cpu_allocation_size_sum`；确认 GPU 侧已逐出之后，再套用预登记规则——外部命中率若仍 <10%，就移除这一层，而不是继续加大。
- 证据：`/Data2/lzp/vllm-eval/results/20260828-workload-offload-sizing/metrics_interim_1787945056.prom`。

### 64 GiB tier 的验收：外部命中 4.59%，按预登记规则判定移除（结论 accepted，变更尚未执行）

- 验收条件已满足：CPU 侧确实在逐出——`vllm:kv_offload_cpu_allocation_size_sum` 在窗口内 9,448 → 10,855 block，自重启累计 10,917，打进 1,337 个槽位即已绕过 **8.2 圈**；`vllm:kv_cache_usage_perc` 0.064 → 0.118，`vllm:num_requests_waiting` 0，`vllm:num_preemptions_total` 0。因此「还没轮转过所以不能判」这条推迟理由已经不成立。
- 窗口数字（01:45:50Z→02:29:23Z，87 个采样）：`vllm:external_prefix_cache_hits_total` **一次都没动**，窗口内 0 / 855,134；`vllm:prefix_cache_hits_total` / `_queries_total` 窗口内 91.44%；GPU→CPU +72.24 GB / 464 次传输，CPU→GPU 平在 11.66 GB。
- 自 18:11Z 重启累计（约 8 小时 24 分）：`vllm:external_prefix_cache_hits_total` / `vllm:external_prefix_cache_queries_total` = 327,712 / 7,133,616 = **4.59%**，低于预登记的 10% 阈值；`vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` = 136,993,024 / 144,126,640 = **95.05%**；`vllm:kv_offload_store_bytes_total` 560.51 GB / 157.55 s 对 `vllm:kv_offload_load_bytes_total` 11.66 GB / 0.67 s，写读比 **48:1**，全程只发生过 12 次 load。
- 净账，这比阈值更有说服力：这一层总共救回 327,712 个 token，按探针实测的冷 prefill 速率 2,752 token/s 折算约 **119 s** 的 prefill（按变更前的 2,430 token/s 折算约 135 s）；代价是 **157.55 s** 的 GPU→CPU 传输时间、560 GB 的 D2H 流量，以及 64 GiB 被钉住的主机内存（unit 设 `MemorySwapMax=0`）。**省下的比花掉的还少。** 如实标注一处不对称：store 是异步的，157.55 s 不等于 157.55 s 的引擎停顿，但它确实占用 decode 同一 NUMA 节点的内存与 PCIe 带宽；而 load 在关键路径上，省下的 prefill 也在关键路径上。
- 判读：把 tier 从 32 GiB 加到 64 GiB 确实把命中率从 0.11% 抬到了 4.59%，方向是对的，但**幅度不足以翻转这笔账**，而继续加大在本机不可行——`/dev/shm` 252 GiB 里另一租户泄漏的四个区域占着 137 GB，回收我们自己那个之后也只剩约 125 GiB。GPU prefix cache 已经答掉约 95%，剩下的 5% 里绝大部分是首次计算的新内容，不是任何 tier 能救的重复 prefill。按预登记规则：**移除**，而不是再加大。
- 已准备、尚未执行的变更：从 profile 删去 `--kv-offloading-size`（vLLM 的 `kv_offloading_size is None` 即完全关闭该连接器）。`scripts/reclaim-offload-regions` 保留——它在无候选时 `nullglob` 下直接 exit 0，且本次移除重启正需要它回收我们自己那个将被 SIGTERM 泄漏的 64 GiB 区域；本机其他租户的泄漏区域也仍然存在，扫除逻辑并不因为本部署不再产生区域而失去用处。`run-vllm`、`health-watchdog`、systemd unit 与其内存限额均不需要改动。patch、verify runbook 与回滚副本见证据路径。
- 精度不受影响：与加大 tier 时同理，这一层只改变「已经算好的 KV 从哪里取」，`v1/kv_offload/cpu/gpu_worker.py` 走的是 `torch.int8` page 视图的纯字节拷贝。移除它只是让那些 token 改回重算，KV dtype、量化、定标、context 长度与 sampler 都不变，GPU KV 池仍是 1,358,930 token / 262,144 满长 5.18×。
- 证据：`/Data2/lzp/vllm-eval/results/20260828-workload-offload-sizing/`（`metrics_acceptance_*.prom` 为本条读数，`remove-kv-offloading.patch`、`verify-removal`、`removal-candidate/` 为待执行的变更）；回滚副本 `~/rollback/kv-offload-removal-20260828T193500Z/`。

### 移除 CPU KV offload tier（accepted）

- 动机：上一条的验收结论——64 GiB 下外部命中 4.59%、写读比 48:1、净账为负，且本机拿不到更大的 tier。
- 变更：profile 删去 `--kv-offloading-size 64`（`kv_offloading_size is None` 即完全不加载该连接器）。同步两处会被这次移除证伪的文档：README 里「每次重启都泄漏一个区域」改为「启用 offload 的部署才会泄漏；本部署不再产生区域，但扫除仍每次运行，因为泄漏的区域比造出它的配置活得久」，并把 **KV offloading size 也进入 compile hash** 这一事实补进 compile-cache 那一段（此前未记，正是两次重启都冷编译的原因）；MEASUREMENT.md 的「KV offload tier 几乎未被测试」已被本次证伪，改为它已被实测并移除。`scripts/`、systemd unit 与其内存限额均未改动。
- `reclaim-offload-regions` 保留，本次启动即证明了它仍是 load-bearing：日志显示它回收了我们自己那个 68,695,359,488 字节的区域，同时对另一租户的四个区域逐个报「owned by another user」而不动。本部署此后不再产生区域，但泄漏的区域仍需被扫除。
- 上线后核验：重启 **418 s**（20:15:57 发起 → 20:22:55 active），`init engine` 291.86 s、`torch.compile` 170.66 s——仍是冷编译（新 hash `3e3f83fd…`），但比 64 GiB 那次的 613 s / 457.69 s 快，差额主要是不再需要分配并 fault-in 那 64 GiB 共享区域。verify 全通过：两个 unit active，三个健康端点 200，未认证 `/v1/models` 401，gateway 列出模型，`kv_offloading_size="None"`，`vllm:kv_offload_*` 系列**一条都不存在**，`/dev/shm` 中已无属于我们的区域（另一租户的四个仍在，符合预期），一次经 gateway 的 `xhigh` 端到端补全 finish_reason stop 且答案正确，`num_preemptions_total` 0，`memory.events` 全零。49 项测试 OK。
- 容量与精度不变：`GPU KV cache size: 1,358,930 tokens`、`Maximum concurrency for 262,144 tokens per request: 5.18x`、`fp8_e4m3`，与移除前逐位相同。这一层从来只决定「已经算好的 KV 从哪里取」，移除它只是让那些 token 改回重算。
- 资源回收：`/dev/shm` 192 GiB → 128 GiB（放掉 64 GiB，且该主机上另一租户仍占着 137 GB）；cgroup `memory.current` 92.56 GiB → **12.0 GiB**，其中 shmem 64.87 GiB → 0.9 GiB。unit 的 `MemoryHigh=112G` / `MemoryMax=128G` 保持不变，现在只是余量更大。
- 并发探针，以及一条必须记下的测量限制：移除后连跑三次。第一次与第二次期间，另一租户 `lxy` 的进程占着 **GPU 7——本服务两块 GPU 之一**（3,080 MiB），同时有一个 2128% CPU 的作业，主机 load average 54。同一份配置在相隔 11 分钟的两次运行里，并发 1 的 per-stream decode 测出 **40.08 与 27.47 tok/s**，相差 31%。**因此这台共享主机在其他租户活动时，本探针根本不具备分辨 8% 级差异的能力**，那两次的数字不能与任何基线比较。等 GPU 7 空出、load 降到约 7 之后的第三次干净测量：per-stream decode 40.08 / 35.17 / 28.08 tok/s（added 1/4/8），aggregate 38.1 / 129.7 / 181.4，TTFT p50 0.24 / 0.31 / 0.82 s，`waiting_max` 0，preemption 0；prefill-interference 为 98,827 token 冷 prefill、TTFT 34.79 s、期间每流 0.683 tok/s、前后 34.94 / 34.93。对照 64 GiB 时的干净测量（40.13 / 34.84 / 30.58，冷 TTFT 35.67 s，期间 0.610 tok/s）：c1 与 c4 在 1% 以内不可区分，prefill-interference 略好，c8 低 8.2%——而 8% 正落在刚刚实测出的 31% 噪声带以内，因此**不主张任何方向的性能结论**，只主张无功能回归。
- 顺带修正：本次 `verify-removal` 首跑报了一个 FAIL，查证后是我自己的断言写错而非系统问题——vLLM 在 `v1/metrics/loggers.py` 里**无条件**注册 `vllm:external_prefix_cache_*` 计数器，与是否配置外部连接器无关，因此该系列不会消失，只会归零。断言已改为「queries / hits / `external_kv_transfer` 三者皆为 0」，改后全绿。真正随连接器消失的是 `vllm:kv_offload_*` 系列，那一条断言本来就是对的。
- compile cache 现有两个条目：64 GiB 的 `02d25764…`（657 MB）与本次 no-offload 的 `3e3f83fd…`（646 MB）。刻意先都留着——回滚窗口内保留前者能让回滚重启是热编译（约 4 分钟）而不是冷编译（约 10 分钟）。回滚窗口关闭后按 README 的办法剪掉多余条目。
- 决定/回滚：接受。回滚为 `git revert` 本次 commit 加一次重启；变更前的 profile 副本另存在 `~/rollback/kv-offload-removal-20260828T193500Z/`，SHA-256 `f8487114…2576`。
- 证据：`/Data2/lzp/vllm-eval/results/20260828-workload-offload-sizing/`（`metrics_post_removal.prom`、`verify-removal`、`probe-removal*.log`、`restart-timeline.txt`）；三次探针 JSON 在消费方 `logs/analysis/concurrency/`。

## 2026-10-22

### 模型 profile、按需实例与共享部署同源（accepted）

- 动机：除了 GPU 6–7 上的共享部署，还需要在调用方已认领的单卡上按需起别的模型（27B 单卡、Qwen3.5 小模型、Qwen3 embedding），而且要和共享部署走同一条启动路径、同一份引擎参数，不能再复制一份 profile。共享部署自 2026-10-19 起停着，这次不得启动它，也不得碰 GPU 2–7。
- 变更（配置）：引擎配置拆为两层。`config/profiles/<name>.conf` 只描述模型——checkpoint、served name、任务、默认 TP 与上下文、默认 vLLM 参数、发布日期、知识截止与一行说明；首批七个：`qwen-3.8-27b-fp8`（单卡、32,768、`fp8_e4m3` KV、prefix caching、`--max-num-seqs 128`、thinking 默认关）、`qwen-3.5-9b`/`4b`/`2b`/`0.8b`（BF16、原生 262,144、thinking 默认关）、`qwen3-embedding-8b`/`0.6b`（pooling，last-token 与归一化取自 checkpoint 的 sentence-transformers 配置）。发布日期取自各自 README 的引用（Qwen3.8 为 2026-08，Qwen3.5 为 2026-02，Qwen3-Embedding 为 2025-06，即 arXiv 2506.05176）；七份 README 都没有写知识截止，profile 里如实留空，`list-profiles` 输出 null。`config/deployments/qwen-3.8-27b-fp8.conf` 变成「引用 profile + 部署自己的覆盖」：端口 8010、GPU 6,7、16 核、`--tensor-parallel-size 2`、`--max-model-len 262144`、`--max-num-seqs 256`、thinking 默认开、alias `high:xhigh`。
- 变更（脚本）：`run-vllm` 是唯一的启动器。带 `--profile/--gpus/--port`（可选 TP、上下文、显存比例与逐个 token 追加的 `--extra-arg`）时起一个前台实例，绑定 `127.0.0.1`，`CUDA_VISIBLE_DEVICES` 取给定的主机卡号，绑定这些卡所在的整个 NUMA 节点；不带参数时读 deployment，把它的覆盖翻译成同一组选项再走同一条路径。API key 仍只经环境变量传入；profile 参数与 extra 参数里出现启动器自己负责的选项（model、served name、host、port、runner、TP、上下文、显存比例、`--api-key`，含下划线写法）一律拒绝，保证每个参数只有一个来源；卡数必须等于 TP。解释器固定为仓库 `.venv/bin/python`，`CUDA_DEVICE_ORDER`、HF 离线与 `PATH` 从 profile 挪进 `run-vllm`。`resolve-placement` 的核数改为可选，缺省即整节点。新增 `list-profiles`，以 JSON 数组输出全部 profile；任何一个 profile 加载失败则整份输出失败、不打印半截。测试 49 → 60 项，新增参数解析、profile 加载、部署与实例同一路径、列表输出的用例，全部不起引擎、不碰 GPU。
- 证据（部署等价，未启动）：用记录器替代解释器，分别跑旧 `main` 与新布局的部署路径，比较 vLLM argv 与环境。环境变量与绑定（node 1、CPU 48–63）完全相同；argv 只多出三处——显式 `--runner generate`（与 `auto` 解析结果相同）、`--max-num-seqs 128` 后接部署的 `256`、`--default-chat-template-kwargs` 先关后开。用 vLLM 自己的参数解析器读新 argv，有效值为 `max_num_seqs=256`（与 L20 上 vLLM 对 API server 的默认值相同）、`enable_thinking=True`（与 chat template 缺省相同）、TP 2、262,144、`fp8_e4m3`。按 MEASUREMENT.md 进入 compile hash 的量都没变，因此预期复用现有 `3e3f83fd…` 条目；但部署没有在新布局下重启过，这一点只是推断。
- 证据（GPU 0，单卡实例）：每次启动前后 `nvidia-smi` 与 `free -h`，GPU 0 起止均为 0 MiB，`available` 从 442 GiB 降到 419 GiB，差额是同机其他进程。
  - 27B 首次启动失败：单卡上 hybrid 模型的线性注意力状态只够 167 个序列，而 vLLM 在 L20 上默认 `max_num_seqs=256`，CUDA graph 捕获拒绝继续。profile 因此设 `--max-num-seqs 128`，部署重申 256。
  - 27B 第二次启动：从启动到 `/health` 200 且 `/v1/models` 列出模型共 339 s。compile cache 冷（`--max-num-seqs` 进入 hash；torch.compile 150.0 s），权重已在 page cache 里（加载 9.7 s；首次失败那次约 17 / 30.9 GB 在缓存中，加载 37.9 s）。引擎自报：权重 28.51 GiB，可用 KV 8.45 GiB，GPU KV 213,598 tokens，32,768 满长 6.52×，CUDA graph 1.63 GiB，`init engine` 259.5 s；启动后占 40,611 MiB，压测后 43,183 MiB。未认证的 `/v1/models` 返回 401。
  - 不带任何 thinking 字段、`temperature: 0` 的 chat completion 返回纯 content、`reasoning` 为 null、4 个 completion token；同一请求加 `"chat_template_kwargs": {"enable_thinking": true}` 后推理文本进入 `reasoning` 字段，content 只剩答案（前面带两个换行）。
  - prompt 吞吐：20 个并发请求，每个约 1,000–1,030 prompt token、首 token 起互不相同（不命中 prefix cache）、`max_tokens: 1`，三轮各 20,335–20,383 token、6.61–6.67 s，即 3,053–3,078 prompt token/s。这是单卡、空载、短 prompt 的预填充上限，不代表混合负载。
  - `qwen3-embedding-8b`：启动到就绪 105 s（compile 冷，38.8 s；权重 14.11 GiB），pooling 解析为 LAST 并归一化。一次请求 512 段短文本（共 8,594 token），客户端计时 3.62–3.79 s（含解析 45 MB 的 JSON 浮点），维度 4096，范数全为 1.0；同样的请求用 `encoding_format: base64` 为 1.59 s、11.2 MB，浮点格式只计传输为 2.78 s。
  - `qwen-3.5-9b` 以原生 262,144 单卡启动的可行性：232 s 就绪，权重 17.66 GiB，GPU KV 666,043 tokens，满长 2.54×，chat completion 返回纯 content。其余三个 Qwen3.5 更小，未逐一启动。
  - 每个实例都以 SIGTERM 发给进程组停止（27B 3 s、embedding 2 s 内整组退出），之后 GPU 0 回到 0 MiB；GPU 1–7 全程未用。
- 清理：仓库内删除 `__pycache__`（根目录、`eval/`、`tests/`）与只含三个空目录的 `.pi/`，二者本就在 `.gitignore` 里；仓库外删除属于本用户的 `/Data/public/Qwen3.8-27B/.cache`（4.5G）与 `/Data/public/Qwen3.8-27B-FP8/.cache`（326M）两处未完成的下载残留、只含一个空日志的 `/Data/public/vllm-logs`，以及悬空的 `~/.config/systemd/user/timers.target.wants/qwen38-watchdog.timer`；compile cache 里删掉了首次失败留下的、单卡 256 序列这一永远起不来的配置条目，保留部署的 `3e3f83fd…` 与本次三个可用配置的条目。
- 遗留，如实记录：profile 指向的模型目录有六个属于其他用户（`wjh`、`gz`、`wmm`），并且都在 `root:root` 0777、无 sticky bit 的 `/Data/public` 下——任何用户都能改名或替换这些目录项，与 2026-08-23 迁走 venv 时是同一个可篡改面，这次没有处理。按需实例的卡必须在同一个 NUMA 节点上（本机 0–3 或 4–7），跨节点 TP 会在启动前失败而不是退化绑定。共享部署在新布局下的首次重启需要人工确认引擎自报的容量与之前逐位相同。
- 决定/回滚：接受。回滚为 `git revert` 本次三个 commit；部署未运行，回滚不需要重启。
