# Linux A800 Bring-up Checklist

本文只描述首次服务器联调顺序，不改变 `EXPERIMENT_SPEC.md` 中的正式实验定义。

## 1. 准备独立环境

安装依赖会占用磁盘，并可能改变当前 Python 环境中的 torch/vLLM 版本。应使用新的 Python 3.10 conda 环境，不覆盖已有可用环境。

```bash
conda create -n infoskill python=3.10 -y
conda activate infoskill
cd /workspace/infoskill
pip install -r requirements-server.txt
pip install -e /workspace/alfworld-master
pip install -e /workspace/SkillRL --no-deps
pip install -e .
```

SkillRL 的 CUDA `DataParallelPPOActor` 会在模块导入时无条件导入
`flash_attn.bert_padding`，所以 M0 也必须安装 flash-attn。当前服务器基线固定使用
SkillRL vLLM-0.8 容器对应的 `2.7.4.post1 + cu12 + torch2.6 +
cxx11abiFALSE + cp310` 预编译 wheel；不要在 ABI 不匹配时盲目源码编译。

安装后先做一个不加载模型的快速检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"
python -c "import flash_attn; from flash_attn.bert_padding import unpad_input; print(flash_attn.__version__, 'OK')"
```

vLLM 0.8.4 还依赖 cachetools 5.x 的私有 LRU 方法；其 requirements 没有约束上限，
而 cachetools 6 删除了该方法。环境固定为 `cachetools==5.5.2`，并在启动 GPU 前检查：

```bash
python -m pip install --no-deps --force-reinstall cachetools==5.5.2
python -c "import cachetools; c=cachetools.LRUCache(1); print(cachetools.__version__, hasattr(c, '_LRUCache__update'))"
```

第二条命令必须输出 `5.5.2 True`。

## 2. 只读运行时盘点

```bash
bash scripts/runtime_doctor.sh
```

这里的第一次 doctor 只盘点当前环境；在第 7 节安装补丁版 vLLM 之前，报告中的
Hybrid Prefix API 可以显示为不可用。保留生成的 `runtime-doctor.json`；它记录准确
包版本、CUDA、vLLM/VERL Python 源文件位置和 SHA-256，不读取权重或数据内容。

## 3. 修改并校验路径

不要直接修改 Git 跟踪的基线 YAML。复制一份机器专用配置，只修改副本顶部的
`paths`，并在当前 shell 导出 `CONFIG`。本地配置已被 `.gitignore` 排除，不会阻塞
以后拉取代码：

```bash
cp configs/alfworld_qwen25_7b.yaml configs/alfworld_qwen25_7b.local.yaml
export CONFIG=configs/alfworld_qwen25_7b.local.yaml
```

`policy_model` 可以放在不同的绝对路径，但训练前必须验证它确实是约定的共同初始化，
而不是另一个同名目录。下面的命令只读模型配置、tokenizer 与权重并输出组合 SHA-256，
不会加载 GPU；首轮 7B 的 `sha256` 必须为
`8305dee0a659a8f9e0650129eaaf584006338a42f237d071ef5cdbaed91fc14a`：

```bash
PYTHONPATH=src python -m infoskill.persistence.model_identity \
  /root/autodl-tmp/wjh/models/Qwen/Qwen2.5-7B-Instruct \
  --model-id qwen2.5-7b-instruct
```

若需要复现旧 SFT 对照，复制
`configs/alfworld_qwen25_7b_sft.yaml`，不要把主配置的模型 ID 临时改回 SFT。

YAML 只保存 `policy_model_id`，可信 revision 与 SHA-256 独立登记在代码中，避免同时
修改模型路径和本地指纹绕过校验。训练入口会在启动 Ray/GPU 前复算并 fail-fast；实际
路径可以变化，内容不能静默变化。新增共同初始化模型必须通过代码评审加入注册表，不能
只改机器专用 YAML。验证后的完整逐文件 manifest 会写入 run 和 checkpoint 的
`provenance.json`。

第一轮使用 0 卡校验：

```bash
GPUS=0 bash scripts/run_alfworld.sh validate no_skill
```

## 4. 本地逻辑测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

这些测试不加载 7B 权重。任何失败都应先修复，不能直接进入长任务。

## 5. 严格专家数据

这一步会遍历全部 train 游戏，运行经过实例身份校验的 ALFWorld planner，并占用输出磁盘，
但不会修改原数据。正式入口使用与已通过的 planner pilot 相同的专家类型；manifest
schema v2 会记录并强制校验 requested/effective expert identity，旧 handcoded 或未验证
产物不能被 M1 加载。

```bash
GPUS=0 \
GROUNDING_WORKER_BATCH_SIZE=64 \
GROUNDING_WORKER_PROCESSES=2 \
RUN_NAME=m1-grounding-planner-formal \
bash scripts/run_alfworld.sh grounding
```

只有 `manifest.json` 同时满足专家成功覆盖率不少于 99%、超过 30 步比例不高于 1%，才允许作为正式 grounding 版本。隔离原因必须检查，不能只删除失败样本后继续。

grounding 每 64 个任务重启一次短生命周期 CPU worker，并把该 worker 的
`TMPDIR` 限定在 run 目录内；worker 退出后立即清理 TextWorld/Fast Downward
临时副本。`grounding-lifecycle.json` 必须显示
`temporary_directories_cleaned=true`、`processed_tasks=3553`；启用双 worker 时还必须
显示 `worker_concurrency=2`、`peak_worker_processes=2`。这个分批与并发只改变资源
生命周期，不减少任务、专家步数或 formal gate。`GPUS=0` 只用于一次性 embedding
检索，专家回放子进程会主动隐藏 GPU。报告中的每个
`free_disk_bytes_during` 是该 shard 逐任务采样到的最低剩余空间，用于确认临时占用
峰值确实受到约束。

若 bounded grounding 的生命周期门通过、但 formal gate 因
`expert_action_not_admissible` 失败，不要直接降低 99% 覆盖率门限。下面的历史专家对比
入口只保留用于复核旧 handcoded 失败归因；它只用 CPU、不修改 grounding 数据，也不允许
在一条轨迹中混用专家：

```bash
SOURCE=$(find "$PWD/runs" -maxdepth 1 -type d \
  -name '*-m1-grounding-formal-bounded' | sort | tail -n 1)

GROUNDING_SOURCE_RUN="$SOURCE" \
GROUNDING_DIAGNOSTIC_TASKS_PER_TYPE=3 \
GROUNDING_DIAGNOSTIC_MAX_REPLAY_STEPS=150 \
INFO_SKILL_CPU_THREADS=1 \
RUN_NAME=m1-grounding-expert-diagnostic \
bash scripts/run_alfworld.sh grounding-expert-diagnostic
```

输出 `expert-diagnostic.json` 会按六类各抽最多 3 条历史隔离任务，并记录历史隔离行、
手写专家首次不匹配动作及内部状态、当前可执行命令、包装器 expert plan，以及独立
planner 重放结果。先核对 `historical_failure_reproduced_count`；只有手写失败能够稳定复现
时，`planner_rescue_count` 才能作为是否重新审议正式 grounding 专家的证据。这个小样本
只用于定位原因，不能直接替代全量 3,553 条 formal grounding。

历史失败诊断确认 planner 是候选方案后，先跑六类各 2 条（共 12 条）的 planner 身份
smoke。它不加载模型、不做技能检索并主动隐藏 GPU；目的不是估计成功率，而是验证
`AlfredExpert("planner")` 在真实 worker 中确实绑定为 planner。旧版 ALFWorld 的调用把
这个位置参数误传给了 `env`，因此 INFO-SKILL 会安装窄兼容保护，并在父进程和每个 worker
中 fail-closed 探测。后台运行：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PWD/logs/m1-grounding-planner-identity-smoke-${STAMP}.log"

nohup env \
PLANNER_PILOT_TASKS_PER_TYPE=2 \
PLANNER_PILOT_MAX_REPLAY_STEPS=150 \
GROUNDING_WORKER_BATCH_SIZE=12 \
INFO_SKILL_CPU_THREADS=1 \
RUN_NAME=m1-grounding-planner-identity-smoke \
bash scripts/run_alfworld.sh grounding-planner-pilot \
> "$LOG" 2>&1 &

PID=$!
echo "$PID" > "${LOG}.pid"
echo "PID=$PID"
echo "LOG=$LOG"
tail -f "$LOG"
```

这 12 条只检查 `expert_identity_gate_passed=true`、requested/effective 均为 `planner`、
guard/corrected 均为 `true`，以及 `handcoded_timeout_signature_count=0`。小样本的覆盖率和
长尾 `pilot_gate_passed` 不用于决策。身份门通过并完成结果复核后，再从完整 train 集按
六类各 50 条做确定性的 300 条 planner pilot。默认每 64 条重启 worker，通常约需
25–50 分钟，按 1 小时预留：

```bash
RUN=$(find "$PWD/runs" -maxdepth 1 -type d \
  -name '*-m1-grounding-planner-identity-smoke' | sort | tail -n 1)
echo "RUN=$RUN"
cat "$RUN/planner-pilot.json"
cat "$RUN/grounding-lifecycle.json"

STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="$PWD/m1-grounding-planner-identity-smoke-${STAMP}.tar.gz"
tar -czf "$ARCHIVE" -C "$RUN" \
  planner-pilot.json \
  planner-pilot-results.jsonl \
  grounding-lifecycle.json \
  console.log
echo "ARCHIVE=$ARCHIVE"
ls -lh "$ARCHIVE"
```

身份 smoke 通过后，不直接开启 300 条并行 pilot。先在同一固定 12 条上各运行一次串行和
双 worker replay，并对完整逐步结果做 exact parity。该入口主动隐藏 GPU；串行与并行
使用相同任务、种子、planner、150 步验证上限和 30 步持久化窗口。`worker-batch-size=6`
确保 12 条被拆成两个可以重叠的 shard：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PWD/logs/m1-grounding-planner-parity-${STAMP}.log"

nohup env \
GROUNDING_PARITY_TASKS_PER_TYPE=2 \
GROUNDING_PARITY_WORKER_BATCH_SIZE=6 \
GROUNDING_PARITY_PARALLEL_WORKERS=2 \
PLANNER_PILOT_MAX_REPLAY_STEPS=150 \
INFO_SKILL_CPU_THREADS=1 \
RUN_NAME=m1-grounding-planner-parity \
bash scripts/run_alfworld.sh grounding-planner-parity \
> "$LOG" 2>&1 &

PID=$!
echo "$PID" > "${LOG}.pid"
echo "PID=$PID"
echo "LOG=$LOG"
tail -f "$LOG"
```

只有 `planner-parity.json` 同时满足 `passed=true`、全部 `field_checks=true`、全部
`identity_checks=true`、全部 `lifecycle_checks=true`、`mismatch_count=0`，并且
`parallel.peak_worker_processes=2`，才允许在 300 条 pilot 中设置
`GROUNDING_WORKER_PROCESSES=2`。`speedup` 只用于估算耗时，不是正确性门槛。结果打包：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
RUN=$(find "$PWD/runs" -maxdepth 1 -type d \
  -name '*-m1-grounding-planner-parity' | sort | tail -n 1)

echo "RUN=$RUN"
cat "$RUN/planner-parity.json"

STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="$PWD/m1-grounding-planner-parity-${STAMP}.tar.gz"
tar -czf "$ARCHIVE" -C "$RUN" \
  planner-parity.json \
  serial-results.jsonl \
  parallel-results.jsonl \
  serial-lifecycle.json \
  parallel-lifecycle.json \
  console.log
echo "ARCHIVE=$ARCHIVE"
ls -lh "$ARCHIVE"
df -h /root/autodl-tmp
```

将该压缩包以及终端打印的 `planner-parity.json`、`df -h` 结果回传。串并行一致性通过后，
六类各 50 条的 300 条 planner pilot 使用已验证的双 worker；默认并发仍为 1，避免旧命令
静默改变执行拓扑：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PWD/logs/m1-grounding-planner-pilot-${STAMP}.log"

nohup env \
PLANNER_PILOT_TASKS_PER_TYPE=50 \
PLANNER_PILOT_MAX_REPLAY_STEPS=150 \
GROUNDING_WORKER_BATCH_SIZE=64 \
GROUNDING_WORKER_PROCESSES=2 \
INFO_SKILL_CPU_THREADS=1 \
RUN_NAME=m1-grounding-planner-pilot \
bash scripts/run_alfworld.sh grounding-planner-pilot \
> "$LOG" 2>&1 &

PID=$!
echo "$PID" > "${LOG}.pid"
echo "PID=$PID"
echo "LOG=$LOG"
tail -f "$LOG"
```

该入口只生成 `planner-pilot.json`、`planner-pilot-results.jsonl`、
`grounding-lifecycle.json` 和 `console.log`，故意不生成正式训练入口要求的
`manifest.json` 与 `grounding_samples.jsonl`。即使 `pilot_gate_passed=true`，也只表示
值得继续跑 3,553 条正式 planner grounding，不能把该目录传给 `GROUNDING_DATA`。
重点先检查 `expert_identity_gate_passed` 和 `expert_binding`，再检查整体及六类
`success_rate`、`over_persist_horizon_rate`、失败原因和 p90/p95/p99 轨迹长度。

完成后用以下命令打印摘要并打包需要回传的文件：

```bash
RUN=$(find "$PWD/runs" -maxdepth 1 -type d \
  -name '*-m1-grounding-planner-pilot' | sort | tail -n 1)
echo "RUN=$RUN"
cat "$RUN/planner-pilot.json"
cat "$RUN/grounding-lifecycle.json"

STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="$PWD/m1-grounding-planner-pilot-${STAMP}.tar.gz"
tar -czf "$ARCHIVE" -C "$RUN" \
  planner-pilot.json \
  planner-pilot-results.jsonl \
  grounding-lifecycle.json \
  console.log
echo "ARCHIVE=$ARCHIVE"
ls -lh "$ARCHIVE"
```

2026-09-12 旧代码得到的 278/300 及其 28 条长 horizon 诊断已经作废。源码核验和逐步
trace 证明，ALFWorld 调用 `AlfredExpert(expert_type)` 时把 `"planner"` 绑定到了 `env`，
实际 `expert_type` 仍为默认 `handcoded`；5,047 个诊断步骤均表现为 handcoded 分支，19 条
失败还命中了其固定 200 步 `Timeout`。这些数字只能描述误绑定的手写专家，不能判断
planner 的覆盖率、长尾或循环。禁止把旧 pilot 目录传给正式 grounding，也禁止继续用它
作为 loop diagnostic 的输入；新入口会要求来源报告通过专家身份门。

只有新的 12 条身份 smoke 与 300 条 pilot 均通过相应复核后，才可以对新 pilot 的失败项
运行长 horizon loop diagnostic。该诊断仍为 CPU-only，不写正式 grounding 样本：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
mkdir -p logs

SOURCE=/root/autodl-tmp/wjh/alfworld_eval/infoskill/runs/20260911T180620Z-m1-grounding-planner-pilot
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PWD/logs/m1-grounding-planner-loop-${STAMP}.log"

nohup env \
GROUNDING_SOURCE_RUN="$SOURCE" \
PLANNER_LOOP_SUCCESS_CONTROLS=6 \
PLANNER_LOOP_MAX_REPLAY_STEPS=300 \
INFO_SKILL_CPU_THREADS=1 \
RUN_NAME=m1-grounding-planner-loop-diagnostic \
bash scripts/run_alfworld.sh grounding-planner-loop-diagnostic \
> "$LOG" 2>&1 &

PID=$!
echo "$PID" > "${LOG}.pid"
echo "PID=$PID"
echo "LOG=$LOG"
tail -f "$LOG"
```

`planner-loop-diagnostic.json` 汇总被更长 horizon 救回、仍失败、状态—动作循环和对照退化
的任务；`planner-loop-traces.jsonl` 则保留每步 observation、完整 admissible commands、
planner plan 前五项、实际动作与状态指纹。状态指纹优先使用 ALFWorld 完整 world facts，
缺失时才退回 observation、admissible commands 与 won。报告把“任意历史位置重复三次”
记为 `historical_repeated_state_action_tasks`，但只有轨迹末尾存在同一状态—动作周期连续
重复至少三次，才记为 `terminal_cycles_detected`；检测周期上限为 10。该信号用于区分
“曾经回访”与“最终卡死”，不作为正式质量门本身。

完成后打印报告并打包这三个文件：

```bash
RUN=$(find "$PWD/runs" -maxdepth 1 -type d \
  -name '*-m1-grounding-planner-loop-diagnostic' | sort | tail -n 1)
echo "RUN=$RUN"
cat "$RUN/planner-loop-diagnostic.json"
tail -n 40 "$RUN/console.log"

STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="$PWD/m1-grounding-planner-loop-${STAMP}.tar.gz"
tar -czf "$ARCHIVE" -C "$RUN" \
  planner-loop-diagnostic.json \
  planner-loop-traces.jsonl \
  console.log
echo "ARCHIVE=$ARCHIVE"
ls -lh "$ARCHIVE"
```

## 6. 评测闭环

先用很少任务做开发 smoke（正式结果仍必须完整 140 条），确认模型加载、环境 reset/step、日志和动作解析。当前 CLI 的正式 `eval` 会强制 140 条：

```bash
GPUS=0 RUN_NAME=qwen25-7b-base bash scripts/run_alfworld.sh eval no_skill
GPUS=0 RUN_NAME=qwen25-7b-raw bash scripts/run_alfworld.sh eval raw_skill_prompt
```

检查运行目录中的：

- `console.log`；
- `traces/valid-seen-*.jsonl.zst`；
- `valid_seen_summary.json`；
- `metrics.jsonl` 和 `metrics.csv`。

若任一基础设施任务失败，整次评测应显示 `is_complete=false`，不能用剩余样本重算成功率。

## 7. vLLM Hybrid Prefix gate

正式 `infoskill` 训练之前必须完成以下四项：

1. vLLM 能把 5 个显式 placeholder 位置替换为对应 soft-prefix vectors；
2. 每条请求使用由 `update/task/rollout/step` 派生的独立 seed；
3. 同后端 transport 门与多案例 Transformers/vLLM 数值门均通过；
4. 4→4、4→2、2→4 恢复测试核对 LoRA、Adam 一二阶矩、scheduler、任务游标和下一 update 身份。

当前适配器在 capability 不存在时会 fail-fast，不会退回 token-only M1。

先构建并安装固定补丁运行时：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
python -m pip install -r requirements-vllm-build.txt
mkdir -p /root/autodl-tmp/wjh/wheelhouse
python -m pip download vllm==0.8.4 --no-deps -d /root/autodl-tmp/wjh/wheelhouse
export VLLM_PRECOMPILED_WHEEL_LOCATION=/root/autodl-tmp/wjh/wheelhouse/vllm-0.8.4-cp38-abi3-manylinux1_x86_64.whl
bash scripts/build_patched_vllm.sh \
  /root/autodl-tmp/wjh/alfworld_eval/vllm-0.8.4 \
  ./dist/vllm \
  --install
bash scripts/runtime_doctor.sh
```

确认 doctor 识别到 `INFOSKILL_HYBRID_PREFIX_API=1` 后，再在单张 GPU 上运行双门禁：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/hybrid_prefix_parity.py \
  --model /absolute/path/to/Qwen2.5-7B-Instruct \
  --output hybrid-prefix-parity.json
```

新版 parity 默认一次完成 8 个固定案例（4 个 prompt × 2 个 prefix seed），
Transformers 和 vLLM 各只加载一次。输出使用 schema v2，并包含两个独立门：

1. `transport_gate`：普通 vLLM token IDs 与等价 token embedding hybrid 输入必须
   100% 首 token 一致，最大 logprob 误差不超过 `1e-4`；两侧分开调用
   `generate`，保证普通对照不进入 hybrid embedding 分支。
2. `cross_backend_gate`：Transformers hybrid 与 vLLM hybrid 必须 100% 首 token
   一致，logprob 误差 P95 不超过 `0.05`、最大值不超过 `0.10`。

`--case-count`、`--base-seed`、`--transport-logprob-atol`、
`--cross-p95-logprob-atol`、`--cross-max-logprob-atol` 与
`--required-token-match-rate` 均可显式修改，但正式实验应固定参数并保存 JSON，
不得看到结果后临时放宽。`bfloat16` 是正式默认精度；`float16` 仅作为显式诊断
选项，不应在看到 BF16 结果后替换正式口径。

上面的 wheel 文件名以实际下载结果为准。该路径复用官方 wheel 中的 CUDA
扩展，只重新打包本项目修改过的 Python 文件；若不设置
`VLLM_PRECOMPILED_WHEEL_LOCATION`，脚本会退回完整源码编译，耗时和临时磁盘占用都明显更高。
构建脚本在卸载当前版本前强制检查成品同时包含 `vllm/_C*.so` 与
`INFOSKILL_HYBRID_PREFIX_API`；任一缺失都会终止，不安装残缺 wheel。

若曾由旧脚本安装过约 2.5 MB、缺少 `vllm._C` 的 Python-only wheel，先恢复
已下载的官方 wheel，再使用新脚本重建：

```bash
unset OMP_NUM_THREADS
python -m pip install --force-reinstall --no-deps \
  /root/autodl-tmp/wjh/wheelhouse/vllm-0.8.4-cp38-abi3-manylinux1_x86_64.whl
python -c "import vllm, vllm._C; print(vllm.__version__, 'native OK')"
```

项目启动脚本不再继承外部 `OMP_NUM_THREADS`/`MKL_NUM_THREADS`，而是用
`INFO_SKILL_CPU_THREADS` 同时设置二者，默认值为 `1`，且启动前要求它是正整数。
这避免 Conda 或宿主机遗留的非法值触发 `libgomp` 警告；该警告本身不是
`vllm._C` 缺失的原因。

## 8. M0 `no_skill` 训练门禁

M0 只训练 Qwen LoRA，不使用技能检索、embedding 模型或 soft prefix。它仍使用
固定 SkillRL commit 的 VERL/Ray/FSDP1/vLLM 运行时。首次运行依次执行：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill

# 只解析参数和检查路径，不占 GPU
GPUS=0,1,2,3 PROFILE=smoke MAX_UPDATES=1 DRY_RUN=1 \
  bash scripts/run_alfworld.sh train no_skill

# 第一门：一个 update
GPUS=0,1,2,3 PROFILE=smoke MAX_UPDATES=1 RUN_NAME=m0-smoke-u1 \
  bash scripts/run_alfworld.sh train no_skill

# 第二门：完整 smoke，两次 update
GPUS=0,1,2,3 PROFILE=smoke RUN_NAME=m0-smoke-u2 \
  bash scripts/run_alfworld.sh train no_skill
```

每次真实训练启动前都会要求 ALFWorld `train` 恰好发现 3553 条任务。所有训练
档位都从这份完整清单按相同种子顺序取任务；不再派生 355 条 train monitor。
正式档位按每 update 8 个任务组形成 445 个 update，最后一个 update 允许不足
8 组。

一个 update 成功的最低检查项：

- `console.log` 和 `metrics.jsonl` 中所有 loss、KL、gradient norm 均为有限值；
- 首个 update 在优化器更新前通过 rollout/recompute 强制门禁：logprob 绝对误差
  mean/median/P95/P99 分别不超过 `0.05/0.01/0.15/0.30`，误差大于 `1`
  的比例不超过 `0.1%`，误差大于 `5` 的比例必须为 `0`，且 ratio mean 位于
  `[0.98, 1.02]`；
- `runtime/training_sample_count + runtime/training_padding_count = runtime/training_padded_sample_count`，且 padded 数能被当前 GPU 数整除；
- `traces/train-update-*.jsonl.zst` 存在，包含该 update 的全部任务组和全部环境步骤；
- `checkpoints/step-*/checkpoint.complete.json` 存在，目录才可用于恢复；
- 第 2 个 update 使用不同于第 1 个 update 的任务，证明游标实际前进。

这些阈值来自 A800、Qwen2.5-7B/Alfworld-7B-SFT 的真实 token-only smoke：修复
VERL `action_stop` 末尾 padding 哨兵后，观测到 mean/median/P95/P99 为
`0.0212/0.00175/0.1028/0.1826`、ratio mean 为 `1.0005`，且无误差大于 `1`
的 token。若门禁失败，训练在 reference 计算和 optimizer update 前停止，错误消息
列出所有超标项，并在 run 根目录写出
`rollout-recompute-alignment-failure.json`。该文件包含完整 summary、固定阈值、全部
失败条件、最坏 token 的 sample/position/token ID、首尾位置标记及 rollout logprob
分桶统计；不得绕过后继续放大训练，也不得在缺少完整分布时只凭一个 P99 数值放宽阈值。

原版 Qwen 的同形状首更新中，no-skill 与 raw/full 的 P99 分别为 `0.24966` 和
`0.28432`；后者同时满足 mean=`0.03010`、median=`0.00267`、P95=`0.14057`、
误差大于 `1` 的比例=`0.0212%`、误差大于 `5` 的比例=`0`、ratio mean=`0.99976`，
且最大误差不在首尾 token。统一 P99 门限因此校准为 `0.30`，其余门限保持不变；这个
门限对所有模式相同，不能按方法单独放宽。

若 update 0 已评测并提交 `step-000000`，但首个训练 update 在该门禁停止，可从这个
checkpoint 原地恢复。恢复会跳过已经完成的 update-0 `valid_seen`，并重新执行首个训练
update；如果问题可复现，新版代码会留下上述诊断文件：

```bash
RUN=/absolute/path/to/failed-raw-skill-run
GPUS=0,1,2,3 \
  PROFILE=pilot \
  MAX_UPDATES=50 \
  RESUME="$RUN/checkpoints/step-000000" \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  ENVIRONMENT_WORKERS=1 \
  POLICY_MAX_TOKENS_PER_GPU=16384 \
  BALANCE_POLICY_TOKENS_ACROSS_RANKS=1 \
  RAW_SKILL_PROMPT_FORMAT=full \
  INFO_SKILL_CPU_THREADS=1 \
  bash scripts/run_alfworld.sh train raw_skill_prompt
```

恢复时必须保持源运行的训练计划和不可变配置一致。若门禁意外通过，该命令会继续原定
训练，而不会在诊断点自动停止。

验证恢复时，源运行必须原本就按 `MAX_UPDATES=2` 规划，并且
`step-000001/checkpoint.complete.json` 已原子提交。源运行可以随后生成 step 2；
命名分叉仍会把不可变的 step 1 作为输入。`MAX_UPDATES=1` 的完整运行不能事后
扩展预算。验证原拓扑恢复：

```bash
GPUS=0,1,2,3 PROFILE=smoke MAX_UPDATES=2 \
  RESUME=/absolute/path/to/m0-smoke-u2/checkpoints/step-000001 \
  bash scripts/run_alfworld.sh train no_skill
```

验证跨拓扑恢复时只改 `GPUS`，并提供新的 `RUN_NAME`。这会从权威 checkpoint
分叉出一个新运行目录，因此不会覆盖源运行已经生成的 step 2：

```bash
GPUS=0,1 PROFILE=smoke MAX_UPDATES=2 \
  RUN_NAME=m0-smoke-4to2 \
  RESUME=/absolute/path/to/m0-smoke-u2/checkpoints/step-000001 \
  bash scripts/run_alfworld.sh train no_skill
```

反向 2→4 使用一个按 `MAX_UPDATES=2` 完成的两卡 smoke 的 step 1：

```bash
GPUS=0,1,2,3 PROFILE=smoke MAX_UPDATES=2 \
  RUN_NAME=m0-smoke-2to4 \
  RESUME=/absolute/path/to/m0-smoke-2gpu/checkpoints/step-000001 \
  bash scripts/run_alfworld.sh train no_skill
```

`RESUME` 必须直接指向 `checkpoints/step-*`。不设置 `RUN_NAME` 表示在源运行中原地
恢复，此时 GPU 数也必须保持一致；同时设置新的 `RUN_NAME` 表示分叉恢复，只允许
GPU 数变化。训练档位、预算、模型、数据及其他 resolved config 在两种方式下都
必须与源 checkpoint 一致。分叉运行的 `provenance.json` 会记录源 checkpoint、
源 GPU 数和 `resume_forked=true`。

新版代码还会在启动 runtime 前继承源 checkpoint 当时已经完成的 `valid_seen` 记录，
并在目标 run 的 `checkpoint_selection.json` 中和后续评测合并。旧版已经完成的命名分叉
若只保留目标 run 的最后一次评测，可只修复这份选择元数据；命令会先保存
`checkpoint_selection.pre-fork-merge.json`，不修改模型、optimizer、trace 或成功率：

```bash
python scripts/repair_forked_checkpoint_selection.py \
  /absolute/source-run/checkpoints/step-000005 \
  /absolute/destination-run
```

输出必须列出完整 `evaluated_steps`，并按注册规则给出整条曲线的 `best_valid`。风险仅是
重写目标 run 的选择 JSON；原文件有一次备份，脚本可重复执行且会校验 manifest 和同一步
指标冲突。

完成 smoke 与两种恢复检查后，运行一个与正式训练相同形状、覆盖完整评测周期的
25-update pilot。`integration` 保留为需要用较小 G 和 batch 定位问题时的可选档位，
不再与 pilot 串行作为必经门禁：

```bash
# 可选：中等形状故障定位
GPUS=0,1,2,3 PROFILE=integration RUN_NAME=m0-integration \
  bash scripts/run_alfworld.sh train no_skill

# 必需：默认即为 25 updates，在 update 0 和 25 评完整 valid_seen 140 条
GPUS=0,1,2,3 PROFILE=pilot RUN_NAME=m0-pilot \
  bash scripts/run_alfworld.sh train no_skill

# 仅在方法实现和共享运行时冻结后启动
GPUS=0,1,2,3 PROFILE=formal RUN_NAME=m0-formal \
  bash scripts/run_alfworld.sh train no_skill
```

### 用固定 140 条 valid_seen 补测旧 M0 pilot

旧版 pilot 若只保存了 train-monitor 结果，不需要重训即可补测。下面的包装脚本会
依次评测共同原版 Qwen 起点（update 0）和 portable checkpoint；两次均使用四卡
VERL/vLLM、固定 140 条 `valid_seen` 与完全相同的确定性随机协议。它不会修改源
checkpoint，但会占用指定 GPU，并在 `runs/` 新建两份完整评测目录。

短任务可以前台运行：

```bash
GPUS=0,1,2,3 \
POLICY_CHECKPOINT=/absolute/run/checkpoints/step-000025 \
BASE_RUN_NAME=m0-pilot-valid-seen-update0 \
CHECKPOINT_RUN_NAME=m0-pilot-valid-seen-update25 \
bash scripts/run_m0_valid_seen_pair.sh
```

远程连接可能中断时使用 `nohup`。先创建日志目录并用时间戳保存日志和 PID；退出
SSH 后进程会继续运行，风险是程序也会继续占用四张 GPU，停止时应只终止记录的 PID：

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/m0-valid-seen-pair-${STAMP}.log"
nohup env \
  GPUS=0,1,2,3 \
  POLICY_CHECKPOINT=/absolute/run/checkpoints/step-000025 \
  BASE_RUN_NAME=m0-pilot-valid-seen-update0 \
  CHECKPOINT_RUN_NAME=m0-pilot-valid-seen-update25 \
  bash scripts/run_m0_valid_seen_pair.sh \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

用 `tail -f "${LOG}"` 查看两个 140-task 进度条。每个结果目录都会保存
`valid_seen_summary.json`、`metrics.{jsonl,csv}` 和包含模型原始响应、动作解析及
ALFWorld 原始输出的 `traces/valid-seen-*.jsonl.zst`。只有 summary 完整、
`evaluated=140` 且 manifest SHA-256 匹配，结果才可用于 update 0/25 对比。

### 正式批量形状的无损性能基准

`benchmark` 只运行一个与正式训练相同形状的 update（8 个任务、每个任务 8
条轨迹），不运行评测，也不能作为实验结果引用。它用于验证工程优化的速度和
输出等价性，不替代 smoke、integration、pilot 或 formal 的任何验证。

先运行旧式的“每个环境步都唤醒/休眠 vLLM”基线，再运行同任务、同随机种子的
持久 rollout session。持久模式仍会在环境步之间清空 prefix cache，以保持与
基线相同的交互式生成语义：

```bash
GPUS=0,1,2,3 PROFILE=benchmark PERSISTENT_ROLLOUT_SESSION=0 \
RUN_NAME=rollout-session-baseline \
  bash scripts/run_alfworld.sh train no_skill

GPUS=0,1,2,3 PROFILE=benchmark PERSISTENT_ROLLOUT_SESSION=1 \
RUN_NAME=rollout-session-optimized \
  bash scripts/run_alfworld.sh train no_skill
```

比较两份轨迹。门禁要求任务、生成 token、原始响应、动作、环境输出和奖励完全
一致，rollout logprob 最大绝对误差不超过 `1e-3`。唯一排除项是 ALFWorld
在每步额外计算、但不进入策略输入、动作解析、奖励、世界状态 checksum 或策略更新
的 `extra.expert_plan`；其余 `info` 字段仍严格比较：

```bash
BASELINE=$(find "$PWD/runs" -maxdepth 1 -type d \
  -name '*-rollout-session-baseline' | sort | tail -n 1)
OPTIMIZED=$(find "$PWD/runs" -maxdepth 1 -type d \
  -name '*-rollout-session-optimized' | sort | tail -n 1)

python scripts/compare_rollout_session_runs.py "$BASELINE" "$OPTIMIZED"
```

两份 `metrics.jsonl` 会额外记录 `perf/rollout_seconds`、
`perf/rollout_generation_worker_seconds`、`perf/old_logprob_seconds`、
`perf/reference_logprob_seconds`、`perf/actor_update_seconds` 和
`perf/core_update_seconds`，用来区分生成、环境和训练张量阶段。只有比较结果
`passed=true` 后，才允许把持久模式设为正式默认。该门禁已通过，因此脚本默认
`PERSISTENT_ROLLOUT_SESSION=1`；需要诊断或回退时可显式设为 `0`。从旧式会话
checkpoint 原地恢复时，必须继续显式使用其保存的 `0`，不能静默改变运行时语义。

### 独立 ALFWorld 环境并发门禁

正式形状的每个 update 包含 64 个相互独立的环境。`ENVIRONMENT_WORKERS` 只控制
这些已加载环境的 `step` 与 `close` 是否并发等待，不改变任务、环境实例、模型
请求顺序、随机种子、动作或奖励。环境创建与 `reset` 保持串行。TextWorld 1.7
不仅在 PDDL loader 中使用进程级共享 Tatsu parser，`step` 收集 admissible commands
和 expert info 时还会进入 `textworld.logic` 的另一个模块级 parser；INFO-SKILL
因此用同一个可重入锁保护 `textgen._parse_and_convert` 与
`logic._parse_and_convert`，环境 step 的其他部分仍可并发。该优化
通过门禁前默认保持为 `1`。以已经通过的持久会话 benchmark 作为串行基线，再运行
一次 64 worker 候选：

```bash
GPUS=0,1,2,3 PROFILE=benchmark PERSISTENT_ROLLOUT_SESSION=1 \
ENVIRONMENT_WORKERS=64 RUN_NAME=environment-workers-64 \
  bash scripts/run_alfworld.sh train no_skill
```

比较时必须显式选择环境并发门禁；比较器会同时要求两侧均使用持久会话、基线
worker 数为 1、候选 worker 数大于 1：

```bash
BASELINE=$(find "$PWD/runs" -maxdepth 1 -type d \
  -name '*-m0-sft-noskill-benchmark-persistent-u1' | sort | tail -n 1)
OPTIMIZED=$(find "$PWD/runs" -maxdepth 1 -type d \
  -name '*-environment-workers-64' | sort | tail -n 1)

python scripts/compare_rollout_session_runs.py \
  "$BASELINE" "$OPTIMIZED" \
  --comparison-mode environment-workers
```

新 trace 必须达到 `passed=true` 才能启用。新指标中的
`perf/environment_create_seconds`、`perf/environment_reset_seconds`、
`perf/environment_step_seconds` 和 `perf/environment_close_seconds` 用于确认串行
加载与交互各自的耗时；若并发不稳定或无收益，保持 `ENVIRONMENT_WORKERS=1`。

### 终端日志级别

训练默认使用精简终端日志：保留 INFO-SKILL 的运行时初始化/就绪、逐 update 指标、
checkpoint、评测汇总、错误 traceback 和所有 `tqdm` 进度条，但不再把每个 Ray
worker 的 Qwen 配置、vLLM 初始化、FSDP 弃用警告及重复消息复制到 driver 终端。
Ray worker 的原始 stdout/stderr 仍保存在 `/tmp/ray/session_latest/logs/`；项目的
`console.log`、`metrics.jsonl`、压缩轨迹、模型原始响应和环境原始输出均不受影响。
需要调查底层运行时问题时，可一键恢复详细输出：

```bash
VERBOSE_RUNTIME_LOGS=1 bash scripts/run_alfworld.sh train no_skill
```

`VERBOSE_RUNTIME_LOGS` 只接受 `0` 或 `1`，默认 `0`。它只改变可观测日志，不改变
任务顺序、随机种子、rollout、优化器或 checkpoint 恢复语义。

`formal` 固定 445 个 update，并在 update 0、每 25 个 update 和训练结束后评测
完整 `valid_seen`。正式训练不接受 `MAX_UPDATES` 的其他值。

## 9. `raw_skill_prompt` 独立对照门禁

raw 对照不是 M1，也不从 M0 checkpoint 继续训练。它从同一份注册的原版
`Qwen2.5-7B-Instruct` 权重独立初始化，
复用 M0 已验证的 GRPO、VERL/vLLM、任务顺序、奖励、checkpoint、恢复和
`valid_seen` 评测管线；唯一方法差异是每个 episode 按任务目标检索一次最多 17 条
技能，并在每个环境步骤写入完整的 full 技能块。默认使用 YAML 的
`embedding`；`template` 仅作为可选诊断，通过 `RETRIEVAL_MODE=template` 切换，
不需要编辑 YAML。`RAW_SKILL_PROMPT_FORMAT=full` 是正式默认值；`compact` 只用于
显式消融。

先拉取代码并运行不占 GPU 的逻辑门和静态预检：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
git pull origin main
python -m unittest \
  tests.unit.test_skill_retrieval \
  tests.unit.test_raw_skill_training_setup \
  tests.unit.test_training_cli -v

GPUS=0,1,2,3 PROFILE=smoke MAX_UPDATES=1 DRY_RUN=1 \
  bash scripts/run_alfworld.sh train raw_skill_prompt
```

然后运行一次 embedding smoke。启动阶段会先批量计算全部 train 目标和固定 140 条
`valid_seen` 目标的检索结果，释放 embedding 模型及其 CUDA cache，再初始化
Ray/FSDP/vLLM；这不是外部 API 调用。full skill block 会用 policy tokenizer
预计算长度，但不设置独立的 1,600-token 上限。运行时最终 prompt 超过 4,096
tokens 时先从最旧历史开始移除；历史清空后仍超限才明确失败，绝不截断技能或减少
Top-K。

```bash
GPUS=0,1,2,3 \
PROFILE=smoke \
MAX_UPDATES=1 \
PERSISTENT_ROLLOUT_SESSION=1 \
ENVIRONMENT_BACKEND=native_batch \
ENVIRONMENT_WORKERS=1 \
POLICY_MAX_TOKENS_PER_GPU=12288 \
BALANCE_POLICY_TOKENS_ACROSS_RANKS=1 \
RAW_SKILL_PROMPT_FORMAT=full \
INFO_SKILL_CPU_THREADS=1 \
RUN_NAME=raw-skill-embedding-smoke-u1 \
bash scripts/run_alfworld.sh train raw_skill_prompt
```

通过条件与 M0 smoke 相同，并额外要求：

- `resolved_config.json` 与 `provenance.json` 的 `mode` 均为
  `raw_skill_prompt`；
- `skill_conditioning` 记录 `retrieval_mode=embedding`、`prompt_format=full`、
  规范化 skill bank SHA-256、
  223 条 train 轨迹来源 manifest、embedding 模型内容校验值、train 与固定 140 条
  `valid_seen` 的逐任务检索计划、Top-K 与 raw skill block token 统计；
- trace 每条轨迹的 `candidate_skill_ids` 非空，同一 episode 的候选 ID 保持不变；
- trace 用 `history_entries_omitted_by_window` 记录默认 H 窗口省略量，用
  `history_entries_omitted_for_prompt_budget` 单独记录 prompt 超限后额外移除量；
  `history_entries_omitted` 保留为两者之和；历史清空后仍超限则保存结构化错误并终止；
- checkpoint 的 `mode` 与 skill conditioning 不匹配时，恢复和评测都必须拒绝。

smoke 通过后再运行 25-update pilot；它会在 update 0 和 25 各评一次固定 140 条
`valid_seen`。长任务使用 `nohup`，断开 SSH 不会停止训练：

在启动 pilot 前，先独立测一次 raw prompt 的 update-0 起点。这个评测不加载
policy checkpoint，用于区分“raw prompt 本身的影响”和“训练后权重的影响”：

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/raw-skill-valid-seen-update0-${STAMP}.log"
nohup env \
  GPUS=0,1,2,3 \
  EVAL_BACKEND=verl \
  CHECKPOINT_STEP=0 \
  POLICY_CHECKPOINT= \
  RETRIEVAL_MODE=embedding \
  RAW_SKILL_PROMPT_FORMAT=full \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  ENVIRONMENT_WORKERS=1 \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=qwen25-raw-skill-embedding-full-valid-seen-update0 \
  bash scripts/run_alfworld.sh eval raw_skill_prompt \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

每次成功启动的 `eval` 都会写出 `provenance.json` 和
`checkpoint-load.json`。update 0 的 checkpoint 状态应为 `not_requested`；加载
portable checkpoint 的评测应为 `loaded`，并记录各 rank 的 base-sync 报告和加载
耗时。`evaluation-timing.json`、`metrics.jsonl` 与
`valid_seen_summary.json` 同时记录 task discovery、skill setup、后端初始化、
checkpoint load、rollout、runtime close、trace write 和总耗时。终端只额外显示一行
精简耗时汇总。

`embedding + full` update-0 为 38/140、macro `0.25145`、非法动作率 `0.09540`；
严格 A/B 的 compact 为 35/140、macro `0.23886`，端到端仅提速约 2%。因此 full
是正式默认，compact 仅为消融。若后续 raw prompt 明显退化，不要立即启动正式训练，
pilot。先运行固定 12 条任务（六类各 2 条）的诊断矩阵，区分检索方式与 prompt
格式的影响。该命令只初始化一次 VERL/vLLM，并补测尚未测量的三个组合：
`embedding + SkillRL concise`、`template + full`、`template + SkillRL concise`。
已有的 140 条 `embedding + full` 结果不重复消耗 GPU。

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/raw-skill-ab-${STAMP}.log"
nohup env \
  GPUS=0,1,2,3 \
  RAW_SKILL_AB_TASKS_PER_TYPE=2 \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=raw-skill-ab-valid-seen-12 \
  bash scripts/run_alfworld.sh raw-skill-ab raw_skill_prompt \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

用 `tail -f "${LOG}"` 查看一个覆盖 36 条轨迹的总进度条。结果目录必须包含
`raw_skill_ab_summary.json`、`provenance.json`、`checkpoint-load.json`、三条
`metrics.jsonl` 记录和三份压缩轨迹。所有这些文件均强制记录
`diagnostic_only=true`、`reportable_as_valid_seen=false`；12 条结果只能定位问题，
不能替代固定 140 条 `valid_seen`、参与 checkpoint 选模或写入论文主表。

这三个变体在固定 12 条任务上均为 0/12，而成对的 `no_skill` 为 2/12。由于
SkillRL 的 GRPO prompt 并不是上面的 SFT/concise prompt，可再运行一次只包含
`skillrl-rl-exact` 的诊断。它从环境提供的规范任务目标做 template 检索，固定
general=6、当前类别全部 task skills、mistakes=5、history=2；step 0 使用 SkillRL
`NO_HIS` 模板且不显示技能，后续步骤才使用其 `WITH_MEMORY` 模板。该命令不加载
embedding 模型、不训练、不保存新 checkpoint，也不会重跑前述三个变体。

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/skillrl-rl-exact-${STAMP}.log"
nohup env \
  GPUS=0,1,2,3 \
  CONFIG=configs/alfworld_qwen25_7b_sft.yaml \
  RAW_SKILL_AB_TASKS_PER_TYPE=2 \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=skillrl-rl-exact-valid-seen-12 \
  bash scripts/run_alfworld.sh skillrl-rl-exact raw_skill_prompt \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

结果仍写入 `raw_skill_ab_summary.json`，但只包含一项
`skillrl-rl-exact`，并生成一份 trace、一条 metrics、`provenance.json` 和
`checkpoint-load.json`。这里的 exact 仅指锁定 commit 中的 ALFWorld GRPO
prompt 文本与初始静态检索形状；评测继续使用本项目的 greedy 解码、30 步上限、
动作解析和只读技能库，不复现 SkillRL 动态技能更新，不能作为正式 140 条结果。

若要判断当前 SFT 模型是否只对其训练时的 instruction 结构敏感，再单独运行
`skillrl-sft-exact`。它依据已审计的发布 SFT parquet，从 step 0 开始注入静态技能，
使用最近 5 步历史、6 条 general skills、当前类别全部 task skills、5 条 mistakes，
并采用发布数据的未加引号逗号分隔动作格式。该诊断仍保留在线环境提供的全部可执行
动作，不将其人为缩减为发布数据中的 10 条，因此只称为 instruction/static-skill
exact。它不加载 embedding 模型、不训练，也不保存新 checkpoint。

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/skillrl-sft-exact-${STAMP}.log"
nohup env \
  GPUS=0,1,2,3 \
  CONFIG=configs/alfworld_qwen25_7b_sft.yaml \
  RAW_SKILL_AB_TASKS_PER_TYPE=2 \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=skillrl-sft-exact-valid-seen-12 \
  bash scripts/run_alfworld.sh skillrl-sft-exact raw_skill_prompt \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

结果仍使用统一的 `raw_skill_ab_summary.json`、trace、metrics、`provenance.json` 和
`checkpoint-load.json`，但只包含 `skillrl-sft-exact` 一项，并强制标记
`diagnostic_only=true`、`reportable_as_valid_seen=false`。运行目录的 provenance
同时记录参考 parquet 校验值、行数、轨迹数、history=5、step-0 技能注入，以及
“在线完整动作池”这一有意保留的边界。

该入口还会把环境任务末尾句号和 TextWorld welcome banner 从策略可见 instruction
中移除，因为二者都未出现在发布 SFT parquet 对应字段中。provenance 会记录
`task_text_normalization=strip-terminal-period`、
`observation_normalization=strip-textworld-welcome-banner` 和类别分类器版本。注意：
发布 parquet 对 `examine ... with desklamp` 使用独立 `Examine Skills`，而当前 SkillRL
仓库的数据生成分类代码不是这一行为；本诊断以发布 parquet 为准。固定 12 条中若出现
发布训练任务文本未覆盖的 `find ...`、`hot ...`，其结果属于分布外措辞诊断，不能据此
宣称 template 分类器逐字复现了数据生成过程，也不能在本入口中静默改用环境 task type。

在做 oracle 分类或替换基座前，先运行三项 SFT 因果诊断。该入口固定复用同一 12 条
任务、`master_seed=0`、动作解析器和一个 VERL/vLLM runtime；三组依次是：SFT
外壳但无技能的 greedy、统一 no-skill prompt 的 `temperature=0.4` 采样、SFT exact
prompt 的 `temperature=0.4` 采样。它不训练、不写 checkpoint，也不改变任何正式评测
默认值。预计四卡总耗时约 13–18 分钟，可后台运行：

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/skillrl-sft-causal-${STAMP}.log"
nohup env \
  GPUS=0,1,2,3 \
  CONFIG=configs/alfworld_qwen25_7b_sft.yaml \
  RAW_SKILL_AB_TASKS_PER_TYPE=2 \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=skillrl-sft-causal-valid-seen-12 \
  bash scripts/run_alfworld.sh skillrl-sft-causal raw_skill_prompt \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

`raw_skill_ab_summary.json`、`resolved_config.json` 与 `provenance.json` 会逐项记录
`policy_mode`、prompt 格式、`do_sample`、temperature、top-p、最大输出长度和 seed；
三份 trace 仍保留每步模型原始输出、动作解析与环境结果。该矩阵强制标记为诊断，不能
替代 140 条正式 `valid_seen`。

三项 SFT 因果诊断完成后，用 `unified-skill-causal` 运行严格的统一 prompt 四格门。
四格都固定同一 12 条任务、seed、greedy、history=2、观察文本、动作列表、解析器和
一个 VERL/vLLM runtime：第一格是正式 `no_skill` 文本，第二格让相同文本经过
`RawSkillPromptConditioner + EmptyRetriever`，第三格只增加 template 检索的完整技能
块，第四格只增加 embedding 检索的完整技能块。空检索格的 policy user message 必须
与第一格逐字一致；它用于发现 conditioning 分支或 runtime 状态污染，不能作为新方法。

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/unified-skill-causal-${STAMP}.log"
nohup env \
  GPUS=0,1,2,3 \
  CONFIG=configs/alfworld_qwen25_7b_sft.yaml \
  RAW_SKILL_AB_TASKS_PER_TYPE=2 \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=unified-skill-causal-valid-seen-12 \
  bash scripts/run_alfworld.sh unified-skill-causal raw_skill_prompt \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

预计四卡约 10–15 分钟。用 `tail -f "${LOG}"` 查看覆盖 48 条轨迹的总进度条。运行器会
在 summary 的 `control_prompt_parity` 中自动逐步核对第一、二格的 prompt token 数、
生成 token、动作与空 skill ID；prompt 文本必须逐字一致，否则命令以非零状态退出。
完成后打包下面这些文件；trace 用来复核该门禁，并查看 template/embedding 实际选择的
skill ID。

```bash
RUN=$(find "$PWD/runs" -maxdepth 1 -type d \
  -name '*-unified-skill-causal-valid-seen-12' | sort | tail -n 1)
STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="$PWD/unified-skill-causal-diagnostics-${STAMP}.tar.gz"
tar -C "$RUN" -czf "$ARCHIVE" \
  raw_skill_ab_summary.json \
  provenance.json \
  resolved_config.json \
  checkpoint-load.json \
  metrics.jsonl \
  metrics.csv \
  console.log \
  traces
echo "run=${RUN}"
cat "$RUN/raw_skill_ab_summary.json"
ls -lh "$ARCHIVE"
```

请提供上面三段终端输出（`run=...`、完整 summary、压缩包大小）和生成的压缩包。
结果仍必须带 `diagnostic_only=true`、`reportable_as_valid_seen=false`，不能据 12 条
诊断直接选择 checkpoint 或报告正式成功率。若 no-skill 与空检索轨迹不一致，应先
判定本次门无效并检查 runtime 状态；只有二者一致后，才解释 template/embedding 差异。

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/raw-skill-pilot-${STAMP}.log"
nohup env \
  GPUS=0,1,2,3 \
  PROFILE=pilot \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  ENVIRONMENT_WORKERS=1 \
  POLICY_MAX_TOKENS_PER_GPU=12288 \
  BALANCE_POLICY_TOKENS_ACROSS_RANKS=1 \
  CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
  RAW_SKILL_PROMPT_FORMAT=full \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=raw-skill-embedding-full-pilot-u25 \
  bash scripts/run_alfworld.sh train raw_skill_prompt \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

`tail -f "${LOG}"` 可持续查看 update 和 `valid_seen` 进度条。只有真实 smoke、
checkpoint 加载评测和 pilot 都通过后，才能称 raw 对照已完成服务器验证；这不会
改变 M1 `infoskill` 已完成代码接线、但仍需通过独立 GPU smoke 与恢复门禁的事实。

## ALFWorld 环境多进程基准

在训练中启用任何进程式环境后端前，必须先运行独立的 CPU 差分基准。它不会加载
Qwen、vLLM、Ray 或 FSDP。`smoke` 使用 2 个任务 × 2 条 rollout × 3 步；`full`
复现正式训练的 8 个任务 × 8 条 rollout × 30 步形状。两者都将候选后端的规范状态、
原始转移输出、终止标志、奖励和世界状态 checksum 与当前串行 `batch_size=1` 实现比较。

```bash
PROFILE=smoke bash scripts/benchmark_alfworld_environment.sh
PROFILE=full bash scripts/benchmark_alfworld_environment.sh
```

`smoke` 门只要求 `semantic_exact=true`；只有 `full` 门额外要求环境工作至少加速
`1.25x`。失败时返回非零退出码，不会改变正式训练入口；结果 JSON 写入 `runs/`。

两个 CPU 门通过后，通过显式开关把候选后端用于训练 smoke：

```bash
GPUS=0,1,2,3 \
PROFILE=smoke \
MAX_UPDATES=1 \
PERSISTENT_ROLLOUT_SESSION=1 \
ENVIRONMENT_BACKEND=native_batch \
ENVIRONMENT_WORKERS=1 \
INFO_SKILL_CPU_THREADS=1 \
RUN_NAME=m0-sft-noskill-native-batch-smoke-u1 \
bash scripts/run_alfworld.sh train no_skill
```

该后端已经依次通过 64 轨迹 benchmark 严格 A/B 和两个连续 benchmark update
寿命门，因此训练入口默认使用 `native_batch`。A/B 比较使用
`--comparison-mode native-batch`；旧 run 未记录该字段时仍按 `individual` 解释，
从旧 checkpoint 恢复时不会静默切换后端。若要诊断或回退，可显式设置：

```bash
ENVIRONMENT_BACKEND=individual ENVIRONMENT_WORKERS=1 \
  bash scripts/run_alfworld.sh train no_skill
```

默认后端的服务器验证结果是：64 条轨迹与 `individual` 的任务、token、logprob、动作、
环境状态和奖励严格一致；单 update 核心耗时从约 `818s` 降到约 `408s`。连续两次
benchmark update 均完成，`perf/environment_forced_terminations=0`，CPU 内存从
约 `50.7 GiB` 增至 `52.9 GiB`，未观察到环境进程泄漏。

TextWorld 1.7.0 的默认 `_ChildEnv.__del__` 会在正常 close 后仍对环境进程发送
`SIGTERM`；如果进程从 Ray driver fork，会继承 Ray 的 signal handler 并打印误导性的
Ray SIGTERM 堆栈。INFO-SKILL 的 native batch 关闭路径先发送 TextWorld 原生 close
控制消息并等待子进程正常退出，只在 2 秒总超时后强制终止。每次运行必须满足
`perf/environment_forced_terminations=0`；非零表示真实的环境进程关闭故障，应阻止后续门。

### 已拒绝的显存回收与激活优化候选

以下候选均不得用于正式实验：

- 关闭 rollout step 之间的 `torch.cuda.empty_cache()`：语义轨迹和 rollout logprob
  完全一致，但单个 benchmark update 从约 `407.90s` 增加到 `412.08s`，且峰值显存
  没有下降。因此该候选没有性能收益。
- 关闭 actor gradient checkpointing：runtime 可以正常启动，但第一次 actor update
  在 Qwen MLP/LoRA 前向传播中发生真实 CUDA OOM。GPU 0 当时已使用约
  `78.85/79.25 GiB`，下一次 `470 MiB` 分配失败。因此必须保持
  `actor_ref.model.enable_gradient_checkpointing=true`。

不要通过减少 rollout 数、任务组、token 上限、训练 batch，或调低 vLLM 显存比例来
“救活”第二个候选；这些改动会改变正式实验定义或牺牲吞吐，不再是同条件工程优化。

### 逐卡 CUDA 显存基线

VERL 原有的 `perf/max_memory_allocated_gb` 会经过 worker 指标归并，且 vLLM sleep
allocator 的逻辑峰值可能超过物理卡容量，不能解释为单卡物理显存峰值。
INFO-SKILL 的显式诊断模式使用后台线程定期调用 CUDA driver 的 `mem_get_info`，分别
记录 rollout 与 old/ref/actor policy 阶段每个 rank 的最小物理 free memory；同时按
VERL 实际连续分片记录每个 rank 的训练 token 数。该监控不改变张量、随机数、batch
或模型状态，但会产生轮询开销。未显式设置时仍为 `0`；200ms 只用于短时显存诊断，
长时 pilot/formal 建议显式使用 `CUDA_MEMORY_POLL_INTERVAL_MS=1000`，若要关闭必须在
运行命令中明确设为 `0` 并接受失去阶段内物理峰值证据。

先按当前正式默认值运行一个完整形状 benchmark：

```bash
GPUS=0,1,2,3 \
PROFILE=benchmark \
PERSISTENT_ROLLOUT_SESSION=1 \
ENVIRONMENT_BACKEND=native_batch \
ENVIRONMENT_WORKERS=1 \
INFO_SKILL_CPU_THREADS=1 \
CUDA_MEMORY_POLL_INTERVAL_MS=200 \
RUN_NAME=cuda-memory-baseline-u1 \
bash scripts/run_alfworld.sh train no_skill
```

汇总最后一个 train update：

```bash
RUN=$(find "$PWD/runs" -maxdepth 1 -type d \
  -name '*-cuda-memory-baseline-u1' | sort | tail -n 1)
python scripts/report_cuda_memory.py "$RUN"
```

报告必须显示 `token_budget_decision_ready=true`。判断训练 token 动态装箱是否还有
放大空间时，只使用 `policy_physical_min_free_gb_min` 和 `tokens/max_to_min_ratio`；
任何超过 `policy_total_gb_min` 的 PyTorch allocator peak 都只是逻辑计数。取得有效基线
前不暴露更大的 token budget；若余量不足则保持 `16384`，不通过缩 batch 或关闭梯度
检查点为候选腾显存。

当前 A800 基线的最差物理余量为 rollout `12.93 GiB`、policy `14.76 GiB`，四个
rank 的训练 token 最大/最小比为 `1.211`。因此首个保守候选只把 old/ref/actor
训练侧动态微批预算从 `16384` 提高到 `20480`，vLLM rollout 调度预算保持 `16384`，
不同时改变两个变量，也不直接尝试 `24576/32768`。候选运行必须保留相同的 200ms
物理显存采样：

```bash
GPUS=0,1,2,3 \
PROFILE=benchmark \
MAX_UPDATES=1 \
PERSISTENT_ROLLOUT_SESSION=1 \
ENVIRONMENT_BACKEND=native_batch \
ENVIRONMENT_WORKERS=1 \
CUDA_MEMORY_POLL_INTERVAL_MS=200 \
POLICY_MAX_TOKENS_PER_GPU=20480 \
INFO_SKILL_CPU_THREADS=1 \
RUN_NAME=token-budget-20480-u1 \
bash scripts/run_alfworld.sh train no_skill
```

用本节 `16384` 物理采样 run 作为 baseline，执行：

```bash
python scripts/compare_rollout_session_runs.py \
  "$BASELINE" \
  "$OPTIMIZED" \
  --comparison-mode token-budget
```

默认门禁同时要求轨迹语义完全一致、rollout logprob 最大绝对误差不超过 `1e-3`、
完整 core update 至少提速 `3%`，并且 candidate 在 rollout 和 policy 两阶段的最差
物理空闲显存都不低于 `8 GiB`。只有 `passed=true` 才允许把 `20480` 提升为正式默认；
OOM、余量不足或收益不足均保持当时的 `16384` 基线。`POLICY_MAX_TOKENS_PER_GPU` 会写入
resolved config，恢复时不得静默修改。后续长时运行使用 1,000ms 低频物理采样；200ms
仍只用于短时诊断。

真实 A/B 结果拒绝了 `20480`：轨迹与 logprob 完全一致，policy 从 `185.76s`
降到 `173.53s`，但 rollout 波动到 `218.32s` 后，core 只从 `400.10s` 降到
`391.85s`，仅提速 `2.1%`，未达到 `3%` 门限；policy 最差物理空闲显存同时从
`14.76 GiB` 降到 `8.06 GiB`，只比安全线高约 `64 MiB`。因此拒绝向上放大到
`20480`；该值只保留为已拒绝候选的显式复现实验参数。随后 raw/full 长 prompt 的
跨模式安全门发现 `16384` 余量不足，并据下文结果把统一默认下调到 `12288`。

### 跨 rank 策略 token 均衡

基线每个 rank 都收到 423 行，但 token 总数分别为 232201、281087、277543 和
254791，最大/最小比为 `1.211`。INFO-SKILL 顶层编排没有调用固定 VERL trainer
已有的长度均衡步骤，因此同步 FSDP 计算可能等待最长 rank。候选复用固定 VERL 的
Karmarkar–Karp 等行数分区算法，并按“同一个同步 GRPO minibatch”分别重排：每次
optimizer step 的全局样本集合、顺序边界和超参数不变，只改变样本所在 rank；padding
位置被显式追踪，不进入首 update 的真实 logprob 门禁。

```bash
GPUS=0,1,2,3 \
PROFILE=benchmark \
MAX_UPDATES=1 \
PERSISTENT_ROLLOUT_SESSION=1 \
ENVIRONMENT_BACKEND=native_batch \
ENVIRONMENT_WORKERS=1 \
CUDA_MEMORY_POLL_INTERVAL_MS=200 \
POLICY_MAX_TOKENS_PER_GPU=16384 \
BALANCE_POLICY_TOKENS_ACROSS_RANKS=1 \
INFO_SKILL_CPU_THREADS=1 \
RUN_NAME=rank-token-balance-u1 \
bash scripts/run_alfworld.sh train no_skill
```

仍使用 `20260906T071133Z-cuda-memory-physical-u1` 作为未均衡 baseline：

```bash
python scripts/compare_rollout_session_runs.py \
  "$BASELINE" \
  "$OPTIMIZED" \
  --comparison-mode rank-token-balance
```

门禁除语义、logprob、至少 `3%` core 提速和至少 `8 GiB` 物理余量外，还要求均衡后
`perf/tokens/max_to_min_ratio <= 1.02`。单 update A/B 达到 `passed=true`：core 从
`400.10s` 降到 `366.99s`，policy 从 `185.76s` 降到 `149.27s`，轨迹和 rollout
logprob 完全一致，policy 最差物理空闲显存从 `14.76 GiB` 增加到 `23.45 GiB`。

随后两个连续正式形状 update 均完整提交 checkpoint。均衡前最大/最小 token 比依次为
`1.211` 和 `1.376`，均衡后为 `1.000015` 和 `1.000022`；core 分别为 `376.86s`
和 `239.88s`，CPU 内存稳定在约 `52.9 GiB`，无 worker、OOM 或环境关闭异常。因此
`BALANCE_POLICY_TOKENS_ACROSS_RANKS=1` 已成为正式默认，`0` 只作为显式回滚开关。
在该默认值启用前创建、且 resolved config 中没有该字段的历史 checkpoint 仍按
`false` 解释；恢复这类 checkpoint 时必须显式设置
`BALANCE_POLICY_TOKENS_ACROSS_RANKS=0`，系统不会在续训中静默改变样本分配。

### 正式 policy token budget 与长时显存监控

`raw_skill_prompt + embedding + full` 的正式形状首更新暴露了比 no-skill 更长的输入，
`POLICY_MAX_TOKENS_PER_GPU=16384` 的阶段快照一度只剩约 `2.51 GiB` 物理空闲显存。
把 old/ref/actor 动态微批预算降到 `12288` 并以 200ms 采样重跑后，policy 与 rollout
最差物理空闲显存分别为 `18.64 GiB` 和 `12.46 GiB`；64 条轨迹、任务、动作、奖励及
rollout/recompute 对齐统计与 `16384` 运行完全一致，optimizer 和 portable checkpoint
均正常提交。代价是该次观测的 core/policy 时间约增加 `13.4%/10.1%`，其中包含高频
200ms 监控开销，不能解释为纯 token-budget 开销。

因此 `12288` 是 `no_skill`、`raw_skill_prompt`、`infoskill` 共用的正式默认；它只改变
old/ref/actor 的动态装箱，不改变全局样本集合、GRPO minibatch、优化目标或 vLLM
rollout 的 `16384` 调度预算。长时运行建议：

```bash
POLICY_MAX_TOKENS_PER_GPU=12288 \
CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
bash scripts/run_alfworld.sh train raw_skill_prompt
```

命令行显式值始终优先。旧 checkpoint 的 resolved config 若记录 `16384`，续训必须继续
显式传入 `16384`；更早、缺少该字段的 checkpoint 也按历史默认 `16384` 解释。若要改用
`12288`，必须创建新 run，并按新的实验分支记录，不能原地恢复后静默切换。

1,000ms 监控的两 update 寿命门随后通过：两个 update 各含 8 个任务、64 条轨迹，任务
集合互斥，游标从 8 前进到 16；core 耗时为 `668.52s` 和 `679.99s`，跨 update 最差
policy/rollout 物理空闲显存为 `18.64/13.02 GiB`，CPU 内存为 `45.86/46.18 GiB`，
`environment_forced_terminations=0`。首更新通过完整 rollout/recompute 门禁，两个
checkpoint 都声明完整 LoRA optimizer/scheduler 状态，最终 checkpoint 标记为 permanent。
全部 3,141 个环境步骤都保留模型、动作和环境输出；没有 prompt-budget 历史删减，只有
18 步以长度上限结束（两个 update 各 9 步），不足以支持扩大 256-token response 上限。

### portable checkpoint 推理效果诊断门

如果同一份固定 `valid_seen` manifest 上，update 0 与非零 checkpoint 的 140 条轨迹、
token 和 logprob 完全一致，先不要重跑完整评测，也不要修改学习率。运行下面的轻量门，
用两个互相独立的 VERL/vLLM runtime 生成 3 条固定 ALFWorld 风格 prompt：checkpoint
分支严格按正式评测顺序在第一次 rollout 之前加载 portable state，baseline 分支不加载：

```bash
GPUS=0,1,2,3 \
POLICY_CHECKPOINT=/root/autodl-tmp/wjh/alfworld_eval/infoskill/runs/20260906T103804Z-m0-sft-noskill-pilot-u25/checkpoints/step-000025 \
CHECKPOINT_EFFECT_MAX_NEW_TOKENS=64 \
RUN_NAME=m0-pilot-update25-checkpoint-effect \
bash scripts/run_alfworld.sh checkpoint-effect no_skill
```

该命令不创建 ALFWorld 环境、不训练也不修改 checkpoint。它依次加载两个 7B runtime，
随后检查四个边界，并把完整结果写到对应 run 的 `checkpoint_effect.json`：

1. 磁盘 `adapter_model.safetensors` 与每个 rank 当前 FSDP LoRA 是否逐张量完全相同；
2. 每个 rank 的 vLLM 是否注册并激活了唯一的非零 LoRA，并在 BF16 量化容差内
   匹配 checkpoint 的 A/B 聚合统计；
3. 同一批确定性 prompt 的生成 token 是否改变；
4. 即使 token 没变，已选 token 的 logprob 是否出现大于 `1e-7` 的变化。

终端只打印一行分类和 JSON 路径。分类含义如下：

- `checkpoint_to_fsdp_mismatch`：portable checkpoint 没有完整进入 actor；
- `fsdp_to_vllm_missing_or_zero`：actor 正确，但 rollout 侧未激活非零 LoRA；
- `checkpoint_effect_below_probe_precision`：两段权重链路均正确，但 3 条探针在当前
  BF16 推理精度下 token/logprob 均未变化；
- `checkpoint_effect_visible`：checkpoint 已在 rollout 输出或概率上产生可测影响。

前三种情况命令返回非零退出码 5，其中前两种属于工程错误；第三种不是加载错误，表示
当前 25 updates 产生的 LoRA 更新太小，不能把旧 355 条不同随机流上的小幅成功率变化
当作策略提升证据。只有最后一种返回 0。

固定 SkillRL 的 `dummy_dtensor` rollout 在第一次 sharding session 只同步冻结基座，
不会注册 LoRA。INFO-SKILL 因此在每次 portable checkpoint 加载前先执行一次不生成
token 的基座同步，再恢复 FSDP LoRA；后续首次正式 rollout 才能直接注册恢复后的
adapter。这个顺序同时适用于独立 checkpoint 评测和断点续训，禁止删除该 warm-up 或
把 checkpoint 恢复提前到它之前。

## M1 `infoskill` 首次服务器门禁

M1 代码接线完成后，先使用已通过 formal gate 的 train-only grounding 目录做静态
预检。`GROUNDING_DATA` 只覆盖本次进程内的配置，不修改受 Git 跟踪的 YAML：

```bash
GROUNDING_DATA=/absolute/path/to/completed-grounding-run \
GPUS=0,1,2 PROFILE=smoke MAX_UPDATES=1 DRY_RUN=1 \
PERSISTENT_ROLLOUT_SESSION=1 \
ENVIRONMENT_BACKEND=native_batch ENVIRONMENT_WORKERS=1 \
POLICY_MAX_TOKENS_PER_GPU=12288 \
BALANCE_POLICY_TOKENS_ACROSS_RANKS=1 \
INFO_SKILL_CPU_THREADS=1 \
bash scripts/run_alfworld.sh train infoskill
```

当前故障恢复期间默认三卡。固定 VERL floor normalization 会把 smoke 配置的 16
动作样本 minibatch 规范化为 15（正式配置 256 规范化为 255）；这不会丢弃轨迹，
只改变同步 minibatch 边界，实际值必须出现在
`runtime/effective_action_minibatch_size`。两卡和四卡精确保持配置值。

静态预检通过后才运行一个真实 update。该门必须同时验证 soft-prefix rollout、
LoRA/projector 联合策略更新、auxiliary 更新、五个 M1 模块和两个 optimizer/scheduler
的 portable checkpoint；任何 auxiliary 有限值错误、grounding manifest 错误或
checkpoint 文件缺失都应 fail-fast：

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PWD/logs/m1-infoskill-smoke-${STAMP}.log"
nohup env \
  GROUNDING_DATA=/absolute/path/to/completed-grounding-run \
  GPUS=0,1,2 PROFILE=smoke MAX_UPDATES=1 \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch ENVIRONMENT_WORKERS=1 \
  POLICY_MAX_TOKENS_PER_GPU=12288 \
  BALANCE_POLICY_TOKENS_ACROSS_RANKS=1 \
  CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=m1-infoskill-smoke-u1 \
  bash scripts/run_alfworld.sh train infoskill \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

首个 smoke 通过后，下一道门是从 `step-000001` 以相同三卡恢复到 update 2；之后才
对该 checkpoint 做固定 140 条 `valid_seen`。不要直接启动 M1 pilot 或 formal。

## Planner grounding 安全提速门

这些命令只使用 CPU；不会占用当前可用的三张 GPU。正式 `grounding` 默认仍是
`individual`，下面所有候选都通过显式环境变量启用。每项都重新运行相同的串行基线，
并比较完整逐步序列，而不是只比较成功率。

先在六类各 2 条上验证 4 个独立 worker。小样本只要求完全一致与实际观察到 4 个环境
并发，不用它判断稳定提速：

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PWD/logs/planner-parity-4worker-${STAMP}.log"
nohup env \
  GROUNDING_PARITY_TASKS_PER_TYPE=2 \
  GROUNDING_PARITY_WORKER_BATCH_SIZE=3 \
  GROUNDING_PARITY_PARALLEL_WORKERS=4 \
  GROUNDING_PARITY_CANDIDATE_BACKEND=process_parallel \
  GROUNDING_PARITY_MINIMUM_SPEEDUP=0 \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=m1-grounding-planner-parity-4worker \
  bash scripts/run_alfworld.sh grounding-planner-parity \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

随后在同一固定 12 条上验证原生 batch size 4。串行与候选各自只有一个 bounded
父 worker，差异仅为逐任务环境与固定四 slot TextWorld batch：

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PWD/logs/planner-parity-native4-${STAMP}.log"
nohup env \
  GROUNDING_PARITY_TASKS_PER_TYPE=2 \
  GROUNDING_PARITY_WORKER_BATCH_SIZE=12 \
  GROUNDING_PARITY_CANDIDATE_BACKEND=native_batch \
  GROUNDING_NATIVE_BATCH_SIZE=4 \
  GROUNDING_PARITY_MINIMUM_SPEEDUP=0 \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=m1-grounding-planner-parity-native4 \
  bash scripts/run_alfworld.sh grounding-planner-parity \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

两项 `passed=true` 后才运行六类各 10 条的 exact、性能与生命周期压力门。这里要求
端到端至少提速 5%；若不足，结果仍可证明一致性，但不能批准 native batch 为正式后端：

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PWD/logs/planner-parity-native4-stress60-${STAMP}.log"
nohup env \
  GROUNDING_PARITY_TASKS_PER_TYPE=10 \
  GROUNDING_PARITY_WORKER_BATCH_SIZE=60 \
  GROUNDING_PARITY_CANDIDATE_BACKEND=native_batch \
  GROUNDING_NATIVE_BATCH_SIZE=4 \
  GROUNDING_PARITY_MINIMUM_SPEEDUP=1.05 \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=m1-grounding-planner-parity-native4-stress60 \
  bash scripts/run_alfworld.sh grounding-planner-parity \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

每次完成后检查 `planner-parity.json`。必须同时满足 `passed=true`、所有
`field_checks/lifecycle_checks/identity_checks=true`、`performance_passed=true`，并且
`parallel.peak_environment_slots` 与候选一致。任何失败都不要启动 3,553 条正式运行；
保留对应 run 目录用于诊断。

## Grounding 卡死保护与断点恢复

提交 `925498c` 生成的旧正式运行没有逐 shard 持久化；即使进度曾到 3,008/3,553，也不能
从该数字安全恢复。更新代码后应先停止旧进程树，再创建一次新的小规模验证运行。新运行会
生成 `grounding-resume.json` 和 `grounding-shards/shard-NNNN/{results.jsonl,complete.json}`。
只有存在并通过 SHA-256、任务顺序及参数校验的 `complete.json` 才算可恢复。

默认无进展上限为 300 秒。这个计时器在每完成一个任务后重置；触发时会终止整个 worker
进程组，把尚未完成的任务改为逐条隔离。单条任务仍超时会明确写为
`expert_wall_timeout`，不会产生伪造标签，也不会放宽 99% formal coverage gate。需要调整时
使用 `GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS`，正式运行建议保持 300。

恢复一个由新版本创建、参数完全相同的 run 时，不要同时设置 `RUN_NAME`：

```bash
GROUNDING_RESUME_RUN=/absolute/path/to/runs/<grounding-run> \
GROUNDING_WORKER_BATCH_SIZE=64 \
GROUNDING_WORKER_PROCESSES=1 \
GROUNDING_REPLAY_BACKEND=native_batch \
GROUNDING_NATIVE_BATCH_SIZE=4 \
GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS=300 \
INFO_SKILL_CPU_THREADS=1 \
bash scripts/run_alfworld.sh grounding
```

程序会先把已提交任务计入进度条，只执行缺失 shard。若工作项、源码校验和、配置文件内容
或任何关键运行参数与原 run 不一致，会在启动 worker 前拒绝恢复；此时应检出原提交后再
恢复，不要手工修改 `grounding-resume.json`。

### 2×3 组合并行正式候选

旧提交 `925498c` 的 3,008 条进度不能恢复；以下命令只适用于更新后新建的 run。组合候选
同时使用 2 个 bounded worker 和每 worker 3 个原生 planner slot，总并发为 6。它不使用
GPU，但会并发创建 6 套 TextWorld 临时环境；要求启动前至少有 23 GiB 空闲磁盘。先运行
固定 12 条完全一致性门：

```bash
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PWD/logs/planner-parity-native2x3-${STAMP}.log"
nohup env \
  CUDA_VISIBLE_DEVICES="" \
  GROUNDING_PARITY_TASKS_PER_TYPE=2 \
  GROUNDING_PARITY_WORKER_BATCH_SIZE=6 \
  GROUNDING_PARITY_PARALLEL_WORKERS=2 \
  GROUNDING_PARITY_CANDIDATE_BACKEND=native_batch_parallel \
  GROUNDING_NATIVE_BATCH_SIZE=3 \
  GROUNDING_PARITY_MINIMUM_SPEEDUP=0 \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=m1-grounding-planner-parity-native2x3 \
  bash scripts/run_alfworld.sh grounding-planner-parity \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

`planner-parity.json` 必须显示 `passed=true`、`mismatch_count=0`、
`parallel.worker_concurrency=2`、`parallel.peak_worker_processes=2` 和
`parallel.peak_environment_slots=6`。通过后用六类各 10 条重新验证一致性、生命周期和
实际提速；预计约 20--45 分钟：

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PWD/logs/planner-parity-native2x3-stress60-${STAMP}.log"
nohup env \
  CUDA_VISIBLE_DEVICES="" \
  GROUNDING_PARITY_TASKS_PER_TYPE=10 \
  GROUNDING_PARITY_WORKER_BATCH_SIZE=30 \
  GROUNDING_PARITY_PARALLEL_WORKERS=2 \
  GROUNDING_PARITY_CANDIDATE_BACKEND=native_batch_parallel \
  GROUNDING_NATIVE_BATCH_SIZE=3 \
  GROUNDING_PARITY_MINIMUM_SPEEDUP=1.20 \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=m1-grounding-planner-parity-native2x3-stress60 \
  bash scripts/run_alfworld.sh grounding-planner-parity \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

60 条门也通过且 `timed_out_tasks` 为空后，检查磁盘并启动 3,553 条正式运行。预期墙钟约
4--6 小时，具体仍取决于 `pick_two_obj_and_place` 长尾：

```bash
FREE_BYTES=$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')
if (( FREE_BYTES < 23 * 1024 * 1024 * 1024 )); then
  echo "可用磁盘不足 23 GiB，拒绝启动。"
  df -h /root/autodl-tmp
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$PWD/logs/m1-grounding-formal-native2x3-${STAMP}.log"
nohup env \
  CUDA_VISIBLE_DEVICES="" \
  GROUNDING_WORKER_BATCH_SIZE=32 \
  GROUNDING_WORKER_PROCESSES=2 \
  GROUNDING_REPLAY_BACKEND=native_batch \
  GROUNDING_NATIVE_BATCH_SIZE=3 \
  GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS=300 \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME=m1-grounding-formal-native2x3 \
  bash scripts/run_alfworld.sh grounding \
  >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${LOG}.pid"
echo "pid=${PID} log=${LOG}"
```

若 SSH 断开或进程异常退出，使用同一代码版本和完全相同参数恢复；不要设置 `RUN_NAME`：

```bash
nohup env \
  CUDA_VISIBLE_DEVICES="" \
  GROUNDING_RESUME_RUN=/absolute/path/to/runs/<grounding-run> \
  GROUNDING_WORKER_BATCH_SIZE=32 \
  GROUNDING_WORKER_PROCESSES=2 \
  GROUNDING_REPLAY_BACKEND=native_batch \
  GROUNDING_NATIVE_BATCH_SIZE=3 \
  GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS=300 \
  INFO_SKILL_CPU_THREADS=1 \
  bash scripts/run_alfworld.sh grounding \
  >"${LOG}" 2>&1 &
```
