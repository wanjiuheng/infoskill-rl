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

这一步会遍历全部 train 游戏，运行 ALFWorld 手写专家并占用输出磁盘，但不会修改原数据。

```bash
GPUS=0 bash scripts/run_alfworld.sh grounding
```

只有 `manifest.json` 同时满足专家成功覆盖率不少于 99%、超过 30 步比例不高于 1%，才允许作为正式 grounding 版本。隔离原因必须检查，不能只删除失败样本后继续。

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

`OMP_NUM_THREADS` 也可显式设为正整数，例如 `export OMP_NUM_THREADS=1`；空值、
非数字或其他非法值会触发 `libgomp` 警告，但不是 `vllm._C` 缺失的原因。

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

每次真实训练启动前都会要求 ALFWorld `train` 恰好发现 3553 条任务。开发档位从
固定 10% train monitor 中排除任务；正式档位使用全部 3553 条，按每 update 8
个任务组形成 445 个 update，最后一个 update 允许不足 8 组。

一个 update 成功的最低检查项：

- `console.log` 和 `metrics.jsonl` 中所有 loss、KL、gradient norm 均为有限值；
- `rollout_recompute/ratio_mean` 接近 1，并保留 mean/max log-prob 偏差用于决定后续门限；
- `traces/train-update-*.jsonl.zst` 存在，包含该 update 的全部任务组和全部环境步骤；
- `checkpoints/step-*/checkpoint.complete.json` 存在，目录才可用于恢复；
- 第 2 个 update 使用不同于第 1 个 update 的任务，证明游标实际前进。

首个 update 的偏差指标暂不硬编码阈值：先取得 A800 上的真实观测值，再把接受
阈值作为新实验决策固定下来。若指标缺失、出现非有限值或明显偏离 1，应停止，
不能继续放大训练。

验证恢复时，源运行必须原本就按 `MAX_UPDATES=2` 规划，并在
`step-000001/checkpoint.complete.json` 原子提交后中断；不要先让它生成 step 2。
`MAX_UPDATES=1` 的完整运行不能事后扩展预算。验证原拓扑恢复：

```bash
GPUS=0,1,2,3 PROFILE=smoke MAX_UPDATES=2 \
  RESUME=/absolute/path/to/m0-smoke-u2/checkpoints/step-000001 \
  bash scripts/run_alfworld.sh train no_skill
```

验证跨拓扑恢复时只改 `GPUS`；权威 checkpoint 不依赖保存时的 world size。这个
测试应使用另一个只存在 step 1 的中断运行，避免覆盖原拓扑恢复已经生成的 step 2：

```bash
GPUS=0,1 PROFILE=smoke MAX_UPDATES=2 \
  RESUME=/absolute/path/to/m0-smoke-u2/checkpoints/step-000001 \
  bash scripts/run_alfworld.sh train no_skill
```

`RESUME` 必须直接指向 `checkpoints/step-*`。恢复时不能同时设置 `RUN_NAME`；
除 GPU 数外，训练档位、预算、模型、数据及其他 resolved config 必须与原运行一致。
完成 smoke 与两种恢复检查后，再依次运行：

```bash
GPUS=0,1,2,3 PROFILE=integration RUN_NAME=m0-integration \
  bash scripts/run_alfworld.sh train no_skill

GPUS=0,1,2,3 PROFILE=pilot RUN_NAME=m0-pilot \
  bash scripts/run_alfworld.sh train no_skill

GPUS=0,1,2,3 PROFILE=formal RUN_NAME=m0-formal \
  bash scripts/run_alfworld.sh train no_skill
```

`formal` 固定 445 个 update，并在 update 0、每 25 个 update 和训练结束后评测
完整 `valid_seen`。正式训练不接受 `MAX_UPDATES` 的其他值。
