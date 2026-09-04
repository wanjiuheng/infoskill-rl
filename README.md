# INFO-SKILL

第一阶段实现以 Qwen2.5-7B-Instruct + ALFWorld 为首个闭环，固定 SkillRL 技能库，只验证：

1. `no_skill`：统一 prompt、无技能输入；
2. `raw_skill_prompt`：episode 开始时检索一次技能，逐步完整写入 prompt；
3. `infoskill`：同一候选集经过状态条件随机压缩并生成 5-token soft prefix。

动态技能库增删改查不在当前阶段。正式协议见 `docs/EXPERIMENT_SPEC.md`，架构决定集中在 `docs/ARCHITECTURE_DECISIONS.md`。

## 目录边界

本项目依赖但不会修改：

- 完整 `alfworld-master` 源码：环境、TextWorld wrapper、内置手写专家；
- ALFWorld 数据目录；
- SkillRL 源码：固定 VERL/FSDP 运行时；
- `SkillRL/memory_data/alfworld/claude_style_skills.json`；
- 本地 Qwen policy 与 Qwen3 embedding 权重。

## Linux 环境

推荐新建 Python 3.10 环境。安装会占用较多磁盘，并且 `torch/vLLM/flash-attn` 必须与服务器 CUDA 驱动匹配；不要在已有稳定训练环境里直接覆盖安装。

```bash
conda create -n infoskill python=3.10 -y
conda activate infoskill

cd /workspace/infoskill
pip install -r requirements-server.txt
pip install -e /workspace/alfworld-master
pip install -e /workspace/SkillRL --no-deps
pip install -e .
```

`requirements-server.txt` 中的 flash-attn 使用 SkillRL vLLM-0.8 容器所固定的
Linux x86_64 / Python 3.10 / Torch 2.6 / CUDA 12 / CXX11 ABI false 预编译 wheel。
SkillRL 的 CUDA actor 在模块导入时就需要它，即使关闭 remove-padding 也不能省略。
若服务器环境不满足这些 ABI 条件，应停止并重新选择 wheel，不能退回源码盲编译。
此外固定 `cachetools==5.5.2`：vLLM 0.8.4 的 LoRA cache 调用了该版本仍存在的
私有 `LRUCache.__update`，cachetools 6 及以后已将它替换，会在 vLLM LoRA dummy
profile 阶段报 `LoRALRUCache` AttributeError。

`infoskill` 模式还需要项目自带的 vLLM Python 补丁。它不会改原始 clone，默认在临时目录构建；具体命令见
[`third_party/patches/vllm-0.8.4/README.md`](third_party/patches/vllm-0.8.4/README.md)。

安装后先检查，而不是直接跑长任务：

```bash
python -c "import torch, transformers, vllm, ray, alfworld; print(torch.__version__, transformers.__version__, vllm.__version__, ray.__version__)"
python -m unittest discover -s tests -v
bash scripts/runtime_doctor.sh
```

补丁安装成功后，`runtime-doctor.json` 中应同时出现
`infoskill_hybrid_prefix_api: 1` 和 `has_infoskill_hybrid_prefix: true`。

如果 `SkillRL` 不在 `infoskill` 同级目录，显式指定其源码根目录，避免加载环境中残留的另一份 VERL：

```bash
SKILLRL_SOURCE=/absolute/path/to/SkillRL bash scripts/runtime_doctor.sh
```

`runtime-doctor.json` 只读取已安装包的版本、入口签名、源码路径和校验值，不读取模型权重或数据内容。实现 Hybrid Prefix runtime 时必须以这份报告对应的 vLLM 源码为准，不能用其他版本的内部接口替代。

## 配置和运行

先修改 `configs/alfworld_qwen25_7b.yaml` 顶部的本地路径。脚本通过 `CUDA_VISIBLE_DEVICES` 选卡；进程内的 `cuda:0` 指向所选列表的第一张物理卡。

```bash
# 只检查路径，不加载模型
GPUS=0 bash scripts/run_alfworld.sh validate no_skill

# 原始/任意本地 Hugging Face 模型，确定性评测完整 valid_seen 140 条
GPUS=0 bash scripts/run_alfworld.sh eval no_skill

# embedding 检索 + 原始技能 prompt
GPUS=0 bash scripts/run_alfworld.sh eval raw_skill_prompt

# 训练后的 LoRA + INFO-SKILL 模块；先在 YAML 中填写 checkpoint/adapter
GPUS=0 bash scripts/run_alfworld.sh eval infoskill

# 生成 train-only 严格专家 grounding 数据
GPUS=0 bash scripts/run_alfworld.sh grounding

# M0 静态预检：解析配置和训练档位，但不加载模型、不启动 Ray
GPUS=0,1,2,3 PROFILE=smoke MAX_UPDATES=1 DRY_RUN=1 \
  bash scripts/run_alfworld.sh train no_skill

# M0 首次真实 smoke：四卡、1 个 update、同一任务的 2 条独立轨迹
GPUS=0,1,2,3 PROFILE=smoke MAX_UPDATES=1 RUN_NAME=m0-smoke \
  bash scripts/run_alfworld.sh train no_skill
```

训练档位固定为 `smoke`、`integration`、`pilot`、`formal`。前三者允许用
`MAX_UPDATES` 缩短联调；`formal` 固定 445 个 update，不能覆盖。推荐按
`smoke(1 update) → smoke(2 updates) → integration(20 updates) → pilot(100 updates)`
逐级推进，通过后再启动正式训练。`smoke`/`integration` 不评测，`pilot` 每 25
个 update 只评固定 train monitor，`formal` 每 25 个 update 评完整 140 条
`valid_seen`。

断点恢复直接指向某个带 `checkpoint.complete.json` 的 `step-*` 目录；不再填写
`RUN_NAME`，卡数可以由 4 改为 2：

```bash
GPUS=0,1 PROFILE=smoke MAX_UPDATES=2 \
RESUME=/absolute/output/run/checkpoints/step-000001 \
  bash scripts/run_alfworld.sh train no_skill
```

恢复会校验原始 resolved config、任务顺序和游标，并恢复 LoRA、完整 AdamW
状态和 scheduler；不匹配时直接停止，绝不会静默重置优化器。这里原运行本身必须
预先按 `MAX_UPDATES=2` 启动并在 step 1 后中断；不能把一个预算为 1 的已完成运行
事后扩展为 2，也不能在同一运行目录里覆盖已经存在的后续 checkpoint。

也可以把模型换为 SFT 或 RL 权重，只修改 `policy_model`/`policy_adapter`，评测状态机、动作规则和 140 条分母保持不变。

每次运行创建独立时间戳目录，包含：

- `console.log`：终端信息；
- `resolved_config.json`：实际参数；
- `traces/*.jsonl.zst`：模型原始响应、动作解析、ALFWorld 原始输出及完整轨迹；
- `metrics.jsonl`、`metrics.csv`、`valid_seen_summary.json`：六类及总体成功率。

正式结果只有在 140 条任务全部得到明确终态、六类固定分母一致且无基础设施失败时才标记 complete。

## 当前实现边界

Transformers 评测后端、ALFWorld/技能/INFO-SKILL 核心模块、M0 `no_skill`
GRPO 训练入口及可移植 LoRA checkpoint 已实现。M0 不使用 soft prefix，因此不受
Qwen2.5-7B BF16 的 cross-backend Hybrid Prefix parity 结论阻塞；但它仍需在
A800 服务器完成真实 smoke、update-0 rollout/recompute 审计和断点恢复验证。
`raw_skill_prompt` 与 `infoskill` 训练当前保持 fail-fast；M1 正式训练仍受 Hybrid
Prefix Input parity gate 约束。
