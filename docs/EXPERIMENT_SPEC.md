# INFO-SKILL Phase-One Experiment Specification

本文集中记录 INFO-SKILL 第一阶段已经确认的训练、评测、运行和可复现性规格。当前目标是在固定技能库上验证状态条件技能压缩与强化学习；技能库演化不属于本阶段。

## Confirmed Defaults and Protocols

**Infrastructure Baseline (M0)**:
Qwen2.5-7B-Instruct 在 ALFWorld 上不使用技能的端到端 GRPO 基线，用于验证多轮采样、策略更新、checkpoint 与评测闭环；它不是 INFO-SKILL 的最终方法。
_Avoid_: 第一阶段完成版、INFO-SKILL baseline

**Fast-Update Method (M1)**:
在固定技能库上联合训练状态条件随机压缩器、soft-prefix 投影器、辅助估计器与 LoRA 策略的 INFO-SKILL 方法；完成 M1 才表示第一阶段完成。
_Avoid_: 普通 GRPO、技能库演化

**Fixed Skill Library**:
在 M1 训练期间允许检索但不允许新增、删除、修改或动态重排内容的只读技能集合。首个正式版本直接采用 SkillRL 由 223 条 ALFWorld `train` 轨迹生成的现有技能；不得使用 `valid_seen`、`valid_unseen` 或测试轨迹生成技能。
_Avoid_: evolving library、dynamic skill bank

**Skill Library Provenance Manifest**:
与每个固定技能库版本一同保存的来源清单，至少记录原始技能文件校验值、生成代码版本、轨迹总数、数据划分（当前为 `train`）、生成模型/方法和生成时间。训练与评测启动时记录 manifest 标识，以便审计数据泄漏和复现实验。
_Avoid_: undocumented skill copy、split-unknown skill bank

**Episode-Level Candidate Retrieval**:
M1 默认在任务开始时仅依据任务目标执行一次固定 embedding 检索：分别选择 6 条 general skills、6 条 task-specific skills，并附加 5 条 common mistakes，候选集最多 17 条，以复现 SkillRL 的候选范围。同一任务的 G 条独立 GRPO 轨迹共享完全相同的候选技能集合。每个环境步骤再由状态条件压缩器结合当前状态与该固定候选集生成当步 soft prefix。逐步重新检索、更小的统一 Top-K 和 template 检索仅作为显式消融，不混入默认主实验。
_Avoid_: per-step retrieval by default、different candidates within one GRPO task group

**Expert-Derived Grounding Target**:
Executable Grounding Head 的默认监督目标是 ALFWorld `train` 文本环境在当前状态给出的规范化专家下一步命令。离线生成器沿训练专家轨迹重放，并保存任务目标、当前 observation、`admissible_commands`、最多 17 条候选技能、专家命令以及可获得的人工 `high_descs`。专家命令是正式 M1 的逐步 grounding target；`high_descs` 只保存用于审计和可选高层文本 grounding 消融。整个过程不调用外部 API 或额外教师模型，`grounding_loss_weight=0` 仍保留为无 grounding 消融。
_Avoid_: validation-derived targets、LLM-generated summaries、concatenated-candidate target

**Strict Expert Replay and Quarantine**:
离线生成器使用 ALFWorld 内置 handcoded expert，而不直接信任包装器的 `extra.expert_plan`：初始状态允许用一次位于当前 `admissible_commands` 的 `look` 启动专家状态机；后续读取专家实际返回动作并要求规范化后与当前可执行命令精确匹配，禁止把包装器静默保留的兜底 `look` 当作标签。生成器最长重放 150 步直到 `won=True` 以验证整条游戏可解，但只持久化策略可达的前 30 步 grounding 状态。任一步出现异常、超时、动作不合法或最终未赢，整个游戏从 grounding 数据中隔离，不保留部分轨迹，也不自动从共同 RL corpus 删除；不混用 planner 专家兜底。manifest 按六类记录总数、成功数、隔离数与原因、轨迹长度分布、超过 30 步的比例和数据/代码校验值；专家成功覆盖率低于 99% 或超过 30 步比例高于 1% 时阻止正式训练并要求检查环境、数据版本或重新审议 `max_steps=30`。
_Avoid_: silent look fallback、partial expert trajectory、mixed expert definitions、unchecked horizon mismatch

**Executable Grounding Head**:
训练期辅助排序头，不负责生成或执行策略动作。它用两层 MLP 将当前状态 embedding 与压缩 latent 映射为 query；同一个冻结 embedding 模型编码并缓存规范化 `admissible_commands`，经线性层得到 command keys。对当前合法命令集合做温度缩放相似度 softmax，并以专家命令索引计算交叉熵。Strict Expert Replay 生成的数据必须保证专家命令位于集合中；数据加载时发现违例应拒绝该数据版本并报告 manifest 不一致，不能在训练中静默跳过。主策略 Qwen 仍独立生成实际环境动作。
_Avoid_: free-text decoder、policy replacement、full-vocabulary generation head

**Canonical Agent State**:
环境适配层产生的结构化单一事实源，至少包含任务/数据划分标识、任务目标、当前步号、当前 observation、最近 H 个已执行动作及对应 observation、当前 `admissible_commands`、终止状态和候选技能标识。ALFWorld 默认 `H=2`，可配置。训练、正式评测与离线 grounding 数据不得各自拼接不一致的状态文本。
_Avoid_: prompt-only state、backend-specific state schema、untracked history truncation

**Agent State Views**:
由 Canonical Agent State 确定性渲染的三个用途视图。`retrieval_view` 只有任务目标，并且每个 episode 仅使用一次；`policy_view` 包含任务目标、最近 H 步历史、当前 observation、步号和全部 `admissible_commands`，与 SkillRL 的策略可见信息一致；`compression_view` 包含前述信息但排除 `admissible_commands`，防止压缩器绕过技能、直接从候选动作形成捷径。Grounding Head 单独把 `admissible_commands` 当作待排序候选集合，而非压缩状态文本。
_Avoid_: admissible-command shortcut in compressor、different train/eval renderers

**Unified ALFWorld Prompt Rendering**:
M0/M1 与三个 Skill-Injection Control Modes 每个环境步骤都把一条英文 ALFWorld 指令作为唯一 `user` message，并调用当前 policy tokenizer 自己的 `apply_chat_template(..., add_generation_prompt=True)`；不额外添加 system message、不硬编码 Qwen 特殊 token。统一指令在 step 0 和后续步骤都包含任务目标、已执行步数、最近 `H` 个按旧到新排列的“动作前 observation + 实际执行动作”、当前一步编号、当前 observation、全部 `admissible_commands` 以及 `<think>/<action>` 输出协议；空历史显式写 `None`，不使用 SkillRL 单独的 `NO_HIS` 模板。`no_skill` 与 `infoskill` 的文本完全相同，后者只在模型输入前端注入 soft prefix；`raw_skill_prompt` 仅在任务目标之后、Current Progress 之前插入 `## Retrieved Relevant Skills`，按 episode 检索固定顺序完整列出最多 17 条技能的 ID、类型和原文。`retrieval_view` 是无标签、无附加说明的原始任务目标；`compression_view` 只含目标、最近历史、当前 observation 和步号，不含 admissible actions，候选技能 embedding 作为 cross-attention 的独立张量输入。
_Avoid_: mode-specific base prompt、system-message drift、raw-skill truncation、admissible leakage into compression、manual special-token assembly

**Frozen Semantic Feature Encoder**:
M1 复用本地 `Qwen3-Embedding-0.6B` 作为不参与反向传播的统一文本特征编码器，用于 episode-level 技能检索、压缩器的状态/技能输入以及 Grounding Head 的命令 keys。正式默认 `semantic_token`：预计算并缓存固定技能的 token-level hidden states，每步批量编码变化的 `compression_view`，同一任务 G 条轨迹共享候选技能缓存；规范化命令表示也按字符串缓存。`semantic_pooled` 每段文本仅保留单个向量，只用于低显存调试和明确消融。三个策略基座共享该语义编码器，只有 soft-prefix projector 适配各自 hidden size。
_Avoid_: trainable retrieval encoder、different semantic encoders per policy base、pooled-only main result

**State-Conditional Stochastic Compressor**:
`semantic_token` 状态与候选技能先投影到 256 维；两层、8 头 cross-attention 以状态 token 为 query、带技能类别与边界信息的技能 token 为 key/value，并严格 mask padding。attention pooling 后分别输出 32 维 `mu` 与 `logvar`，其中 `logvar` clamp 到 `[-10, 4]`；重参数采样得到 latent。两层 `Linear -> SiLU -> Linear` projector 将 latent 映射为 5 个、宽度等于当前策略 hidden size 的向量，再经 RMSNorm 与初始值 0.01 的可学习标量 gate 形成 soft prefix；projector 使用小方差初始化，并记录 gate/prefix RMS/最大值，越界时报警而非静默裁剪。正式默认固定 `latent_dim=32`、`soft_prefix_length=5`、`cross_attention_layers=2`；`latent_dim={16,32,64}`、prefix `{1,5,10}`、层数 `{1,2,4}` 只作为显式消融，不在首轮联合搜索。
_Avoid_: unbounded log-variance、single unmasked skill sequence、silent hyperparameter sweep

**No Grounding Warm-Start**:
正式 M1 先离线生成 train-only 专家 grounding 数据，但不单独预训练 compressor 或 projector；GRPO、fidelity、rate 与 grounding 从 update 1 起联合执行，以免额外 imitation-learning 阶段混淆与 M0 的归因。Grounding 梯度仍只进入 compressor/prior/grounding head，`grounding_weight=0` 是必做消融；近零 prefix gate 用于稳定随机初始化而非额外预训练。
_Avoid_: expert-action pretraining before GRPO、unreported imitation warm-start、missing no-grounding ablation

**Latent Sampling Mode**:
训练 rollout 对同一任务的 G 条独立轨迹分别执行重参数采样；训练重算必须使用 Rollout Replay Record 中保存的原 latent，不能再次采样。正式单次评测默认 `latent_mode=mean`，直接使用 posterior `mu`，并配合策略确定性解码以保证复现。可选 `latent_mode=sample` 评测必须记录种子并明确标记 stochastic evaluation。常规日志只保存 `mu/logvar` 汇总统计和 KL，不输出完整 latent 张量。
_Avoid_: resampled training ratio、stochastic result reported as deterministic、full-tensor routine logs

**Policy Decoding Mode**:
GRPO 训练 rollout 默认 `do_sample=true`、`temperature=1.0`、`top_p=1.0` 且不额外截断 top-k，以复现 SkillRL/VERL 的训练探索设置；同组每条轨迹使用独立、可由全局种子与任务/rollout 标识复现的随机流。正式评测默认 `do_sample=false`、`temperature=0`，并与 `latent_mode=mean` 配对。随机策略评测仅作为显式可选模式，不能混入正式确定性成功率。
_Avoid_: greedy GRPO rollout、sampled main evaluation、untracked worker RNG

**Generation Limits and Stops**:
M0/M1 训练与评测默认文本 prompt 上限 4,096 tokens、每个环境步骤 response 上限 256 tokens；soft prefix 计入模型实际序列长度但不计入文本 prompt 上限。生成 EOS 或完整 `</action>` 时立即停止；缺少标签的基础模型允许生成到 EOS/长度上限后交给 fallback parser。超长 prompt 优先移除最旧历史并记录，任务目标、当前 observation 与合法命令仍超限时显式报错而非静默改变动作空间。日志记录 finish reason、生成长度分位数、达到长度上限比例和因截断解析失败比例；128/384/512 仅为显式可选上限，三个正式对比方法保持一致。
_Avoid_: 512-token default per step、silent right truncation、different train/eval generation caps

**Structured Trajectory Trace**:
每个正式 update 的全部 8 个任务组×G=8、共 64 条训练轨迹都按 update/rank 分片写入压缩 `.jsonl.zst`；每次 `valid_seen` 则完整保存 140 条轨迹。记录 canonical state、模型原始响应、finish reason、token 数、解析/执行动作、`admissible_commands`、环境原始 observation/reward/done/won、合法性/格式、检索技能 ID/分数/版本、轨迹 reward/advantage 与必要训练统计。运行目录另存 resolved config、来源 manifest、console/JSONL/CSV/TensorBoard 指标和六类汇总；外部 W&B 默认关闭。
_Avoid_: one-group-only routine trace、single monolithic log、external-logger-only evidence

**Tensor Persistence Policy**:
当前 update 内存中的 Rollout Replay Record 保留算法所需精确张量。磁盘常规持久化 compact replay：32 维 exact latent、action token IDs、token-level old logprob、mask，以及 soft-prefix checksum/均值/方差/范数/最大绝对值和梯度/KL/ratio/clip 汇总。完整 soft prefix、current/old 对照、中间 hidden state、attention map 与完整梯度仅在 `tensor_dump=on-anomaly`（默认）或显式 `sampled/full` 时保存；全词表 logits 默认永不例行保存。磁盘剩余低于 10GB 时完成原子写入、保存紧急 step-boundary checkpoint 并明确安全停止，不静默降级日志或自动删除数据。
_Avoid_: routine full-vocabulary logits、silent trace loss、mid-update recovery dependency

**Checkpoint Retention and Resume**:
每 5 个完成的 optimizer update 原子保存一次完整恢复 checkpoint，并只轮换保留最近 2 个；它包含 LoRA、INFO-SKILL 模块、两个优化器与调度器、随机状态、task sampler 顺序与游标、已消费任务标识、update 计数和 resolved config，但不复制冻结的 Qwen 基座或语义编码器权重。update 0、每 25 个 update 和最终 update 445 另永久保留轻量评测 checkpoint，只含可移植 LoRA/INFO-SKILL 权重、配置、来源 manifest 和指标；`best-valid` 与 `last` 用 manifest 引用已有 checkpoint，不重复复制权重。默认恢复粒度是已完整提交的 update 边界：进程在 update 中途失败时丢弃不完整轨迹，从最近完整恢复点确定性重做，因此通常最多重算 4 个已完成但尚未形成恢复 checkpoint 的 update；轻量评测 checkpoint 可用于评测或分支初始化，但不承诺恢复原优化器轨迹。磁盘低于 10GB 时在安全边界写入不参与轮换的紧急完整 checkpoint 后停止。
_Avoid_: base-weight duplication、evaluation-only resume、partial-update commit、best-checkpoint copy

**Authoritative Portable Resume Format**:
完整恢复 checkpoint 的唯一权威来源是 rank-0 可移植格式，而不是 VERL 以保存时 world size 命名的原生 FSDP model shards。它不保存冻结 Qwen 基座或冻结 embedding encoder；manifest 记录二者路径、revision 与 checksum，恢复时重新加载并 fail-fast 校验。rank 0 保存完整 LoRA adapter 权重、由 FSDP1 汇聚且按未展平参数语义映射的 LoRA AdamW full optimizer state、projector/auxiliary 模块及其两个 AdamW state、三个 scheduler state、trainer committed-update/task-order/cursor/consumed IDs、semantic RNG counters、resolved config 和 provenance/runtime manifests。保存采用同一文件系统临时目录，所有 rank 完成并校验后最后原子提交 completion manifest；缺少该 manifest 的目录不得恢复。恢复到新的 2/4 卡 world size 时，先从原始基座重建模型和 LoRA，再创建目标 FSDP 拓扑，将 full LoRA optimizer state scatter/reshard 到新 ranks，并广播 DDP 小模块和 optimizer states；mutable per-rank CUDA RNG 仅作同 world-size 诊断，跨 world-size 连续性以命名 stateless semantic streams 为准。VERL 原生 world-size shards 只允许作为默认关闭、可删除的同拓扑快速缓存，不能成为 source of truth。契约测试必须覆盖 4→4、4→2、2→4，并核对 LoRA/小模块参数、Adam 一二阶矩、scheduler、任务游标和下一 update 身份；若固定 FSDP1 不能可靠恢复 full optimizer state，必须阻断 elastic resume，禁止清空 optimizer 后标记为 resume。
_Avoid_: authoritative native shards、full-base checkpoint、non-atomic directory、optimizer-reset resume、untested world-size reshard

**Algorithmic Resume Equivalence**:
断点恢复的验收标准是任务 ID、任务顺序、命名随机流、数据游标、已提交 update 数、模型/优化器/调度器状态连续，且没有跳过或重复提交更新；恢复后的首个 update 必须记录所加载 checkpoint ID 与预期 next-update ID。相同 GPU world size 和软件版本应尽量复现实验轨迹，但不要求 CUDA/vLLM/FSDP 浮点运算逐 bit 相同；从 4 卡切换到 2 卡时允许因分片与归约顺序产生微小数值差异，只要样本身份与算法语义保持不变。恢复集成测试至少覆盖同 world-size 中断恢复和 4→2 卡 checkpoint 重分片加载。
_Avoid_: bitwise cross-world-size criterion、silent sampler reset、partial-update reuse、duplicate committed update

**INFO-SKILL Research Layer**:
由本项目拥有的论文方法实现，包括核心模块、训练编排、环境适配、配置、测试和入口；它通过适配层使用训练运行时，但不拥有通用分布式训练基础设施。
_Avoid_: SkillRL fork、VERL copy

**VERL Runtime Adapter**:
隔离 INFO-SKILL Research Layer 与固定版本 VERL 分布式训练运行时的集成边界；只有扩展点无法承载 soft prefix 时，才允许形成有来源记录的最小 fork。
_Avoid_: direct SkillRL dependency、monkey patch

**Pinned Runtime Baseline**:
首版运行时固定使用 SkillRL 仓库 commit `8e66726ed866a4e0a7f053586a41022798192e6c` 所提供、包名为 `verl` 的训练基础设施，并锁定 Python 3.10、PyTorch 2.6.0、Transformers 4.51.1 与 vLLM 0.8.4；CUDA wheel/build 变体必须在部署服务器上依据完整 NVIDIA driver/CUDA 兼容信息确定并写入最终 lock，不允许猜测。INFO-SKILL 只能经 `integrations/verl/` 调用该 runtime，禁止依赖 SkillRL 的技能生成、动态技能库或实验入口。依赖解析结果保存精确 lock、源 commit 和构建 manifest，启动时执行 fail-fast 版本检查；任何核心版本升级都必须形成新 runtime ID 并重新通过 rollout compatibility、checkpoint resume 与 smoke tests。
_Avoid_: floating dependency range、unrecorded runtime source、latest-version substitution、SkillRL experiment coupling

**Reproducible vLLM Patch Build**:
现有 SkillRL 和环境中的 `site-packages` 保持只读。项目保存仅针对 vLLM 0.8.4 Hybrid Prefix Input 的最小 patch 与构建脚本；脚本检出固定上游 tag/commit、验证源 checksum、应用 patch、构建带本项目本地版本标识的 wheel，并在 manifest 中记录上游 commit、patch checksum、编译环境与 wheel checksum。校验不匹配时停止，不模糊应用补丁；普通 M0/no-skill 可使用未补丁 vLLM 进行诊断，但进入统一正式对比时所有模式记录同一个 runtime ID。
_Avoid_: direct site-packages edit、mutable source checkout、unverified patch application、anonymous wheel

**Project Module Structure**:
项目采用 `src/infoskill` package layout，顶层深 Module 为 `domain`、`episode`、`conditioning`、`rollout`、`learning`、`evaluation`、`persistence` 与 `integrations`；配置、运行脚本、vLLM patch 和测试分别位于 `configs/`、`scripts/`、`third_party/patches/vllm-0.8.4/` 与 `tests/{unit,contract,integration}/`。`Environment` Interface 由 ALFWorld 及后续 WebShop/Search QA Adapter 实现；`SkillConditioner` Interface 由 no-skill、raw-skill-prompt 与 INFO-SKILL Adapter 实现；`RolloutBackend` Interface 由 Transformers 与 patched vLLM Adapter 实现。`TrajectoryCollector` 隐藏 G 条环境的 episode 采集，`LearningEngine` 隐藏 GRPO 与 auxiliary 更新，`EvaluationRunner` 只返回完整评测或 `incomplete`；Persistence 当前只有本地文件实现，不为假想远端存储创建额外 Interface。依赖只能从 entrypoint 流向 learning/evaluation、episode、conditioning/rollout 再到 domain；`integrations` 实现 Interface，但 domain 不得反向导入外部 runtime。
_Avoid_: shallow pass-through packages、VERL types in domain、backend-specific training logic、single-adapter abstraction

**INFO-SKILL-Owned Training Orchestration**:
顶层 `InfoSkillTrainer` 由本项目实现，不继承、复制或修改 SkillRL `RayPPOTrainer.fit()`；它显式编排 `collect trajectories -> finalize rewards -> old/reference logprob -> grouped advantage -> policy update -> auxiliary update -> trace/evaluate/checkpoint`，三个 Skill-Injection Control Modes 共用同一状态机。`VERLRuntime` Interface 只暴露 worker 初始化、逐步生成、actor/reference logprob、policy optimizer update、rollout 权重同步和分布式 checkpoint 操作；`integrations/verl` 内部负责 INFO-SKILL dataclass 与 VERL `DataProto` 转换，任何 `DataProto`、历史 `ppo_*` 配置或 SkillRL reward/dynamic-memory 逻辑不得越过该 seam。Compressor Optimizer、auxiliary losses、评测分母、任务游标与提交 update 状态归 INFO-SKILL Trainer 所有。
_Avoid_: overridden RayPPOTrainer.fit、DataProto domain model、VERL-owned auxiliary step、duplicated mode loops

**Hybrid Soft-Prefix Rollout**:
M1 的主推理路径：压缩器按环境步骤生成连续 soft prefix，vLLM 使用 prompt-embedding 输入执行高速 rollout；训练阶段由 Transformers/FSDP 重算新 logprob。纯 Transformers rollout 仅作为低速正确性验证和故障排查后端。
_Avoid_: token-only rollout、重新采样 latent 后计算概率比

**Rollout Compatibility Gate**:
现有 SkillRL/VERL vLLM worker 只接受 `prompt_token_ids`，仓库中没有可直接复用的连续 `prompt_embeds` rollout，因此 M1 必须先实现纯 Transformers soft-prefix 正确性路径，再为固定版本 VERL/vLLM 增加带来源记录的最小 prompt-embedding 扩展。正式 445-update M1 训练启动前，固定样本上的两条路径必须使用相同文本 token、soft prefix、position IDs、attention mask、停止条件和解析器，并核对生成 token、old logprob 与 finish reason；测试阈值由实现期的数值基线固化，差异超限时只能运行冒烟/诊断，禁止静默改用纯 Transformers 形成另一套正式实验。兼容报告保存依赖版本、代码 commit、GPU/CUDA 信息、输入 checksum 和逐项误差。
_Avoid_: assumed vLLM embedding support、formal run before parity、silent backend substitution

**Hybrid Prefix Input Transport**:
正式 vLLM rollout 不传输整个文本 prompt 的 embedding，而是在 Qwen chat template 所产生第一个文本 token 之前显式加入 5 个计入上下文长度的占位位置，并随请求只传 `[5, policy_hidden_size]` 的 soft-prefix vectors 与同长度 prefix mask；不把 prefix 插入 system/user 消息文本或模板标记内部。vLLM 调度器和 KV cache 按 `5 + text_tokens` 分配；model runner 完成普通 token embedding lookup 后，仅在 mask 位置替换为 soft-prefix vectors，生成响应的 token IDs、解码和 action logprob mask 均排除占位位置。Transformers 正确性路径使用相同的序列布局、position IDs 与 attention mask。动态 soft prefix 请求默认关闭 prefix caching，除非未来实现把 prefix checksum 纳入 cache identity 并重新通过兼容门；总长度检查使用 `soft_prefix_length + text_prompt_tokens + max_response_tokens`。占位 token 的具体 ID 不承载语义、不得加入 tokenizer 或扩大词表，并由 prefix mask 而非 token ID 本身决定替换位置。
_Avoid_: full-prompt embedding main transport、unaccounted prefix positions、token-id-only replacement、dynamic-prefix cache reuse

**Rollout Replay Record**:
为每个环境步骤保存行为策略实际使用的 latent 样本、旧 soft prefix、动作 token 和旧 logprob。训练阶段重放同一个 latent，由当前 projector 重新计算 prefix 和动作 logprob；旧 prefix 用于一致性审计与复现。
_Avoid_: regenerate latent、prefix-free replay

**Separated Gradient Mode**:
M1 的默认梯度路由。GRPO 动作损失更新 Qwen LoRA 与 soft-prefix projector，并在保存的 latent 处停止梯度；随机 encoder、fidelity predictor、state-conditioned prior 与 Executable Grounding Head 仅由 CIB/fidelity/grounding 辅助目标更新。GRPO ratio 默认不包含高斯 latent 密度。
_Avoid_: accidental encoder policy gradient、latent-density ratio by default

**Reference-Policy KL Constraint**:
M0/M1 默认在有效生成动作 token 上使用 `low_var_kl`，系数 `0.01`，reference policy 为关闭 LoRA 的同一冻结 Qwen 基座；KL 不加入环境 reward。M1 的 actor 与 reference KL 分支接收同一个当前 soft prefix，但两侧 prefix 都 detach，使 KL 只约束 LoRA 漂移，不更新 compressor/projector；GRPO policy loss 仍可更新 LoRA 与 projector。
_Avoid_: KL-shaped environment reward、different actor/reference prefixes、projector suppression by KL

**Shared-Base Reference Execution**:
不按现有 VERL 默认路径创建和 CPU-offload 第二套 7B reference model。`InfoSkillActorWorker` 在同一个 actor FSDP Module 上顺序计算 actor 与 reference logprob：actor 分支开启 LoRA；reference 分支在所有 rank 同步进入 PEFT adapter-disable context，以 `torch.no_grad()` 使用相同 token、position/attention mask 和同一个 detach 后的当前 soft prefix，随后无条件恢复 LoRA 与原 train/eval 状态。共享基座只在一次 policy update 内顺序访问，禁止 actor/reference 并发。正式训练前的契约测试必须将该 reference 输出与单独加载的原始 Qwen 基座逐 token 比较并固化数值容差，同时验证 reference 无梯度、context 退出后 actor 输出与状态不变、异常路径也能恢复 adapter、所有 rank 切换次序一致；任何一项失败都阻断正式训练并要求显式选择独立 reference 方案，不能静默创建或 offload 第二模型。M0、M1 和所有 control modes 使用相同实现。
_Avoid_: separately materialized reference、reference CPU swap、prefix mismatch、adapter-state leak、concurrent shared-module forward

**Shared LoRA Policy Configuration**:
M0/M1 和三个策略基座都冻结原始 Qwen 参数，并在 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` 上使用相同 LoRA：`rank=16`、`alpha=32`、`dropout=0`、`bias=none`；embedding 与 `lm_head` 不训练。零 dropout 用于避免 rollout 与训练重算产生非参数更新导致的 logprob 随机差异。正式 checkpoint 保存 LoRA adapter、INFO-SKILL 模块、优化器/调度器、随机状态和完整配置，不复制基座权重；rank 8/32 仅作显式消融。
_Avoid_: full-backbone fine-tuning、LoRA dropout in policy ratio、different ranks across main baselines

**Separated Optimizers**:
算法上保留 Policy Optimizer 与 Compressor Optimizer 两个逻辑优化域，物理上使用三个 AdamW 状态，避免一个 optimizer state 同时跨越 FSDP 与 DDP。`LoRAOptimizer` 位于 VERL FSDP worker，只管理 LoRA，使用 `lr=1e-6, weight_decay=0`；`ProjectorOptimizer` 管理 DDP soft-prefix projector，使用 `lr=1e-4, weight_decay=0.01`；两者由 `PolicyUpdateCoordinator` 作为一个原子 Policy Optimizer 编排，对同一 GRPO graph backward、共享 policy update 计数和 scheduler 进度，合并计算 LoRA 分片梯度与 projector 复制梯度的全局范数并统一按 `1.0` 缩放，只有两侧梯度均有限且准备完成时才一起 step，否则一起跳过。`AuxOptimizer` 管理 stochastic compressor、state prior、fidelity predictor 与 Executable Grounding Head，使用 `lr=1e-4, weight_decay=0.01`，只执行 auxiliary update。三个物理 AdamW 均使用 `betas=(0.9,0.95)`、`eps=1e-8`，前 3% 各自逻辑 update 线性 warmup 后保持常数学习率；分别记录 zero-grad、backward、梯度范数、clip、step、skip reason 和 scheduler。合并 norm 时，FSDP LoRA shard 只计一次全局平方和，DDP projector 的已同步梯度也只计一次，禁止按 rank 重复累计。
_Avoid_: cross-FSDP-DDP optimizer state、partial policy step、separate LoRA/projector clipping、replica-multiplied norm、unlogged gradient routing

**Normalized Auxiliary Objective**:
Compressor Optimizer 默认最小化 `L_aux = L_fidelity + 0.001 * L_rate + 0.1 * L_ground`。其中 fidelity 是对 detached task-group trajectory advantage 的步骤级 MSE；rate 是 posterior 与 state-conditioned prior 的 32 维高斯 KL，先按 latent 维求和再按有效步骤平均；ground 是专家命令在当前合法命令集合上的交叉熵，只按具有合法标签的离线样本平均。各项先按自身有效样本数归一化再加权，并同时记录原始值、加权贡献与梯度范数。`0.1` 来自草案 `alpha2/alpha1`，独立 auxiliary optimizer 不再重复乘整体 `alpha1`；默认不使用 KL annealing。
_Avoid_: double alpha1 scaling、length-dependent auxiliary weights、unlogged weighted loss

**Auxiliary Update Batch**:
每个 GRPO update 执行一次 Compressor Optimizer step。在线部分使用当前 64 条轨迹的全部有效步骤，先在每条轨迹内平均 fidelity/rate，再对轨迹平均，避免长失败轨迹权重更大；rollout 的冻结语义 token features 只在内存复用至本次 update。离线部分从 256 个 train 专家游戏中各均匀抽取一个状态，计算 grounding 与 offline rate；总 rate 为 online/offline rate 各 0.5。辅助 forward 复用 rollout 保存的 32 维重参数 `epsilon`，使同一噪声下的 latent 对 compressor 保持梯度；通过 micro-batch 累积完成一次归一化 optimizer step，完成后释放中间 features。
_Avoid_: step-count-weighted trajectories、second online latent noise、multiple implicit aux steps per GRPO update

**Auxiliary-Weight Tuning Rule**:
首个端到端实现锁定 `fidelity/rate/ground = 1/0.001/0.1`。只有 raw/weighted loss、分支梯度范数、每维 KL、grounding Top-1 accuracy、fidelity 与 advantage 的相关性等训练诊断持续显示失衡时，才允许在固定 train-only 监控集上做单因素调整：`beta={1e-4,1e-3,1e-2}` 或 `grounding_weight={0.03,0.1,0.3}`，不默认执行完整网格搜索。禁止使用 140 条 `valid_seen` 选择权重；权重确定后，三个正式基座实验共用同一设置。
_Avoid_: validation-set tuning、simultaneous coefficient sweep、metric-free weight changes

**Train-Only Monitor Split**:
从六类 ALFWorld `train` 游戏中分别按相对路径稳定哈希排序选取 10%（约 355 条），固化为带源数据校验值的 `train_monitor_manifest.json`。开发与调参阶段该集合同时从 RL 更新和 grounding auxiliary batch 中排除，只用于确定性监控；超参数冻结后，正式训练重新纳入全部 3,553 条 `train` 游戏。该集合不是独立论文测试集，最终结果仍只报告 140 条 `valid_seen`。
_Avoid_: random split per run、monitor examples in pilot gradients、reported monitor generalization

**Periodic Valid-Seen Evaluation**:
正式训练在 update 0、之后每 25 个 optimizer updates 以及训练结束时，对完整 140 条 `valid_seen` 执行无梯度确定性评测（M1 `latent=mu`、策略 greedy）。评测记录六类 success、macro success、overall success、非法动作率和平均步数，但不得用于调整 loss、学习率或其他超参数。每次保留对应 checkpoint，并同时报告固定预算结束的 `last` 与按预注册规则选择的 `best-valid`：先最大化六类 macro success，再比较 overall success、较低非法动作率，最后选更早 checkpoint。使用同一集合选模并报告属于 validation-selected performance，必须明确披露；所有基座与对比方法采用相同频率和规则。
_Avoid_: final-only health check、valid-seen hyperparameter tuning、highest-score-only reporting

**Complete Valid-Seen Denominators**:
正式评测 manifest 固定包含 140 条本地 `valid_seen` 游戏：`pick_and_place_simple=35`、`pick_two_obj_and_place=24`、`look_at_obj_in_light=13`、`pick_clean_then_place_in_recep=27`、`pick_cool_then_place_in_recep=25`、`pick_heat_then_place_in_recep=16`。overall success 为成功总数除以 140；每类 success 使用对应固定分母；macro success 为六类 success 的不加权算术平均。`done=True, won=False`、30 步耗尽、非法输出或生成长度上限等模型行为属于正常失败并保留在分母。Ray worker 崩溃、CUDA OOM、环境初始化异常、游戏文件缺失或序列化错误等基础设施故障以相同任务、checkpoint 和确定性配置最多重试 2 次；仍失败则整次评测标记 `incomplete`，不得产生正式成功率、参与 `best-valid` 或删除该样本缩小分母。有效正式评测必须满足 `evaluated=140` 且六类计数逐项匹配 manifest。
_Avoid_: partial evaluation、dynamic denominator、infrastructure failure counted as loss、failed-game omission

**Reference Hardware Profile**:
正式默认服务器为单节点 4×NVIDIA A800 80GB PCIe、约 1TiB RAM，GPU 可独占。Transformers/FSDP 使用 4 卡 bf16；rollout 默认 4 个 `tensor_parallel_size=1` 的 vLLM replica，初始 `gpu_memory_utilization=0.45`；每卡复制冻结的 Qwen3-Embedding-0.6B 与固定技能 cache。开启 gradient checkpointing，不默认启用参数/优化器/embedding offload。保守后备为 2 个 TP=2 rollout replica，纯 Transformers 单卡只用于正确性验证。
_Avoid_: required quantization、default CPU offload、unnecessary TP for 7B

**Distributed Module Placement**:
Qwen2.5-7B 冻结基座与可训练 LoRA 统一由 VERL FSDP worker 以 FULL_SHARD/ZeRO-3 管理，默认不为 Qwen 主体实现或启用 DDP replica；这是为了给共卡 vLLM、长序列激活和语义编码器保留稳定显存余量，并复用固定 runtime 已有的 FSDP-vLLM 权重同步及 checkpoint 路径。每个选中 GPU 同时复制一份冻结的 Qwen3-Embedding-0.6B、固定技能 semantic-feature cache 与 INFO-SKILL 小模块：projector 作为 Policy Optimizer 参数组使用 DDP 梯度同步，stochastic compressor、state prior、fidelity predictor 与 Executable Grounding Head 作为 Compressor Optimizer 参数组使用 DDP 梯度同步；这些模块不进入 vLLM，rollout 时只把本卡计算出的 5 个 soft-prefix vectors 传给本卡 vLLM。LoRA 更新后通过 VERL sharding manager 同步到 vLLM，小模块只依靠 DDP 保持副本一致。所有 rank 必须执行相同顺序的 collectives；全局损失按跨 rank 的 eligible-sample numerator 与 denominator 求和后归一化，禁止平均各 rank 的局部均值。FSDP actor checkpoint 使用分布式分片并另存 rank-0 可移植 LoRA；DDP 小模块及其优化器状态由 rank 0 保存、校验并在恢复时广播。
_Avoid_: replicated 7B DDP actor、driver-only conditioner、rank-local collective skip、mean-of-rank-means、tiny-module vLLM ownership

**Pinned FSDP Implementation**:
M0 与 M1 首轮正式实验统一使用固定 VERL 的 `strategy=fsdp`，即 FSDP1 FULL_SHARD；不在主对比中启用 `strategy=fsdp2`。虽然 FSDP2 是后续运行时方向，但当前固定 commit 的 LoRA 提取和 vLLM 权重同步仍依赖 FSDP1 的 `FSDP.summon_full_params()` 与 `_fsdp_wrapped_module` 结构，直接切换会同时引入运行时迁移变量。未来采用 FSDP2 时必须生成新 Runtime ID，并重新通过 LoRA-vLLM weight-sync、Hybrid Prefix Input parity、checkpoint save/resume、2/4 卡加载和短程训练测试；不得让不同主对比模式使用不同 FSDP 实现。
_Avoid_: FSDP2-by-config-only、different sharding implementations across M0/M1、runtime migration hidden as method change

**Elastic GPU Selection**:
启动脚本通过单一 `--gpus` 参数接受物理 GPU 列表并自动派生 `CUDA_VISIBLE_DEVICES`、FSDP world size 和 TP=1 rollout replica 数，例如 `--gpus 0,1,2,3` 或 `--gpus 0,1`。两卡模式保持 G、全局任务组 batch 和算法配置不变，必要时自动调整 micro-batch/梯度累积，只降低并行吞吐。checkpoint 必须同时提供可跨 world-size 重分片的训练状态和 rank-0 可移植 LoRA/INFO-SKILL 权重，允许 4 卡与 2 卡之间恢复。
_Avoid_: hard-coded world size、manual YAML edits for GPU count、world-size-locked checkpoints

**Environment Parallelism Profile**:
参考服务器的 112 个 CPU 核中，Ray 默认声明 96 核并为系统、driver、日志和数据加载预留 16 核；设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，数据加载 worker 默认为 8。正式 rollout batch 含 8 个任务组，每组 G=8，共 64 个独立 ALFWorld 环境且每个声明 1 个逻辑 CPU；冒烟/联调分别用 1/4 个任务组。若监控证明 GPU 持续等待环境，可显式扩展到 12 组/96 环境，但不作为默认主配置。两卡正式模式仍保留全局 8 组，仅降低吞吐。
_Avoid_: 0.1-CPU default workers、nested CPU oversubscription、world-size-dependent global group batch

**Formal Training Budget**:
M0/M1 首版正式训练各自对 3,553 个 train 任务执行一次带种子完整遍历：每个完整 update 含 8 个任务组，共 444 个完整 batch 和最后 1 个单任务组部分 batch，合计 445 个 optimizer updates、28,424 条 G=8 训练轨迹；环境硬上限为每条 30 步。冒烟、集成和 Qwen2.5-7B pilot 分别默认 2/20/100 updates。扩展到两遍只能通过统一 `num_train_passes` 参数，并在三个基座及对应 M0/M1 对比中保持同一预算。
_Avoid_: ambiguous total_epochs、dropped final task、method-specific update budget

**Paired Base Initialization**:
每个基座的 M0 与 M1 都从同一份未经 ALFWorld 训练的原始 Qwen 权重独立开始，使用相同 LoRA 初始化种子、train 任务顺序、G、预算、奖励、解码与评测协议；M1 自有模块使用单独固定种子并记录 `initialization_manifest`。update-0 的共同基座 `valid_seen` 结果只计算一次供两者引用。M1 不允许从已训练 M0 warm-start；`init_from_m0` 若实现只能标记为额外继续训练实验，不能进入主对比。
_Avoid_: 890-vs-445 update comparison、different LoRA seeds、duplicated base evaluation

**Paired Randomness Protocol**:
首轮 Qwen2.5-7B 正式对比统一使用 `master_seed=0`，并为 `no_skill`、`raw_skill_prompt` 与 `infoskill` 派生互不串扰但可成对复现的命名随机流。train 任务顺序、LoRA 初始化、环境、策略采样、INFO-SKILL 模块初始化、latent epsilon 与离线 grounding 抽样分别以稳定语义键派生种子；每步策略与 latent 随机键至少包含 `global_update/task_id/rollout_id/env_step`，不得依赖进程号、rank、执行先后或其他模式是否额外消耗随机数。checkpoint 保存并恢复 Python、NumPy、Torch CPU/CUDA RNG 状态、数据游标和语义计数器；两卡与四卡恢复应保持样本身份和随机流一致。正式评测使用 greedy policy 与 latent mean，不消耗随机评测流。首轮只跑一个配对 seed；只有 Seven-Billion Go/No-Go Gate 判为不明确时，才最多增加 `master_seed=1`，且相关对比方法全部成对补跑，不能只补 INFO-SKILL。
_Avoid_: one global mutable RNG、rank-dependent samples、method-only rerun、unpaired seed comparison

**Skill-Injection Control Modes**:
统一框架至少支持三种成对模式：`no_skill` 仅 LoRA+GRPO；`raw_skill_prompt` 使用与 INFO-SKILL 相同的 episode-level embedding 检索和最多 17 条候选技能，但将原文直接放入每步 policy prompt，不启用 compressor/projector/aux loss；`infoskill` 对相同候选集执行状态条件随机压缩并注入 5-token soft prefix。三者从同一基座独立开始，使用相同任务顺序、G、训练预算、奖励、LoRA、解码和评测协议，以分别测量技能检索增益与压缩注入增益。Raw 模式不设独立技能 token 上限，17 条技能必须完整保留；启动时按当前策略 tokenizer 预计算长度，运行时只检查最终 4,096-token prompt，禁止截断技能、减少 Top-K 或过滤任务，超限时显式报错并保存样本。
_Avoid_: M0-vs-M1-only attribution、different retrieval candidates across controls、SkillRL-external-only control

**Phase-One Experimental Scope**:
当前只对 Qwen2.5-7B-Instruct 正式运行 `base_eval/no_skill/raw_skill_prompt/infoskill_full`；确认完整方法产生正向结果后，再在同一 7B 基座运行 `no_fidelity/no_rate/no_ground` 必要消融。Qwen2.5-3B-Instruct 与 Qwen3-1.7B-Instruct 暂不执行 445-update 正式训练，但模型、projector hidden-size、启动配置、checkpoint 与评测接口必须从首版保持可扩展；后续先用 2–20 updates 验证兼容性，再决定跨基座正式实验。单基座结果不得支持跨模型规模普适性声明。
_Avoid_: premature multi-base compute、7B-specific code paths、cross-scale claim from one base

**Seven-Billion Go/No-Go Gate**:
工程通过要求 445 updates 与 step-boundary resume 完整、无 NaN/Inf、首次更新前 ratio 近 1、vLLM/Transformers 无系统性 logprob 偏差、140 条 valid_seen 都有明确终态且六类 denominator 可核对。效果以 `delta = infoskill_full best-valid macro success - max(no_skill, raw_skill_prompt)` 判断：`delta>=+3` 个百分点才进入必要消融；`-3<delta<+3` 为不明确，先诊断并最多补一个配对 seed；`delta<=-3` 不扩展基座。另要求 last 比 best-valid 下降不超过 5 个百分点、无单类被平均值掩盖的严重崩溃，且 grounding accuracy、fidelity correlation、KL/prefix 指标证明 compressor 确实学习。该门槛是单 seed 工程决策，不构成统计显著性或跨规模结论。
_Avoid_: post-hoc success threshold、scale-out on ambiguous result、success-only compressor claim

**Joint Latent-Ratio Mode**:
预留的实验模式，将 latent 视作随机策略的一部分，并在 GRPO ratio 中同时计入新旧高斯 latent 密度；不作为第一阶段默认正式实验配置。
_Avoid_: default training mode、untracked latent likelihood

**Task-Grouped Episodic GRPO**:
对同一个 ALFWorld 任务创建 G 个初始任务相同但状态互相独立的环境实例，每个实例采样一条完整轨迹；仅在这 G 条轨迹内部按终局回报计算 group-relative advantage，并将每条轨迹的 detached advantage 广播到该轨迹所有有效动作 token 和步骤级 fidelity target。提前结束的轨迹通过 mask 排除后续位置。正式 M0/M1 训练固定 `G=8` 以对齐 SkillRL 并降低稀疏奖励下同组全同回报的概率；`G=2` 只用于冒烟测试，`G=4` 只用于小规模联调。
_Avoid_: sequential actions as group samples、cross-task normalization、fake padded environment steps

**Clipped GRPO Policy Update**:
INFO-SKILL 不使用 critic、value loss 或 GAE；advantage 只来自同任务 G=8 完整轨迹的组内标准化回报。策略更新使用 old/new action-token logprob ratio 与 `clip_low=clip_high=0.2` 的 clipped GRPO surrogate，加上既定 reference KL。每批 rollout 默认只做 `grpo_update_epochs=1`，动作响应 minibatch 为 256（不足时使用全部有效样本），启用动态 token batch、每 GPU 上限 16,384 tokens、minibatch shuffle、entropy coefficient 0.001，并用 `seq-mean-token-mean` 防止较长 `<think>` 自动获得更大权重。INFO-SKILL 配置、日志和文档统一使用 `grpo_*` 名称；只有 VERL Runtime Adapter 映射到其历史 `ppo_*` 字段。
_Avoid_: PPO critic semantics、token-count-weighted reasoning、public ppo naming

**RL Training Corpus**:
M0/M1 正式强化学习共同使用 ALFWorld `train` 中全部 3,553 个可运行 TextWorld 游戏，按游戏等概率进行带种子的 shuffle/sampling，并记录抽样顺序以支持成对比较。223 条轨迹只定义固定技能库的来源，不限制 RL 任务范围；`valid_seen` 的 140 条游戏只用于正式评测，绝不进入训练或 grounding 数据生成。按六类任务均衡重采样只作为显式消融，不属于默认主实验。
_Avoid_: skill-source-only RL subset、validation training、silent task balancing

**Step-Relative Optimization**:
按环境步骤构造相对优势的 GiGPO/step-relative 变体，只作为后续可选实验与消融项，不属于 M0/M1 默认算法。
_Avoid_: default GRPO semantics、silent replacement of episodic advantage

**Resolved Action**:
动作解析器从模型响应中得到的规范化 ALFWorld 命令。优先读取 `<action>...</action>`，缺少标签时允许 fallback 解析；是否使用 `<think>`、是否带 `<action>` 标签和响应语言只属于格式统计，不直接决定动作合法性。

**Deterministic Action Fallback**:
完整 `<action>...</action>` 是唯一优先解析来源：单个标签直接取内容；多个标签只有在内容经大小写与连续空白规范化后完全相同时才可接受，不同则判为歧义。没有完整标签时只检查最后一个非空行，允许依次移除 `Action:`/`Assistant:` 前缀、Markdown 列表符号、包围整行的代码反引号以及单个未闭合 `<action>` 前缀，清理后的整行必须与唯一一条当前 `admissible_commands` 在相同规范化后精确相等。不扫描整段 reasoning 中的命令子串，不自动去除句号，不做模糊/最近动作匹配；无法唯一解析时 `resolved_action=None` 并执行 Invalid-Action Sentinel。
_Avoid_: prose substring extraction、last-of-conflicting-tags、punctuation correction、fuzzy action projection
_Avoid_: raw model response、XML-only action

**Executable Action Validity**:
Resolved Action 成功产生且与当前环境提供的某一条 `admissible_commands` 规范化后精确匹配。解析失败或不在当前可执行动作集合中才记作非法动作，每个环境步骤至多计一次。
_Avoid_: tag validity、Chinese-text penalty、environment progress heuristic

**Invalid-Action Execution**:
Resolved Action 合法时向环境提交当前 `admissible_commands` 中对应的规范原文；无法解析或不匹配时，不提交模型原始候选，也不替换为 `look` 或最近合法动作，而是统一提交固定 `__invalid_action__` 哨兵。哨兵消耗一个环境步骤、计一次非法动作并接受 ALFWorld 的原始 parser feedback，但必须保持世界状态不变。启动兼容测试需在六类任务上验证哨兵不会被识别为别名、推进目标、结束 episode 或导致环境异常；失败时阻止训练并要求显式修复环境适配。日志同时记录原始响应、解析候选、实际提交命令、执行前后状态校验值与环境原始输出。
_Avoid_: look fallback、nearest admissible correction、raw invalid alias、free retry

**Format Compliance**:
模型是否按推荐协议输出 `<action>` 等标签的独立日志指标；它不等于 Executable Action Validity，也不单独进入默认奖励。
_Avoid_: action legality、task success

**Primary Training Reward**:
M0/M1 默认轨迹奖励为 `won - 0.01 * invalid_action_count`。不加入逐步长度惩罚、goal-condition success、information bonus；轨迹长度只记录并由 `max_steps` 设置硬上限。正式评测始终只以 `won` 计算成功率。
_Avoid_: step penalty、format penalty、shaped evaluation success
