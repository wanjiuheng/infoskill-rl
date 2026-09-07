# INFO-SKILL Domain Language

本文件定义 INFO-SKILL 第一阶段代码和论文共同使用的领域术语。具体参数、训练流程和评测协议统一见 [`docs/EXPERIMENT_SPEC.md`](docs/EXPERIMENT_SPEC.md)。

## Experimental Systems

**Infrastructure Baseline (M0)**:
不使用技能、仅通过端到端 GRPO 训练策略的基线系统，用来验证训练与评测基础设施。
_Avoid_: INFO-SKILL baseline、phase-one full method

**Pinned VERL Runtime**:
由固定 SkillRL 源码 commit 提供、通过适配层使用的 VERL 训练运行时；其版本与本项目方法代码独立记录和校验。
_Avoid_: latest VERL、copied SkillRL application

**INFO-SKILL Trainer**:
由本项目拥有的顶层训练状态机，统一编排轨迹采集、GRPO、辅助更新、评测和持久化，并通过 Runtime Interface 调度 VERL worker。
_Avoid_: RayPPOTrainer subclass、VERL-owned method loop

**Distributed Module Placement**:
Qwen 策略主体使用 FSDP FULL_SHARD，INFO-SKILL 的 projector 与辅助模块使用复制式 DDP；冻结语义编码器和技能特征 cache 在每张选中 GPU 上各保留一份。
_Avoid_: full Qwen DDP replica、FSDP-wrapped tiny modules、driver-centralized compression

**Pinned FSDP Strategy**:
首版 M0/M1 固定采用 VERL `strategy=fsdp` 对应的 FSDP1 FULL_SHARD；FSDP2 属于需要新 Runtime ID 和完整兼容性复验的后续升级。
_Avoid_: mixed FSDP versions across controls、unvalidated FSDP2 migration

**Policy Update Coordinator**:
把 FSDP LoRA 与 DDP projector 的两个物理 AdamW 组织成一个原子策略更新：共享 update 计数与调度语义，合并计算全局梯度范数，并保证一起 step 或一起跳过。
_Avoid_: cross-wrapper optimizer state、partial policy step、separate clipping semantics

**Authoritative Portable Checkpoint**:
不复制冻结基座、由 rank 0 保存完整 LoRA、可重分片 LoRA optimizer、小模块及训练游标的权威恢复格式；VERL 原生 world-size 分片仅可作为非权威缓存。
_Avoid_: frozen-base duplication、world-size-bound source of truth、silent optimizer reset

**Shared-Base Reference Policy**:
reference logprob 复用 actor 的同一个冻结 FSDP Qwen 基座，通过同步、顺序执行的 LoRA-disable context 得到，不创建独立 reference model。
_Avoid_: second frozen base、CPU-offloaded reference swapping、concurrent actor/reference access

**Unified Policy Prompt Protocol**:
每步把统一英文 ALFWorld 指令作为单条 `user` message 交给当前模型 chat template；no-skill 与 INFO-SKILL 文本完全相同，raw-skill 只多固定位置的完整技能块。
_Avoid_: extra system message、step-zero special template、model-specific special tokens

**Fast-Update Method (M1)**:
在固定技能库上联合使用状态条件随机压缩、soft prefix 和策略强化学习的 INFO-SKILL 第一阶段完整方法。
_Avoid_: ordinary GRPO、skill-library evolution

**Skill-Injection Control Mode**:
共享同一训练评测框架、但改变技能信息如何进入策略的实验模式；首阶段包括 `no_skill`、`raw_skill_prompt` 和 `infoskill`。

2026-09-07 的 `raw_skill_prompt` embedding smoke 完成工程链路，但其 step-1
portable checkpoint 在固定 140 条 `valid_seen` 上为 0/140；全部任务跑满 30 步，
轨迹以重复 `look` 和重复旧动作循环为主。该 smoke update 的组内 policy signal 为
零，因此在改变已登记 prompt 设计前，必须先补测相同 raw prompt 的 update 0，区分
prompt 条件效应和 checkpoint 效应。评测现已要求独立写出 provenance、结构化
checkpoint-load 状态和分阶段耗时。
update-0 的 `embedding + full` 已确认同样为 0/140，且诊断显示格式合规、没有
prompt 截断，但检索类别匹配弱、轨迹大量退化为 `look`/旧动作循环。下一步固定用
六类各 2 条 `valid_seen`、同一模型和环境种子，在一次 runtime 中补测
`embedding + SkillRL concise`、`template + full`、`template + SkillRL concise`；
这些 12-task 结果只用于归因，明确不得作为正式 `valid_seen` 指标。

上述三个 raw 变体在固定 12 条任务上均为 0/12，而同一任务、模型、种子和评测
框架的 `no_skill` 为 2/12，说明当前“每步统一 prompt + 可见技能块”本身已经造成
负向条件效应。进一步核对锁定 SkillRL 源码后确认，其 GRPO prompt 与 SFT prompt、
本项目正式 raw control 都不相同：step 0 使用独立无历史模板且不注入技能，只有后续
步骤才加入 `Retrieved Relevant Experience` 和最近 2 步历史。新增
`skillrl-rl-exact` 仅用于复现这套 GRPO prompt 与初始静态 template 检索；它仍使用
本项目的确定性解码、30 步上限、动作解析和固定技能库，不属于第四个正式 control，
也不代表完整复现 SkillRL 的动态技能更新或训练算法。

对发布的 `Jianwen/SkillRL-SFT-Data` ALFWorld parquet 完整审计确认：文件包含
7,486 行、500 条轨迹和 237 个唯一任务文本；每行从 step 0 起都显示技能，使用
最近 5 步历史、6 条 general skills、当前类别全部 skills、5 条 mistakes，以及
10 条未加引号的逗号分隔 admissible actions。六类静态技能块与当前固定技能库按
上述规则渲染的结果逐字一致。新增 `skillrl-sft-exact` 只复现该 SFT instruction
结构与静态技能选择，用于固定 12 条的归因诊断；在线评测仍保留环境当步提供的全部
可执行动作，而不把动作空间人为抽样成 10 条，因此不得称为完整 SFT 数据生成复现，
也不得作为第四个正式 control 或 140 条正式结果。

发布 parquet 还确认了两项输入规范化边界：训练 instruction 中任务文本没有末尾句号，
初始 observation 也不含 TextWorld 的 welcome banner，因此 `skillrl-sft-exact` 在渲染
当前 observation 和历史 observation 时移除该 banner，并移除任务末尾句号。类别选择
以发布 parquet 实际出现的六种技能块为最高依据；当前 SkillRL 仓库的数据生成脚本会把
`examine` 映射到 `look_at_obj_in_light`，但发布 parquet 对 `examine ... with desklamp`
使用独立 `Examine Skills`，两者并不逐字一致。`valid_seen` 中的 `find ...`、`hot ...`
等表达也未出现在已审计 SFT 任务文本中，属于训练文本分布外措辞，不允许在 exact 诊断
中静默使用环境 task type 进行 oracle 分类；如需验证类别归因，应另做显式 oracle A/B。
_Avoid_: unrelated baseline、different evaluation pipeline

## Skills and State

**Fixed Skill Library**:
训练和评测期间只读、带可审计来源版本的技能集合；首阶段不允许在线增删改查。
_Avoid_: evolving library、dynamic skill bank

**Episode-Level Candidate Retrieval**:
仅依据任务目标在 episode 开始时检索一次候选技能，并由同一任务组的所有轨迹共享候选集合。
_Avoid_: per-step retrieval、rollout-specific candidates

**Canonical Agent State**:
由环境适配层产生的结构化状态单一事实源，供训练、评测、检索和压缩视图共同派生。
_Avoid_: prompt-only state、backend-specific state

**Agent State View**:
从 Canonical Agent State 确定性渲染、针对检索、策略或压缩用途裁剪的信息表示。
_Avoid_: independently assembled prompt、untracked state rendering

**Frozen Semantic Feature Encoder**:
不参与梯度更新、统一产生状态、技能和命令语义表示的编码模型。
_Avoid_: trainable retriever、policy-specific encoder

## INFO-SKILL Modules

**State-Conditional Stochastic Compressor**:
依据当前状态选择并压缩候选技能语义、输出随机 latent 的 INFO-SKILL 模块。
_Avoid_: static summarizer、deterministic retrieval only

**Executable Grounding Head**:
训练期根据状态与 latent 对当前可执行命令进行排序的辅助监督头；它不生成或替代策略动作。
_Avoid_: policy decoder、free-text action generator

**Strict Expert Replay**:
使用 ALFWorld 内置手写专家完整验证 train 游戏，并只从通过动作可执行性和终局成功校验的轨迹生成 grounding 监督；失败游戏整体隔离而不保留部分标签。
_Avoid_: wrapper-fallback labels、partial failed demonstration

**Hybrid Soft-Prefix Rollout**:
rollout 侧用连续 soft prefix 高速采样、训练侧重算动作概率的执行模式。
_Avoid_: token-only rollout、prefix-free recomputation

**Hybrid Prefix Input**:
由完整文本 token IDs、固定数量的占位位置、对应 soft-prefix vectors 和 prefix mask 组成的 vLLM 输入；调度器显式看到全部序列位置，模型在 embedding lookup 后替换占位 embedding。
_Avoid_: full-prompt embedding transport、hidden uncounted prefix

**Patched vLLM Runtime**:
从固定 vLLM tag 可重复构建、仅承载 Hybrid Prefix Input 所需最小变更且带独立 patch 标识的 rollout runtime。
_Avoid_: edited site-packages、unversioned local fork

**Rollout Compatibility Gate**:
正式 M1 训练前先以 vLLM token 输入对 vLLM token-embedding hybrid 输入验证 transport，再以多个固定 prompt/seed 验证 vLLM rollout 与 Transformers 重算具有一致的序列语义和可接受的概率误差。
_Avoid_: single synthetic sample、conflated transport/kernel error、unverified fast rollout

**Rollout Replay Record**:
连接行为策略 rollout 与训练重算的逐步记录，保存复现当时策略分布所需的精确信息。
_Avoid_: regenerated rollout state、trajectory log only

**Separated Gradient Mode**:
GRPO 策略目标与压缩器辅助目标沿明确边界更新不同模块的默认训练语义。
_Avoid_: implicit joint gradient、latent-density policy ratio

## Reinforcement Learning and Actions

**Task-Grouped Episodic GRPO**:
以同一任务的多条独立完整轨迹组成 group，并按终局回报计算组内相对优势的训练方法。
_Avoid_: cross-task normalization、step-relative GRPO

**Shaped Policy Advantage**:
由主训练回报在同任务轨迹组内标准化得到、只用于 GRPO 策略更新的优势信号；它可以同时反映任务成功与奖励整形。
_Avoid_: fidelity target、unshaped success signal

**Task-Success Fidelity Target**:
仅由每条轨迹是否完成任务在同任务组内标准化得到的辅助学习目标；只有同时包含成功与失败轨迹的组才提供非零目标。
_Avoid_: shaped policy advantage、invalid-action target

**Shaping-Only Group**:
任务结果在组内完全相同、因奖励整形差异仍产生非零 Shaped Policy Advantage 的轨迹组；它能训练策略，但不提供任务成功区分信号。
_Avoid_: mixed-outcome group、zero-signal group

**Resolved Action**:
动作解析器从模型原始响应中确定并规范化后的 ALFWorld 命令。
_Avoid_: raw model response、XML block

**Executable Action Validity**:
Resolved Action 是否与当前环境提供的某条可执行命令规范化后精确匹配。
_Avoid_: tag validity、format compliance

**Invalid-Action Sentinel**:
模型动作无法解析或不属于当前可执行命令集合时，环境适配层提交的固定无效命令；它消耗一步、产生原始环境反馈，但不得推进世界状态。
_Avoid_: look fallback、nearest-action correction、raw illegal alias

**Format Compliance**:
模型输出是否遵守推荐标签协议的独立观测指标，不代表动作是否合法或任务是否成功。
_Avoid_: executable validity、task success

## Data and Evaluation

**Unified Valid-Seen Checkpoint Evaluation**:
pilot 与 formal 在约定 checkpoint 上使用同一个固定 140 条 `valid_seen` manifest 做无梯度确定性评测；不再派生或保留 355 条 train monitor。该曲线用于训练趋势、选模和最终 valid-seen 报告，必须披露为 validation-selected performance，而不是独立隐藏测试。
_Avoid_: per-update evaluation、train-monitor curve、hidden-test claim

**Complete Benchmark Evaluation**:
只有固定 manifest 中全部 140 条 `valid_seen` 游戏均产生可归类终态时才成立的正式评测；模型失败计入固定分母，基础设施故障则使整次评测无效。
_Avoid_: partial denominator、infrastructure-as-model-failure

**Paired Randomness Protocol**:
不同对比方法从同一主种子派生成对且互不串扰的语义随机流，使样本身份不依赖进程、GPU 数或执行顺序。
_Avoid_: global mutable RNG、rank-dependent randomness

**Algorithmic Resume Equivalence**:
恢复训练前后保持任务身份、随机流、更新边界和优化状态连续，不要求不同 GPU world size 的浮点结果逐 bit 相同。
_Avoid_: bitwise cross-world-size identity、restart-from-scratch equivalence
