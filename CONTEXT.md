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

**Actor-Only Recovery Fork**:
仅用于已命名、有限区间的恢复实验：从权威可移植 checkpoint 恢复完整 LoRA、INFO-SKILL 模块、optimizer、scheduler、RNG 与任务游标，但冻结 conditioning/projector，只让 LoRA actor 继续更新。该分叉必须固定源 checkpoint 内容身份、学习率、停止 update、评测协议与 drift guard，并写入新的 run；它不属于正式 M1 的默认联合更新语义，也不得用来改写 `Policy Update Coordinator` 的原子 step 约束。
_Avoid_: unnamed actor-only continuation、reset optimizer/RNG/cursor、in-place source mutation、treating recovery results as default M1 semantics

**Authoritative Portable Checkpoint**:
不复制冻结基座、由 rank 0 保存完整 LoRA、可重分片 LoRA optimizer、小模块及训练游标的权威恢复格式；VERL 原生 world-size 分片仅可作为非权威缓存。
_Avoid_: frozen-base duplication、world-size-bound source of truth、silent optimizer reset

**Shared-Base Reference Policy**:
reference logprob 复用 actor 的同一个冻结 FSDP Qwen 基座，通过同步、顺序执行的 LoRA-disable context 得到，不创建独立 reference model。
_Avoid_: second frozen base、CPU-offloaded reference swapping、concurrent actor/reference access

**Unified Policy Prompt Protocol**:
每步把统一英文 ALFWorld 指令作为单条 `user` message 交给当前模型 chat template；no-skill 与 INFO-SKILL 文本完全相同，raw-skill 只多固定位置的 full 技能块，compact 仅作显式消融。
_Avoid_: extra system message、step-zero special template、model-specific special tokens

**Fast-Update Method (M1)**:
在固定技能库上联合使用状态条件随机压缩、soft prefix 和策略强化学习的 INFO-SKILL 第一阶段完整方法。
_Avoid_: ordinary GRPO、skill-library evolution

M1 实现已完成首轮端到端代码接线，等待目标 Linux/A800 GPU 门禁。训练 replay 会把 5 个显式
prefix 占位位置、exact detached latent、rollout 时旧 prefix、动作 token 与行为策略
logprob 放入同一个 tensor batch；旧 prefix 仅用于审计，训练前向重算器会用保存的
detached latent 和当前 projector 生成 inputs embeddings，并只向 projector 回传梯度；
该重算现已通过模型 embedding forward 内部的受控注入点接入 FSDP worker，避免在
FSDP forward 外直接访问已分片 token-embedding 权重。grounding 产物已有 train-only 正式门禁
加载器，并按“不同专家游戏各抽一个状态”确定性采样。LoRA 与 projector 的两个
optimizer 已有共同 finite gate、合并梯度范数、统一裁剪和一起 step/skip 的
`PolicyUpdateCoordinator`。`infoskill` 训练与 VERL 评测入口已经开放；grounding 数据
可由共享 YAML 或 `GROUNDING_DATA` 启动参数提供，并在 Ray/GPU 初始化前校验。

worker 条件化边界现已落地：driver 只负责 episode-level 候选检索，并把压缩视图、
候选 ID、latent seed 与模式作为小型 work item 发送到 Ray；每个 VERL worker 自己加载
冻结语义编码器、预热完整固定库的技能特征 cache、compressor 和 projector，返回 soft
prefix 及 exact replay trace。该专属加载与初始化使用隔离 RNG，不改变 control modes
或后续配对采样的随机流。RPC 会按 world size padding 并核验返回顺序，M1 配置缺少 hybrid-prefix、
语义模型或技能库路径时在启动前失败。

策略侧分布式接线现已继续完成：compressor 与 projector 在每个 worker 上以复制式
DDP 管理；projector 拥有独立 AdamW（`lr=1e-4`、`weight_decay=0.01`、
`betas=(0.9,0.95)`）并与 LoRA optimizer 共享一次 finite gate、合并 norm、统一裁剪及
step/skip。M1 专用 actor 在 dynamic micro-batch 中显式保留 replay latent 和 prefix mask；
GRPO/entropy 分支使用保存的 detached latent 与当前 projector，actor/reference KL 分支
使用同一当前 prefix 但在 projector 输出处 detach，因此 KL 只约束 LoRA。没有 M1 张量的
batch 会 fail-fast，避免 partial policy update；`no_skill` 与 `raw_skill_prompt` 未启用
M1 worker modules，仍委托固定 VERL 原路径。auxiliary 的五个复制式 DDP 模块、独立
optimizer/scheduler、全局归一化梯度累积和有限值门已经接通；可移植 checkpoint 会保存
五个 M1 模块、LoRA、两个 optimizer、两个 scheduler 与 RNG，并在恢复时逐 rank 审计。
首次三卡 M1 smoke 已完成一个真实 update：soft-prefix rollout、LoRA/projector 原子
策略更新、auxiliary 更新、grounding 离线采样和完整 portable checkpoint 均通过；
rollout/recompute P99 为 `0.23041`，低于统一 `0.30` 门限。该诊断 run 显式使用历史
`16384` policy token budget，rollout 最差物理空闲显存仅约 `8.22 GiB`，因此只作为链路
证据，新的 pilot/formal 仍使用正式 `12288`。下一步是从 step 1 恢复到 step 2，再做
固定 140 条 `valid_seen`，不是继续补写训练算法。三卡因 VERL floor normalization 将
配置 minibatch 256 规范化为 255（smoke 16 规范化为 15），实际值写入
`runtime/effective_action_minibatch_size`；所有样本仍参与后续 minibatch。恢复配置门只在
延长 `max_updates` 不改变 3% 整数 warmup 步数时允许该目标变化，其他配置继续严格锁定。

**Skill-Injection Control Mode**:
共享同一训练评测框架、但改变技能信息如何进入策略的实验模式；首阶段包括 `no_skill`、`raw_skill_prompt` 和 `infoskill`。

同一 M1 step-205 checkpoint 的独立评测现已确认存在 fresh-runtime generation 漂移：分叉前
prompt、技能、latent 与 soft prefix 一致，eager 和 CUDA Graph 均可观察到，因此首先排查动态
LoRA checkpoint→FSDP→vLLM 边界。`m1-lora-reproducibility` 会用两个 checkpoint runtime、两个
LoRA-B 全零 base control、逐 tensor 精确 SHA-256 和重复固定 hybrid-prefix generation 一次区分
权重加载/同步漂移与相同权重下的执行漂移；该门完成前不依据单次 140 条结果采用梯度裁剪候选。
`m1-lora-isolation` 将 Graph/eager、前缀/无前缀及三卡填充布局统一到一次四-runtime
后台诊断，连续重复采样并校验实际 vLLM 输入指纹；输出不是正式成功率指标。

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

以上 0/140 均是发布的 `Alfworld-7B-SFT/checkpoint-140` 上的历史诊断，不能外推到
当前登记的原版 Qwen 起点。修复有限动作格式兼容后，原版
`Qwen2.5-7B-Instruct` 在统一框架的 `no_skill` update 0 为 33/140、macro
`0.21777`、非法动作率 `0.10108`；相同模型的 `embedding + full` raw prompt 为
38/140、macro `0.25145`、非法动作率 `0.09540`。因此 raw 技能并未在当前基座上
造成整体崩溃，但完整字段使 prompt 明显增长。随后在候选 ID、顺序、Top-K 和检索
provenance 不变的严格 A/B 中，compact 将技能块 token 缩短约 26%，端到端仅提速约
2%，而结果从 full 的 38/140、macro `0.25145` 降至 35/140、macro `0.23886`。
因此正式 raw control 恢复 `full`；compact 仅作为显式消融，不能与 full 曲线或
checkpoint 混用。

2026-09-09 的 `raw_skill_prompt + embedding + full` 50-update 扩展 pilot 在正确复现
update-0（38/140、macro `0.25145`）并提交 `step-000000` 后，于首个 policy update
前被 rollout/recompute 门禁停止：唯一打印的超标项为 P99=`0.28432` > `0.25`，因此
没有 optimizer step、训练轨迹或 update-1 checkpoint。该结果不能当作模型训练失败或
放宽阈值的依据；代码现会将完整对齐分布、固定阈值、全部失败条件、最坏 token 与
logprob 分桶写入 `rollout-recompute-alignment-failure.json`。下一步从现有 step 0
原地恢复以复现首个 update，先用该产物区分 full 长上下文的正常 BF16 尾差与
padding/EOS、rank 或样本重排错误，再决定是否修改任何门禁或实现。

复现 benchmark 保存的完整分布确认只有 P99 超过旧门限：mean=`0.03010`、
median=`0.00267`、P95=`0.14057`、P99=`0.28432`、误差大于 `1` 的比例
=`0.0212%`、误差大于 `5` 的比例=`0`、ratio mean=`0.99976`，且最大误差不在首尾
token。作为对照，同一原版 Qwen 的 no-skill 首更新 P99=`0.24966`，旧 `0.25`
门限只留 `0.00034` 余量，实际过度贴合单次 no-skill 观测。经确认，统一 P99 门限
校准为 `0.30`，其他六项门限不变，且所有模式必须使用同一阈值；下一步以单个正式
形状 benchmark update 验证 optimizer 与 checkpoint 链路，不直接恢复 50 updates。

该 raw/full benchmark 在 P99=`0.30` 下已完成 optimizer 与 portable checkpoint，
但 `POLICY_MAX_TOKENS_PER_GPU=16384` 的阶段快照一度仅剩约 `2.51 GiB` 物理显存。
使用 `12288` 与 200ms 物理采样重跑后，policy/rollout 最差空闲显存为
`18.64/12.46 GiB`，64 条轨迹和完整对齐统计与 `16384` 运行一致；core/policy 耗时
观测增加约 `13.4%/10.1%`，且包含高频采样开销。经确认，三个训练模式的正式 policy
动态微批默认统一为 `12288`，vLLM rollout 预算仍为 `16384`；长时 pilot/formal 建议
显式使用 `CUDA_MEMORY_POLL_INTERVAL_MS=1000`。旧 checkpoint 继续按其 resolved config
或历史缺省 `16384` 恢复，不能在原 run 中静默切换。

`12288 + CUDA_MEMORY_POLL_INTERVAL_MS=1000` 的两个连续 raw/full benchmark update
现已完成。两批各含 8 个互不重复任务和 64 条完整轨迹，训练游标依次为 8、16；首更新
通过完整 rollout/recompute 门禁，两次 optimizer 指标均有限，step 1/2 可移植 checkpoint
均提交，最终 step 2 为 permanent。跨两次 update 的最差 policy/rollout 物理空闲显存为
`18.64/13.02 GiB`，core 耗时为 `668.52/679.99s`，CPU 内存从 `45.86` 增至
`46.18 GiB`，无环境强制终止。全部 3,141 个步骤都保留模型响应、动作决议和环境输出，
没有 prompt-budget 历史删减；每个 update 只有 9 步因 256-token 长度上限停止。该寿命门
通过，下一步应从原版 Qwen 起点运行新的 25-update raw/full pilot，并只用 update 0 与
update 25 的相同 140 条 `valid_seen` 判断训练趋势。

该 25-update raw/full pilot 后来从四卡 `step-000005` 权威 checkpoint 分叉到两卡并
完整结束：任务游标为 200，updates 1–25 连续，LoRA、完整 optimizer/scheduler 与
portable checkpoint 均通过结构审计。固定 140 条 `valid_seen` 从 update 0 的
38/140、macro `0.25145` 变为 update 25 的 35/140、macro `0.24332`；非法动作率从
`0.09540` 小幅降至 `0.09137`。17 个任务发生结果翻转（7 个失败转成功、10 个成功转
失败），clean 类下降而 heat/cool 类上升，因此这是类别间迁移而非稳定净提升。200 个
训练任务组中仅成功混合信号组占 `45.5%`，shaping-only 组占 `52.5%`；最后 5 updates
后者升至 `65%`。这支持“策略更多学到动作合法性而非终局成功”的诊断假设，但尚不能
单凭一次 pilot 断言因果，也不能据此用 `valid_seen` 调奖励权重。两卡阶段物理最小空闲
显存一度仅约 `1.32 GiB`，只能视为故障恢复路径，不能作为后续 M1 长跑的安全默认。

审计同时发现历史实现的命名分叉只在目标目录记录 update 25，因而把 update 25 错标为
`best-valid`；合并整条曲线后真实最佳是 update 0。现已规定分叉恢复必须继承源 checkpoint
之前的评测历史并保留来源路径。该问题只影响 checkpoint 选择元数据，不改变上述权重、
轨迹或成功率。

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

下一步先做三项单因素 SFT 因果诊断，而不是直接改技能或换基座：固定同一 12 条任务、
同一 seed 和单个 runtime，分别测 SFT 外壳但无技能的确定性版本、统一 no-skill
prompt 的 `temperature=0.4` 采样版本、SFT exact prompt 的 `temperature=0.4` 采样版本。
这三项用于区分“提示词外壳”“确定性解码”“技能内容”三种可能原因；正式评测仍保持
greedy，诊断结果不得当作 140 条 `valid_seen` 成功率。

三项 SFT 因果诊断确认：同一 update-0 SFT 模型在统一 no-skill prompt 下，无论
greedy 还是 `temperature=0.4` 都在相同 2/12 任务成功；SFT 外壳即使不注入技能也
降为 0/12，SFT exact 加技能后同样为 0/12。采样降低重复但没有恢复成功；动作标签
解析率接近 100%，因此不是 parser 或 GRPO checkpoint 问题。SFT 外壳无技能主要
退化为重复 `go` 和非法旧动作，SFT exact 技能则把行为推向更合法但不推进任务的
重复 `look`。下一门是 `unified-skill-causal`：固定 12 条、greedy、history=2 和同一
在线动作呈现，只比较 no-skill、经过 raw conditioner 的空检索、template 技能文本、
embedding 技能文本。前两格必须逐字生成相同 policy prompt；后两格都只在统一 prompt
的固定位置增加完整 skill block。该门仍只用于归因，不重新定义正式 140 条结果。
_Avoid_: unrelated baseline、different evaluation pipeline

## Skills and State

**Fixed Skill Library**:
训练和评测期间只读、带可审计来源版本的技能集合；首阶段不允许在线增删改查。
_Avoid_: evolving library、dynamic skill bank

**WebShop Imitation Audit**:
在 GPU warm-start 前用真实策略 tokenizer 对已准备 WebShop SFT 数据执行的 fail-closed 审计；统一检查
轨迹隔离、步骤连续性、动作可执行性、response 合同、manifest 计数和序列长度。不同 ID 但共享
步骤内容的轨迹必须分到同一 split；同 split 官方重复记 warning，跨 split 内容重复才是硬失败。
_Avoid_: character-length estimate、sample-only inspection、post-training validation

**WebShop Phased Skill Library**:
只从 train human demonstrations 登记来源和动作族、正文按 query 到 purchase 六阶段固定生成且不含
商品实例答案的 WebShop 技能库。
_Avoid_: ASIN memory、validation-derived skill、ALFWorld planner skill reuse

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

**Bounded Grounding Worker**:
只处理固定数量 Strict Expert Replay 任务的短生命周期 CPU 子进程；退出时释放 TextWorld/Fast Downward 的进程级临时资源，父进程按原任务顺序合并结果。
_Avoid_: one-process full replay、larger TMPDIR as a leak workaround

2026-09-12 的 3,553 条 bounded grounding 实跑已确认生命周期修复有效：56 个短生命周期
worker 全部退出并清理临时目录，历史 `OSError` 与 factory `OSError` 均降为 0，最低剩余
磁盘约 18.47 GiB。剩余 1,541 条隔离全部是
`expert_action_not_admissible`，手写专家成功覆盖率为 56.63%，其中双物体任务为
0/813；这属于专家方法门未通过，不再归因于磁盘或 worker 生命周期。正式 M1 仍被
grounding gate 阻止。随后在 CPU-only、六类各 3 条的历史隔离任务上完成了手写专家与
ALFWorld planner 的同任务同种子诊断：18/18 手写失败被精确复现，planner 成功 17/18；
唯一失败为双物体任务在 150 步达到 replay limit。17 条成功轨迹中位数 28 步，6 条超过
30 步，说明 planner 明显优于当前手写专家，但还不能由定向失败样本直接外推全量。

2026-09-12 的 300 条所谓 planner pilot（278/300）及其 28 条长 horizon 诊断均已判定
无效：固定 ALFWorld 源码用 `AlfredExpert(expert_type)` 调用签名为
`AlfredExpert(env=None, expert_type="handcoded")` 的包装器，导致 `"planner"` 被绑定到
`env`、实际专家仍为 handcoded。全部 5,047 个诊断步骤都请求了只有 handcoded 分支使用
的 facts，19 条失败还命中该专家固定 200 步的 `Exception("Timeout")`。因此不得用旧
278/300、3/22 rescued 或 28/28 cycle 数字评价 planner。

INFO-SKILL 现由自身的运行时兼容层窄修正该位置参数，并在父进程与每个短生命周期 worker
创建环境前探测 requested/effective expert type；身份不符即 fail closed。pilot schema v2
记录 `expert_binding`、`expert_identity_gate_passed` 和手写专家 Timeout 指纹。循环统计也
拆成历史重复与轨迹末尾连续至少三次的精确周期，旧的“任意同对出现三次”不再叫作卡死。
六类各 2 条的 CPU-only 真 planner 身份 smoke 已通过：requested/effective 均为 planner，
12/12 成功、轨迹 4--9 步、无手写专家 Timeout 指纹。当前下一门是在相同固定 12 条上
分别运行串行与双 worker planner replay，对完整逐步序列化结果做 exact parity；只有状态、
合法命令、专家动作、终局与异常全部一致且实际观察到双进程重叠，才用双 worker 重跑六类
各 50 条的 300 条 pilot。之后才决定是否运行新 loop diagnostic 或 3,553 条正式
grounding。

串并行逐步一致性门随后通过，双 worker 的正确 planner pilot 也在固定六类各 50 条上
达到 300/300：六类均为 50/50，轨迹 4--11 步、没有超过 30 步、身份门与 pilot 门均
通过；5 个短生命周期 shard 全部清理，实际峰值并发为 2，最低空闲磁盘约 20.98 GiB。
该结果批准进入 3,553 条正式方法门，但不等于正式数据已经生成。审计同时发现历史正式
`grounding` CLI 未显式把 `planner` 传给 bounded worker，仍会使用默认 handcoded；现已
移除 worker 的专家默认值，正式入口显式验证并传入 planner，formal manifest 升为 schema
v2 并记录/强制校验专家身份。旧 schema v1 或未验证的 handcoded grounding 不能被 M1
加载。

Grounding 提速候选包括 4 个独立 bounded worker 的 `process_parallel`，以及单 worker 内
固定 slot 的 `native_batch` planner replay。后者会逐 slot 保存与串行相同的状态、专家动作
和终局数据，完成 slot 的占位动作不会落入样本；运行时按环境 slot 预留磁盘并以 0.5 秒
周期执行 4 GiB 硬熔断。12 条四 worker 和 12 条 native-batch exact parity 已通过，用户随后
批准以 `native_batch_size=4` 启动正式运行；原计划中的 60 条生命周期压力门没有执行，因而
该次正式运行仍承担了尚未覆盖的长尾存活性风险。

首个 `native_batch_size=4` 的 3,553 条正式运行在 3,008 条后暴露 planner 单核搜索长期
不返回；磁盘、inode、内存与整机 CPU 均非瓶颈。该历史运行由旧提交 `925498c` 产生，只在
父进程内存保留已完成结果，因此不能恢复。当前修复增加逐 shard 原子提交、严格校验恢复、
默认 300 秒无任务完成熔断、POSIX 子进程组清理，以及卡住批次的 individual 隔离；单任务
仍超时则以 `expert_wall_timeout` quarantine，99% formal gate 不放宽。该机制已通过本地
故障注入单元测试，仍需服务器小样本超时/恢复 smoke 后才能重新启动正式 grounding。

下一轮正式候选改为 2 个 bounded worker、每个 worker 内 3 个原生 batch slot（总计 6 个
planner slot），shard 大小为 32。这个组合不改变任务、专家、种子、horizon 或 formal gate，
但必须先通过固定 12 条完整字段 exact parity 和 60 条生命周期/性能门。按现有磁盘公式需要
至少 22 GiB 空闲，启动前采用 23 GiB 操作门槛。某个原生 batch 超时时，其未完成任务会在
最多 3 个独立临时目录中并行 individual 重试，完成后仍按原任务顺序提交；这样避免一次
长尾 shard 在 fallback 阶段再串行等待数小时。

该 `native_batch_parallel` 正式运行随后完整处理并提交 3,553/3,553 条、112/112 shards，
结构完整性、任务唯一性、结果校验和、planner 身份和临时目录清理均通过；总耗时约 5 小时。
3,510 条成功，43 条均为双物体任务的 `expert_wall_timeout`，覆盖率 98.7898%，距离 99%
formal 门只差救回 8 条。其余五类为 100%，成功轨迹最长 11 步，horizon 门通过。因此这不是
全量结果损坏，也不应降低 99% 门或重跑已成功的 3,510 条。

当前增加 `grounding-timeout-rescue`：它在读取源结果前校验 manifest、train/skill/work-item
身份、resume plan 及全部 shard marker/结果 SHA，只重跑 43 条 timeout。默认采用 4 个
individual CPU worker、每条一次 600 秒无进展上限；单条首次超时立即 quarantine，不再进行
旧逻辑中的重复第二次等待。成功救援才按 task ID 原位替换，随后重新生成完整 3,553 条派生
manifest，并记录源 manifest SHA 和救援生命周期。理论最坏约 110 分钟，磁盘并发启动门约
16 GiB，救援过程可按一任务一 shard 断点恢复。只有派生 manifest 重新通过 formal gate 后
才能作为 M1 grounding 输入。

M1 首个 3-GPU smoke 已完成两个 update，并在固定 140 条 `valid_seen` 上验证了 update 0 与
update 2。update 2 的 portable checkpoint 在三个 rank 上均加载 LoRA 与 INFO-SKILL 状态；
主指标由 37/140、macro 0.25618 变为 36/140、macro 0.25043，无效动作率由 0.08787 降至
0.07949。两个 update 只构成链路 smoke，不据此宣称学习改善或退化。两次评测分别约 40 与
70 分钟，但轨迹步数、prompt token 和 response token 几乎相同，证明额外耗时不是样本工作量
增加。代码审计发现 native-batch collector 每一步仍按 task 分别调用 M1 conditioning，单次
140 条评测约产生 3,500 个小型分布式调用；3-GPU 下单请求还需补齐到 world size。

现提供仅限评测、默认关闭的 `GROUPED_INFOSKILL_CONDITIONING=1` 候选，把同一环境批次中各
task 的 conditioning 合并为一次 RPC，同时保留每条请求自己的候选技能。旧训练与评测默认
语义不变。候选必须在相同模型、checkpoint、GPU 数、环境后端、检索计划和 140 条 manifest
上通过完整 trajectory/token/logprob 对比，并至少达到 1.05x rollout 提速，才可讨论升级默认；
失败时保持开关为 0，不影响现有结果。

首轮跨任务 batch 候选实测把 conditioning RPC 降到 540 次，并将 rollout 从
`2412.33s` 降到 `2190.86s`（`1.101x`），但门禁正确阻止了采用：相同任务第一步的
32 维 latent 最大差约 `0.0171`，继而造成动作、轨迹与 logprob 分叉。根因是优化同时改变了
worker 内冻结语义编码器和 compressor 的 batch 几何，而不只是减少 RPC。修正版继续默认关闭，
只在一个 RPC 中让每个 data-parallel rank 接收相同逻辑序列，并逐条以 batch-size 1 计算；
返回时仍选择与历史单条调用一致的 rank-0 副本。默认训练路径继续使用原批处理实现，不受该
评测候选影响。修正版必须重新通过同一 140 条 exact parity 与 1.05x 性能门，首轮失败 run
不得作为方法结果或性能依据。

为缩短后续评测提速迭代，新增默认关闭的 batch-size 两阶段门。第一阶段只使用固定压力型
12 条（六类各两条，来自已注册 M1 update-0 中各类 response token 数最高且均达到 30 步的
轨迹），在相同 portable checkpoint 上比较 batch 8 与 batch 12，同时验证所有 rank 权重
加载、完整轨迹/token/logprob 一致性、至少 1.10x rollout 提速及至少 8 GiB 物理显存余量。
报告明确标记为不可作为 valid_seen 结果。只有全部通过时脚本才自动启动 batch 12 的完整
140 条；任一门失败则 fail closed，正式 YAML 默认仍为 batch 8。

首轮 3-GPU、update-2 checkpoint 压力门实测拒绝 batch 12：虽然 rollout 从
`528.69s` 降至 `293.48s`（`1.80x`），生成阶段达到 `1.93x`，物理显存最低仍余
`29.91 GiB`，但 12/12 条轨迹均发生 token/语义分叉。两边在首次分叉前的状态、prompt、
候选技能、conditioning replay 与 soft-prefix 统计逐项相同，漂移来自 batch 8（每 rank
3 个 padding 后请求）变为 batch 12（每 rank 4 个请求）后的 BF16/vLLM batch 几何与 kernel
数值路径变化；greedy 解码在近似并列 token 处放大成后续环境分叉。因此正式评测继续固定
batch 8，不以统计相近替代 exact parity。比较器也明确把 token 或 logprob 长度漂移标为
`logprob_comparison_valid=false`，避免零个可对齐 token 时误报 `logprobs_close=true`。

首轮 M1 445-update 正式训练按实验方要求，把同一 run 内 update 0、25、50……的周期监控
显式设为 batch 12，并自动原子刷新 `valid_seen_learning_curve.svg`。由于 batch 8/12 未通过
逐轨迹 exact parity，这条曲线只用于同 run 趋势观察；所有产物写明
`nonregistered_monitoring_curve`，最终跨方法报告仍对选中的 M1 checkpoint 补 batch-8 固定
140 条评测。训练进程另提供 checkpoint-boundary graceful pause：向
`training-control.json` 中记录的 PID 发送 SIGINT/SIGTERM 后，完成当前 update、提交可恢复
checkpoint 并退出；同配置且不设置新 RUN_NAME 即可在原 run 中续跑。

update 38 现场剖面显示 M1 每轮约 `920s`，主要由 vLLM generation（约 `347s`）和 policy
actor/update（约 `444/509s`）构成；环境、conditioning 与 auxiliary 不是主瓶颈。现增加两项
默认关闭、仅允许命名分叉恢复改变的吞吐候选：跳过 old-logprob 路径未被消费的 entropy，及
显式增大 rollout scheduler token 容量。另增加 `SEGMENT_END_UPDATE`，使同一 step-50 source
可各运行一个完全相同的 step 51 control/candidate，提交 portable checkpoint 后自动暂停，
无需竞速发送信号，也不会触发非 25 边界的 140 条评测。比较器同时门禁完整训练轨迹、rollout
logprob、LoRA/M1/optimizer/scheduler/RNG/trainer state、至少 8 GiB 物理显存和至少 1.05x core
提速；未通过时继续 control。整 update 预缓存 reference 的方案因会跨 optimizer minibatch
冻结旧 projector、改变既定 KL 数学语义而被明确拒绝。

首次 step-50 组合门中，`32768` scheduler 候选改变了 rollout：候选训练 batch 的即时成功率
更高且平均轨迹更短，但这既不是固定验证集效果，也使原始 wall-time 失去同工作量可比性。因此
exact parity 现只用于区分等价工程优化与 behavior-changing 候选；后者不再被表述为“效果有害”，
而是必须补相同评测协议的 macro/overall success 效果门后才能采用。

下一阶段转向两个默认关闭的深层候选。`HYBRID_PREFIX_CUDA_GRAPH=1` 让 patched vLLM 使用固定
地址的 persistent `inputs_embeds` 缓冲区捕获 hybrid-prefix CUDA Graph；它属于等价工程候选，
必须同时通过相同 source checkpoint 的完整轨迹/logprob、最终 checkpoint、物理显存和吞吐门。
`FUSE_KL_PPO_FORWARD=1` 则复用 PPO actor logprob，删除一次 actor KL 前向；为了只做一次
backward，KL 会同时正则化 LoRA 与 projector，明确改变 D009 的梯度边界，属于算法候选。即使
当前 rollout 完全相同也不得被等价门批准，必须用固定 `valid_seen` 的 Macro success 为首要、
Overall success 为次要指标验证效果，并补多 update 稳定性后才可接回 formal。

固定真实 Qwen、真实 step-50 trace、4 个 prompt、两种 prefix、重复与反序请求的执行层诊断进一步
定位了首版 Graph 候选的数值漂移。persistent eager 与 eager 完全一致，Inductor compile-only 与
标准 Graph 完全一致；关闭 Inductor 但继续使用 vLLM 强制的 `custom_ops=["none"]` 后仍保留同一
最大 logprob 偏差 `0.231836`。只把 kernel policy 恢复为 `custom_ops=["all"]` 即与 eager 达到
16/16 序列和全部 logprob 零误差，随后加回 Graph capture 仍为零误差。因此该最小诊断中的因果
变量是 vLLM V1 的 kernel substitution，不是 persistent prefix 或 Graph replay。正式 Graph
路径现精确采用已验证组合：`use_inductor=False`、`custom_ops=["all"]`、Graph capture 开启；旧版
Graph run 的 Inductor/`custom_ops=["none"]` 语义在 resume 配置中保持可区分，只能命名分叉到
新候选。该修复
仍须通过同一 step-50 的完整 140 条 Macro/Overall、稳定性、显存和吞吐门后才能恢复 formal。

同一 step-50、同一 140 条 `valid_seen`、batch 12 的 checkpoint-aware 效果门现已完成。
独立 eager 为 37/140（Macro 0.2613、Overall 0.2643、总耗时约 62.6 分钟）；修正后的
CUDA Graph 两次独立运行分别为 30/140 和 44/140，合计 Overall 恰为 74/280 = 0.2643，
第二次相对 eager 的 Macro/Overall 分别高 3.33/5.00 个百分点，并通过同 checkpoint、任务
manifest、评测协议和加载状态门。Graph 总耗时稳定在约 26--27 分钟，约 2.3x，最低物理
显存余量约 29.36 GiB。由此不再把首轮 30/140 解释为稳定的 Graph 效果损害；M1 可从
step 50 命名分叉，显式采用 `HYBRID_PREFIX_CUDA_GRAPH=1`，但不同时混入尚未通过成功率门的
fused KL/PPO、entropy skip 或更大 scheduler token budget。日常曲线可单次 Graph 评测，关键
checkpoint 使用两次 Graph 汇总，降低独立 runtime 数值扰动经 greedy 多步轨迹放大的噪声。

进一步在同一 step-50 checkpoint、固定 140 条和修正 CUDA Graph 上完成 batch 12/64 各两次
独立完整评测。batch 12 两轮平均为 37/140、Macro 0.253524、Overall 0.264286、总耗时约
1606.8 秒；batch 64 两轮平均为 38/140、Macro 0.258275、Overall 0.271429、总耗时约
837.1 秒，约 1.92x 更快，最低物理显存余量约 29.36 GiB，且无 forced termination。虽然
batch 64 第二轮单独对比偏高的 batch 12 第二轮时严格门返回失败，但按预先声明的两轮平均
效果规则通过。后续 M1 命名分叉的周期评测采用 `EVAL_BATCH_SIZE=64`；step 50 作为 batch 12
到 batch 64 的协议桥，曲线和报告必须标记该边界。训练 batch、rollout batch 与训练算法不变。
旧 step-50 resolved config 尚未记录后来新增的 `evaluation_manifest.execution_mode`；命名性能
分叉现在把该字段与 eval batch/CUDA Graph 开关一并视为允许变化，但无名字的原地 resume 仍严格
拒绝执行模式变化。

后续从 step 50 新建的 M1 formal 分叉采用有界 checkpoint 策略：最近 5 个、当前
`best-valid` 和最终 checkpoint，重合项只保存一份。该策略必须以
`CHECKPOINT_KEEP_RECENT=5`、`CHECKPOINT_KEEP_BEST_VALID=1` 显式开启；旧 run 默认和原地恢复
继续沿用原规则。轮换只删除当前新 run 内已完整提交的标准 checkpoint，并在删除前后分别
`fsync` delete-intent/deleted 到 `checkpoints/retention-audit.jsonl`，源 step-50 及其他 run
不在清理范围。

正式训练现为每个 update 额外写入 `task-outcomes/train-update-NNNNNN.jsonl`，并原子刷新
`task-outcomes-summary.json`。索引覆盖全部任务组，但只保存任务信息、8 条 rollout 的紧凑结果
和完整 trace 的引用；按 `won` 分为 0/8 `hard_failed`、1--7/8 `partial`、8/8 `mastered`，数量
异常单列 `incomplete`。恢复较早 checkpoint 时，更高 update 的旧索引会被可恢复地隔离到
`stale-after-resume-*`，对应完整 trace 也同步归档并保持引用有效，不进入汇总。它不参与当前采样或 loss，未来错题重训只能作为独立命名
分叉。

`valid_seen_learning_curve.svg` 现在为每一个点直接标注 Macro/Overall 百分比，并随点数动态扩宽。
每条 checkpoint evaluation 自带 batch 与 eager/CUDA Graph 协议；从源 run 继承时保留旧协议，
并在 step 50 后 batch 12/eager → batch 64/CUDA Graph 的位置画出明确分界。

训练期间还会在每个完成的 update 后原子刷新 `training_rollout_steps_curve.svg`。上半图分别显示
全部、成功和失败 rollout 的平均环境步数，下半图显示达到 rollout horizon 的比例；每个点带
update/value 提示。命名分叉会从源 `metrics.jsonl` 接续历史曲线，但旧指标只有
`rollout/mean_steps` 时不虚构成功/失败拆分。该图只用于训练行为诊断，因为不同 update 的任务
组成会变化；固定 140 条 `valid_seen` 的 Macro/Overall 曲线仍是效果判断依据。
图表读取或原子写入失败只记录 warning，不得阻断后续训练或 checkpoint 提交。

M1 formal 的 update 75--175 固定 batch-64/CUDA-Graph Macro success 只在约
0.276--0.299 间波动；截至 update 190 训练连续、optimizer 正常、无 NaN/Inf，所以当前问题定义为
“早期提升后的学习平台”，不是训练失效。约 58% task groups 只有 shaping 信号，双物体任务约
95% 没有成功差异；同时后期 projector policy gradient 常显著大于 LoRA actor gradient。当前新增
默认关闭的 `POLICY_GRADIENT_CLIP_MODE=separate` 命名分叉候选，用同一 checkpoint 的 joint A/B
验证共同裁剪是否压制 actor。它保持共享 finite gate 与两个 optimizer 原子 step，只分开裁剪域并
增加可观测指标。未经固定 140 条 Macro/Overall 效果门，不得用于继续 formal 或与 reward/curriculum
改动混跑。

step-205 的 layer-localization v1 已把 LoRA 复现问题从 raw model logits 继续前移，但不同 rank 的
首个异常层分别为 24、0、10，不能再用全局最早层代替逐 rank 执行顺序。层 0 的有效证据显示：
LoRA shrink FP32 buffer 每轮变化，只有 rank 1 的 `gate_up_proj` 变化越过 BF16 expand/combined
output 舍入边界，随后 `down_proj` input 才变化。固定 vLLM 0.8.4 的 shrink 对所有 token 数都使用
split-K（小于 128 token 为 64，否则为 8），并用 atomic add 合并 partial；这构成强机制假说，
但不能只凭 hash 宣布根因。`m1-lora-layer-localization` v2 因而在同一个三卡只读任务中增加逐
rank 首层、有效 token 行、FP32 reference 中间值、probe/rank 轮换，以及 reference-shrink、
reference-expand、full-reference 三种实际推理替换。v2 实跑确认 reference shrink/full 稳定、
reference expand 仍漂移，因此只能严格定位到 native shrink；仅凭默认实现使用 split-K/atomic
add 还不能把 split-K 写成既定根因。v3 在相同 checkpoint-eager runtime 内增加
`native_split_k_one`：复用完全相同的 vLLM Triton shrink kernel、metadata、block/warp/stage，
只把 `SPLIT_K` 改为 1。只有该路径逐边界 exact 时才允许输出
`native_shrink_split_k_atomic_nondeterminism`；否则明确报告 split-K=1 未消除漂移。所有干预仍只在
诊断 session 内生效，不改变训练/评测默认路径，也不能替代 140 条成功率。v3 通过后，正式
train/eval 新增默认关闭的 `LORA_SHRINK_SPLIT_K_ONE=1` 候选：它只复用已经验证的
`native_split_k_one` 干预，进入持久 rollout session 后在每个 rank 安装、离开或异常时恢复，
并要求所有 worker 回报 active 才允许生成。该候选不改 checkpoint、基础权重或 vLLM wheel；
只能通过命名 resume fork 改变，必须从同一 checkpoint 做固定任务与 140 条效果/吞吐门后再晋升。
正式 step-205 重复门随后得到 44/140 与 47/140；轨迹中 139 条生成、130 条动作序列不一致，且
首次动作分叉前输入完全一致。这否证了“split-K=1 已解决全部跨 runtime 漂移”，但不否定 v3 对
native shrink atomic 漂移的局部因果定位。当前候选继续保持非默认，先运行一次性 fresh-runtime
matrix（Graph/eager split-K=1 + eager reference-full，均含 checkpoint A/B 与 base A/B）再决定
下一修复边界，不能从 step-205 继续 formal 长训。
中央 `scripts/run_alfworld.sh` 必须直接执行显式 `PYTHON`（未设置时才回退到 `python`），不能只依赖
外层临时修改 `PATH`；`bash -lc` 会读取登录环境并可能重置 PATH，曾导致预检与真正训练使用不同
解释器。该约束覆盖 validate、eval、诊断、grounding 和 train 的所有 CLI 分支。

首轮 step-200 gradient-clip A/B 暴露了一处配置路由错误：resolved config 已登记 candidate 为
`separate`，但 worker 构造 actor 时只传入 `actor` 子配置，而裁剪实现从该子配置读取了存放在
`model` 下的开关，因此实际仍执行 `joint`。修复后由 worker 从 model config 显式解析并传给
actor；运行指标必须报告 `policy/separate_gradient_clipping=1` 才承认候选真正生效。独立训练
分叉的随机 rollout 不要求动作/token 完全相同，基础设施门改为要求 `(task_id, rollout_id)`
工作负载完全一致；完整 trace 差异保留为诊断并强制进入固定 valid-seen 效果门。

该救援首次实跑在 15 条已提交结果中救回 5 条、其余 10 条仍为 600 秒
`expert_wall_timeout`，证明延长窗口有效，但等待全部 43 条没有实验价值。正式覆盖率仍只需
累计救回 8 条。当前支持从仍在增长的 rescue run 读取 checksum 校验通过的 committed-shard
快照，并写到一个独立派生目录；快照与源 formal run 均逐项校验，只有重算后的完整 3,553 条
manifest 通过原 99% coverage、1% horizon 和 planner identity 门才返回成功。该路径不修改
正在运行的救援目录，也不把未完成任务视为成功。

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

**Actor Learning-Rate Fork**:
从同一 portable checkpoint 创建的命名短程分支；保留 LoRA optimizer moments、scheduler 进度、任务游标和随机状态，只在恢复完成后显式覆盖 actor optimizer/scheduler 的目标学习率。固定 `1e-6 / 3e-6 / 1e-5` 分支必须使用相同训练工作负载和同一 140 条评测协议。
_Avoid_: in-place LR mutation、optimizer reset、config-only LR change

**Training Drift Guard**:
训练 update 完整提交后，根据多个训练指标的联合、连续越界条件锁存安全暂停；触发时必须先保存可恢复 checkpoint，并把阈值、连续次数、观测值和暂停原因写入产物。
_Avoid_: single-metric spike stop、mid-update termination、silent early exit
