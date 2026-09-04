# INFO-SKILL Architecture Decisions

本文按编号集中记录 INFO-SKILL 已确认、难以逆转且会跨模块影响实现或论文实验定义的架构决策。具体实验参数见 [`EXPERIMENT_SPEC.md`](EXPERIMENT_SPEC.md)，统一术语见 [`../CONTEXT.md`](../CONTEXT.md)。后续若替代某项决定，应保留原条目并标记 `Superseded by Dxxx`，不要静默改写历史。

## D001：研究层与训练运行时分离

INFO-SKILL 保持为独立项目，自有方法模块、训练编排、环境适配、配置、测试和入口；首版固定由 SkillRL commit `8e66726ed866a4e0a7f053586a41022798192e6c` 中包名为 `verl` 的代码提供分布式 GRPO、Ray/FSDP、rollout 与 checkpoint runtime，并且只能通过 `infoskill/integrations/verl/` 使用。不复制或整体修改 SkillRL，也不依赖其技能生成和动态技能库；若连续 soft-prefix 注入无法通过稳定扩展点实现，只允许维护从固定 vLLM 0.8.4 源码与 checksum 可重复构建的最小 patch wheel，禁止直接修改 `site-packages`。这样既复用与参考实验最接近的基础设施，又把来源耦合限制在可替换适配边界内。

## D002：混合 soft-prefix rollout 与训练重算

M0 使用 VERL 原生 vLLM token rollout；M1 在每个环境步骤生成状态条件连续 soft prefix，通过 vLLM prompt-embedding 输入进行高速采样。每步保存 exact latent、旧 soft prefix、动作 token 和行为策略 logprob，GRPO 优化时由 Transformers/FSDP 重放同一 latent，用当前 projector 重算 prefix 与新 logprob；旧 prefix 仅用于一致性审计和复现。纯 Transformers rollout 保留为较慢的正确性基准与故障排查后端，而不是 7B 主 rollout 引擎。由于现有 SkillRL worker 没有连续 prompt embedding 接口，正式 M1 前必须先实现 Transformers 基准，再以“完整 token IDs + 5 个显式占位位置 + 短 soft-prefix vectors + prefix mask”的最小 VERL/vLLM 扩展避免传输完整 prompt embedding。兼容性门分为两个不可互相替代的部分：先在同一 vLLM 实例中把普通 token-ID 输入与等价 token embedding hybrid 输入分批比较，隔离 transport；再用多个固定 prompt 与 prefix seed 比较 Transformers 重算和 vLLM hybrid rollout，统计 token 一致率及 logprob median/P95/max，避免以单个随机 prefix 混淆补丁错误和 BF16 kernel 漂移。

## D003：默认分离策略与压缩器梯度

M1 默认在 replayed latent 处截断 GRPO 动作梯度：GRPO 更新 Qwen LoRA 与 soft-prefix projector，随机 encoder、fidelity predictor、state-conditioned prior 和 Executable Grounding Head 只由 fidelity、rate/CIB 与 grounding 辅助目标更新。默认 GRPO ratio 只包含动作 token 概率，不包含 Gaussian latent density；另保留非默认 `joint_latent_ratio` 实验模式，用旧 encoder 统计量计算 latent likelihood ratio。选择默认分离模式是为了获得更稳定、可解释的首阶段归因，同时不永久排除联合随机策略解释。

## D004：用同任务的独立完整轨迹构造 GRPO group

每个采样 ALFWorld 任务创建 G 个底层任务与初始条件相同、但状态互相独立的环境实例，并从每个实例采样一条完整轨迹。group-relative advantage 只在这 G 条轨迹间归一化，并广播到对应轨迹的所有有效动作 token；detached trajectory advantage 同时作为该轨迹各访问状态的 fidelity target，提前终止通过 mask 表示而不伪造环境步骤。step-relative/GiGPO 只作为后续显式实验，不能静默替换 M0/M1 的 episodic GRPO 定义。

## D005：按环境可执行性定义非法动作且默认不加步数惩罚

系统不沿用 SkillRL 以 XML 标签和响应语言判定语义合法性的 `is_action_valid`。解析器优先读取完整 `<action>`；多个完整标签只有内容相同才接受，内容冲突即判歧义。缺少完整标签时只允许从最后一个非空行去除有限的格式前缀后，与当前 `admissible_commands` 做大小写/连续空白规范化后的整行精确匹配；不扫描 reasoning 子串、不纠正标点、不做模糊或最近动作投影。只有无法解析动作，或规范化命令不能唯一匹配当前可执行命令时，才计为非法动作，标签、`<think>` 和语言仅作格式统计。合法动作提交对应规范命令，非法动作统一提交必须消耗一步且不推进世界状态的 `__invalid_action__` 哨兵，不用 `look`、最近合法动作或可能被环境解释成别名的模型原文进行免费修正。默认轨迹奖励为 `won - 0.01 * invalid_action_count`，保留可配置硬步数上限，但不加入逐步惩罚、goal-condition reward 或 information bonus；正式评测成功率始终只依据未塑形的 `won`。

## D006：首阶段使用仅来自 train 的固定技能库

首轮正式 M1 直接采用项目负责人确认由 223 条 ALFWorld `train` 轨迹生成的 SkillRL 技能库，禁止 `valid_seen`、`valid_unseen` 或测试轨迹参与技能生成。技能库在本阶段不可变，并携带 provenance manifest，记录原文件校验值、源代码版本、轨迹数量与划分、生成方法和版本标识；每次训练与评测均记录该标识。替换或重新生成技能库必须产生新版本和 manifest，以便在保持 SkillRL 可比性的同时审计数据泄漏。

## D007：INFO-SKILL 拥有顶层训练编排

INFO-SKILL 自己实现顶层 `InfoSkillTrainer`，统一拥有轨迹采集、奖励与 advantage、policy/auxiliary 两类更新、评测、日志和断点提交状态，不继承或覆盖 SkillRL 高度耦合的 `RayPPOTrainer.fit()`。固定 VERL runtime 仍提供 Ray worker group、FSDP actor/reference、vLLM rollout、权重同步和分布式 checkpoint primitives，但只能通过窄 `VERLRuntime` Interface 调用，`DataProto` 转换完全留在 Adapter 内。相比直接复用现成循环，这需要额外训练编排代码，却能防止动态技能、旧 reward manager 和 `ppo_*` 语义渗入论文方法，并把未来 runtime 升级集中在一个 seam。

在该编排下，Qwen 冻结基座与 LoRA actor 使用 FULL_SHARD/ZeRO-3 FSDP，以保留与共卡 vLLM、长序列激活和语义编码器之间的显存余量，并直接复用 VERL 已有的 FSDP-vLLM 权重同步与分布式 checkpoint 实现；不为首版 Qwen 主体新增 DDP 路径。每卡复制的 INFO-SKILL 小模块则使用 DDP：projector 归 Policy Optimizer，compressor/prior/fidelity/grounding 归 Compressor Optimizer，冻结 embedding encoder 无 optimizer。所有 rank 按固定次序参与 collectives，损失通过全局 eligible numerator/denominator 归一化；FSDP actor 分片保存，rank 0 保存可移植 LoRA 和复制式小模块状态并在恢复时广播。这样把复杂的大模型分片封装在 Runtime Adapter 内，同时避免用 FSDP 管理体积极小、需要独立优化器语义的 INFO-SKILL 模块。

首版进一步固定使用 VERL `strategy=fsdp` 所对应的 FSDP1 FULL_SHARD，并要求 M0/M1 及三个 control modes 使用相同实现。固定 SkillRL commit 虽然已有 `fsdp2` 分支，但其 LoRA 提取与 vLLM 同步仍依赖 FSDP1 内部结构；在方法验证阶段切换会扩大 patch 面并降低归因清晰度。FSDP2 仅作为未来带新 Runtime ID 的显式升级，必须重新通过 rollout、权重同步、checkpoint 和跨 world-size 兼容性门，不能通过单个配置项静默进入正式实验。

## D008：以可移植状态而非原生 FSDP 分片作为恢复依据

完整恢复 checkpoint 由 INFO-SKILL Persistence 定义为 rank-0 权威可移植状态：保存完整 LoRA、可在恢复时重新分片的 LoRA AdamW full optimizer state、DDP projector/auxiliary 权重与优化器、scheduler、训练游标、语义随机流和完整 manifests，但不复制冻结的 Qwen 或 embedding 权重。恢复时从经 checksum 校验的原始基座重建模型，再按目标 2/4 卡拓扑创建 FSDP 并 scatter optimizer state；VERL 当前按 `world_size/rank` 命名的原生分片只可作为默认关闭的同拓扑缓存。该设计增加了 checkpoint 时在 CPU rank 0 汇聚少量可训练状态的成本，却避免约 14–16GB 冻结基座重复落盘，并使 4→2、2→4 成为可验证的真实续训而不是权重 warm-start。任何 full optimizer state 映射失败都必须让兼容门失败，不允许静默重置动量或伪装成 resume。

## D009：reference policy 复用 actor 的冻结 FSDP 基座

reference policy 不再按 VERL 默认方式创建并 CPU-offload 第二套 Qwen，而是在 actor 的同一 FSDP Module 上临时关闭 LoRA、以无梯度模式顺序计算。由于 Qwen 基座永久冻结，关闭 LoRA 后该模型在定义上就是固定 reference；M1 两分支还必须接收同一个 detach 后的当前 soft prefix。该选择消除第二份 7B 权重和反复 CPU/GPU 搬运，并让 checkpoint 无需持久化 reference，但要求严格的 adapter context 生命周期和全 rank 一致调用顺序。正式训练前必须证明共享实现与独立原始基座的逐 token reference logprob 在固化容差内一致，且异常退出也不会泄漏 adapter/train-mode 状态；若失败只能阻断并重新决策，不能静默退回独立 reference。

由于 LoRA 处在 FSDP seam 内而 projector 处在 DDP seam 内，二者不放入同一个物理 optimizer state。INFO-SKILL 保留 Policy/Compressor 两个逻辑优化域，但以 `LoRAOptimizer`、`ProjectorOptimizer` 和 `AuxOptimizer` 三个 AdamW 状态实现；`PolicyUpdateCoordinator` 把前两者封装成一个原子策略更新，共享 update 与 scheduler 进度，合并计算不重复计数的跨 FSDP/DDP 全局梯度范数，并保证一起 step 或一起跳过。该内部拆分不改变两组 policy 参数各自的 Adam 数学更新，却显著简化 checkpoint、跨 world-size 重分片与故障恢复，且不会把分布式实现细节暴露给 `InfoSkillTrainer` Interface。
