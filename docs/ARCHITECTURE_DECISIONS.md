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

其中“同一 detached trajectory advantage 同时作为 fidelity target”的部分由 D011 取代；其余 group 构造与广播语义仍然有效。

## D005：按环境可执行性定义非法动作且默认不加步数惩罚

系统不沿用 SkillRL 以 XML 标签和响应语言判定语义合法性的 `is_action_valid`。解析器优先读取完整 `<action>`；多个完整标签只有内容相同才接受，内容冲突即判歧义。为公平支持未经 ALFWorld SFT 的 Qwen，缺少完整 XML 标签时允许从最后一个非空行识别显式 `[action]` 或 `[action>` 起始标记，并移除可选的 `[/action]`、`</action>` 或实测出现的 `</action>]` 结束标记；这种兼容动作仍记为格式不合规。若没有上述显式标记，只允许从最后一个非空行去除有限的格式前缀后，与当前 `admissible_commands` 做大小写/连续空白规范化后的整行精确匹配；不扫描 reasoning 子串、不纠正标点、不做模糊或最近动作投影。只有无法解析动作，或规范化命令不能唯一匹配当前可执行命令时，才计为非法动作，标签、`<think>` 和语言仅作格式统计。合法动作提交对应规范命令，非法动作统一提交必须消耗一步且不推进世界状态的 `__invalid_action__` 哨兵，不用 `look`、最近合法动作或可能被环境解释成别名的模型原文进行免费修正。默认轨迹奖励为 `won - 0.01 * invalid_action_count`，保留可配置硬步数上限，但不加入逐步惩罚、goal-condition reward 或 information bonus；正式评测成功率始终只依据未塑形的 `won`。

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

## D010：pilot 与 formal 统一使用固定 valid_seen 评测

取消从 ALFWorld train 派生的 355 条内部 monitor；所有训练档位都从完整 3,553 条 train 清单按同一规则取样。pilot 在 update 0 和 25、formal 在 update 0、每 25 个 update 及最终 update 445，使用同一个以 SHA-256 预注册身份的固定 140 条 `valid_seen` manifest、确定性解码和完整六类分母；身份不符时必须在加载模型前失败。这样缩短 pilot 的评测时间，并让 pilot 与正式曲线直接可比；代价是曲线、`best-valid` 和最终报告使用同一集合，因此必须标注为 validation-selected performance，且不得根据该曲线调整损失权重、学习率等超参数。旧 train-monitor pilot 只保留为工程稳定性证据，不与新曲线拼接。

跨拓扑或命名分叉恢复时，目标 run 必须继承源 checkpoint 所在 update 及之前的全部有效评测历史，并把源 run 的相对 checkpoint 路径改写为带来源的绝对路径；随后目标 run 的新评测与继承历史合并，再按同一预注册规则选择 `best-valid`。只在分叉目录内比较恢复后的评测会把较差的最后 checkpoint 错标为最佳，因此属于必须阻断或修复的结果选择错误，尽管它不改变模型权重、优化器或已计算成功率。

## D011：策略整形优势与任务成功 fidelity 目标分离

M0/M1 的 Policy Optimizer 继续使用由 `won - 0.01 * invalid_action_count` 在同任务组内标准化得到的 Shaped Policy Advantage，以保留合法动作密集反馈；M1 的 fidelity predictor 改用仅由二值 `won` 组内标准化得到的 Task-Success Fidelity Target。两者使用同一批轨迹但不能共用一个无语义区分的 `advantage` 接口。每个 update 同时记录混合结果组、全失败组、全成功组、Shaping-Only Group、零策略信号组以及按任务类型拆分的任务成功信号覆盖率。

75-update M0 审计发现 596 个具有非零策略优势的组中有 351 个组内 `won` 恒定，其梯度只来自非法动作数差异；由于组内标准化会消除正比例系数，单纯继续减小 `0.01` 不能削弱这些组中的相对整形信号。若 fidelity 继续拟合同一个整形优势，压缩器可能主要学习动作合法性而不是技能对任务成功的贡献。完全去除策略整形会丢失稀疏成功奖励下的可用反馈并改变已验证的 SkillRL 相近基线，因此不采用；分离两个目标保留策略学习信号，同时让 fidelity 的含义可解释且可审计。

## D012：首轮 7B 主实验改用原版 Qwen2.5-7B-Instruct 初始化

首轮 M0、`raw_skill_prompt` 与 M1 的共同初始化由 `Alfworld-7B-SFT/checkpoint-140` 改为注册指纹的原版 `Qwen2.5-7B-Instruct`。在固定 140 条 `valid_seen`、统一 no-skill prompt、greedy 解码和同一 VERL/vLLM 评测链上，原版 Qwen 在有限显式动作标记兼容后达到 macro success `0.21777`、overall success `33/140`、非法动作率 `0.10108`；SFT 模型为 macro `0.14713`、overall `25/140`、非法动作率 `0.55350`。原版 Qwen 结果也与旧独立框架的 `31/140` 接近，且所有兼容动作仍要求与当步 `admissible_commands` 精确匹配。因此正式主对比统一从原版 Qwen 独立开始，SFT 权重只保留为明确配置的附加对照；此前以 SFT 为起点的 M0 训练仅作为工程闭环证据，不与新主实验曲线拼接或用于选模。

## D013：raw-skill 保留 full 序列化，compact 作为显式消融

`raw_skill_prompt` 的正式模型可见技能块继续采用 `full`：保留检索所得候选的 ID、type、category 及全部原始语义字段。`compact` v1 保留为显式消融；它不改变检索算法、候选数、候选 ID、候选顺序或逐任务检索计划，只从模型可见文本移除存储 ID、重复 type 与 `why_it_happens`。

固定原版 Qwen、140 条 `valid_seen`、检索计划和其余设置的 update-0 A/B 中，`full` 为 38/140、macro `0.25145`，`compact` 为 35/140、macro `0.23886`；compact 虽将技能块缩短约 26%，但端到端只提速约 2%，且主次成功率均下降。因此节省不足以抵消观测到的效果风险，正式 raw control 恢复 `full`，`RAW_SKILL_PROMPT_FORMAT=compact` 仅用于消融。格式名称必须写入 run provenance、resolved config 与 portable checkpoint 兼容条件；compact 与 full checkpoint/曲线不能混用。

## D014：统一校准 rollout/recompute P99 门限为 0.30

首个 optimizer step 前的 rollout/recompute 联合门禁保留 mean、median、P95、P99、误差大于 `1`/`5` 的比例及 ratio mean，不删除尾部检查。相同原版 Qwen 起点和正式 update 形状下，no-skill 的 P99=`0.24966`，raw/full 的 P99=`0.28432`；raw/full 同时满足 mean=`0.03010`、median=`0.00267`、P95=`0.14057`、误差大于 `1` 的比例=`0.0212%`、误差大于 `5` 的比例=`0`、ratio mean=`0.99976`，最大误差也不在首尾 token。旧 P99≤`0.25` 因而过度贴合单次 no-skill 观测，并会把长但合法的 full prompt 数值尾差误判为序列错位。P99 上限统一校准为 `0.30`，其余门限不变；所有 control modes 与 M1 必须使用同一阈值，失败仍在 reference 与 optimizer 前停止并保存完整诊断。不得为某个方法设置专用阈值，也不得删除 P99 后只依赖平均误差。

## D015：统一采用 12,288 policy token budget

`no_skill`、`raw_skill_prompt` 与 `infoskill` 的 old/ref/actor 动态微批预算统一为每 GPU `12,288` tokens；vLLM rollout 调度预算继续保持 `16,384`，两者不得混为一个参数。正式形状的 raw/full 运行在 `16,384` 下曾观测到约 `2.51 GiB` 的 policy 阶段物理空闲显存；`12,288` 的 200ms 监控重跑把 policy/rollout 最差余量提高到 `18.64/12.46 GiB`，同时保持 64 条 rollout 与对齐统计完全一致，并完成 optimizer 与 portable checkpoint。观测到的 core/policy 耗时约增加 `13.4%/10.1%`，其中含 200ms 监控开销；在可接受的吞吐代价下优先保留跨模式、长 prompt 与后续 M1 的显存安全余量。随后 1,000ms 监控的两个连续 raw/full 正式形状 update 均完成：跨 update 最差 policy/rollout 余量为 `18.64/13.02 GiB`，core 为 `668.52/679.99s`，CPU 内存仅从 `45.86` 增至 `46.18 GiB`，无环境强制终止，两个可移植 checkpoint 的游标依次为 8 和 16。长时 pilot/formal 因此建议显式设置 `CUDA_MEMORY_POLL_INTERVAL_MS=1000`，但代码默认仍为 `0`，显式参数优先。历史 checkpoint 缺少该字段时仍按旧默认 `16,384` 解释，禁止在原地 resume 中静默改成 `12,288`。

## D016：grounding 使用有界短生命周期 worker

ALFWorld Strict Expert Replay 按固定任务顺序执行，但每 64 个游戏更换一个短生命周期 CPU 子进程；每个子进程使用 run 目录内独占的临时目录，并在退出后由父进程验证清理。候选技能在父进程一次性确定，任务随机种子由 master seed 与 task ID 稳定派生，因此输出不依赖 shard 边界。父进程按原始任务顺序合并全部结果，再统一生成 `grounding_samples.jsonl`、`quarantine.jsonl` 与 formal manifest。

采用该边界是因为单进程回放实测到 Fast Downward 为每次 TextWorld 环境初始化复制 `libdownward.so`，临时占用在进程退出前持续累积：第 722 个任务时数据盘被写满，同时 RSS 从约 411 MiB 增至约 3.4 GiB。单纯改用更大磁盘或 `/dev/shm` 只会推迟故障，并可能把磁盘泄漏改成主存耗尽。worker 分批不得改变 3,553 条 train 全集、150 步专家验证、30 步持久化窗口或 99% formal gate；生命周期报告缺失、任务顺序变化、worker 失败或临时目录未清理均视为基础设施失败。

## D017：ALFWorld 专家类型必须按实际包装器实例 fail closed

固定 ALFWorld 源码以 `AlfredExpert(expert_type)` 创建签名为 `AlfredExpert(env=None, expert_type="handcoded")` 的包装器，导致请求 planner 时字符串被绑定到 `env`、有效专家静默保持 handcoded。INFO-SKILL 不直接改写外部仓库，而在 ALFWorld 适配边界安装只识别 `handcoded|planner` 位置字符串的窄兼容保护；父进程与每个短生命周期 worker 都必须用源码中的真实类执行同形调用，验证 requested/effective type、保护器状态和位置参数修正，任一不符即在回放前失败。pilot 报告还把 handcoded 独有的 `environment_step/Exception/Timeout` 作为第二道身份门，并记录实际模块路径。旧误绑定产生的 planner pilot 与后续诊断全部作废，不得用作方法判断或训练数据。

长 horizon 诊断将“历史状态—动作重复”和“末尾周期卡死”分开：前者只表示轨迹曾回访，后者要求轨迹结尾存在长度不超过 10 的完全相同状态—动作周期连续重复至少三次。保留旧 `cycles_detected` 字段供读取器兼容，但 schema v2 中它与严格的 terminal cycle 同义，不再等同于任意位置第三次出现。

## D018：grounding 任务并行必须先通过逐步串并行一致性门

Grounding 的并发单位是相互独立的短生命周期 CPU worker，而不是单个 TextWorld 环境内部的线程。默认 `worker_processes=1` 保留历史串行行为；只有固定 train 任务、相同任务种子、相同 planner 身份、相同 150 步验证上限和 30 步持久化窗口的串行/并行差分运行，对完整序列化结果逐字段完全一致时，才允许在后续 pilot 或 formal 显式启用并发。比较范围包括每一步 canonical state、history、`admissible_commands`、候选技能 ID、专家动作、终局、隔离原因、异常与动作不匹配信息；只比较最终成功率不足以通过。

并行 worker 各自使用独占临时目录，父进程按预注册任务顺序合并乱序完成的 shard，并记录实际峰值并发。并发启动前为每个 worker 额外预留 3 GiB 临时磁盘，同时始终保留 4 GiB 硬下限；运行中跌破硬下限立即失败。该优化不得改变任务集合、专家、标签、gate 或样本顺序，性能提升也不是一致性门通过的必要条件。

固定串并行逐步一致性门通过后，双 worker 在六类各 50 条的正确 planner pilot 上达到 300/300，轨迹均不超过 11 步，身份门、覆盖率门、horizon 门与临时资源清理门全部通过。由此批准双 worker 用于 3,553 条正式 grounding。正式入口必须与 pilot 使用相同的 verified planner：bounded worker 的 `expert_type` 不再有默认值，调用方必须显式指定；formal manifest 使用 schema v2 保存 `expert_type`、完整 binding 报告及身份门结果，M1 加载器拒绝旧 schema、handcoded 或身份未经验证的产物。这一约束防止 pilot 正确但 formal 因调用遗漏静默退回 handcoded。

进一步提速只允许作为显式候选进入：`process_parallel` 用多个独立 bounded worker，
`native_batch` 则在一个 bounded worker 内用固定 TextWorld slot 并行回放多个 planner
任务。原生批处理必须保持 task/slot 映射、完整 canonical state、专家动作、终局和隔离
信息逐字段一致；已结束 slot 的 transport filler 不得进入训练样本。两条候选都按实际
环境 slot 数在启动前预留临时磁盘，并在 worker 存活期间每 0.5 秒执行 4 GiB 硬下限
监控；低于下限时终止 worker、保留失败报告，不允许等待磁盘写满。正式默认仍为
`individual`，只有 12 条 exact parity 与 60 条生命周期/性能门同时通过，才可在
3,553 条正式运行中显式选择更快后端。

## D019：grounding 分片必须可恢复，并对 planner 无进展设置墙钟熔断

2026-09-12 的首个 3,553 条 `native_batch_size=4` 正式运行在 3,008 条后停止推进：
第 48 个 64 条 shard 已创建输入但尚未写出任何结果，其中一个 TextWorld/Fast Downward
子进程持续占满单核超过一小时。磁盘仍有约 24 GiB、inode 充足、系统 I/O wait 为零且
绝大多数 CPU 空闲，因此根因不是资源总量不足，而是 planner 的同步
`policy_commands`/搜索调用缺少墙钟上限；逻辑 `max_replay_steps=150` 只能限制已经返回的
环境步数，无法中断一次不返回的 reset/step。父进程原先又只在全部 3,553 条结束后写正式
结果，导致已完成的 3,008 条只存在内存，进程终止后无法可靠恢复。

此后 bounded grounding 在 run 目录保存不可变的 `grounding-resume.json`，并在每个 shard
完成后先原子写 `grounding-shards/shard-NNNN/results.jsonl`，再提交带工作项、结果和计划
SHA-256 的 `complete.json`。恢复时逐项验证任务顺序、候选技能、种子、源码校验和、配置、
后端、批大小、并发、horizon 与超时设置；任一字段或校验和不符即 fail closed，绝不混合
两套标签。没有
这些文件的历史运行不可恢复。

worker 默认连续 300 秒没有完成任何任务即视为无进展。父进程以独立 POSIX process group
启动 worker，并在超时时终止整棵 TextWorld 子进程树；已 flush 且已经宣布完成的前缀结果
予以保留，未完成部分改用 `individual` 逐任务隔离。单任务再次超时不会伪造专家动作，而以
`expert_wall_timeout` 进入 quarantine，并保留异常类型和阶段。formal 的 99% 覆盖率门保持
不变，所以超时过多会让数据生成明确失败，而不会静默降低训练数据质量。

故障现场还表明，112 核主机在单个 `native_batch_size=4` worker 卡住时绝大多数 CPU 空闲；
固定 12 条测试中，4 个 independent worker 与单 worker/native-batch-4 的耗时分别约为
168 秒和 165 秒，说明主要吞吐近似取决于同时活跃的 planner slot，而不是某一种包装方式。
因此允许把两种已逐步验证的并行层组合为 `native_batch_parallel` 候选，但组合后仍必须重新
通过 12 条 full-row exact parity 和 60 条生命周期/性能门，不能由两项单独测试直接推断。

当前正式候选固定为 2 个 bounded worker、每个 worker 内 3 个 native batch slot、每 shard
32 条。总计 6 个环境 slot，使现有磁盘保护要求为 4 GiB 硬下限加 6×3 GiB 并发预留，即
22 GiB；操作命令在至少 23 GiB 空闲时才启动。若原生 batch 触发无进展超时，未完成任务在
该 worker 原有的最多 3 个 slot 内使用互相隔离的临时目录并行 individual 重试，结果按原始
任务顺序重组后再原子提交。该优化只缩短故障恢复墙钟时间，不改变专家动作、seed、状态、
样本字段、覆盖率门或 quarantine 语义。

## D020：正式 grounding 的超时长尾采用校验后定向救援

2026-09-12 的 `native_batch_parallel` 正式运行完整提交 112/112 个 shard 和 3,553/3,553
条结果，临时目录全部清理，最低空闲磁盘约 21.90 GiB；3,510 条成功，43 条
`pick_two_obj_and_place` 在 native batch 超时后又于 300 秒 individual 隔离重试中超时。
覆盖率为 98.7898%，其余五类全部成功、没有轨迹超过 30 步，formal gate 的唯一失败是
`success_coverage_below_threshold`。99% 门要求至少 3,518 条成功，因此只差 8 条，重跑全部
3,553 条既不增加对已成功标签的信心，也会浪费约五小时。

采用 `grounding-timeout-rescue` 派生式救援：先重建全部任务的候选技能和 seed，并逐项验证
源 manifest、train manifest、技能库、完整 work-item SHA、resume plan，以及每个已提交 shard
的 marker、结果 SHA 和全局任务顺序；任一不符即 fail closed。随后只选择源结果中的
`expert_wall_timeout`，以单任务 individual worker、默认 4 并发和 600 秒无进展上限重放。
成功重放才替换对应源结果，失败重放不改变源 quarantine；合并后重新计算 3,553 条 formal
manifest，并同时保存源 manifest SHA、旧/新源码 SHA、救援结果和生命周期。99% 覆盖率、
1% horizon 门和 planner 身份门均不放宽。救援 run 自身按一任务一 shard 原子提交，可以在
SSH 或父进程中断后校验恢复。

单条 individual worker 首次超时后不得再自动重试同一条，否则 600 秒救援的最坏墙钟会被
静默翻倍。43 条、4 worker 的理论最坏时间约为 `ceil(43/4)×600s=110min`，并发磁盘启动门
为 4 GiB 硬下限加 4×3 GiB 预留，即 16 GiB。只有派生 manifest 的
`formal_gate_passed=true` 时，救援目录才可作为 M1 grounding 数据输入。

救援不要求为得到 formal 数据而等待全部 timeout 重试结束。已完成任务继续按一任务一 shard
原子提交；独立 finalization 入口可在任意时刻读取 committed marker 快照，逐 shard 复核 plan、
work-item 和结果 SHA，并只合并其中成功的重放。一旦累计救回 8 条使完整 3,553 条派生 manifest
通过原门禁，即可生成新的只读正式目录；尚未提交或再次超时的任务继续沿用源 quarantine。
finalization 不在活跃 rescue run 内写派生产物，避免与并发提交互相覆盖。

## D021：M1 正式评测保持 batch 8，batch 12 未通过 exact parity

固定 12 条压力集、同一 update-2 portable checkpoint 和三卡拓扑的首轮门禁显示，batch 12
相对 batch 8 将 rollout 从 `528.69s` 降至 `293.48s`，且最低物理显存仍余约
`29.91 GiB`；但 12/12 条轨迹均产生 token/语义分叉。首次分叉前，两边的 canonical state、
policy user message、候选技能、conditioning replay 和 soft-prefix 统计均一致；可归因的变化
只剩每 rank 本地 batch 几何及其 BF16/vLLM kernel 数值路径。greedy 解码并不保证跨 batch
几何逐 bit 不变，近似并列 logits 会把细小误差放大为不同动作和环境状态。

因此正式 M1 `valid_seen` 继续使用 batch 8。不得因成功率相近、平均概率误差较小或吞吐收益
而放宽完整轨迹/token exact gate；batch 12 只有在未来运行栈变化后重新通过同一门禁才能启用。
当 token 或 logprob 长度漂移时，完整 logprob 比较定义为不可用，报告必须给出
`logprob_comparison_valid=false` 与 `logprobs_close=false`，不能以零个可比较 token 的默认
零误差宣称一致。

## D022：首轮 M1 正式训练用 batch 12 生成同 run 监控曲线

首轮 445-update M1 运行经实验方明确选择，在 update 0、25、50……及 445 结束点的周期
`valid_seen` 评测中显式使用 `eval_batch_size=12`。这些点使用相同 batch 几何，适合判断该次
训练内部的变化趋势，并在每次完整评测提交后原子刷新 `valid_seen_learning_curve.svg`；图中
同时显示 Macro success 与 Overall success。`resolved_config.json`、`provenance.json`、每个
评测 summary、metrics 和 `checkpoint_selection.json` 都必须把它标记为
`nonregistered_monitoring_curve`，不得冒充 D021 的 batch-8 注册结果。

该选择不修改正式 YAML 默认值，也不推翻 batch 8/12 exact parity 失败的事实。需要与 M0、
raw control 或其他 run 做最终定量比较的关键 M1 checkpoint，仍须另行用固定 140 条、batch 8
重评。训练进程接收 SIGINT 或 SIGTERM 时只登记暂停请求：当前 optimizer update 完成后写出
该 update 的 portable checkpoint，跳过额外的非周期评测并安全关闭运行时；恢复必须使用同一
run 的该 checkpoint 和完全相同配置。这样允许提前停止 445-update 运行而不产生半个 optimizer
step，也不把硬杀进程当作正常断点。

## D023：M1 吞吐优化必须由同 checkpoint 单 update 分叉门批准

首轮 M1 formal 的实测中位耗时约为每 update `920s`：rollout 约 `407s`，policy update
约 `509s`；其中 vLLM generation 约 `347s`，old/reference/actor 分别约
`62/65/444s`。环境 reset/step、driver conditioning 和 auxiliary update 合计只占小部分，
所以继续增加 CPU worker 或改 auxiliary 参数不是主要提速方向。

首批候选保持默认关闭，只允许在命名分叉恢复中改变：一是 M1 old-logprob 前向不再计算 PPO
没有消费的 entropy tensor；二是把 vLLM rollout 的 `max_num_batched_tokens` 从注册默认
`16384` 提高为显式候选值。正式 policy 动态微批预算仍固定为 D015 的 `12288`，不借提速名义
降低显存安全余量。曾考虑把 reference logprob 在整个 update 开头一次性预计算，但一个 update
内包含多个 optimizer minibatch，projector 会在其间更新；缓存会让后续 minibatch 不再使用
“当前 projector + detached prefix”的 D009 KL 定义，因此拒绝实现。

采用候选前，必须从同一个 portable checkpoint 分叉 control 和 candidate，各只完成一个相同
global update 并自动提交 checkpoint 后暂停。门禁要求：除两项候选外 resolved config 完全相同；
任务、轨迹、token 与 rollout logprob 一致；LoRA、M1 modules、两个 optimizer、两个 scheduler、
RNG 和 trainer state 在严格数值容差内一致；policy/rollout 物理显存均至少余 `8 GiB`；core
至少提速 `1.05x`。任一项失败就从 control checkpoint 继续，不把 candidate 接入 formal。
`SEGMENT_END_UPDATE` 只是单次调用边界，不改变注册的 445-update 目标、warmup 或 checkpoint
语义；原地 resume 仍禁止静默改变候选设置。

轨迹 exact parity 是“等价工程优化”的分类门，而不是效果优劣的替代指标。若 scheduler、batch
geometry 等候选改变 rollout，比较器必须标记为 behavior-changing，并要求在相同固定评测协议下
另做以 macro success 为首要指标的效果门；不能因轨迹改变直接断言效果下降，也不能把单个训练
batch 的即时成功率当成泛化证据。轨迹长度不同的两次 update，其原始 wall-time 不构成同工作量
吞吐比较，必须明确标记 performance comparison 无效，直到效果门提供端到端决策依据。

## D024：深层吞吐候选分成等价实现与算法变更两条门

M1 hybrid-prefix rollout 默认仍使用 eager vLLM。CUDA Graph 候选只在显式设置
`HYBRID_PREFIX_CUDA_GRAPH=1` 时启用，并要求 patched runtime 的第二版能力标记。捕获和执行都
通过 vLLM 自己的 persistent `inputs_embeds` 缓冲区传递 soft prefix，避免把每轮新分配的
embedding 地址固化进 graph。该候选不改变训练目标，因此只有完整 rollout trace、logprob 和
portable checkpoint 均等价，且物理显存余量与吞吐门同时通过时，才可视为可替换实现。

目标 A800 上的执行层差分证明，首版候选相对 eager 的概率漂移来自 vLLM V1 在 piecewise 编译
路径中强制使用 `custom_ops=["none"]`，而不是 Graph capture：相同 compile 路径加减 Graph 的
logprob 完全一致；`use_inductor=False` 下只恢复 `custom_ops=["all"]` 可消除全部观测误差，加回
Graph 后仍保持零误差。故此开关的候选语义修订为固定使用 eager adaptor、注册的自定义 CUDA
kernels 和 Graph capture。运行产物必须同时记录 `hybrid_prefix_cuda_graph_custom_kernels=true`
与 `hybrid_prefix_cuda_graph_use_inductor=false`。历史 Graph checkpoint 缺少这些字段时按旧语义
解释，禁止原地静默采用新语义；命名分叉仍需经过固定 140 条效果和资源门。

Policy 侧默认继续分别执行 reference、detached-prefix actor KL、trainable-prefix PPO 三次前向。
`FUSE_KL_PPO_FORWARD=1` 复用 PPO actor logprob 计算 KL，从而删除 detached-prefix actor KL
前向；代价是 KL 梯度也进入 projector。这个差异是显式算法假设，不是数值实现细节。比较器即使
观察到当前 update 的 rollout 完全一致，也必须输出 `algorithm_change_requested=true` 和
`efficacy_gate_required=true`，不得以 checkpoint 不一致判定“效果变差”，也不得仅凭吞吐批准。
候选采用条件是相同训练起点、相同预算和固定 valid-seen 协议下，按 Macro success 首排、Overall
success 次排不劣，并在多 update 运行中无崩溃、显存越界或学习曲线退化。两个开关都保持默认
关闭，只允许命名恢复分叉改变，原 formal run 的注册语义不被静默修改。

## D025：修正后的 CUDA Graph 由重复效果门批准用于 M1 命名分叉

2026-09-14 使用同一 step-50 portable checkpoint、同一固定 140 条 `valid_seen` manifest、
同一 batch 12 监控协议比较独立 eager 与修正后的 CUDA Graph。比较器要求两边完整、加载同一
checkpoint 路径和 step，并只允许 `hybrid_prefix_cuda_graph`、custom-kernel policy 与
Inductor policy 三个 runtime 字段不同。独立 eager 为 37/140，Macro/Overall 分别为
0.261341/0.264286；Graph 的两次独立运行分别为 30/140 和 44/140，后者为
0.294658/0.314286 并通过效果门。两次 Graph 合计 74/280，平均 Overall 0.264286，与 eager
一致；平均 Macro 0.253524，比单次 eager 低 0.007817。首轮低值因此不能解释为稳定的 Graph
负效应，而应视为独立 runtime 的微小浮点差异经 greedy action 和多步环境交互放大的运行间波动。

性能收益在两次 Graph 中稳定：完整评测约 26--27 分钟，而 eager 约 62.6 分钟；第二次 Graph
相对 eager 的 rollout、generation 和总耗时分别约 2.70x、3.20x 和 2.38x，最低物理显存余量
约 29.36 GiB。批准的执行组合固定为 Graph capture、`use_inductor=false`、
`custom_ops=["all"]` 和 batch 12。全局默认继续为 eager；只有显式命名分叉可采用该候选，确保
历史 run 仍可复现和回退。M1 从 step 50 恢复时只启用这一项已经通过成功率门的优化，不同时启用
尚未完成固定验证集效果门的 fused KL/PPO、entropy skip 或 scheduler token-budget 候选。

日常学习曲线允许每个评测点运行一次 Graph；关键 checkpoint 与最终报告至少运行两次 Graph 并
汇总 Macro/Overall。单次结果仍完整保留，但小于独立 runtime 波动尺度的差异不得被解释为确定性
模型提升或退化。eager 保留为审计基线，不再承担每个周期点的常规监控。

## D026：M1 CUDA Graph 周期评测采用 batch 64，并在 step 50 建立协议桥

2026-09-14 在 D025 的修正 CUDA Graph 路径上，使用同一 step-50 checkpoint、同一固定 140 条
`valid_seen`、同一三卡拓扑和相同 200 ms 显存采样间隔，对 batch 12 与 batch 64 各完成两次
独立完整评测。batch 12 两轮平均为 37/140、Macro 0.253524、Overall 0.264286、总耗时约
1606.8 秒；batch 64 两轮平均为 38/140、Macro 0.258275、Overall 0.271429、总耗时约
837.1 秒。batch 64 相对两轮 batch 12 平均没有效果退化，反而高 1 个成功任务，Macro/Overall
分别高约 0.48/0.71 个百分点，并取得约 1.92x 端到端提速。其两轮最低物理显存余量均约
29.36 GiB，forced termination 为 0。

单轮严格比较器把第二轮 batch 64（40/140）与偏高的第二轮 batch 12（44/140）比较时返回
`passed=false`；该结果必须保留，但它不是预先声明的重复实验汇总规则。按“两轮平均 Macro
下降不超过 2 个百分点且 Overall 平均少不超过 2 个成功任务”的采用条件，batch 64 通过。

因此从 step 50 之后的 M1 命名分叉开始，周期评测显式采用修正 CUDA Graph 与
`eval_batch_size=64`。step 50 已同时具有 batch 12 和 batch 64 的重复结果，作为评测协议切换
桥点；step 0/25/50 的 batch-12 曲线与后续 batch-64 曲线不得无标记地视为完全同协议序列。
训练算法、训练 batch、rollout batch 和 checkpoint 内容不因该评测 batch 变更而改变。最终关键
checkpoint 仍至少重复评测两次，并按 Macro success 首排、Overall success 次排汇总。
`evaluation_manifest.execution_mode` 是同一评测性能协议的派生身份字段。旧 step-50 checkpoint
创建时尚无该字段，因此只有显式命名且开启 performance-candidate 兼容的 fork 可以增加或更改它；
无名字的原地 resume 仍要求该字段完全一致。这样允许已批准的 eager → CUDA Graph 协议桥，又不
放宽模型、数据、训练算法或原地恢复门。

## D027：长时正式分叉采用最近 5 个、当前最佳和最终 checkpoint 的有界保留

从 step 50 启动的新 M1 formal 命名分叉显式采用 `CHECKPOINT_KEEP_RECENT=5` 与
`CHECKPOINT_KEEP_BEST_VALID=1`。历史默认仍是最近 2 个加每 25 update 永久 checkpoint，保证
旧 run 原地恢复语义不变；新策略只作用于新 run 自己的 checkpoint 目录，不修改作为恢复起点的
源 step 50。启用后，普通周期 checkpoint 的保留集合是最近 5 个与当前 `best-valid` 的并集，
最终 checkpoint 另作永久项；重合路径不复制。当前最佳沿用固定 `valid_seen` 的 Macro success
首排、Overall success 次排、非法动作率更低再次排、较早 update 最后优先的规则。

自动清理只在新 checkpoint 完成原子提交后进行，只接受当前 checkpoint 根目录的
`step-NNNNNN` 已提交直接子目录；先把 delete-intent 追加并 `fsync` 到
`retention-audit.jsonl`，删除完成后再追加并 `fsync` deleted 事件。外部或源 run 路径
不能登记为本地受保护 checkpoint，更不能被轮换器删除。命名分叉可以显式更改保留策略；原地
resume 若与原 resolved config 不同则 fail-fast。按当前约 551 MiB/checkpoint 估算，完成态最坏
约 4.3 GiB（最近 5 + 独立最佳 + 永久 update 0 + 最终），另需预留一个原子写入中的临时
checkpoint；这些集合若重合则实际占用更少。

## D028：训练错题采用旁路全量索引，曲线逐点披露结果与协议

每个训练 update 在完整 `.jsonl.zst` 轨迹之外生成一个紧凑、原子、可覆盖的任务结果索引。
索引包含全部任务组而非只保存失败题，以免将来无法区分遗漏与成功；但只保存任务元数据和每条
rollout 的小型结果摘要，并以相对路径引用现有完整 trace，不重复保存 prompt、token、状态或
环境 observation。正确性只由终局 `won` 定义：正式 G=8 下 0/8 为 `hard_failed`、1--7/8 为
`partial`、8/8 为 `mastered`；数量异常单列 `incomplete`。该数据不在当前 formal 中改变任务
顺序、采样概率、reward 或 optimizer。写入采用临时文件 `flush/fsync` 后替换并同步目录；从较早
checkpoint 原地恢复时，把更高 update 的旧索引及对应压缩 trace 一并移入
`stale-after-resume-*` 目录而不删除，并改写归档索引的 trace 引用，避免陈旧错题污染汇总或
在重跑同一 update 后错误指向新轨迹。未来若做错题重训必须另开命名实验分叉。

`valid_seen_learning_curve.svg` 在每个 checkpoint 点旁同时显示 Macro 与 Overall 百分比，按点数
动态扩宽并在两条文字靠近时自动错位。checkpoint selection 的每条评测记录携带自己的 batch 和
执行模式；命名分叉继承旧点时保留源协议，而不是套用目标 run 的全局设置。连续点协议变化时图中
明确画出分界，例如 step 50 之后从 batch 12/eager 切换到 batch 64/CUDA Graph。图仍由原生 SVG
生成，不新增 matplotlib 等运行依赖。

训练步数趋势同样以旁路指标实现：`metrics.jsonl` 是唯一事实来源，每个 train update 记录全部、
成功、失败 rollout 的数量与平均环境步数，以及 horizon exhaustion rate；随后原子重建
`training_rollout_steps_curve.svg`。命名分叉依次读取源 run 和当前 run 的 metrics，重复 step 以
当前 run 最后一个完整 train 记录为准，并忽略恢复点之后的陈旧数据。历史记录缺少分组字段时只画
存在的总体均值，不把零值伪装成“没有成功轨迹的平均步数”。该监控不参与采样、reward、loss、
checkpoint selection；由于每个 update 的 train 任务组成不同，它不能替代固定 valid_seen 曲线。
曲线作为旁路监控必须故障隔离：解析或写入异常只记录带堆栈 warning，不能越过训练回调阻止
checkpoint 提交。
