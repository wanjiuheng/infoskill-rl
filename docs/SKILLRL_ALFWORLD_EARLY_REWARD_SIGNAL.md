# SkillRL 在 ALFWorld 训练初期的奖励与学习信号

## 结论摘要

必须区分两个不同对象：

1. **SkillRL 主方法不是从 Qwen2.5-7B-Instruct 直接做纯 GRPO。**论文算法明确先用教师模型生成 7,500 条 skill-augmented reasoning traces，对基座做 3 个 epoch 的 SFT，再把该 SFT 模型同时作为 RL 初始策略与 KL reference。换言之，主方法用 imitation cold start 主动打破了“难题没有成功样本，因此 GRPO 没有正优势”的死循环。
2. **论文表中的 `GRPO 77.6 / PickTwo 64.7` 是从 Feng et al.（GiGPO）复现/引用的外部基线，不是 SkillRL 主方法的早期训练日志。**它从 Qwen2.5-7B-Instruct 起步，依靠模型已有的非零成功概率、每题 8 条随机 rollout、50 步 horizon、全参数更新及跨任务迁移，逐渐放大稀有成功；但公开论文与代码没有给出 PickTwo 在第几个 update 首次成功、早期 mixed-reward group 比例或逐 update 曲线。因此不能从最终的 64.7% 反推出它早期一直有充足的 PickTwo 正奖励。
3. **SkillRL 的 ALFWorld 文本环境没有 goal-progress dense reward。**代码中 TextWorld reward 只有终局成功时 `10 * won`，失败通常为 0；额外只有每个非法动作 `-0.1`。后者能教模型减少格式/非法动作，但不能直接教会任务规划。
4. **GRPO 的任务成功信号仍然稀疏。**同一任务的 8 条轨迹按总 reward 做组内标准化；如果 8 条都成功或都失败且非法动作数也相同，advantage 为 0。SkillRL 主方法之所以不完全受制于这一点，主要靠 SFT 先提高成功概率，而不是靠隐藏的 dense reward。
5. **动态 skill evolution 不是 reward shaping。**它每 5 step 在验证集上查找成功率低于 0.4 的类别，分析失败轨迹并增加/改进 skill，随后通过 prompt 改变未来 rollout；它没有给失败轨迹增加中间奖励。论文消融中去掉 cold-start SFT，ALFWorld 从 89.9 降到 65.2；去掉动态 evolution 降到 84.4。冷启动贡献远大于动态更新。

## 论文主方法：SFT 后再进行 SkillRL

论文 Algorithm 1 的顺序是：base policy rollout → 蒸馏初始 skill bank → 教师生成 SFT 数据 → `SFT(base, D_SFT)` → 将 SFT 模型设为 actor 和 KL reference → 每题采样 `G` 条完整轨迹做 GRPO → 定期依据验证失败更新 skill bank。

附录给出的 ALFWorld 配置是：

- 基座：Qwen2.5-7B-Instruct；
- 教师：OpenAI o3；
- SFT：7,500 examples，learning rate `1e-4`，batch size 16，3 epochs；
- RL：learning rate `1e-6`，batch size 64，KL coefficient 0.01，invalid-action penalty 0.1，150 epochs；
- 每题 rollout group size：代码为 8；
- skill retrieval：top-K 6；
- validation interval：5 steps；skill update threshold：0.4。

官方 README 也要求 ALFWorld SkillRL 启动脚本的 `MODEL_PATH` 指向 `YOUR_SFT_CKPT`，而不是原始 instruct 模型。

当前发布的 SFT 数据生成代码只处理 `type == "all_success"` 的成功轨迹，保留成功动作序列，并让 o3 为这些动作生成带 `<think>` / `<action>` 的推理。它证明公开实现的 SFT 是“成功动作轨迹 + 教师推理/skill”的 imitation 数据；它**没有证明论文使用了 ALFWorld planner expert**，因此不应把两者等同。

## 代码中的实际 reward 与 GRPO advantage

ALFWorld TextWorld 路径调用：

```python
reward = 10.0 * float(info['won'])
```

只有 multi-modal 路径才额外加 `goal_condition_success_rate`；官方 ALFWorld RL 脚本使用的是 `AlfredTWEnv` 文本环境。因此文本训练不是 dense goal-progress reward。

trainer 随后在 episode reward 上减去：

```python
0.1 * invalid_action_count
```

GRPO outcome advantage 先将每条 trajectory 的 token rewards 求和，再按相同 UID（同一题的 8 条 rollout）计算均值和标准差。因此：

- 组内既有成功又有失败：产生最直接、最强的目标学习信号；
- 全部失败但非法动作数不同：仍可能产生“减少非法动作”的相对信号；
- 全部失败、reward 完全相同：该组 advantage 为 0，无法从这次更新学到如何完成任务；
- 全部成功且 reward 相同：同样没有组内相对优势。

代码库中存在 DAPO 式 all-equal group filtering/resampling，也存在 GiGPO 的 step-level advantage，但 `run_alfworld_skills.sh` 使用 `adv_estimator=grpo`，且没有启用 `filter_groups`。因此不能把这些机制算作 SkillRL 主方法的早期奖励来源。

## PickTwo 初期的信号从哪里来

### SkillRL 主方法

主要来源按重要性排序：

1. 7,500 条 teacher-generated SFT demonstrations 先教会基本任务行为与 skill 使用；
2. 初始 skill bank 为 rollout 提供显式多步策略；论文举的 PickTwo 技巧是先确认第一个物体已安全放置，再寻找第二个；
3. SFT 后的随机采样更容易产生同题“成功/失败混合组”，GRPO 才获得目标成功优势；
4. 动态 evolution 针对低成功率类别补 skill，间接提高后续成功概率；
5. 非法动作罚分提供有限的行为规范信号。

这与“一个完全不会 PickTwo 的模型仅靠二元 GRPO 自举”不是同一个设定。论文消融是直接证据：去掉 cold-start SFT 后仍能到 65.2，但比完整 SkillRL 的 89.9 低 24.7 个百分点。

### 论文表中的纯 GRPO 基线

SkillRL 表 1 报告 Qwen2.5-7B-Instruct 原始 PickTwo 成功率 3.2%，纯 GRPO 最终为 64.7%。若仅作独立采样近似，单条成功率 3.2%、group size 8 时，至少出现一条成功的概率为：

`1 - (1 - 0.032)^8 ≈ 22.9%`

这说明它并非绝对零启动；稀有成功理论上足以让部分组产生优势。但该计算只是直觉上限/近似，rollout 并不保证独立同分布，而且不同 PickTwo 实例难度差异很大。公开材料没有提供早期实际 mixed-group 比例，所以不能把 22.9% 当作测量结果。

此外，纯 GRPO 基线使用 50 步而非 30 步 horizon，且官方 GiGPO 实验描述为直接训练 Qwen2.5-Instruct，并未声明 LoRA；这与当前 InfoSkill 的 LoRA、30 步截断和数据调度并非严格同条件。

## 对当前 InfoSkill 决策的含义

- 用户指出的“零成功导致死循环”在纯二元 GRPO 中是成立的；invalid-action penalty 只能缓解格式/合法性，不能替代任务成功示范。
- 截稿时间紧时，最有依据的高概率路线不是继续等待稀疏 GRPO 自举，而是复刻 SkillRL 的关键前提：**保存一份可复用的 actor imitation warm-start checkpoint，然后直接进入 M1**。
- 现有 3,521 条 ALFWorld planner 成功轨迹可以作为动作监督基础，再生成/拼接与部署时一致的 skill-conditioned reasoning 格式；但应在论文中清楚写明这是本项目的 planner-supervised warm start，不冒充 SkillRL 的 o3 synthetic trajectory protocol。
- warm start 后仍需保留固定协议的 update-0 评测，才能把“模仿初始化收益”和后续 M1 RL 收益分开报告。
- 若仍要公平对标纯 GRPO，应至少对齐：Qwen2.5-7B-Instruct、全参/LoRA方式、50步 horizon、group size 8、任务采样与总环境交互量，并记录每类 mixed-reward group rate；仅比较 update 数并不是同等数据量。

## 证据边界

目前可以确定：主 SkillRL 有 SFT cold start；文本 ALFWorld 没有 dense progress reward；GRPO group size 8；invalid penalty 为 0.1；动态 skill update 通过 prompt 而不是 reward 起作用。

目前不能确定：纯 GRPO 基线早期 PickTwo 首次成功的具体 update、每步 mixed group 比例、各类别采样量，以及论文最终数字对应的逐 seed 训练轨迹。公开论文与仓库没有提供这些日志。

## 第一方来源

- [SkillRL 论文（arXiv）](https://arxiv.org/abs/2602.08234)，重点见 Algorithm 1、§3.3、Table 1、Table 3、Appendix B。
- [SkillRL 官方仓库与 SFT 说明](https://github.com/aiming-lab/SkillRL)
- [ALFWorld SkillRL 启动脚本](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/examples/grpo_trainer/run_alfworld_skills.sh)
- [ALFWorld reward 实现](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/agent_system/environments/env_package/alfworld/envs.py#L48-L53)
- [invalid-action penalty 实现](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/verl/trainer/ppo/ray_trainer.py#L200-L220)
- [GRPO outcome advantage 实现](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/verl/trainer/ppo/core_algos.py#L113-L173)
- [SFT 数据生成说明](https://github.com/aiming-lab/SkillRL/tree/8e66726ed866a4e0a7f053586a41022798192e6c/examples/sft_data_generation)
- [成功轨迹蒸馏代码](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/examples/sft_data_generation/distillation/distill_alfworld.py#L274-L329)
- [GiGPO 论文（纯 GRPO 基线来源）](https://arxiv.org/abs/2505.10978)
- [GiGPO / verl-agent 官方仓库](https://github.com/langfengQ/verl-agent)
