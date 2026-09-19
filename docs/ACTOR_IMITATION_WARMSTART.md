# Actor imitation warm-start 与 M1 handoff

## 决策

ALFWorld 的新主线不再从原始 Qwen2.5-7B-Instruct 直接开始 GRPO。先用经身份门和正式门验证的 planner 成功轨迹做 actor imitation，再把 LoRA 权重冻结成内容寻址的 `m1-handoff`，最后从该 handoff 启动新的 INFO-SKILL M1。

这是一条新的实验分支，不能与旧的“所有方法从原始模型开始”结果混报。旧 M0/M1 checkpoint 仍可用于历史分析，但不能标记为 imitation warm-start 的对照。

## 数据和技能边界

- ALFWorld provider 只接受 train split、verified planner、formal gate passed 的正式 grounding。prepare 同时生成 `alfworld-imitation-data-grounding`，将旧 candidate IDs 改写为新 bank IDs；新 M1 必须使用这份派生 grounding。
- 先按完整 trajectory/task ID 划分，再展开为逐步 SFT 样本，防止相邻状态泄漏。
- SFT prompt 复用线上 `render_policy_message`；loss 只覆盖 `<think>…</think><action>…</action>` response。
- planner skill bank 使用六个原生 task type。PickTwo 包含 first object、first delivery、second object、second delivery 四阶段，并记录停止条件和防循环规则。
- WebShop 与 Search 不复用 ALFWorld planner，而由各自 demonstration provider 产生成功轨迹。

## Handoff 语义

`m1-handoff` 包含 adapter 权重/配置、SFT 训练清单、数据清单、skill bank 及其 provenance manifest、`m1-handoff.json` 和 `checkpoint.complete.json`。清单绑定 base model ID、LoRA rank/alpha、训练协议和所有内容的 SHA-256；目录存在时拒绝覆盖。

加载时只恢复 actor LoRA。GRPO optimizer/scheduler 和所有 M1 模块均重新初始化。`--warmstart-handoff` 与 `--resume` / `--policy-checkpoint` 互斥。

## 服务器执行顺序

```bash
GROUNDING=/root/autodl-tmp/wjh/alfworld_eval/infoskill/runs/20260912T153809Z-m1-grounding-formal-rescued-finalized

GROUNDING_DATA="$GROUNDING" \
PYTHON=/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python \
bash scripts/run_actor_imitation.sh prepare

GPUS=0,1,2 \
PYTHON=/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python \
IMITATION_LEARNING_RATE=1e-4 IMITATION_EPOCHS=2 \
bash scripts/run_actor_imitation.sh train

# 若 SFT 中断，只需显式指向最近的 Trainer checkpoint 后重跑 train：
# IMITATION_RESUME="$PWD/runs/actor-imitation-warmstart/checkpoint-<step>"

PYTHON=/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python \
bash scripts/run_actor_imitation.sh finalize

HANDOFF="$PWD/artifacts/m1-handoff" \
GPUS=0,1,2 PYTHON=/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python \
bash scripts/run_m1_handoff_update0_eval.sh

HANDOFF="$PWD/artifacts/m1-handoff" \
GROUNDING_DATA="$PWD/artifacts/alfworld-imitation-data-grounding" \
TARGET_UPDATES=200 GPUS=0,1,2 \
PYTHON=/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python \
bash scripts/run_m1_from_handoff.sh
```

新 M1 沿用成功率/步数曲线、错题记录、最近 5 个 + 当前最佳 + 最终 checkpoint，以及 checkpoint 边界优雅暂停。
