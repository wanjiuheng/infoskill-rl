# WebShop Bring-up and Data Contract

WebShop 与 ALFWorld 共享 policy、LoRA warm-start、InfoSkill conditioning 和
VERL 训练基础设施，但不共享专家轨迹生成器。WebShop warm-start 使用官方 human
demonstrations；不能把 ALFWorld planner 接到 WebShop 上。

## 当前实现边界

已经落地的深模块接口：

- `integrations.webshop.splits`：官方 goal-index 划分；
- `OfficialWebShopHumanDemonstrationProvider`：官方 IL JSONL 到统一成功轨迹；
- `render_webshop_policy_message`：warm-start 与在线 WebShop prompt 的共同合同；
- `prepare-webshop`：按整条轨迹切分后生成 response-only SFT 数据；
- `webshop_asset_doctor.py`：不加载模型的资产/依赖/索引预检；
- WebShop skill bank 的固定 SHA-256 provenance；
- InfoSkill 自有分区合同固定从索引 1500 开始训练，防止 validation 500--1499 泄漏。

尚未宣称完成的部分：正式 WebShop VERL rollout adapter、warm-start/在线 prompt
逐字段 parity、完整 500 条 test 评测、M1 handoff 效果门。这些必须在真实资产到位后完成，
不能用小型 fixture 代替。

锁定的 SkillRL 上游代码保持只读；它的旧 WebShop `is_train` 范围仍从 500 开始，不能直接
作为正式训练入口。后续 InfoSkill rollout adapter 必须显式使用本项目的分区合同，不能把
上游旧范围透传进实验。

## 官方划分

以 `human_goals.json` 的稳定索引为准：

- test: `[0, 500)`；
- validation/eval: `[500, 1500)`；
- train: `[1500, goal_count)`。

warm-start provider 默认只接收 train demonstrations，再从这些训练轨迹内部按 trajectory
ID 做确定性 98/2 SFT train/validation 切分。官方 validation/test 不进入 SFT。

## 代码与数据位置

锁定的 WebShop 代码根目录保持只读：

```text
/root/autodl-tmp/wjh/alfworld_eval/SkillRL/
  agent_system/environments/env_package/webshop/webshop
```

WebShop 数据不写入上述代码树，统一放在：

```text
/root/autodl-tmp/wjh/data/webshop
```

正式运行要求的数据布局：

```text
data/items_shuffle.json
data/items_ins_v2.json
data/items_human_ins.json
baseline_models/data/il_trajs_finalized_images.jsonl
baseline_models/data/human_goals.json
search_engine/indexes/<non-empty index>
```

先做只读预检：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
PYTHON_BIN=/path/to/webshop/python \
bash scripts/run_webshop_imitation.sh doctor
```

`formal_ready=false` 时不要启动长训练。doctor 返回 2 是 fail-closed，不会修改数据。
正式门还要求 Java 11、锁定的 Python 包版本、`en_core_web_sm`、可导入的
`web_agent_site`、非空 Lucene 索引，以及至少 15 GiB 剩余空间。

## 分阶段取得官方 human trajectories

不要先运行上游 `setup.sh -d all`；它会连续安装依赖、下载全量产品、生成四套中间
resources/索引，峰值空间不可控。先单独下载 WebShop README 链接的官方
`all_trajs.zip`，校验 ZIP 后只保存文件清单，不立即解压：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
bash scripts/stage_webshop_human_archive.sh
```

脚本把下载工具放在独立的 `/root/autodl-tmp/wjh/webshop-tools`，把归档放在
`/root/autodl-tmp/wjh/data/webshop/raw/all_trajs.zip`。前者不是数据集目录；它不修改
InfoSkill Python 环境，也不使用 GPU；
可用空间低于 30 GiB 时会拒绝下载。取得归档清单与 SHA-256 后，再实现并验证从官方
原始 session logs 到 imitation provider 的转换，不能静默改用未登记的第三方预处理数据。

## 环境隔离

WebShop 官方栈依赖较旧的 Gym/Flask/Pyserini，并需要 Java 11。不要在正在使用的
`my_new_env/infoskill` 中直接安装或降级这些包。应创建独立 WebShop 环境，并在 ALFWorld
训练完成或 GPU 空闲后再做 import/runtime smoke。资产下载和索引构建主要使用 CPU、磁盘和
Java，但会产生额外磁盘占用。

## 生成 warm-start 数据

数据到位后：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
PYTHON_BIN=/path/to/webshop/python \
bash scripts/run_webshop_imitation.sh prepare
```

入口默认要求恰好 1,012 条 train human trajectories；数量不符会拒绝继续。输出
`artifacts/webshop-imitation-data/manifest.json` 会记录两个源文件 SHA-256、轨迹 ID
SHA-256、train/validation 轨迹数和步骤样本数。

在 prompt parity gate 通过前，不启动正式 warm-start。通过后可用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTHON_BIN=/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python \
bash scripts/run_webshop_imitation.sh train
```

这一步会占用三张 GPU；不得与 ALFWorld 正式训练并发。

## 数据来源

- 官方代码与设置说明：<https://github.com/princeton-nlp/WebShop>
- 官方 baseline 的 `train_choice_il.py` 定义上述 goal-index 划分和 IL 字段。

下载前应核对官方许可与文件来源；不把第三方镜像的未知修改静默当作正式数据。
