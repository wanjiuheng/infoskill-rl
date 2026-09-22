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
- `audit-webshop`：逐样本验证格式、动作可执行性、轨迹切分与真实 tokenizer 长度；
- `build-webshop-skill-bank`：从 train demonstrations 生成阶段化、无商品实例泄漏的技能库；
- `webshop_asset_doctor.py`：不加载模型的资产/依赖/索引预检；
- `build_webshop_search_index.py`：只为完整商品库构建一份外部 Lucene 索引；
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
search_engine/resources/documents.jsonl
search_engine/index-manifest.json
```

先做只读预检：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
PYTHON_BIN=/path/to/webshop/python \
bash scripts/run_webshop_imitation.sh doctor
```

`formal_ready=false` 时不要启动长训练。doctor 返回 2 是 fail-closed，不会修改数据。
正式门还要求 Java 11、锁定的 Python 包版本、`en_core_web_sm`、可导入的
`web_agent_site`、非空 Lucene 索引及其完整构建清单，以及至少 15 GiB 剩余空间。

在 H20 的独立 `ifs-webshop` 环境中构建完整索引（无需 GPU）：

```bash
cd /models/wanjh2/data/wjh/alfworld_eval/infoskill
PYTHON_BIN=/data/wanjh2/miniconda3/envs/ifs-webshop/bin/python \
PATH=/data/wanjh2/miniconda3/envs/ifs-webshop/bin:$PATH \
WEBSHOP_INDEX_THREADS=4 \
bash scripts/run_webshop_imitation.sh build-index
```

该入口明确读取数据目录中的完整 `items_shuffle.json`、`items_ins_v2.json` 与
`items_human_ins.json`，不采用上游默认的 1,000 件商品小样本，也不生成其他三套
调试索引。输出仅写入外部 `data/webshop/search_engine`；构建过程中使用隐藏暂存目录，
完成索引可检索检查后才发布。源数据与 SkillRL 代码树不会被修改。构建前要求数据盘
至少空余 50 GiB；若中途失败，会保留暂存目录供排查，并拒绝下一次直接覆盖。
构建可能消耗大量 CPU、内存和共享磁盘 I/O，宜在系统负载允许时后台运行。

`doctor` 变为 `formal_ready=true` 只代表资产和环境预检通过，不表示下文尚未完成的
正式 WebShop rollout、prompt parity 和效果验收已经通过。

资产通过后，先运行单进程、CPU-only 的真实环境 smoke：

```bash
cd /models/wanjh2/data/wjh/alfworld_eval/infoskill
PYTHON_BIN=/data/wanjh2/miniconda3/envs/ifs-webshop/bin/python \
PATH=/data/wanjh2/miniconda3/envs/ifs-webshop/bin:$PATH \
bash scripts/run_webshop_imitation.sh smoke-online
```

入口在本进程初始化上游 `WebAgentTextEnv` 时，显式注入外部商品文件、人类指令文件和
Lucene 索引，不修改 SkillRL 源码或复制 5.5 GB 数据。它把运行时 goal 映射回官方
`human_goals.json` 的最早同文索引，从 train/validation/test 各抽一条，检查 reset、
可用动作、prompt 渲染和真实搜索转移。输出默认是项目根目录下的
`webshop-online-smoke.json`，只保存 goal 索引、动作数量与文本 SHA-256，不保存原始
instruction。通过只证明这条环境/路径链路，尚不证明 warm-start prompt parity 或正式评测。
上游加载完整商品会占用大量主机内存和共享盘 I/O，但不使用 GPU。

环境 smoke 通过后，可运行只读的示范重放 prompt parity 门：

```bash
cd /models/wanjh2/data/wjh/alfworld_eval/infoskill
PYTHON_BIN=/data/wanjh2/miniconda3/envs/ifs-webshop/bin/python \
PATH=/data/wanjh2/miniconda3/envs/ifs-webshop/bin:$PATH \
bash scripts/run_webshop_imitation.sh prompt-parity
```

它从已准备的 train 示范中确定性选择 3 个不同的官方 goal index，优先覆盖搜索、
结果选择、商品页、选项选择和购买五个阶段，并复核来源 SHA-256，
并在真实 `WebAgentTextEnv` 中逐步执行整条示范（默认上限 100 步）。每个决策点比较
goal、规范化观察、最近两步历史、可执行动作和完整策略 prompt；示范动作不再可执行时
立即停止该轨迹并保存首个分叉点。WebShop 会重新随机产生价格上限，因此报告把两项结论
分开：`structural_passed` 使用示范中的完整 task 文本检查其余在线状态，
`exact_online_prompt_match` 则使用真实在线 task 文本检查逐字一致性。正式门以结构一致性
和整条示范可执行为准，但必须披露真实 task 文本产生的 prompt 差异。报告只保存索引、
布尔判定和文本 SHA-256，不保存商品、instruction 或 observation 原文。它仍只是 3 条抽样，
不证明全部 1,010 条示范、模型 policy rollout 或 500 条正式评测。
和 smoke 一样，它会再次读取完整商品库，消耗较多 CPU 内存与共享盘 I/O，但不使用 GPU。

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

归档上传并通过 ZIP 校验后，先执行不解压、不输出原始 instruction 的结构审计：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
PYTHON_BIN=/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python \
bash scripts/run_webshop_imitation.sh audit-archive
```

报告写入 `/root/autodl-tmp/wjh/data/webshop/raw/all_trajs.audit.json`，用于决定哪些
session 可直接恢复 goal 索引，以及环境重放需要支持哪些页面转移。原始网页日志不能
绕过重放直接伪装成已经包含状态和 admissible actions 的 baseline IL 数据。

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

入口默认要求恰好 1,010 条 train human trajectories；该数量由官方 baseline 的
`process_goal`、`human_goals.index` 和 `[1500, 12087)` 划分在锁定数据快照上重新审计得到。
入口还会校验 `human_goals.json` 与 IL JSONL 的注册 SHA-256；数量或源身份不符都会拒绝继续。输出
`/root/autodl-tmp/wjh/data/webshop/processed/imitation-data-content-grouped-v2/manifest.json`
会记录两个源文件 SHA-256、轨迹 ID
SHA-256、train/validation 轨迹数和步骤样本数。

准备完成后先运行 CPU 审计和技能库生成；这两步不加载 7B 模型权重，不占用 GPU：

```bash
cd /root/autodl-tmp/wjh/alfworld_eval/infoskill
PYTHON_BIN=/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python \
bash scripts/run_webshop_imitation.sh audit
PYTHON_BIN=/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python \
bash scripts/run_webshop_imitation.sh build-skill-bank
```

准备阶段按步骤内容连通分组后再确定性切分：共享任一完整 prompt/action 的轨迹必须位于同一
split。这样保留官方 1,010 条示范，同时避免重复目标或页面跨 train/validation 泄漏。同一 split
内的官方重复会在审计报告中记录为 warning，不会擅自删除或重复计为泄漏。

审计默认以 Qwen2.5-7B tokenizer 和训练 `max_length=16384` 计算完整 chat-template
token 长度；轨迹跨 split 泄漏、步骤不连续、response 合同错误、示范动作不在当前动作空间、
manifest 计数不一致或样本超长都会令 `passed=false` 并返回非零状态。16,384 来自注册语料的
真实上界：首次审计最大完整样本为 16,305 tokens。为控制长序列峰值显存，WebShop warm-start
默认使用每卡 batch 1、梯度累积 16。报告默认写入
`/root/autodl-tmp/wjh/data/webshop/processed/imitation-data-content-grouped-v2-audit.json`。

技能库只读取 `train.jsonl`，不读取内部 SFT validation。它固定覆盖 query、结果筛选、商品
核验、选项选择、回退和购买六个阶段；示范只用于登记动作族数量和来源校验，不把商品名、
ASIN 或具体选项写进技能文本。技能库与 provenance manifest 默认写入
`/root/autodl-tmp/wjh/data/webshop/processed/webshop-skill-bank-content-grouped-v2.json` 及同名
`.manifest.json`。技能库生成和 GPU warm-start 都会重新核对审计报告的 `passed`、数据目录、
三份输入 SHA-256；warm-start 还要求 tokenizer 路径与 `max_length` 完全一致。数据或训练配置
在审计后发生变化时会拒绝启动，必须重新运行 `audit`。训练完成后的 adapter manifest 会记录
审计报告路径与 SHA-256，供 `m1-handoff` 继续追溯。

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
