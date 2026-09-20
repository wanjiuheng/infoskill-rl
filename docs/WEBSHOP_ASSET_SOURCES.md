# WebShop 缺失资产：可用来源与完整性核验

核验日期：2026-09-21。

## 结论

Princeton NLP 官方仓库仍是文件名、用途和原始 Google Drive ID 的权威来源，但其
`setup.sh` 没有备用源、版本固定或校验和。用户报告的三个 Drive 链接均已返回“文件不存在”；
Princeton 官方仓库的 [issue #61](https://github.com/princeton-nlp/WebShop/issues/61) 也在
2026-07-21 报告 `items_shuffle` 等数据链接无法访问，且截至核验日没有维护者给出替代链接；
本次环境访问 `drive.google.com` 又超时，故不能把旧链接继续作为可靠的自动化来源。

目前最省事的单一替代源是固定提交
[`Skyler215/VERL_WEBSHOP@b40c2398d6544b6f80d104575e4228c17082fddb`](https://huggingface.co/datasets/Skyler215/VERL_WEBSHOP/tree/b40c2398d6544b6f80d104575e4228c17082fddb)。
它同时包含三个环境资产以及已经整理好的 `human_goals.json` 和
`il_trajs_finalized_images.jsonl`。不过这是个人账号上传的第三方镜像，README 为空；
因此应固定提交并在下载后校验哈希，不能称为“Princeton 官方发布”。

## 官方来源所能确认的事实

Princeton 官方 [`setup.sh`](https://github.com/princeton-nlp/WebShop/blob/master/setup.sh#L28-L42)
明确规定全量模式下载：

| 目标文件 | 官方 Drive ID | 用途 |
|---|---|---|
| `items_shuffle.json` | `1A2whVgOO0euk5O13n2iYDM0bQRkkRduB` | 全量商品抓取信息 |
| `items_ins_v2.json` | `1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi` | 商品属性 |
| `items_human_ins.json` | `14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O` | 人工指令 |

官方 README 也要求全量 WebShop 使用前两个精确文件名，并说明 WebShop 有 118 万商品与
12,087 条众包文本指令（[README](https://github.com/princeton-nlp/WebShop/blob/master/README.md#overview)，
[全量路径配置](https://github.com/princeton-nlp/WebShop/blob/master/README.md#setup)）。官方仓库采用
[MIT License](https://github.com/princeton-nlp/WebShop/blob/master/LICENSE.md)，但该许可文件只明确
覆盖“software and associated documentation”；对于抓取的商品内容及众包数据，不能据此推断
第三方内容权利已全部清理。

## 推荐下载源（固定 revision）

以下链接均固定到同一提交，避免 `main` 漂移：

| 文件 | 固定下载链接 | HF 页面显示大小 | 完整性信息 |
|---|---|---:|---|
| `items_shuffle.json` | [下载](https://huggingface.co/datasets/Skyler215/VERL_WEBSHOP/resolve/b40c2398d6544b6f80d104575e4228c17082fddb/items_shuffle.json?download=true) | 5.48 GB | 期望 SHA-256 `2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8`；精确大小 5,479,720,229 bytes |
| `items_ins_v2.json` | [下载](https://huggingface.co/datasets/Skyler215/VERL_WEBSHOP/resolve/b40c2398d6544b6f80d104575e4228c17082fddb/items_ins_v2.json?download=true) | 186 MB | 期望 SHA-256 `1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e`；精确大小 186,295,270 bytes |
| `items_human_ins.json` | [下载](https://huggingface.co/datasets/Skyler215/VERL_WEBSHOP/resolve/b40c2398d6544b6f80d104575e4228c17082fddb/items_human_ins.json?download=true) | 5.14 MB | 本地锁定 SHA-256 `cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3` |
| `human_goals.json` | [下载](https://huggingface.co/datasets/Skyler215/VERL_WEBSHOP/resolve/b40c2398d6544b6f80d104575e4228c17082fddb/human_goals.json?download=true) | 999 kB | 本地锁定 SHA-256 `b68746ed66cd31fdc5f70eb3f5831a46b38163563b57a66ddcab8fd60ee0cbdc` |
| `il_trajs_finalized_images.jsonl` | [下载](https://huggingface.co/datasets/Skyler215/VERL_WEBSHOP/resolve/b40c2398d6544b6f80d104575e4228c17082fddb/il_trajs_finalized_images.jsonl?download=true) | 97.5 MB | SHA-256 `0f3ef1890245a283f8116b7abcabebd4acdf355d773edd99977e8ed6de63ec6c` |

Skyler 的仓库文件页显示该提交为 verified、7 个文件、总大小 5.77 GB，并列出上述各文件
（[文件树](https://huggingface.co/datasets/Skyler215/VERL_WEBSHOP/tree/main)）；轨迹文件页公开了
[97.5 MB 与 SHA-256](https://huggingface.co/datasets/Skyler215/VERL_WEBSHOP/blob/main/il_trajs_finalized_images.jsonl)。
“verified”这里只表示 Hugging Face 提交签名/状态，不等于 Princeton 对内容背书。

两个大文件的期望哈希并非只来自单一上传者：

- [`YWZBrandon/webshop-data`](https://huggingface.co/datasets/YWZBrandon/webshop-data/blob/main/items_shuffle.json)
  显示 `items_shuffle.json` 为 5.48 GB，SHA-256 与上表一致；其
  [`items_ins_v2.json`](https://huggingface.co/datasets/YWZBrandon/webshop-data/blob/main/items_ins_v2.json)
  也显示相同的 186 MB 与 SHA-256。
- 独立镜像 [`HongbangYuan/webshop`](https://huggingface.co/datasets/HongbangYuan/webshop/commit/cd7e57faf269fd2323648f1ef5c238f489f9a57a)
  的 LFS pointer 给出 `items_shuffle.json` 精确大小与相同 SHA-256；其
  [`items_ins_v2.json` 提交](https://huggingface.co/datasets/HongbangYuan/webshop/commit/03f4b8ad6dfca15a88679e662a91043149ff755a)
  给出精确大小 186,295,270 bytes 与相同 SHA-256。
- Utah Spark Lab 的
  [`timewarp-env-data` 数据卡](https://huggingface.co/datasets/sparklabutah/timewarp-env-data/blob/main/README.md)
  声称其 WebShop 文件是原始数据的逐字节镜像，并记录这两项哈希与上述两个独立镜像一致。
  这增强了内容一致性信心，但仍不是 Princeton 官方签名。

如果不希望使用个人 Skyler 镜像，`sparklabutah/timewarp-env-data` 是组织账号、带 provenance
说明的高信任备用源；其三个文件位于 `webshop/` 子目录。它没有包含本项目 warm-start
所需的两项整理后 IL 文件，因此不如 Skyler 的单提交组合方便。

## 能否避免 raw replay

**可以避免 warm-start 数据构建阶段的 raw replay，但不能免掉正式环境准备。**

本项目的 `OfficialWebShopHumanDemonstrationProvider` 直接消费
`il_trajs_finalized_images.jsonl` 的 `states`、`available_actions`、`action_idxs` 以及
`actions`/`actions_translate`，再用 `human_goals.json` 稳定映射 goal index；它不要求读取原始
session archive，也不在转换时启动 WebShop 环境。Skyler 数据集的 HF 预览错误反而明确列出了
这些轨迹字段，并把两项文件固定到 revision `b40c2398…`
（[数据集预览诊断](https://huggingface.co/datasets/Skyler215/VERL_WEBSHOP)）。因此，把这两个文件
放到 `baseline_models/data/` 后，可直接运行现有 `prepare-webshop` 路径。

限制如下：

1. Skyler README 为空、仅 1 位贡献者/2 次提交，未解释轨迹如何由 Princeton 原始日志生成；
   所以这是“可复现的第三方预处理快照”，不是已经证明的官方 human demonstrations。
2. 在线 rollout、完整 test 评测与 reward 计算仍需要三个环境资产、Java/Python 运行时以及非空
   Lucene 索引；整理后的 IL JSONL 不能替代这些。
3. 首次采用时应把五个文件的实际 SHA-256 全部写入项目 manifest；缺少独立上游校验的文件
   至少以固定 revision + 本地哈希锁定。当前准备入口已固定 IL JSONL 与
   `human_goals.json` 的 SHA-256。

锁定快照共 1,571 条 IL 轨迹。按官方 baseline 的原始 `process_goal`、首次精确
`human_goals.index` 与 goal-index 划分重放筛选后，得到 test 507、validation 54、train
1,010 条；当前 provider 得到完全相同的结果，没有缺失目标、跨 split 映射或归一化偏差。

## 建议决策

为尽快恢复流程：使用 Skyler 固定 revision 下载五个文件；下载完成后强制校验两个大文件和
IL JSONL 的上表哈希，并记录另两项哈希。这样可直接走 `prepare-webshop`，无需先取得并重放
`all_trajs.zip`。如果实验必须满足“仅官方原始数据可追溯”的更高证据门槛，则应继续向
Princeton 作者索取恢复后的官方链接或官方校验和；第三方镜像不能独自满足该门槛。
