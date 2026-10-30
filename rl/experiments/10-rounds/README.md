# Harmonies：正式 10 轮 RL 实验版本

这里集中管理**实际完成 10 轮 self-play → 训练 → 晋级测试**的实验。版本号表示完整实验方案，不表示网络 schema，也不是游戏内启发式 AI 的 v1/v2/v3。最后整理日期：2026-10-30。

目前共 **3 个已完成并核验的10轮版本**。各版原始轨迹、replay、全部checkpoint、日志、看板和核验文件单独存放在该版的 `run/` 中。v003的正式训练核验已通过；其400局独立对战已完成。

| 版本 | 改动概要 | 状态 | 看板 | 机器可读记录 |
| --- | --- | --- | --- | --- |
| [v001-fixed32](v001-fixed32/README.md) | 全状态 32 模拟、每轮 1000 局、固定训练 400k 行 | 10/10，已核验 | [打开](v001-fixed32/run/dashboard.html) | [manifest](v001-fixed32/manifest.json) |
| [v002-playout-cap](v002-playout-cap/README.md) | 2000 局、25% expensive64 / 75% cheap16，仅 expensive 训练，预算上限 2W | 10/10，已核验 | [打开](v002-playout-cap/run/dashboard.html) | [manifest](v002-playout-cap/manifest.json) |
| [v003-spatial](v003-spatial/README.md) | 5 组空间 kernel ×16 channel、FFN 512→128、每个 tower 增加 64 隐层 | 10/10，数据已核验；UI静态核验 | [看板](v003-spatial/run/dashboard.html) | [manifest](v003-spatial/manifest.json) |

## 方案对比

| 配置 | v001-fixed32 | v002-playout-cap |
| --- | --- | --- |
| 每轮 self-play 局数 | 1000 | 2000 |
| 每个原子决策新增 MCTS 模拟 | 全部 32 | 独立抽样：25% 为 64，75% 为 16 |
| Root Dirichlet noise 权重 | 全部 0.25 | expensive 为 0.25，cheap 为 0 |
| 选招温度 / 回合内树复用 | 1 / 开启 | 1 / 开启 |
| Replay 收录 | 全部决策状态 | 仅 expensive 状态 |
| 每轮训练行数 | 固定 400000 | min(400000, 2 × 实际窗口行数) |
| 窗口采样 | 每遍无放回；不足预算则开始下一遍 | 每遍无放回；每轮每行最多两次 |
| Gatekeeper | 100 局；serving 128 模拟；50% 晋级 | 相同 |

v001/v002 共同配置：A 面、两人、无自然之灵；数值特征 1614 维，5 个市场卡牌 ID 使用共享 8 维 embedding；共享 FFN `1654 → 128 → 128`，policy 和三个 value head；总参数 343555。玩家持牌为固定 32 卡位 × 3 特征。AdamW、学习率 0.001、batch size 64；window 参数 `min_rows=100000`、`taper_scale=50000`、`expand_window_per_row=0.4`、`taper_window_exponent=0.65`。v003 保留旧输入，追加5组空间kernel×16channel（3680维），主干扩展至5334→512→128，四个tower各增加64维ReLU层，共2,900,467参数；其余采样与训练规则同v002。

v001/v002均以 seed 1729 随机初始化，初始权重 SHA256 完全相同，各自拥有独立 replay；v002 并非从 v001 checkpoint 续训。每轮由 best 收集数据，从 latest 续训，即使上一轮未晋级也继续训练 latest。v003同样使用seed1729全新初始化且独立生成replay；架构改变，初始权重不与前版相同。晋级对局关闭噪声、温度 0、每动作新增 128 模拟，平局半胜，测试数据不进入 replay。

## 实际完成量与结果

| 指标 | v001-fixed32 | v002-playout-cap | v003-spatial |
| --- | ---: | ---: | ---: |
| 训练 self-play 总局数 | 10000 | 20000 | 20000 |
| Replay 样本数 | 953730 | 487277 | 485403 |
| 累计训练消费行数 | 4000000 | 2799410 | 2794616 |
| 优化器更新次数 | 62500 | 43746 | 43671 |
| 晋级测试总局数 | 1000 | 1000 | 1000 |
| 晋级轮次 | 1–10 | 1、2、3、5、6、7、8、10 | 1、3、4、5、6、8、9 |
| 最终 best / latest | 第10轮 / 第10轮 | 第10轮 / 第10轮 | 第9轮 / 第10轮 |
| 总耗时（逐轮合计） | 14331.49秒（3.981小时） | 25434.49秒（7.065小时） | 56376.49秒（15.660小时） |
| 末轮 self-play：分数 / 入住动物 / 完整牌 | 35.886 / 2.313 / 0.1195 | 40.046 / 2.49275 / 0.16375 | 39.90325 / 2.489 / 0.16275 |
| 末轮晋级双方：分数 / 入住动物 / 完整牌 | 48.645 / 3.34 / 0.285 | 51.235 / 3.72 / 0.34 | 50.55 / 3.78 / 0.335 |

以上终局指标均为每局双方各计一次的均值。self-play 是本轮训练前 best 的表现，晋级统计混合本轮候选与此前 best，不能视为最终模型单独的能力。各版采样策略、训练量或晋级对手不同，不能仅凭这些均分判断棋力。

2026-10-29 已完成双方最终 best 的 **100 局独立直接对战**：v002 **58 胜 / 0 平 / 42 负，得分率 58%**；先手 25/0/25（50%），后手 33/0/17（66%）。配对种子 bootstrap 95% 区间为 **47%–68%**，尚不足以确认稳定优势。均分 v002 51.08、v001 48.74。配置沿用晋级测试：50 个新种子交换先后手、serving 每动作新增 128 模拟、无根噪声、温度 0、回合内复用树。完整动作已独立规则重放核验，原训练数据和模型未改变。详见 [对比报告](comparisons/v002-vs-v001-20261029/README.md) / [结构化结果](comparisons/v002-vs-v001-20261029/result.json)。

后续版本的主要对比指标采用：**第 10 轮结束时 best 对固定基准的独立测试得分率 `(胜 + 0.5 × 平) / 总局数`**。固定基准为 v002 的第 10 轮模型，默认统一 serving 每原子动作新增 128 模拟、噪声 0、温度 0，至少 400 局按 seed 交换先后手；种子独立于训练与晋级，报告按配对 seed 计算的置信区间。模型扩容对比另注明此预算不等于相同推理耗时。v002-v001的历史100局按用户指定的晋级口径执行，保留原结论；v003-v002的400局基准单独记录，不能与历次晋级胜率混用。

## v003 独立对战与扩容结果

最终best（第9轮）对固定v002 best（第10轮）完成400局：**214胜 / 5平 / 181负，得分率54.12%；配对seed bootstrap 95%区间49.25%–58.88%**。双方均分52.610 / 51.053。200个新seed交换先后手，serving新增128模拟、noise0/temp0、回合内复用树。详见[独立对战](comparisons/v003-vs-v002-20261029/README.md)。

v003正式训练总耗时为v002的约2.217倍；采集0.4465局/秒（v002为0.9894），训练4573行/秒（v002约21919）。这是完整流程的不同轨迹计时，不能替代[同输入跑图测速](v003-spatial/README.md)。末轮在线rank MSE为0.216867（v002为0.376095），但没有新增固定checkpoint的独立value精度测试，不能据此宣称泛化或过拟合问题已解决。

全部21,000局、2,041,642动作独立规则重放通过；预算、样本映射、replay隔离、模型与冻结源码哈希、所有checkpoint架构和三项终局统计通过核验。浏览器禁止本地file: URL；v003看板只完成双tab和10轮数据静态核验，浏览器渲染/交互仍未验证。详见[完整逐轮结果](v003-spatial/README.md)、[结构化结果](v003-spatial/results.json)、[核验及局限](v003-spatial/run/verification.json)。

## 目录与文件

```text
rl/experiments/10-rounds/
  README.md                      总索引、配置与结果对比、版本规则
  v001-fixed32/
    README.md                    本版目的、配置、结果、入口
    manifest.json                结构化登记：配置、逐轮结果、模型哈希
    artifacts.sha256             run 内全部文件的 SHA256 清单
    relocation.json              此次目录迁移核验
    run/                         完整原始实验产物（本地大文件）
  v002-playout-cap/
    ...
  v003-spatial/                  正式10轮已完成并核验
    README.md / manifest.json / results.json / artifacts.sha256
    benchmark.json / verification.json  实现与测速核验
    run/                        正式数据、看板及 run/verification.json
    run/benchmark-20261029/      本地跑图数据与模型
  comparisons/
    v003-vs-v002-20261029/         400局固定v002基准对战
    v002-vs-v001-20261029/
      README.md                  配置、胜平负、置信区间、复现入口
      result.json                结果、逐局简表、核验与文件哈希
      run_match.py / verify.ts / summarize.py
      run/                       本地完整对局、日志、核验及指标
```

README、manifest、哈希清单和迁移核验进入 Git 管理；`run/` 沿用原实验大文件不进 Git 的策略。克隆仓库能获取实验记录，完整对局、数据库与 checkpoint 需要另行备份/传输，不能把 Git 当作这些产物的备份。

旧路径 `rl/runs/selfplay-10rounds-20261015`、`rl/runs/playout-cap-10rounds-20261018` 保留为指向新 `run/` 的相对符号链接。它们不是第二份实验。历史日志与 JSON 中的旧绝对路径保留原样，避免改变原始记录。核验脚本仅做必要的路径调整，原脚本保存在各 run 的 `_relocation-originals/`；每个未修改原文件都经搬迁前后完整 SHA256 核对。

可在任意版本的 `run/` 下执行 `shasum -a 256 -c ../artifacts.sha256` 检查登记时产物完整性。核验脚本会更新自身输出，重新执行后需审阅变化再更新清单；不要把生成时间变化误判为训练数据损坏。

## 后续版本规则

1. 使用下一个连续编号，v003 已登记为空间网络方案；每个版本有自己的 `run/`。正在运行的版本明确标记状态，只有 state 为 complete、10 轮 metrics 齐全且核验通过，才计入“已完成版本”。
2. 启动前记录相对前版的改动、代码 commit（如有未提交改动，保存 patch）、完整配置、初始化模型与哈希。运行中的当前 HEAD 不能冒充历史实验的代码版本。历史v001/v002没有独立 launch commit 证据，因此 manifest 不猜测填充。
3. 完成后补齐本版 README、manifest、逐轮窗口/消费量/晋级记录、best/latest 与哈希、完整性清单、看板及核验入口，并更新本索引。
4. 已登记的 10 轮结果固定保存；继续到更多轮、修改配置或重新训练，应建立新实验目录并明确父版本、checkpoint 和 replay 来源，不能覆盖现有结果。
5. 小规模 smoke、拟合审计、value 准确率与条件方差实验放在诊断目录，通过版本文档关联；不单独占正式训练版本号。v003的测速、正式10轮训练和独立版本对战分别保存与核验。

2026-10-29目录整理的历史核验（仅v001/v002）见 [organization-verification.json](organization-verification.json)：配置与逐轮结果一致、模型哈希与 replay 行数一致、49 个文档链接有效、3 个 TypeScript 审计入口可构建。两个看板 HTML 与原始已验证版本逐字节相同，双 tab 元素与 10 轮数据静态检查通过。本次浏览器重检受环境限制未完成：headless Chrome 退出，浏览器工具禁止 `file:` URL；未将历史 UI 核验冒充为本次执行。

流程细节与全部实测记录见 [self-play 文档](../../../docs/ai-selfplay-loop.md)。
