# Self-play loop

`rl/loop.py` 串行调度现有 TypeScript 自对弈与 PyTorch 训练，不复制游戏规则。支持 A 面、两人、无自然之灵。每轮用最优模型采样，从最新 checkpoint 续训，再进行 serving 对战晋级测试。

## 运行

在仓库根目录安装 `rl/requirements.txt`，然后执行：

```bash
.venv-rl/bin/python rl/loop.py --out rl/runs/selfplay --iterations 10
```

`--iterations` 是希望累计完成的 self-play 轮数。重复执行相同命令会恢复未完成的工作；改为 `--iterations 20` 则继续到第 20 轮。已完成目标不会重复采样或训练。以下均为默认值：

| 参数 | 默认值 |
| --- | ---: |
| `--games-per-iteration` | 1000 |
| `--simulations` | 32 |
| `--gate-games` | 100（实际对局数，必须为正偶数） |
| `--gate-simulations` | 128 |
| `--min-rows` | 100000 |
| `--taper-scale` | 50000 |
| `--expand-window-per-row` | 0.4 |
| `--taper-window-exponent` | 0.65 |
| `--train-rows` | 400000 |
| `--batch-size` | 64 |
| `--lr` | 0.001 |
| `--hidden` | 128 128 |
| `--embedding-dim` | 8 |
| `--seed` | 1729 |
| `--device` / `--threads` | cpu / 1 |

默认从随机模型开始；`--initial path/to/model.pt` 可从已有模型及 Adam 状态开始。首轮把它保存成 run 内部的 `initial.pt/.json`。恢复时省略的配置自动从 `config.json` 读取；禁止修改已有 run 的实验参数。初版使用 POSIX 文件锁，支持本项目 macOS/Linux 运行环境。

累计数据不足 `min-rows` 时，默认使用已有全部行立即重复抽样训练，不等待 10 万行；窗口不足 400k 时同样通过重新打乱取样补足。只有显式传入 `--wait-for-min-rows` 才启用 KataGo 式积累门槛。这里按用户要求保留窗口增长公式，并调整启动行为。

每个原子动作新增 32 次 MCTS 模拟，并复用本回合子树；不是每局或每整回合 32 次。自对弈沿用生产采样设置：根噪声权重 0.25、温度 1。所有真实原子动作都生成样本，policy 来自搜索，三个 value label 在对局结束后统一用真实终局结果回填。超出动作上限或规则异常的未完局不进入 buffer。

## Gatekeeper 与模型选择

调度器分别维护 `bestModel` 和 `latestModel`。初始模型同时作为两者。

1. 使用 `bestModel` 完成训练 self-play，加入 replay buffer。
2. 从 `latestModel` 恢复模型与 Adam 状态，训练 400k 行，保存本轮候选 checkpoint。无论随后是否晋级，它都成为新的 `latestModel`。
3. 候选与此前的 `bestModel` 进行 **100 局**测试：50 个环境 seed，每个交换先后手各一次。候选的玩家 ID 始终是 0，当前最优模型为 1，指标按模型身份汇总，不会因先后手交换而颠倒。
4. 使用生产 `mctsAgent` 的 serving 路径：根噪声 0、选招温度 0、每原子动作新增 128 次模拟、复用回合内子树。搜索预算由独立的 `--gate-simulations` 控制，不继承 self-play 的 explore 设置。
5. 100 局全部完成后，`(wins + 0.5*draws) / 100 >= 0.5` 则把候选设为新 `bestModel`。**恰好 50% 也晋级**，不提前接受或拒绝。胜负依据引擎最终排名，包含动物数量破同分规则。

例如第 3 轮候选未晋级：第 4 轮继续使用旧最优模型采样，但训练从第 3 轮 checkpoint 及其优化器状态继续。所有候选 checkpoint 保留以便续训和复查；此处维护单个最优模型，不做 top-K 模型池。

测试用独立 seed 序列（`seed + 1000000000 + (iteration-1)*gate_games/2` 起），只保存对局、决策诊断和结果，**不进入 replay buffer**，不消耗额外训练行数。`gatekeeper.json` 包括候选/对手路径及内容哈希、胜平负、晋级率、分数、整体指标、按模型分组的指标。测试单独执行的 CLI：

```bash
npm run rl -- gate --model candidate.json --opponent incumbent.json \
  --games 100 --simulations 128 --seed 1000001729 --output gate-games.jsonl
```

注意 `gate --games 100` 是 100 局；旧 `evaluate --games 50` 中的数字是 seed 数，交换先手后才是 100 局。调度器使用 `gate` 以支持逐局恢复。

恢复旧版无 gatekeeper 的 run 时，会补充上述两个 gate 参数；旧 run 最新 checkpoint 作为启用 gatekeeper 时的初始对手。从下一轮开始执行晋级测试，历史轮次保持“无晋级测试记录”，不回填虚构胜率或晋级结果。

## KataGo 窗口与抽样

窗口公式参考 [KataGo `python/shuffle.py` 的 `compute_desired_num_rows`，379–398 行](https://github.com/lightvector/KataGo/blob/master/python/shuffle.py#L379)，核对日期 2026-10-14。该文件的 `taper-window-scale` 对应本项目的 `taper-scale`。`add_to_data_rows=0`，不额外设窗口上限。公式在本项目 `rl/replay.py::window_size` 中独立实现。

令累计收集的有效样本数为 `N`，`m=100000`、`s=50000`、`a=0.4`、`e=0.65`。当 `N >= m`：

```text
W(N) = min(N, max(m, floor(m + a × ((N - m + s)^e - s^e) / (e × s^(e-1)))))
```

它在 `N=m` 时取 `m`，初始增长斜率为 `a`，随后次线性增长。不是简单的 `0.4*N^0.65`。`N<m` 时实际可用窗口为全部 `N` 行，是否开始训练由启动门槛控制。

| 累计 N | 最近窗口 W |
| ---: | ---: |
| 100000 | 100000 |
| 150000 | 117512 |
| 200000 | 132072 |
| 1000000 | 277827 |
| 10000000 | 1029451 |

SQLite 中每一行是一个原子动作样本，按“self-play 轮 → 对局 → 对局内原子动作”顺序递增 ID。窗口严格取最新的 `W` 行，允许在对局内部截断。窗口增长使用累计总行数，独立于训练消费行数。

训练从该窗口均匀无放回抽取最多 400k 行。如果窗口不足，先按随机排列消费一整遍窗口，再重新独立打乱，重复直到累计消费恰好 400k 行。最后一遍只取剩余额度；窗口用完之前不会出现重复行。重排边界的短片段与下一遍拼接 minibatch，保证 `ceil(400000 / batch_size)` 次更新，默认 **6250 steps**。这 400k 行只消费一遍，不再嵌套额外 epochs。

每轮保存实际训练顺序 `sampled-rows.i64`（little-endian int64）和 SHA-256、抽样 seed、整遍数、最后不足一遍的行数，可直接检查是否无放回。训练不划出验证集，以保证 400k 全部用于梯度更新；每轮另做上述晋级测试，也可使用现有 `evaluate` CLI 单独评测。窗口取样方案遵循本次需求，不复制 KataGo 的分片近似保留机制。

Buffer 以 zlib 压缩 float32 数据保存在 SQLite，按 minibatch 读取，不把历史 JSON 样本全部展开进内存。原始轨迹和旧行保留用于复查，采样只读最新窗口；磁盘历史不自动删除。

## 监控

打开 run 下的 `dashboard.html` 查看并列的 **“训练self-play” / “晋级测试”** tabs。前者显示训练采样、loss 和梯度；后者显示累计测试局数/决策数、各轮胜平负与晋级结果、轮次数、搜索、value、policy、consistency 和 legality。晋级测试另按候选/最优模型分开显示决策指标，不显示训练 loss/梯度。

`state.json` 在采样和测试期间每局更新，测试 tab 每局更新；刷新可查看进展，URL `#gatekeeper` 保留当前 tab。完整测试结束前 `promoted=null`，看板显示“进行中”。控制台输出逐局进度和每 100 steps 的训练进度。看板和 `state.json` 同时列出当前 `bestModel`、`latestModel`；每轮指标的 `selfplayModel` 是采样模型，`sourceModel` 是续训起点，`model` 保留旧字段兼容性，等于 `latestModel`。

| 指标 | 计算口径 |
| --- | --- |
| 总 game / rows | 所有成功入库对局和样本的累计数量；同时显示本轮数量 |
| per-game 轮次数 | 引擎终局 `round` 的 mean/P10/P50/P90，百分位线性插值；不是原子动作数 |
| P1 / P2 / draw | P1 为先手座位、P2 为后手座位；按引擎最终排名（包含动物数量破同分）判断；与原始玩家 ID 无关 |
| root 宽度 | 合法根动作数与已访问根动作数分别记录，按实际原子决策平均 |
| 平均分支深度 | 本次新增模拟到新叶/终局的路径长度，单位为原子动作；先在一次 search 内平均，再在本轮实际决策上平均 |
| 最大分支深度的平均 | 每次 search 的新增模拟路径最大深度，再在本轮平均；不是保留树全部节点的结构深度 |
| root value | 分别记录 search / network utility 的均值、总体 std 等；均为当前行动玩家视角 |
| policy | 加根噪声前、合法 mask 后的 NN 分布，以及搜索监督分布，分别统计 top1 probability、top5 mass、entropy（自然对数） |
| search − network | 搜索根平均回传 utility 减 NN utility，记录 signed mean/std、meanAbs、RMS 等 |
| as-root − inherited | 复用子节点提升为根后搜索的平均回传 utility，减提升前该子节点的平均回传 utility；仅统计继承 visits > 0 的子节点，同时记录有效 count；无继承不记零 |
| 各参数 gradnorm | 每个 `named_parameters()` 张量的梯度 L2 norm，裁剪前统计 mean/max/last；另有全局裁剪前后范数 |
| value label/predict avg | 三个 head 均按 `[自己, 对手]` 分开记录，使用原始模型输出尺度；按实际训练行数加权，预测为对应 step 更新前的预测 |
| 各 head loss | policy 为合法 mask 交叉熵，value 为 MSE；按实际训练行数加权 |
| legality | 搜索扩展时检查动作 ID 一一映射、范围；实际转移由规则引擎检查；收集与训练检查 target 和 mask 一致性。记录检查数量及异常次数 |

`utility = rank + 0.25*relative + 0.05*absolute`，沿用当前模型搜索目标；两个得分 head 的单位为分数 `/100`。搜索根均值为 `sum(edge.sum) / sum(edge.visits)`，包含复用 visits。原 `SearchResult.value` 为缓存 NN 预测，继续保持其原语义；新增 `diagnostics.searchValue` 才是搜索回传值。

继承值使用子节点自己的边统计，不使用父边 Q：父边通常还包含创建子节点时的一次 NN bootstrap，直接比较父边会多混入一个样本。

非法动作或 mask 冲突会记录至 `errors.jsonl` 并终止当前轮，不会静默跳过、伪造完成样本或继续训练。`state.json`/大盘显示失败和累计错误数。非零的非法动作 logits 本身不是冲突，合法 mask 排除它们是正常行为。

## 文件与恢复

```text
config.json              固定实验配置
state.json               当前阶段/轮次/逐局进度/错误
metrics.json             全部已完成轮次的汇总
dashboard.html           本地可视化大盘，无外部资源
replay.sqlite3           压缩样本、逐局统计与入库记录
initial.pt/.json         初始模型及优化器/TS 推理权重
iteration-0001/
  games.jsonl            本轮完整原始轨迹
  collection.log         TS 收集日志
  errors.jsonl           发生异常时创建
  sampled-rows.i64       训练实际消费的 row ID 顺序
  training.jsonl         每一步 loss/梯度范数/标签和预测均值
  training-result.json   训练阶段完成标记，包含输入和输出 checkpoint 哈希
  model.pt/.json         本轮训练后的模型（warmup 轮没有）
  gate-games.jsonl       晋级测试完整对局与双方决策诊断
  gate.log              晋级测试 CLI 进度
  gate-errors.jsonl      晋级测试异常（发生时创建）
  gatekeeper.json        晋级测试结果和指标，逐局刷新
  metrics.json           训练与晋级测试汇总，也是整轮完成标记
```

入库按完整对局事务提交，以轮次/局序号及内容摘要去重。采样中断后继续尚未完成的局数，环境与搜索 seed、先手交替和未中断运行一致。只有末尾未写完的 JSONL 记录会移除；完整但损坏的记录直接报错。

训练中途没有阶段完成标记时，从上一轮最新模型及 Adam 状态重做本轮 400k 行；抽样 seed 固定，已采集的数据不重跑、不重复入库。训练完成后写入 `training-result.json`，恢复时核验源模型、窗口、配置和候选 `.pt/.json` 哈希，一致则跳过训练。

晋级测试中断后保留已完成对局，按原 seed/先后手继续剩余局数；不会重新训练或重复测试已完成对局。模型与全部测试指标写完后才提交整轮完成标记；未完成测试不会提前更新最优模型。run 目录文件锁阻止两个调度进程并发修改同一个 buffer。

## 验证

```bash
npm test
npx tsc -p tsconfig.rl.json --noEmit
.venv-rl/bin/python -m unittest discover -s rl -p 'test_*.py' -v

# 少量真实对局，多轮训练/恢复；正式预算是上述默认值
.venv-rl/bin/python rl/loop.py --out rl/runs/loop-smoke --iterations 2 \
  --games-per-iteration 2 --simulations 4 --train-rows 1000 \
  --min-rows 100 --taper-scale 50 --hidden 16 16 --gate-games 4 --gate-simulations 8
```

覆盖窗口公式、逐遍无放回、400k/64 恰好 6250 batches、回放张量与原训练入口一致、真实多轮模型更替、统计聚合、warmup、采样断点恢复、训练断点重跑一致性，以及搜索回传/NN/继承值与实际树统计的一致性。

Gatekeeper 测试另覆盖恰好 50%、平局半胜、低于 50% 拒绝、未完成不晋级、拒绝后采样/续训模型分离、serving 动作与现有 agent 一致、交换先后手的模型身份归属、测试断点恢复不重训，以及旧 run 兼容。

2026-10-14 实测：86 项 TypeScript 测试、10 项 Python 测试、生产构建和 RL TypeScript 类型检查通过。`rl/runs/loop-validation-20261014/` 使用默认模型、32 次模拟、默认窗口参数及每轮 400k 训练行，只把每轮对局数缩为 8，完成两轮：

| 轮次 | 新 rows | window | 训练 rows / steps | 自对弈耗时 | 训练与模型保存耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 707 | 707 | 400000 / 6250 | 6.58s | 14.68s |
| 2 | 754 | 1461 | 400000 / 6250 | 7.31s | 15.11s |

两轮所有样本都少于 100k，均立即训练，验证了启动阶段重复抽样的约定。逐行核对采样 ID、无放回边界、数量及 SHA-256，并从实际抽中行独立重算 value label 均值，均一致；非法动作和 mask 冲突均为 0。另以缩小的 `min-rows=100/taper-scale=50` 跑过两轮，验证窗口截断与增长。上述耗时来自小数据集、CPU 单线程测试，不是完整 1000 局/轮的大窗口性能测量。

2026-10-15 Gatekeeper 验证：87 项 TypeScript 测试、14 项 Python 测试、生产构建和 RL TypeScript 类型检查通过。另在 `rl/runs/gatekeeper-validation-20261015/` 完成两轮真实运行：每轮采集 4 局、32 次模拟，网络缩为 `16 16`，其余保留默认窗口及 400k 训练行；每轮晋级测试使用完整 **100 局、每动作新增 128 次模拟**。

| 轮次 | 胜 / 平 / 负 | 晋级胜率 | 结果 | 测试原子决策数 | 测试对局耗时 |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | 61 / 0 / 39 | 61% | 晋级 | 9710 | 325.19s |
| 2 | 57 / 0 / 43 | 57% | 晋级 | 9602 | 333.14s |

从逐局记录独立核验了 50 个 seed 的先后手配对、模型哈希和身份、胜率与晋级指针、每次决策的 serving 配置和新增模拟数，以及原始搜索指标与汇总均值/标准差的一致性。最终 replay 仅有训练 self-play 的 **8 局、720 行**，未包含 200 局晋级测试；两轮累计优化器步数为 12500。非法动作和 mask 冲突均为 0。核验脚本和结果保存在 run 下的 `verify.py` / `verification.json`。

浏览器验证覆盖双 tab、六张测试趋势图、刷新后保留 tab、键盘切换、手机宽度无页面溢出，未发现 JavaScript 错误。此验证用于确认调度和统计正确性，不是默认 1000 局/轮、128×128 网络的正式训练或棋力结论；拒绝晋级后的模型分流由自动化测试单独覆盖。
