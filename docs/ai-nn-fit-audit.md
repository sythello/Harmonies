# 末回合 NN 拟合实验

此实验复用生产 `Model`、`losses`、JSON 导出、`Network`、`SearchTree`；独立实验代码位于 `scripts/audit/nn-fit.ts`、`scripts/nn-fit.ts` 和 `rl/fit_audit.py`。不修改生产搜索或训练的默认行为。

## 数据与真值

1. A 面、2 人、无自然之灵。500 局随机自对弈，环境种子 102000–102499，先手身份交替。每步在完整合法原子动作集合上均匀随机。
2. 重放到每局最后一轮先手行动前，明确设置 `finalRound=round`。用精确搜索找到自己最终得分最大的整回合；在真实游戏的袋子、牌堆上执行该方案，再取后手此刻的局面，并同样优化。后手输入因此与原随机对局可能不同。真实推进到终局才生成双方三个 value 标签，排名含引擎动物数破同分规则。
3. 每局仅两个训练样本，均为回合起点，共 1000 个。按局、固定洗牌种子 1729 划分 400 局训练 / 100 局测试，即 800 / 200 样本。同局两个视角不会跨集合。

精确搜索使用生产 PUCT 的 `exploration` 公式与默认 cpuct/FPU，均匀 prior、当前己方得分 `/100` 作为叶估值，回合结束即终点。前 512 次模拟允许重复访问终点；之后不再选择已证明完整的分支，直到根下每条合法边都得到精确的后续最高分。中间状态按完整的己方回合规则状态合并为 DAG；每个动作仍被证明，且记录所有首动作的最佳可达分数。该 DAG 共享统计，因此它的前 512 次访问并不要求与生产不合并状态的树逐项一致。

合并键包括版图、持牌及完成数、供应、市场、拿牌/拿片标记和待放片；同色片排列与持牌顺序不影响规则。保存颜色/卡牌 ID 编码的动作，重放时重新解析实际 pending 下标。隐藏补充只在结束回合发生，不改变己方本轮最终分数。超时直接失败，不输出未经证明的标签。

Policy target 按已确认的“模拟路径次数”口径：

```text
N_target(a) = 前 512 次模拟中经过根边 a 的次数 + 512 * I(a 属于选定最优路径)
policy(a) = N_target(a) / 1024
```

每条最优路径上的父边也得到额外 512 次访问；`boostedPath` 记录每个父边的前缀访问与追加数，`prefixPaths` 保留 512 条实际路径。训练只使用根部 target，不把中间节点变成额外样本。并列最优时按引擎稳定动作顺序选一条用于加权，命中指标则接受任一同分最优路径。此自定义监督不再应用生产 policy 的访问修正，以保持指定的 512+512 分配。

`dataset.jsonl` 是每局两条监督记录的稀疏数据集，带完整终局 scores/ranks，兼容现有训练数据校验；其中 `actions` 仅对应两条样本的首动作，完整末回合行动单独存于 `finalActions`。原随机对局完整动作保存在 `random-games.json`。

## 训练和逐 epoch 评测

从固定随机初始化开始，标准 128×128 FFN、共享 8 维市场牌 embedding，343,555 参数。50 epochs，batch 64、CPU 单线程、AdamW lr 0.001 / weight decay 0.0001、梯度裁剪 5。四个损失等权：合法 mask 后 policy 软交叉熵及三个 value MSE。每个 epoch 完成后，使用当时固定的权重重新评测全部训练/测试样本。

| 指标 | 定义 |
| --- | --- |
| policyHit | 合法动作内 NN argmax 的首动作可到达任一全局最优整回合 |
| policyCE / KL | 与 1024 访问软目标的交叉熵 / KL；KL 去掉 target 本身的熵 |
| servingFirstHit | 生产 MCTS 选择的首动作仍可达到最优分 |
| servingHit | 实际逐步运行生产 MCTS 到回合结束，最终得分等于穷举最优分 |
| servingRegret | 穷举最优分 − serving 实际分数 |
| greedyTurnHit / Regret | 额外对照：每一步直接选合法 policy argmax，完整执行一回合 |
| value exact | rank 取最近 {-1,0,1}；relative/absolute 还原 ×100 后四舍五入为整数，和真值比较 |
| exactOwn / Opponent / Both | 己方、对手、两个输出同时精确命中；`exact` 是两个输出平均 |
| value MSE | 原始网络输出的 MSE，分数头在 `/100` 单位；`msePoints` 再乘 10000，另存双方分项 |

Serving 每个原子动作新增 128 次模拟，回合内复用树，噪声和温度均为 0；种子为游戏 seed+1000000，每个原子决策加 1009。使用生产默认综合效用 `rank + .25*relative + .05*absolute`，不向 NN 或搜索注入精确答案。精确基准只最大化自身得分，两者目标不完全相同：先手拿走哪些资源可能影响对手终局。每个 epoch 还逐样本检查 PyTorch 与 TS 的 value 预测误差和 policy argmax 一致性（允许极小浮点误差导致的并列）。

只监督回合起点意味着中间节点的 policy/value 没有训练标签；policy 首动作命中率高，不能推出整回合策略好。测试集反映不同随机局面的泛化能力，不能用训练命中率替代。得分 exact 是精确到 1 分的离散评测，和归一化 MSE 同时看。

## 运行与输出

```bash
npm run rl:audit-fit -- collect --out rl/runs/nn-fit-20261012 --games 500 --test-games 100 --seed 102000 --workers 4
npm run rl:audit-fit -- verify --out rl/runs/nn-fit-20261012 --workers 2
.venv-rl/bin/python rl/fit_audit.py --out rl/runs/nn-fit-20261012 --epochs 50 --workers 4
# 独立的容量对照：前 16 个训练样本，全 batch 1000 步，750 步后学习率降至 1/10
.venv-rl/bin/python rl/fit_audit.py --out rl/runs/nn-fit-20261012 --tiny-overfit --seed 1730
```

`verify` 使用既有独立 DFS 检查全部 1000 局面、每个合法首动作的最佳可达分，再核对全部并列最优动作集合；每个最优方案再用独立计分实现核对。单测用不合并状态的朴素穷举核查所有首动作最优值、转置状态、512+512 分配、小于 512 个节点的情况、超时拒绝标签及双方真实时序。

输出均在忽略的 `rl/runs/` 目录：

- `report.html`：无外部依赖的曲线与逐 epoch 完整指标表。
- `epochs.csv` / `epochs.json`：全部逐 epoch 聚合数据，包括 value 每个输出分项。
- `evaluations/epoch-NNN.json`：全部样本的 TS 预测、所选动作序列、得分、命中、复用访问数和耗时。
- `models/epoch-NNN.pt` / `.json`：每个 epoch 的 checkpoint 与实际评测权重。
- `dataset.jsonl`、`games/`、`random-games.json`：输入、真值、搜索路径及完整可重放来源。
- `manifest.json`、`training-config.json`、`collection.json`、`verification.json`、`timing.json`：协议、配置、速度与核查结果。

实验读入全部数据后运行；重复执行会从头重跑并覆盖同名输出，需要新实验时指定新目录。

为减少逐 epoch 的搜索评测时间，生产 TS FFN 跳过 one-hot 与 ReLU 输出中的零值乘法，保留非零项求和顺序。全量 1000 个输入上，与原密集求和实现的所有 logits / value 逐项完全一致；独立单测与 PyTorch 导出一致性测试继续覆盖此性质。输入、模型、搜索预算及监督信号没有改变。

## 本次结果（50 epochs）

本地完整报告：[report.html](../rl/runs/nn-fit-20261012/report.html)；逐 epoch 数据：[CSV](../rl/runs/nn-fit-20261012/epochs.csv)、[JSON](../rl/runs/nn-fit-20261012/epochs.json)。全部 50 个 checkpoint、50,000 次逐局面 serving 结果和精确监督数据均已保存；报告是第 50 轮结果，不按测试集选择最佳模型。

| 指标 | 训练（800 样本） | 测试（200 样本） |
| --- | ---: | ---: |
| Policy 首动作最优命中 | 100.00% | 32.50% |
| Serving 首动作最优命中 | 99.25% | 33.50% |
| Serving 整回合最优命中 | 10.75% | 4.50% |
| Serving 平均失分 | 7.8425 | 9.7350 |
| 逐步贪心 policy 整回合最优命中 | 2.375% | 0.00% |
| Policy KL | 0.001228 | 0.550977 |

| Value head | 训练 exact | 测试 exact | 训练 MSE | 测试 MSE |
| --- | ---: | ---: | ---: | ---: |
| rank | 99.3125% | 56.25% | 0.00580656 | 1.00813007 |
| relative | 36.00% | 4.00% | 0.000119644 | 0.011837104 |
| absolute | 31.25% | 4.00% | 0.000161201 | 0.007010388 |

Value exact 为己方/对手两个输出的平均准确率。分数 MSE 使用原始网络 `/100` 单位；换成分²，relative 为 **1.1964 / 118.3710**，absolute 为 **1.6120 / 70.1039**（训练 / 测试）。完整输出另存己方、对手及同时 exact。

正确性核查：1000 个输入的每个合法首动作、所有并列最优动作均与独立 DFS 一致；完整重放 500 局的 46,218 个动作，输入与终局 value 标签全部一致。50,000 次跨语言根部预测的最大 value 差异为 `7.14e-7`，policy argmax 无不一致。测试集与训练集无相同特征输入。83 个 TypeScript 测试、2 个 PyTorch 流程测试和 2 个指标测试通过，构建与类型检查通过。

16 样本的独立容量对照中，policy 和三个 value head 均达到 100% exact，说明模型能拟合小数据。主实验中 policy 在第 13 轮首次达到训练集 100%，但测试最终 32.5%，均匀随机首动作基线约 25.77%，显示明显的泛化差距。

整回合弱还存在明确的监督范围限制：800 个训练输入只有拿地形的 5 个 policy 输出和入住的 86 个输出进入过合法 mask，**138 个放置输出全部没有 policy 监督梯度**；待放片与拿牌/拿片标记也始终为零。训练集 serving 首动作 99.25% 而完整回合只有 10.75%，因此根部拟合成功不能说明后续放置策略学会了。下一轮若要检验完整回合学习，应增加回合中间状态的 policy/value 监督；本次没有改变已确认的根部 1000 样本口径。

本机实测：随机 500 局 **4.64 秒**；4 worker 生成全部穷举标签 **191.27 秒**，最慢单局面 **12.23 秒**。50 epochs 纯梯度训练合计 **4.94 秒**（约 0.099 秒/epoch）；全量 serving 评测合计 **106.21 分钟**（平均 127.46 秒/epoch），整个训练评测循环 **106.62 分钟**。评测期间部分轮次与独立验证并行，前几轮使用推理优化前的实现；这些是本次实际运行耗时，不是隔离环境下的性能基准。搜索评测是主要耗时。
