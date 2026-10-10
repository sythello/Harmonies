# NN + RL 初版

支持 A 面、2 人、无自然之灵。PyTorch 训练，TypeScript 执行 FFN 推理、MCTS、真实规则转移与对局；不维护第二套游戏规则。入口为现有 `AIObservation → Action`，逐个原子动作决策。目录：

| 文件 | 职责 |
| --- | --- |
| `src/game/ai/rl/encoding.ts` | 特征、固定动作编号、版本化 schema |
| `src/game/ai/rl/network.ts` | 导出权重校验与 FFN 推理 |
| `src/game/ai/rl/mcts.ts` | PUCT、随机补充、价值回传、策略目标及选招 |
| `src/game/ai/rl/trajectory.ts` | 完整自博弈、终局标签、交换先后手评测 |
| `src/game/ai/rl/worker.ts` | 浏览器模型初始化及逐动作搜索接口 |
| `scripts/rl.ts` | 收集、评测、单次决策、HTTP serving |
| `rl/train.py` | 网络、损失、训练、验证、checkpoint 与 JSON 导出 |

## 快速运行

要求 Node.js 22+、Python 3.10+。所有命令在仓库根目录执行；生成物放入已忽略的 `rl/runs/`。

```bash
npm install
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install -r rl/requirements.txt

# 随机初始化，生成 initial.pt 和 initial.json
.venv-rl/bin/python rl/train.py init --out rl/runs/initial

# 真实完整对局；每个原子动作 128 次模拟；默认根噪声 0.25、选招温度 1
npm run rl -- collect --model rl/runs/initial.json --games 20 --simulations 128 --seed 1729 --output rl/runs/iteration-0.jsonl

# 恢复网络和 Adam 状态，学习搜索分布及真实终局结果
.venv-rl/bin/python rl/train.py train --resume rl/runs/initial.pt --data rl/runs/iteration-0.jsonl --epochs 5 --out rl/runs/iteration-1

# --games 是种子数量，每个种子交换先后手，实际共 20 局
npm run rl -- evaluate --model rl/runs/iteration-1.json --opponent rl/runs/initial.json --games 10 --simulations 128 --seed 5000 --output rl/runs/evaluation-1.json

# 使用新模型继续收集；训练可传入多个历史文件作为 replay buffer
npm run rl -- collect --model rl/runs/iteration-1.json --games 20 --simulations 128 --seed 2000 --output rl/runs/iteration-1.jsonl
.venv-rl/bin/python rl/train.py train --resume rl/runs/iteration-1.pt --data rl/runs/iteration-0.jsonl rl/runs/iteration-1.jsonl --epochs 5 --out rl/runs/iteration-2

npm run rl -- serve --model rl/runs/iteration-2.json --simulations 128 --port 8765
```

无 checkpoint 时也可以显式用 `--model uniform` 收集首批数据；它返回均匀 policy 和零 value，是 bootstrap 基线。通常从 `init` 的小随机网络开始即可。`--hidden 64 64 --embedding-dim 8` 可以缩小模型；`--device cuda` / `--device mps` 可选择 PyTorch 后端，默认 CPU。

`collect` 向 JSONL 追加已完成的整局。中断或超过 512 个原子动作会报错，未完成局不输出终局标签。默认每批奇偶局交换先手；环境 seed 和搜索 seed 分离，后者每步递增 1009。数据量大时自行滚动文件，并在训练的 `--data` 中选择最近若干文件，初版不做分布式采样或自动模型晋升。

评测对手支持 `random`、`uniform`、另一个 JSON 模型，以及现有 `heuristic`。例如：

```bash
npm run rl -- evaluate --model rl/runs/iteration-1.json --opponent heuristic --heuristic-ms 100 --games 10 --simulations 128 --seed 5000
```

启发式对手每次请求调用当前 `searchTurn`，执行其第一步，下次再请求。其参数是回合搜索的软时间预算，经引擎约束为 100–30000 ms，实际配置保存在评测报告；NN 对手使用模拟次数预算，报告不代表相同 CPU 时间的公平强度对比。胜负遵循引擎的分数、动物数量破同分规则，完全相同算平局。`winRate=(wins+0.5*draws)/games`，另报绝对分及分差。

## 模型与输入

默认架构：

```text
1614 个数值特征 + 5 个市场牌 ID 的共享 8 维 embedding
→ concat (1654)
→ Linear(1654,128) + ReLU
→ Linear(128,128) + ReLU
├─ policy:   Linear(128,885)
├─ rank:     Linear(128,2) + tanh
├─ relative: Linear(128,2)
└─ absolute: Linear(128,2)
```

后三个 head 均输出 `[当前行动玩家, 对手]`，共享 FFN，head 参数分离。只有市场牌使用 embedding；零 ID 表示空市场槽位，对应 embedding 恒为零。默认模型共 343,555 个参数。具体输入：

- 双方版图按引擎 A 面 23 格固定顺序展开。14 种合法非空堆叠，每种分“未入住／已入住”，另加空格，共 29 类 one-hot。保留树／山高度及建筑底层，而非仅地形顶层颜色；入住动物的具体 ID 不影响后续放置或计分，计分所需完成数在卡牌特征中。
- 双方持牌分别使用固定 `32×3` 矩阵，第 i 行始终表示第 i 号动物牌，记录 `[是否持有, 入住数量, 总目标数]`。未持有为 `[0,0,0]`，已完成牌为 `[1,总目标数,总目标数]`；入住数量和总目标数均使用原始整数，不转成完成比例。卡牌 ID 由行位置确定，不使用持牌 embedding，不随拿牌或持牌数组重排而移动。额外记录空牌位 `/4`、已完成回合数 `/20`、当前分数 `/100`。
- 市场 5 张卡牌共享同一 embedding；5 组地形供应各使用 6 色计数 `/3`。组内顺序没有规则意义。
- 待放地形的 6 色计数 `/3`、是否已拿牌／拿片、轮次 `/20`、当前玩家在引擎先后手序列中的 one-hot、最终轮标记及最终轮数。
- 公开剩余地形计数 `/23`、牌堆数量 `/32`、32 张动物牌的未见 mask。用于建模补充概率；没有真实袋子／牌堆顺序。

数值特征总数为 `1334（双方版图）+192（双方持牌）+6（双方额外信息）+82（市场、回合、公开剩余资源等）=1614`。再拼接 `5×8=40` 维市场 embedding，得到 1654 维 FFN 输入。

`rl/schema.json` 由 TypeScript 的 `schema` 命令生成，Python 读取同一文件。当前 schema 为 **`harmonies-rl-v2`**；这次固定 ID 三特征编码替代了 v1 的排序持牌 embedding，旧 `.pt`、JSON 模型和轨迹均会被拒绝。需要重新 `init`、采样和训练，使用新的输出文件，不能把旧数据追加进去。改动特征语义、动作编号或缩放必须提升版本，不能只修改维度。测试会检查双方 schema 完全一致。

## 动作空间

不枚举整个回合的组合动作，不做启发式 top-k。完整合法集合来自现有 `legalActions`，softmax 只在合法动作上归一化：

| ID | 动作 |
| --- | --- |
| 0–4 | 拿第 i 组地形 |
| 5–9 | 拿市场第 i 张牌 |
| 10–147 | 放置：`10+(颜色-1)*23+格子` |
| 148–883 | 入住：`148+(卡牌ID-1)*23+格子` |
| 884 | 结束回合 |

放置用颜色编号而非会变动的 `pending` 下标；执行时使用合法动作枚举返回的真实 Action。相同颜色的多个片只保留一个动作。卡牌可与放片、入住交错；结束回合也属于搜索动作。

## Value 定义及回传

真实自博弈走到 `phase=ended` 后，为每个历史状态从当时行动玩家视角填入：

```text
rank     = [赢:+1 / 平:0 / 输:-1, 对手对应值]
relative = [(自己最终分-对手最终分)/100, 相反数]
absolute = [自己最终分/100, 对手最终分/100]
```

排名使用引擎 `ranking`，总分相同但动物更多时仍为胜。得分回归是线性的，不截断高分。搜索默认组合效用：

```text
U_i = 1.0*rank_i + 0.25*relative_i + 0.05*absolute_i
```

系数是本项目的初始选择，可通过 `SearchConfig` 修改；它同时优化胜负、分差和少量自身得分。`absoluteWeight=0` 可去掉绝对得分对选招的影响，绝对分 head 仍可作为辅助训练目标。

每条边保存其行动玩家的平均 U，叶节点预测先映射为双方固定座位效用，再逐边回传。同一玩家连续原子操作不换视角；换手也不把绝对分取负。对手节点最大化对手自己的效用。终局搜索叶直接使用规则结果；非终局叶使用 NN，不做额外随机走子 rollout。训练标签始终来自真实完整 rollout，而非这些 NN 叶预测。

## KataGo 参考与采用算法

固定参考 **KataGo v1.16.4 / `4b8de63bea2bd8790db96cd6f8daf86dc87be6f7`**。以下链接均为该提交，可按函数定位；本项目采用公式并针对 Harmonies 重写，没有引入 KataGo C++ 运行时。

| 需求 | 源码具体位置 | 本项目实现 |
| --- | --- | --- |
| PUCT 展开 | [`searchexplorehelpers.cpp`，`cpuctExploration`、`getExploreScaling`、`getExploreSelectionValue`，9–53 行](https://github.com/lightvector/KataGo/blob/4b8de63bea2bd8790db96cd6f8daf86dc87be6f7/cpp/search/searchexplorehelpers.cpp#L9) | `exploration` 及 `search` 的边选择；合法动作全展开，每次访问等权 |
| 未访问边的 FPU | [同文件 `getFpuValueForChildrenAssumeVisited`，265–322 行](https://github.com/lightvector/KataGo/blob/4b8de63bea2bd8790db96cd6f8daf86dc87be6f7/cpp/search/searchexplorehelpers.cpp#L265) | 采用父 NN 价值减去已访问 policy 概率质量平方根惩罚；相当于 parent NN 权重 1、loss proportion 0 |
| 自博弈根噪声 | [`searchhelpers.cpp`，`computeDirichletAlphaDistribution`、`addDirichletNoise`，78–149 行](https://github.com/lightvector/KataGo/blob/4b8de63bea2bd8790db96cd6f8daf86dc87be6f7/cpp/search/searchhelpers.cpp#L78) | 仅根节点混入 Dirichlet；初版用对称 alpha，省略 KataGo 按 policy 形状分配 alpha |
| 搜索 policy target | [`play.cpp`，`Play::extractPolicyTarget`，804–843 行](https://github.com/lightvector/KataGo/blob/4b8de63bea2bd8790db96cd6f8daf86dc87be6f7/cpp/program/play.cpp#L804) | 取搜索后的 play selection weights，归一化为浮点软目标；不把执行动作变成 one-hot |
| 修正访问次数 | [`searchresults.cpp`，`getPlaySelectionValues`，114–194 行](https://github.com/lightvector/KataGo/blob/4b8de63bea2bd8790db96cd6f8daf86dc87be6f7/cpp/search/searchresults.cpp#L114)；[`getExploreSelectionValueInverse`，58–78 行及 `getReducedPlaySelectionWeight`，229–263 行](https://github.com/lightvector/KataGo/blob/4b8de63bea2bd8790db96cd6f8daf86dc87be6f7/cpp/search/searchexplorehelpers.cpp#L58) | `selectionWeights` 反解 PUCT，去掉根据当前 Q 看来过量的探索访问，保留稳定参考动作 |
| 策略训练损失 | [`metrics_pytorch.py`，`cross_entropy`，10–11 行及 `loss_policy_player_samplewise`，55–59 行](https://github.com/lightvector/KataGo/blob/4b8de63bea2bd8790db96cd6f8daf86dc87be6f7/python/katago/train/metrics_pytorch.py#L10) | 对搜索软目标计算交叉熵；额外按引擎合法动作 mask logits |
| Serving 选招 | [`searchresults.cpp`，`getChosenMoveLoc`，572–596 行](https://github.com/lightvector/KataGo/blob/4b8de63bea2bd8790db96cd6f8daf86dc87be6f7/cpp/search/searchresults.cpp#L572)；[`searchhelpers.cpp`，`chooseIndexWithTemperature`，12–75 行](https://github.com/lightvector/KataGo/blob/4b8de63bea2bd8790db96cd6f8daf86dc87be6f7/cpp/search/searchhelpers.cpp#L12) | 同一组结合访问次数与 Q 的权重，按温度选动作；serving 默认温度 0、根噪声 0 |

令 `N=Σ_a N_a`、`P_a` 为合法归一化（根节点可加噪声）的 prior：

```text
C(N) = cpuct + cpuctLog * ln((N+cpuctBase)/cpuctBase)
E(N) = C(N) * sqrt(N+0.01)
FPU  = U_NN(s) - fpuReduction * sqrt(Σ_已访问边 P_a)
选择 argmax_a [(N_a>0 ? Q_a : FPU) + E(N)*P_a/(1+N_a)]
```

默认 `cpuct=1.25`、`cpuctLog=0.5`、`cpuctBase=19652`、`fpuReduction=0.2`。自博弈根 prior 为 `0.75P+0.25η`，`η~Dirichlet(10/合法动作数)`。这些是可调整初始参数，不声称最优。

搜索结束后，访问最多附近的稳定参考动作 `b` 由 `max(0,N_b-1)+2P_b` 最大确定，只允许已访问动作作参考。令：

```text
T = Q_b + E(N)*P_b/(1+N_b)
w_b = N_b
w_a = N_a                                      若 T <= Q_a
w_a = min(N_a, ceil(max(0,E(N)*P_a/(T-Q_a)-1)))  其余已访问动作
w_a = 0                                        未访问动作
π_a = w_a / Σw
```

`π` 是训练监督。实际自博弈动作按 `w^(1/temperature)` 采样（默认温度 1）；serving 选最大 w，相同 w 时选较大 Q，再按引擎稳定枚举顺序。温度只作用于执行动作，不再次改变训练目标。访问次数已通过 PUCT 受 NN value 引导，修正权重又显式使用搜索平均 Q。

为保持最小实现，未实现 KataGo 的方差自适应 cpuct、LCB 提升、非等权 value aggregation、强制 playout、额外 chosenMoveSubtract/Prune、playout cap randomization、辅助 optimistic policy、多线程或跨请求树复用。上述差异不是“完整复刻 KataGo”。尤其本版没有强制 playout，所以没有相应的计数扣除；采用的是源码中的通用 retrospective weight reduction。

## 隐藏信息与随机补充

KataGo 的围棋没有市场抽样；这一部分是 Harmonies 的适配。真实游戏只在 `observe` 边界暴露公开信息。每次 MCTS 模拟独立调用 `sampleGame(view, seed)`，根据公开剩余计数均匀打乱不可见袋子与牌堆，使用引擎 `applyAction` 推进。

确定性动作共享同一子节点。`endTurn` 后以公开局面键区分不同补充结果，同一父边聚合所有抽样回报；补充结果本身不参与 PUCT 竞争。NN 始终只接收公开 observation。搜索既不固定一份“已知未来”，也不把不同随机补充错误合并成同一个子状态。抽样分支较多时，每个分支通常只到 NN 叶，属于初版的效率限制。

## 数据、损失与验证

每行 JSONL 是一局：schema、模型文件 SHA-256、环境 seed、first、完整搜索参数、动作序列、终局 scores/ranks，以及每个原子动作的数值输入、卡牌 ID、合法动作 ID、原始 visits、修正 policy、选中动作及三个终局 value target。模型、种子、规则版本相同时可以重放和重新搜索。`scores/ranks` 按游戏座位顺序；样本内所有 value 对按该状态行动玩家在前。

```text
L = CE(π, softmax(legal_logits))
  + MSE(rank, terminal_rank)
  + MSE(relative, terminal_relative)
  + MSE(absolute, terminal_absolute)
```

损失各项系数为 1。AdamW：默认 lr `1e-3`、weight decay `1e-4`，梯度范数裁剪 5，batch 64，默认 5 epochs。checkpoint 包含架构、schema、网络、optimizer 和更新步数；训练结束同时导出 JSON，TypeScript 无 PyTorch 运行时依赖。

加载数据时检查完整终局、特征维度、合法 mask、概率和为 1，以及 value 与终局结果一致。验证集按环境 seed 划分（默认约 10%，至少保留一个训练 seed），同一局或配对局的状态不会分散到训练／验证两边；只有一个 seed 时不报告验证指标。小数据集全量读入内存。验证损失和对局强度是两种不同评测，需另用保留种子跑对局。

## Serving 接入

HTTP 仅监听 `127.0.0.1`，`GET /health` 返回模型哈希和预算，`POST /action` 的 body 是 `observe(game)` 的 JSON。响应包含 `action`、`actionId`、根 NN `value`、各动作 prior/visits/q/selectionWeight/target、节点数和模拟数。`q` 是搜索平均组合效用；`value` 明确是根 NN 三个 head 的原始预测。

```bash
npm run rl -- act --model rl/runs/iteration-1.json --input observation.json --simulations 128
curl http://127.0.0.1:8765/action -H 'Content-Type: application/json' --data-binary @observation.json
```

也可直接在 TypeScript 调用：

```ts
import { observe } from "./src/game/ai/observation";
import { Network } from "./src/game/ai/rl/network";
import { search } from "./src/game/ai/rl/mcts";
import { applyAction } from "./src/game/engine";

const network = new Network(exportedWeights);
const result = search(observe(game), network, { simulations: 128, seed: 1729 });
game = applyAction(game, result.action);
// 使用真实补充后的 game 再请求下一步，不执行抽样未来的回合计划。
```

浏览器使用新增 Worker：先 `postMessage({weights})` 等待 `{ready:true}`，随后 `postMessage({id,view:observe(game),config})` 接收 `{id,result}`。它返回单步 Action，不冒充原启发式搜索的 `TurnPlan` 或 `AIDecisionLog`。当前游戏 UI 的设置面板尚未增加 NN 模型上传／切换控件，可通过这些 API 接入；原默认对局仍由现有启发式 AI 执行。

## 已执行验证

旧 v1 编码的 128×128 模型小规模采样、训练速度及完整产物审计见 [ai-rl-benchmark.md](ai-rl-benchmark.md)，不代表 v2 的实测速率。可用 `rl/benchmark.py` 对当前版本重新测试；采样 CLI 和训练 CLI 会输出结构化分阶段计时。

```bash
npm test
npm run build
npx tsc -p tsconfig.rl.json
.venv-rl/bin/python rl/test_pipeline.py
```

新增测试覆盖固定卡牌 ID 的三特征、拿低 ID 牌不移动已有行、持牌顺序不变性、已完成牌保留、旧 schema 拒绝、合法 mask、动作编号、隐藏顺序不可见、不同补充结果的节点隔离、连续操作与换手价值回传、绝对分不取负、终局同分规则、policy weight 修正、完整轨迹重放、截断拒绝、真实采样→训练→导出，以及双方推理一致性（rtol `2e-5`、atol `2e-6`）。PyTorch 测试验证所有 head、共享层及市场牌 embedding 更新，并验证小批次能拟合、非法动作 logits 不获得 policy 梯度。

v2 使用默认 128×128 模型完成了两局采样（177 条样本、每步 8 次模拟）、两轮训练、断点续训一轮，以及训练后模型再次采样两局。产物保存在本地 `rl/runs/schema-v2-smoke/`，可用以下命令复现：

```bash
.venv-rl/bin/python rl/benchmark.py --output rl/runs/schema-v2-smoke \
  --games 2 --simulations 8 --epochs 2 --train-repeats 1 \
  --seed 63000 --threads 1 --device cpu
```

此前还跑通了配对评测和 HTTP 请求。生成的 smoke checkpoint 仅验证框架可运行，未进行足量强化学习或稳定强度测量；没有把短期 smoke 胜率当作模型能力结论。
