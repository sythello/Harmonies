"""Summarize an audited match without changing either experiment's best/replay."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'rl/gatekeeper.py').exists())
sys.path.insert(0, str(ROOT / 'rl'))
from gatekeeper import summarize_gate
from loop_metrics import atomic_json
from run_match import digest

run = HERE / 'run'
config = json.loads((run / 'config.json').read_text())
state = json.loads((run / 'state.json').read_text())
verification = json.loads((run / 'replay-verification.json').read_text())
games = [json.loads(line) for line in (run / 'games.jsonl').open()]
assert state['phase'] == 'complete' and verification['passed'] and len(games) == 100
assert digest(run / 'games.jsonl') == verification['gamesSha256']
for path, before in config['replaySha256Before'].items():
    assert digest(ROOT / path) == before
for model in (config['candidate'], config['opponent']):
    assert digest(ROOT / model['path']) == model['sha256']


def summary(rows):
    wins, draws, losses = (sum(g['result'] == result for g in rows) for result in (1, .5, 0))
    return {'games': len(rows), 'wins': wins, 'draws': draws, 'losses': losses,
            'scoreRate': (wins + draws / 2) / len(rows), 'strictWinRate': wins / len(rows),
            'meanV002Score': float(np.mean([g['candidateScore'] for g in rows])),
            'meanV001Score': float(np.mean([g['incumbentScore'] for g in rows])),
            'meanMargin': float(np.mean([g['candidateScore'] - g['incumbentScore'] for g in rows]))}


overall = summary(games)
seats = {name: summary([g for g in games if g['first'] == seat])
         for seat, name in enumerate(('first', 'second'))}
pair_means = np.array([np.mean([g['result'] for g in games[i:i+2]]) for i in range(0, 100, 2)])
rng = np.random.default_rng(20261029)
interval = np.percentile(rng.choice(pair_means, size=(20000, 50), replace=True).mean(axis=1), [2.5, 97.5]).tolist()
gate = summarize_gate(games, 100)
assert [gate[k] for k in ('wins', 'draws', 'losses')] == [overall[k] for k in ('wins', 'draws', 'losses')]
assert gate['winRate'] == overall['scoreRate']
atomic_json(run / 'gate-metrics.json', gate)
terminal = {}
for version, player in (('v002', 0), ('v001', 1)):
    terminal[version] = {key: float(np.mean([g[field][(player - g['first']) % 2] for g in games]))
                         for key, field in [('score', 'scores'), ('settledAnimals', 'settledAnimals'),
                                            ('completedAnimalCards', 'completedAnimalCards')]}
verification.update({'replayDatabasesUnchanged': True, 'modelsUnchanged': True,
                     'trainingAndGateSeedOverlap': config['trainingAndGateSeedOverlap'],
                     'outcomesMatchProductionGateSummary': True})
atomic_json(run / 'verification.json', verification)
result = {'formatVersion': 1, 'id': HERE.name, 'date': '2026-10-29', 'status': 'complete',
          'configuration': config, 'overall': overall, 'byV002Seat': seats,
          'scoreRateConfidenceInterval': {'level': .95, 'bounds': interval,
              'method': 'percentile bootstrap of 50 paired-seed means; 20000 resamples; RNG seed 20261029'},
          'terminalMeans': terminal, 'elapsedSeconds': state['elapsedSeconds'],
          'gamesPerSecond': len(games) / state['elapsedSeconds'],
          'sumGameSeconds': sum(g['collectionSeconds'] for g in games),
          'verification': verification,
          'source': json.loads((run / 'source.json').read_text()),
          'artifacts': {name: digest(run / name) for name in ('config.json', 'games.jsonl', 'source.json',
              'replay-verification.json', 'verification.json', 'gate-metrics.json')},
          'perGame': [{k: g[k] for k in ('seed', 'first', 'result', 'candidateScore', 'incumbentScore')}
                      for g in games]}
atomic_json(HERE / 'result.json', result)

lines = [
    '# v002 vs v001：100 局独立对战（2026-10-29）', '',
    '[返回版本索引](../../README.md)', '',
    f'**v002：{overall["wins"]} 胜、{overall["draws"]} 平、{overall["losses"]} 负，得分率 {overall["scoreRate"]:.1%}。** '
    f'按配对种子 bootstrap 的 95% 区间为 {interval[0]:.1%}–{interval[1]:.1%}。', '',
    '点估计有利于 v002，区间包含 50%，这 100 局尚不足以确认其稳定强于 v001。按现有 gatekeeper 的 50% 阈值会通过；这里只做版本对比，不修改任何 best/latest。', '',
    '## 配置', '',
    '- 双方均为各自第 10 轮结束时的 best，即 `iteration-0010/model.json`；精确哈希见 result.json。',
    '- A 面、两人、无自然之灵。直接调用生产 `scripts/rl.ts gate` → `evaluateGame` → serving `mctsAgent`。',
    '- 每个原子动作新增 128 次 MCTS 模拟，根 Dirichlet noise 为 0，温度为 0，回合内复用树；其余 PUCT/value 权重沿用生产配置。',
    '- 50 个独立环境种子，每个交换先后手，共 100 局。种子为 2026102900–2026102949，与双方训练及历次晋级种子无重叠。',
    '- v002 固定 player ID 0、v001 ID 1；每局按实际座位映射。总分相同时按引擎入住动物数决胜，名次仍相同才计平局。',
    '- 四个工作进程并行、各自负责不重叠的成对种子。完整轨迹仅保存在本对比目录，不进入训练 replay。', '',
    '## 结果', '', '| v002 顺位 | 局数 | 胜 / 平 / 负 | 得分率 | v002 均分 | v001 均分 | 平均分差 |',
    '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
for label, data in [('先手', seats['first']), ('后手', seats['second']), ('合计', overall)]:
    lines.append(f'| {label} | {data["games"]} | {data["wins"]} / {data["draws"]} / {data["losses"]} | '
                 f'{data["scoreRate"]:.1%} | {data["meanV002Score"]:.2f} | {data["meanV001Score"]:.2f} | {data["meanMargin"]:+.2f} |')
lines += ['', '得分率 = (胜 + 0.5 × 平) / 局数；本次无平局，因此与胜率相同。区间以 50 个种子对为抽样单位，保留同种子两局的相关性；重复抽样 20000 次。', '',
          '| 模型 | 平均终局分数 | 平均入住动物 | 平均完整动物牌 |', '| --- | ---: | ---: | ---: |']
for version, values in terminal.items():
    lines.append(f'| {version} | {values["score"]:.2f} | {values["settledAnimals"]:.2f} | {values["completedAnimalCards"]:.2f} |')
lines += ['', '## 核验与复现', '',
          f'100 局、{verification["actions"]:,} 个原子动作全部按真实规则独立重放通过；每个动作实际新增 128 模拟。非法动作及 mask 冲突记录均为 0。双方模型和 replay 数据库的执行前后 SHA256 相同。', '',
          f'对战采集墙钟耗时 {state["elapsedSeconds"]:.2f} 秒（4 个工作进程，含启动及轮询，不含事后核验）。', '',
          '- [结构化结果与逐局简表](result.json) / [完整对局](run/games.jsonl)',
          '- [独立核验](run/verification.json) / [生产 gate 汇总指标](run/gate-metrics.json)',
          '- [运行配置](run/config.json) / [执行代码指纹](run/source.json)', '',
          '仓库根目录复现：', '', '```bash',
          '.venv-rl/bin/python rl/experiments/10-rounds/comparisons/v002-vs-v001-20261029/run_match.py',
          'node --import tsx rl/experiments/10-rounds/comparisons/v002-vs-v001-20261029/verify.ts',
          '.venv-rl/bin/python rl/experiments/10-rounds/comparisons/v002-vs-v001-20261029/summarize.py', '```', '',
          '采集脚本保留已完成记录并支持恢复；已有完整结果不会重新对战。`run/` 为本地原始产物，Git 只保存脚本、本文和 result.json。', '',
          '本次按用户要求使用晋级测试的 100 局口径；版本索引中预定的至少 400 局基准评测仍未执行。单次训练种子、不同采样/训练量等因素仍限制因果解读，不能把结果完全归因于 playout cap randomization。', '']
(HERE / 'README.md').write_text('\n'.join(lines))
print(json.dumps({'overall': overall, 'byV002Seat': seats, 'confidenceInterval': interval,
                  'terminalMeans': terminal, 'verificationPassed': verification['passed']}, ensure_ascii=False, indent=2))
