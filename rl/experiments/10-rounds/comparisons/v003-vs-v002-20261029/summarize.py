"""Summarize an audited match without changing either experiment's best/replay."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'rl/gatekeeper.py').exists())
sys.path.insert(0, str(ROOT / 'rl/experiments/10-rounds/v003-spatial/run/source/rl'))
from gatekeeper import summarize_gate
from loop_metrics import atomic_json
from run_match import digest

run = HERE / 'run'
config = json.loads((run / 'config.json').read_text())
state = json.loads((run / 'state.json').read_text())
verification = json.loads((run / 'replay-verification.json').read_text())
games = [json.loads(line) for line in (run / 'games.jsonl').open()]
assert state['phase'] == 'complete' and verification['passed'] and len(games) == 400
assert digest(run / 'games.jsonl') == verification['gamesSha256']
for path, before in config['replaySha256Before'].items():
    assert digest(ROOT / path) == before
for model in (config['candidate'], config['opponent']):
    assert digest(ROOT / model['path']) == model['sha256']


def summary(rows):
    wins, draws, losses = (sum(g['result'] == result for g in rows) for result in (1, .5, 0))
    return {'games': len(rows), 'wins': wins, 'draws': draws, 'losses': losses,
            'scoreRate': (wins + draws / 2) / len(rows), 'strictWinRate': wins / len(rows),
            'meanCandidateScore': float(np.mean([g['candidateScore'] for g in rows])),
            'meanOpponentScore': float(np.mean([g['incumbentScore'] for g in rows])),
            'meanMargin': float(np.mean([g['candidateScore'] - g['incumbentScore'] for g in rows]))}


overall = summary(games)
seats = {name: summary([g for g in games if g['first'] == seat])
         for seat, name in enumerate(('first', 'second'))}
pair_means = np.array([np.mean([g['result'] for g in games[i:i+2]]) for i in range(0, 400, 2)])
rng = np.random.default_rng(20261029)
interval = np.percentile(rng.choice(pair_means, size=(20000, 200), replace=True).mean(axis=1), [2.5, 97.5]).tolist()
gate = summarize_gate(games, 400)
assert [gate[k] for k in ('wins', 'draws', 'losses')] == [overall[k] for k in ('wins', 'draws', 'losses')]
assert gate['winRate'] == overall['scoreRate']
atomic_json(run / 'gate-metrics.json', gate)
terminal = {}
for version, player in (('v003', 0), ('v002', 1)):
    terminal[version] = {key: float(np.mean([g[field][(player - g['first']) % 2] for g in games]))
                         for key, field in [('score', 'scores'), ('settledAnimals', 'settledAnimals'),
                                            ('completedAnimalCards', 'completedAnimalCards')]}
provenance=json.loads((run / 'source.json').read_text())
for path, expected in provenance['sourceSha256'].items():
    assert digest(ROOT / path)==expected, ('Execution source changed',path)
verification.update({'executionSourcesUnchanged': True, 'replayDatabasesUnchanged': True, 'modelsUnchanged': True,
                     'trainingAndGateSeedOverlap': config['trainingAndGateSeedOverlap'],
                     'outcomesMatchProductionGateSummary': True})
atomic_json(run / 'verification.json', verification)
result = {'formatVersion': 1, 'id': HERE.name, 'date': '2026-10-30', 'status': 'complete',
          'configuration': config, 'overall': overall, 'byCandidateSeat': seats,
          'scoreRateConfidenceInterval': {'level': .95, 'bounds': interval,
              'method': 'percentile bootstrap of 200 paired-seed means; 20000 resamples; RNG seed 20261029'},
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


interpretation = ('区间下界高于 50%，本组独立对战支持 v003 优于固定 v002。' if interval[0] > .5 else
                  '区间上界低于 50%，本组独立对战支持 v003 弱于固定 v002。' if interval[1] < .5 else
                  '区间包含 50%，本组对战尚不能确认两者存在稳定胜负优势。')
lines = [
    '# v003 vs v002：400 局独立对战', '', '[返回版本索引](../../README.md)', '',
    f'**v003：{overall["wins"]} 胜、{overall["draws"]} 平、{overall["losses"]} 负，得分率 {overall["scoreRate"]:.2%}。** '
    f'配对种子 bootstrap 95% 区间：{interval[0]:.2%}–{interval[1]:.2%}。{interpretation}', '',
    '## 配置', '',
    '- v003 使用 10 轮完成后的 best（第 9 轮）；v002 使用固定 best（第 10 轮）。JSON/PT 身份见版本 manifest，实际对战 JSON 哈希见 result.json。',
    '- A 面、两人、无自然之灵；调用 v003 run/source/ 中冻结的生产 gate CLI。双方每原子动作新增 128 次模拟，根噪声 0、温度 0、回合内复用树。相同模拟预算不代表相同推理耗时。',
    '- 200 个独立环境种子 3000001729–3000001928，每个交换先后手，共 400 局。已核对与双方 self-play 和各轮晋级种子无重叠。',
    '- v003 固定 player ID 0、v002 ID 1；每局按物理座位映射。总分相同时按入住动物数决胜，名次仍相同才计平局。',
    '- 4 个 worker 各负责 100 局（50 个完整种子对），完整记录独立存放，不进入 replay，不修改 best/latest。', '',
    '## 结果', '', '| v003 顺位 | 局数 | 胜 / 平 / 负 | 得分率 | v003 均分 | v002 均分 | 平均分差 |',
    '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
for label, data in [('先手', seats['first']), ('后手', seats['second']), ('合计', overall)]:
    lines.append(f'| {label} | {data["games"]} | {data["wins"]} / {data["draws"]} / {data["losses"]} | '
                 f'{data["scoreRate"]:.2%} | {data["meanCandidateScore"]:.3f} | {data["meanOpponentScore"]:.3f} | {data["meanMargin"]:+.3f} |')
lines += ['', '得分率 = (胜 + 0.5 × 平) / 局数。区间以 200 个种子对为抽样单位，保留同种子两局的相关性；重复抽样 20000 次，RNG seed 20261029。该区间只描述固定的两个训练模型在本对战设置下的不确定性，不包含重新训练的种子间方差。', '',
          '| 模型 | 平均终局分数 | 平均入住动物 | 平均完整动物牌 |', '| --- | ---: | ---: | ---: |']
for version, values in terminal.items():
    lines.append(f'| {version} | {values["score"]:.3f} | {values["settledAnimals"]:.4f} | {values["completedAnimalCards"]:.4f} |')
lines += ['', '## 核验与复现', '',
          f'400 局、{verification["actions"]:,} 个原子动作全部独立规则重放通过；每动作实际新增 128 模拟。非法动作、mask 冲突记录均为 0。双方模型和 replay 数据库执行前后的 SHA256 相同。', '',
          f'采集墙钟耗时 {state["elapsedSeconds"]:.2f} 秒（4 worker，含启动与轮询，不含事后核验）；逐局用时合计 {result["sumGameSeconds"]:.2f} 秒；吞吐 {result["gamesPerSecond"]:.4f} 局/秒。', '',
          '- [完整结构化结果及逐局简表](result.json) / [完整对局](run/games.jsonl)',
          '- [独立核验](run/verification.json) / [生产 gate 指标](run/gate-metrics.json)',
          '- [实际配置](run/config.json) / [执行代码指纹](run/source.json)', '',
          '仓库根目录执行（已完成的对局不会重复生成）：', '', '```bash',
          '.venv-rl/bin/python rl/experiments/10-rounds/comparisons/v003-vs-v002-20261029/run_match.py',
          'node --import tsx rl/experiments/10-rounds/comparisons/v003-vs-v002-20261029/verify.ts',
          '.venv-rl/bin/python rl/experiments/10-rounds/comparisons/v003-vs-v002-20261029/summarize.py', '```', '',
          '本实验同时增加空间特征、主干宽度和 tower 深度，不能将结果归因于其中某一项；也不能用在线训练 loss 代替独立测试的 value 预测精度。', '']
(HERE / 'README.md').write_text('\n'.join(lines))
print(json.dumps({'overall': overall, 'byCandidateSeat': seats, 'confidenceInterval': interval,
                  'terminalMeans': terminal, 'verificationPassed': verification['passed']}, ensure_ascii=False, indent=2))
