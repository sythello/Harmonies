"""Independent streaming aggregation of conditional rollout outcome variance."""
import csv
import html
import json
import math
import re
from pathlib import Path

import numpy as np

HEADS = ('rank', 'relative', 'absolute')


def summarize(rows):
    result = {'states': len(rows), 'sourceGames': len({r['gameIndex'] for r in rows}), 'heads': {}}
    for head in HEADS:
        values = np.array([[v['value'][head] for v in r['rollouts']] for r in rows], dtype=np.float64)
        prediction = np.array([r['prediction'][head] for r in rows], dtype=np.float64)
        means, variances = values.mean(1), values.var(1, ddof=1)
        for i, r in enumerate(rows):
            for seat in range(2):
                saved = r['statistics'][head][seat]
                assert math.isclose(saved['mean'], means[i, seat], abs_tol=1e-10)
                assert math.isclose(saved['variance'], variances[i, seat], abs_tol=1e-10)
        scale = 1 if head == 'rank' else 100
        result['heads'][head] = []
        for seat in range(2):
            x, m, v, pred = values[:, :, seat] * scale, means[:, seat] * scale, variances[:, seat] * scale**2, prediction[:, seat] * scale
            item = {'mean': float(m.mean()), 'meanWithinVariance': float(v.mean()),
                    'rootMeanWithinVariance': float(np.sqrt(v.mean())), 'meanWithinStd': float(np.sqrt(v).mean()),
                    'varianceP10': float(np.quantile(v, .1)), 'varianceP50': float(np.quantile(v, .5)),
                    'varianceP90': float(np.quantile(v, .9)), 'zeroVarianceFraction': float((v < 1e-12).mean()),
                    'betweenStateMeanPopulationVariance': float(m.var()), 'pooledPopulationVariance': float(x.var()),
                    'modelMSE': float(((x - pred[:, None])**2).mean()),
                    'mseAgainstRolloutMean': float(((m - pred)**2).mean()),
                    'meanWithinVarianceNormalized': float(variances[:, seat].mean())}
            n = values.shape[1]
            assert math.isclose(item['pooledPopulationVariance'], (n-1)/n*item['meanWithinVariance'] + item['betweenStateMeanPopulationVariance'], abs_tol=1e-9)
            assert math.isclose(item['modelMSE'], (n-1)/n*item['meanWithinVariance'] + item['mseAgainstRolloutMean'], abs_tol=1e-9)
            if head == 'rank':
                item['outcomeFrequency'] = {label: float((x == value).mean()) for label, value in [('win', 1), ('draw', 0), ('loss', -1)]}
            result['heads'][head].append(item)
    return result


def build_report(directory):
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    rows, source_games, actions, seconds = [], [], 0, 0
    for i in range(config['games']):
        folder = directory/f'game-{i:04d}'
        base = json.loads((folder/'base.json').read_text())
        source_games.append(base)
        with (folder/'states.jsonl').open() as stream:
            current = []
            for line in stream:
                r = json.loads(line)
                assert r['gameIndex'] == i and r['step'] == len(current)
                assert len(r['rollouts']) == config['repeats']
                actions += sum(len(v['actions']) for v in r['rollouts'])
                seconds += r['seconds']
                # Retain only values for summary; full action traces remain in the journal.
                r['rollouts'] = [{'value': v['value']} for v in r['rollouts']]
                current.append(r)
        assert len(current) == min(len(base['actions']), config['stateLimit'] or math.inf)
        rows.extend(current)
    groups = []
    for round_, seat in sorted({(r['round'], r['seat']) for r in rows}):
        group = summarize([r for r in rows if r['round'] == round_ and r['seat'] == seat])
        groups.append({'round': round_, 'seat': seat, **group})
    modes = {mode: sum(d['mode'] == mode for b in source_games for d in b['decisions'])
             for mode in ('expensive', 'cheap')}
    result = {'config': config, 'baseGames': len(source_games), 'states': len(rows),
              'continuations': len(rows)*config['repeats'], 'continuationActions': actions,
              'continuationSimulations': actions*config['simulations'], 'summedStateSeconds': seconds,
              'sourceModes': modes,
              'all': summarize(rows), 'groups': groups,
              'method': {'conditioning': 'public observation; independently resampled hidden deck/bag per continuation',
                         'perspective': '[starting active player, opponent], held fixed until terminal',
                         'variance': 'unbiased per-state sample variance (n-1); class mean weights states equally',
                         'units': 'rank in -1/0/1; scores in points, variances in points squared; normalized variance also stored',
                         'grouping': 'engine round and physical seat: 0=first player, 1=second player',
                         'policy': 'both players fixed same checkpoint, all-cheap 16/noise0/temp1, fresh initial tree and within-turn reuse',
                         'scope': f"{config['games']} source games; {'all' if not config['stateLimit'] else config['stateLimit']} atomic pre-decision states per game, including expensive and cheap states",
                         'caution': f"{config['repeats']} continuations estimate variance noisily; states within a source game are correlated; all-cheap targets differ from mixed self-play and serving targets"}}
    (directory/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (directory/'summary.csv').open('w') as f:
        writer = csv.writer(f)
        columns = list(groups[0]['heads']['relative'][0])
        writer.writerow(['round', 'seat', 'states', 'sourceGames', 'head', 'perspective', *columns])
        for g in groups:
            for h in HEADS:
                for seat, label in enumerate(('own', 'opponent')):
                    writer.writerow([g['round'], g['seat']+1, g['states'], g['sourceGames'], h, label, *[g['heads'][h][seat][c] for c in columns]])
    def table(headers, data):
        return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str, r))+' |' for r in data])
    body = '# Self-play 条件 Value 方差实验\n\n'
    body += f"使用固定模型 `{config['modelSource']}`；共 {len(source_games)} 局、{len(rows)} 个原子决策前状态、{result['continuations']} 次完整续局、{actions} 个续局动作。\n\n"
    body += '源对局按当前 self-play：25% expensive（64 模拟、根噪声 .25）/75% cheap（16 模拟、无根噪声），温度 1。每个状态另做 10 次全程 cheap 续局：双方相同模型、每动作新增 16 模拟、无根噪声、温度 1、回合内复用树；每次续局重新抽样隐藏袋子/牌堆并从新树开始。\n\n'
    body += '**统计口径：**先计算每个状态 10 次结果的样本方差（分母 9），再对组内所有状态等权平均。先/后手指当前行动方在该局的物理顺位。己方/对手视角固定为该起始状态，而非续局最终行动方。得分以分表示，得分方差以分²表示；rank 为 -1/0/1。\n\n'
    body += f"源对局实际包含 {modes['expensive']} 个 expensive 状态、{modes['cheap']} 个 cheap 状态，均纳入本实验。续局共新增 {result['continuationSimulations']:,} 次模拟。\n\n"
    if (directory/'verification.json').exists():
        verification = json.loads((directory/'verification.json').read_text())
        elapsed = verification['wallSeconds']
        body += f"总墙钟时间 {elapsed:,.2f} 秒（含源局、续局、独立核验与初版汇总），整体 {result['continuations']/elapsed:.2f} 次续局/秒、{actions/elapsed:.2f} 个续局动作/秒。全部动作独立重放及标签核验通过。\n\n"
    body += '## 全部状态\n\n'
    body += table(['目标', '结果均值', '平均状态内方差', '平均方差的平方根', '状态均值间方差', '模型对单次结果 MSE', '模型对10次均值 MSE'],
                  [[f'{h}/{name}', *[f"{result['all']['heads'][h][j][k]:.5f}" for k in ('mean','meanWithinVariance','rootMeanWithinVariance','betweenStateMeanPopulationVariance','modelMSE','mseAgainstRolloutMean')]] for h in HEADS for j,name in enumerate(('己方','对手'))])
    body += '\n\n## 按轮次及先后手分组\n\n'
    body += '每项依次列出平均方差、对单次结果的 MSE、对 10 次结果均值的 MSE；宽表可横向滚动。\n\n'
    group_targets = [('rank', 0, '排名（己方）'), ('relative', 0, '相对分（己方）'),
                     ('absolute', 0, '己方绝对分'), ('absolute', 1, '对手绝对分')]
    group_metrics = [('meanWithinVariance', '平均方差'), ('modelMSE', 'MSE（单次）'),
                     ('mseAgainstRolloutMean', 'MSE（均值）')]
    main_rows = [[g['round'], '先手' if g['seat']==0 else '后手', g['states'], g['sourceGames'],
                  *[f"{g['heads'][h][s][key]:.5f}" for h,s,_ in group_targets for key,_ in group_metrics]] for g in groups]
    headers = ['轮次','行动方','状态数','来源局数',
               *[f'{label}：{metric}' for _,_,label in group_targets for _,metric in group_metrics]]
    body += table(headers, main_rows)
    body += '\n\n每个指标均并列展示平均状态内方差、MSE（单次）和 MSE（均值）。单次 MSE：先对每个状态的 10 次终局标签分别计算预测平方误差并平均，再对组内状态平均；均值 MSE：先计算该状态 10 次标签的均值，再计算预测与该均值的平方误差，最后对组内状态平均。得分方差和得分 MSE 均为分²。\n\n'
    body += 'rank 和相对分的双方标签互为相反数，因此双方方差相同；表中这两项 MSE 使用起始状态己方 head 的预测，绝对分则分别展示双方 head。完整 JSON/CSV 包含全部 6 个输出的分组 MSE，以及均值、方差 P10/P50/P90、零方差状态占比和平均标准差。\n\n'
    body += '## 波动与预测误差\n\n'
    body += table(['目标','模型对单次结果 MSE','样本内波动项','模型对10次均值 MSE','波动项占比'],
        [[f'{h}/{name}', f"{a['modelMSE']:.5f}", f"{.9*a['meanWithinVariance']:.5f}",
          f"{a['mseAgainstRolloutMean']:.5f}", f"{.9*a['meanWithinVariance']/a['modelMSE']:.1%}" if a['modelMSE'] else '—']
         for h,j,name in [('rank',0,'己方'),('relative',0,'己方'),('absolute',0,'己方'),('absolute',1,'对手')]
         for a in [result['all']['heads'][h][j]]])
    body += '\n\n此处的波动项是 0.9 × 状态内样本方差，对应这 10 次结果相对其样本均值的平均平方偏差；占比是本次观测数据的误差分解，不能解释为已精确测得真实噪声下限。对样本均值的 MSE 也包含有限次采样的误差。\n\n'
    if len(groups) > 2:
        late = [g for g in groups if g['sourceGames'] >= len(source_games)/2][-1]
        rank = late['heads']['rank'][0]
        body += f"在来源局数不少于全部源局一半的最晚分组（第 {late['round']} 轮{'先手' if late['seat']==0 else '后手'}，{late['sourceGames']} 个源局、{late['states']} 个状态），排名平均方差为 {rank['meanWithinVariance']:.5f}，模型 MSE 为 {rank['modelMSE']:.5f}，对续局均值的 MSE 为 {rank['mseAgainstRolloutMean']:.5f}。这说明在本次 cheap 分布下，模型误差不能全部归因于目标随机性；本实验不单独判定欠拟合或过拟合。\n\n"
        last_round = max(g['round'] for g in groups)
        last_games = max(g['sourceGames'] for g in groups if g['round'] == last_round)
        body += f"第 {last_round} 轮仅有 {last_games} 个源局；末尾分组的小幅回升不宜视作稳定趋势。各轮参与统计的源局集合也可能不同，应结合来源局数理解变化。\n\n"
    body += '## 解读限制\n\n当前估计的是给定可见状态、固定模型和 cheap 续局策略后的结果波动，包含隐藏补充、搜索与选招随机性。它不能直接当作之前 serving 晋级对战或混合 self-play 的精确噪声下限。每状态仅 10 次，单状态方差估计较粗；同一源对局内状态相关。\n\n'
    body += '“平均状态内方差”和“状态均值间方差”分别表示同一输入的结果波动、不同输入的难度/预期结果差异。合并所有结果的总体方差 = 0.9 × 平均状态内样本方差 + 状态均值间总体方差。模型对单次结果 MSE = 0.9 × 平均状态内样本方差 + 模型对十次均值 MSE；后者仍包含有限次采样均值的误差。\n\n'
    body += '所有源对局和续局动作保留在 `game-*/base.json`、`game-*/states.jsonl`。独立规则重放为 `replay-verification.json`，汇总与方差恒等式核验为 `verification.json`。\n'
    (directory/'report.md').write_text(body)
    # Render our own small Markdown subset; the report needs no server or external assets.
    def inline(text):
        text = html.escape(text)
        return re.sub(r'`([^`]+)`', r'<code>\1</code>', re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text))
    parts = []
    for block in body.strip().split('\n\n'):
        if block.startswith('|'):
            lines = block.splitlines()
            cells = lambda line: [s.strip() for s in line.strip('|').split('|')]
            if cells(lines[0]) == headers:
                table_head = '<tr>'+''.join('<th rowspan="2">'+inline(x)+'</th>' for x in headers[:4])
                table_head += ''.join('<th colspan="3">'+inline(label)+'</th>' for _,_,label in group_targets)+'</tr><tr>'
                table_head += ''.join('<th>'+inline(metric)+'</th>' for _ in group_targets for _,metric in group_metrics)+'</tr>'
            else:
                table_head = '<tr>'+''.join('<th>'+inline(x)+'</th>' for x in cells(lines[0]))+'</tr>'
            table_class = ' class="group-metrics"' if cells(lines[0]) == headers else ''
            parts.append('<div class="scroll"><table'+table_class+'><thead>'+table_head+'</thead><tbody>'
                         +''.join('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in cells(line))+'</tr>' for line in lines[2:])+'</tbody></table></div>')
        elif block.startswith('#'):
            level = len(block) - len(block.lstrip('#'))
            parts.append(f'<h{level}>'+inline(block[level:].strip())+f'</h{level}>')
            if block == '## 按轮次及先后手分组' and (directory/'variance-by-round.png').exists():
                parts.append('<figure style="margin:16px 0"><a href="variance-by-round.svg"><img src="variance-by-round.png" style="width:100%;height:auto" alt="三个 value 目标的平均状态内方差随轮次变化；区分先手与后手"></a><figcaption>第 10 轮仅有 5 个来源局；点击图片可查看矢量图。</figcaption></figure>')
        else:
            parts.append('<p>'+inline(block)+'</p>')
    (directory/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Value 条件方差</title><style>body{font:15px system-ui;background:#f3f6f4;color:#20352c;margin:24px}main{max-width:1300px;margin:auto}p{line-height:1.8}code{overflow-wrap:anywhere}.scroll{overflow:auto}table{border-collapse:collapse;background:white;white-space:nowrap;width:100%}td,th{padding:12px;text-align:right;border-bottom:1px solid #dde5df}th{background:#e5eee8}.group-metrics td,.group-metrics th{padding:10px 6px;font-size:13px;font-variant-numeric:tabular-nums}.group-metrics th[colspan]{text-align:center;border-left:2px solid #ccd9d0}.group-metrics td:nth-child(3n+4){border-right:2px solid #ccd9d0}a{color:#246b50}h2{margin-top:36px}</style><main>'
        +f'<p><a href="report.md">Markdown 报告</a> · <a href="summary.csv">CSV</a> · <a href="summary.json">JSON</a> · 模型哈希 {config["modelHash"][:12]}</p>' + ''.join(parts)+'</main></html>')
    return result
