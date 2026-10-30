"""Rebuild the formal-training summary after independent audits; never train or change models."""
from pathlib import Path
import json, hashlib
from datetime import datetime

HERE=Path(__file__).resolve().parent
run=HERE/'run'
read=lambda p:json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
metrics=read(run/'metrics-verification.json');replay=read(run/'replay-verification.json');extra=read(run/'checkpoint-log-verification.json');ui=read(run/'ui-verification.json');state=read(run/'state.json')
assert all(x['passed'] for x in [metrics,replay,extra]) and ui['staticPassed']
assert replay['games']==21000 and replay['actions']==metrics['decisions']+metrics['gateDecisionRows']
rows=read(run/'metrics.json')['iterations'];process=read(run/'process.json');manifest=read(HERE/'manifest.json')
assert state['phase']=='complete' and state['completedIterations']==10 and process['exitCode']==0
items=[]
for r in rows:
    g=r['gatekeeper'];t=r['training'];s=r['selfplay']
    items.append({'iteration':r['iteration'],'newRows':s['rows'],'decisions':s['decisions'],'cheap':s['playoutCaps']['cheap'],'cumulativeRows':r['totals']['rows'],'windowRows':r['window']['rows'],'trainingRows':t['rows'],'steps':t['steps'],'loss':t['loss'],'gate':{k:g[k] for k in ['wins','draws','losses','winRate','promoted','meanCandidateScore','meanIncumbentScore']},'best':r['bestModel'],'selfplayTerminal':s['terminal'],'gateTerminal':g['telemetry']['terminal'],'gateCandidateTerminal':g['byModel']['candidate']['terminal'],'gateIncumbentTerminal':g['byModel']['incumbent']['terminal'],'selfplaySeconds':s['collectionSeconds'],'trainSeconds':t['seconds'],'gateSeconds':g['telemetry']['collectionSeconds'],'seconds':r['seconds']})
durations={key:sum(x[field] for x in items) for key,field in [('selfplaySeconds','selfplaySeconds'),('trainingSeconds','trainSeconds'),('gateSeconds','gateSeconds'),('iterationSeconds','seconds')]}
durations['processWallSeconds']=(datetime.fromisoformat(process['endedAt'])-datetime.fromisoformat(process['startedAt'])).total_seconds()
durations.update(selfplayGamesPerSecond=20000/durations['selfplaySeconds'],gateGamesPerSecond=1000/durations['gateSeconds'],trainingRowsPerSecond=metrics['trainingRowsConsumed']/durations['trainingSeconds'])
verification={'passed':True,'scope':'complete formal training data, rules, checkpoints and static dashboard checks','browserInteractionVerified':False,'browserLimitation':ui['browserLimitation'],'trainingGames':20000,'gateGames':1000,'actions':replay['actions'],'replayRows':metrics['replayRows'],'trainingRowsConsumed':metrics['trainingRowsConsumed'],'trainingSteps':metrics['trainingSteps'],'best':manifest['best'],'latest':manifest['latest'],'frozenSourceVerified':True,'initialAndAllCheckpointsVerified':True,'illegalMoves':0,'maskConflicts':0,'nonFiniteTrainingRecords':0,'reports':{name:{'sha256':digest(run/name)} for name in ['metrics-verification.json','replay-verification.json','checkpoint-log-verification.json','ui-verification.json']}}
write(run/'verification.json',verification)
summary={'formatVersion':1,'version':manifest['version'],'trainingStatus':'complete-verified','config':read(run/'config.json'),'architecture':manifest['architecture'],'trainingGames':20000,'gateGames':1000,'replayRows':metrics['replayRows'],'expensive':metrics['replayRows'],'cheap':metrics['decisions']-metrics['replayRows'],'expensiveFraction':metrics['expensiveFraction'],'trainingRowsConsumed':metrics['trainingRowsConsumed'],'trainingSteps':metrics['trainingSteps'],'selfplaySimulations':sum(r['selfplay']['simulations'] for r in rows),'gateSimulations':metrics['gateDecisionRows']*128,'promotedIterations':[x['iteration'] for x in items if x['gate']['promoted']],'best':manifest['best'],'latest':manifest['latest'],'duration':durations,'iterations':items,'anomalies':{'illegalMoves':0,'maskConflicts':0,'nonFiniteTrainingRecords':0,'clippedBatches':extra['clippedBatches'],'maximumGlobalGradnorm':extra['maximumGlobalGradnorm']},'verification':'run/verification.json','browserInteractionVerified':False,'benchmarkNotFormalTraining':True}
comparison=HERE.parent/'comparisons/v003-vs-v002-20261029/result.json'
if comparison.exists():
    c=read(comparison);assert c['status']=='complete' and c['verification']['passed']
    summary['independentComparison']={'path':'../comparisons/v003-vs-v002-20261029/result.json',**{k:c[k] for k in ['overall','byCandidateSeat','scoreRateConfidenceInterval','terminalMeans','elapsedSeconds']}}
write(HERE/'results.json',summary)
manifest.update(status='complete' if comparison.exists() else 'training-verified-comparison-running',completedIterations=10,results='results.json',iterations=items,trainingTotals={k:summary[k] for k in ['trainingGames','gateGames','replayRows','trainingRowsConsumed','trainingSteps']},duration=durations,formalTrainingVerified=True,browserInteractionVerified=False)
manifest['plannedTrainingConfigNote']='已按该配置完成并核验；冻结执行源码、启动清单和 patch 保存在 run 中。'
if comparison.exists():manifest['independentComparison']=summary['independentComparison']
write(HERE/'manifest.json',manifest)

lines=['## 正式 10 轮训练结果（2026-10-29–30）','','使用 seed 1729 从随机权重独立训练，未加载测速权重或旧 replay。5 组空间 kernel ×16 channel，主干 512→128，四个独立 tower64，共 2,900,467 参数。生产执行代码冻结在 `run/source/`，配置、patch 和逐文件 SHA256 均保留；启动 HEAD 含未提交实现，不能用 HEAD 单独复现。','','每轮 2000 局；每个原子动作独立 25% expensive（新增64模拟、根噪声0.25）/75% cheap（新增16、噪声0）；温度1、回合内复用树，只有 expensive 进入 replay。训练预算 min(400000,2W)，逐遍无放回，每行每轮最多2次；AdamW、batch64、lr0.001、CPU单线程。窗口参数 100000/50000/0.4/0.65 与 v002 相同。每轮100局 serving gate，50个配对种子，新增128模拟、噪声0、温度0；得分率≥50%（平局半胜）晋级，best采样/latest续训。','',f'完成 20,000 局 self-play、1,000 局晋级。共 {summary["expensive"]:,} 个 expensive、{summary["cheap"]:,} 个 cheap 决策（expensive {summary["expensiveFraction"]:.4%}）；训练消费 {summary["trainingRowsConsumed"]:,} 行、{summary["trainingSteps"]:,} 个更新步。**最终 best 是第9轮，latest 是第10轮。** 第10轮46胜0平54负，未晋级。','','### 逐轮样本、预算与晋级','','| 轮 | 新 rows | 累计 rows | Window | 训练消费 | Steps | Gate 胜/平/负 | 得分率 | 晋级 | 本轮秒数 |','| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |']
for r in items:
    g=r['gate'];lines.append(f'| {r["iteration"]} | {r["newRows"]:,} | {r["cumulativeRows"]:,} | {r["windowRows"]:,} | {r["trainingRows"]:,} | {r["steps"]:,} | {g["wins"]}/{g["draws"]}/{g["losses"]} | {g["winRate"]:.1%} | {"是" if g["promoted"] else "否"} | {r["seconds"]:.2f} |')
lines+=['','所有轮次均未触及400,000行上限，实际消费恰好为2W。跨轮仍可重复消费留在窗口内的样本。','','### 在线训练 loss','','| 轮 | Policy CE | Rank MSE | Relative MSE | Absolute MSE |','| --- | ---: | ---: | ---: | ---: |']
for r in items:lines.append('| '+str(r['iteration'])+' | '+' | '.join(f'{r["loss"][k]:.6f}' for k in ['policy','rank','relative','absolute'])+' |')
lines+=['','这是训练过程中按消费行数加权的在线 loss，既非训练结束后固定模型的拟合误差，也非独立测试 value 精度。relative/absolute 使用分数÷100的网络单位，MSE乘10000才是分²。状态与标签分布逐轮变化，不能单凭loss下降认定泛化改善。','','### 三项终局统计','','每局每位玩家等权；self-play 使用本轮训练前best，gate混合候选与此前best，不能用这些均分代替最终best的独立对战。','','| 轮 | SP分数 | SP入住动物 | SP完整牌 | Gate双方分数 | Gate入住动物 | Gate完整牌 | Gate候选/对手均分 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
for r in items:
    v=[r[group][k]['mean'] for group in ['selfplayTerminal','gateTerminal'] for k in ['score','settledAnimals','completedAnimalCards']];g=r['gate'];lines.append(f'| {r["iteration"]} | '+' | '.join(f'{x:.5f}' for x in v)+f' | {g["meanCandidateScore"]:.3f}/{g["meanIncumbentScore"]:.3f} |')
lines+=['','### 耗时与核验','',f'逐轮累计 {durations["iterationSeconds"]:.2f} 秒（{durations["iterationSeconds"]/3600:.3f} 小时）；监督进程墙钟 {durations["processWallSeconds"]:.2f} 秒。self-play {durations["selfplaySeconds"]:.2f} 秒（{durations["selfplayGamesPerSecond"]:.4f} 局/秒）；训练 {durations["trainingSeconds"]:.2f} 秒（{durations["trainingRowsPerSecond"]:,.1f} 行/秒）；gate {durations["gateSeconds"]:.2f} 秒（{durations["gateGamesPerSecond"]:.4f} 局/秒）。上述是正式完整流程计时，不能混用前面的同输入小规模跑图计时。','',f'全部21,000局、{replay["actions"]:,}个原子动作独立规则重放通过；无非法动作、mask冲突或非有限训练数值。所有状态模式/新增模拟/噪声、expensive样本映射、replay逐字节张量、gate隔离、窗口公式、消费ID次数、晋级胜率、best/latest哈希及三项终局统计已核验。冻结源码与初始加全部10轮JSON/PT架构、参数、权重完全一致性均通过。','',f'43,671个batch仅1个触发梯度阈值5裁剪：第2轮step9，裁剪前{extra["maximumGlobalGradnorm"]:.6f}，记录的裁剪后5.000191（浮点计算误差量级）；未调整配置或重跑训练。','','看板两个tab、10轮数据、expensive/cheap列、窗口/预算列及120个终局均值静态核对通过。浏览器工具安全策略拒绝本地file: URL，因此未确认浏览器渲染、交互或移动端布局；没有将静态核验标为浏览器通过。','','- [完整看板](run/dashboard.html) / [结构化结果](results.json) / [manifest](manifest.json)','- [全量核验](run/verification.json) / [独立指标核验](run/metrics-verification.json) / [规则重放](run/replay-verification.json)','- [checkpoint与日志核验](run/checkpoint-log-verification.json) / [UI静态核验及局限](run/ui-verification.json)','- [冻结源码清单](run/source-snapshot.json) / [配置](run/config.json) / [进程退出状态](run/process.json) / [完整性清单](artifacts.sha256)','', '### 与 v002 的独立对战','']
if comparison.exists():
    x=c['overall'];ci=c['scoreRateConfidenceInterval']['bounds'];lines+=[f'最终best第9轮 vs 固定v002 best第10轮，400局（200新种子交换先后手，serving新增128、noise0/temp0）：**{x["wins"]}胜/{x["draws"]}平/{x["losses"]}负，得分率{x["scoreRate"]:.2%}，配对seed bootstrap 95%区间{ci[0]:.2%}–{ci[1]:.2%}**。', '', '[完整独立对战报告](../comparisons/v003-vs-v002-20261029/README.md) / [结构化结果](../comparisons/v003-vs-v002-20261029/result.json)。相同模拟数不是相同耗时；单训练种子且同时改变空间特征、宽度和tower深度，不能分别归因。']
else:lines+=['400局独立对战正在进行；完成规则重放和配对bootstrap后再登记结果。']
section='\n'.join(lines)+'\n'
(run/'summary-section.md').write_text(section)
text=(HERE/'README.md').read_text();prefix=text.split('## 正式 10 轮训练')[0]
prefix=__import__('re').sub(r'\*\*状态：.*?\n\n','**状态：10轮训练已完成并通过全量数据核验；'+('400局独立对战已完成。' if comparison.exists() else '400局独立对战进行中。')+'**\n\n',prefix,count=1,flags=__import__('re').S)
(HERE/'README.md').write_text(prefix+section)
print(json.dumps({'trainingVerified':True,'comparisonComplete':comparison.exists(),'totals':manifest['trainingTotals'],'duration':durations},ensure_ascii=False))
