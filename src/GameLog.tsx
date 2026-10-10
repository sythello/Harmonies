import { Download } from "lucide-react";
import { CARDS, type Session } from "./game/engine";
import { downloadGameLog } from "./game/ai/log";
import type { CandidateLog, Evaluation } from "./game/ai/types";

const number = (n: number | undefined) =>
  n === undefined ? "—" : Number(n.toFixed(3));
const vector = (values: number[]) => values.map(number).join(" / ");
function Estimate({ value }: { value: Evaluation }) {
  return (
    <div className="estimate-detail">
      <p>
        当前 {number(value.actualScore.total)} + 潜力 {number(value.potential)}{" "}
        = <b>{number(value.value)}</b> · R={value.R} · B={value.B} · F={value.F}
        {value.L !== undefined && (
          <>
            {" "}
            · 预计第 {value.L} 轮结束 · 剩余回合 T={value.T}
          </>
        )}
      </p>
      {value.greedyValue !== undefined && value.ignoredCards?.length ? (
        <p>
          保留不使用{value.ignoredCards.map((id) => CARDS[id].name).join("、")}
          的计划：原贪心值 {number(value.greedyValue)} → {number(value.value)}。
        </p>
      ) : null}
      <p className="vector-note">向量顺序：水 / 灰 / 棕 / 绿 / 黄 / 红</p>
      <p>
        H=[{vector(value.H)}] · A=[{vector(value.A)}]<br />
        p=[{vector(value.p)}]
      </p>
      {value.projects.map((p, i) => (
        <details key={i}>
          <summary>
            项目 {i + 1} ·{" "}
            {p.project.kind === "animal"
              ? `卡牌 #${p.project.cardId} 入住格 ${p.project.targetCell! + 1}`
              : "放置地形"}{" "}
            · G={number(p.G)} · m={p.m} · W={number(p.W)}
          </summary>
          <p>
            {p.terrainGain !== undefined && (
              <>
                地形 {number(p.terrainGain)} + 动物 {number(p.animalGain!)} +
                自然之灵 {number(p.spiritGain!)} = G {number(p.G)}
                <br />
              </>
            )}
            B={p.B} · F={p.F} · q={number(p.q)}
            <br />
            m[c]=[{vector(p.mByColor)}]<br />
            d=[{vector(p.d)}]<br />
            p=[{vector(p.p)}]<br />
            H=[{vector(p.H)}]<br />
            A=[{vector(p.A)}]
          </p>
          <p>
            追加：
            {p.project.additions
              .map((a) => `格 ${a.cell + 1}：${a.colors.join("→")}`)
              .join("；") || "无需补建"}
          </p>
        </details>
      ))}
    </div>
  );
}
function Candidate({ candidate }: { candidate: CandidateLog }) {
  return (
    <details className="candidate-log">
      <summary>
        {candidate.label} ·{" "}
        {candidate.scoreKind === "order" ? "便宜评分 S" : "V"}=
        {number(candidate.value)} · Δ=
        {number(candidate.gain)}
        {candidate.freeCardSlots !== undefined && (
          <> · 空牌位 {candidate.freeCardSlots}</>
        )}
      </summary>
      <Estimate value={candidate.evaluation} />
    </details>
  );
}
export function GameLog({ session }: { session: Session }) {
  const labels = {
    takeTokens: "拿地形",
    takeCard: "拿动物牌",
    place: "放地形",
    settle: "入住",
    chooseSpirit: "选择自然之灵",
    endTurn: "结束回合",
  };
  return (
    <div className="game-log">
      <div className="log-toolbar">
        <span>
          {session.game.log.length} 条操作 · {session.aiLog?.length ?? 0} 条 AI
          决策
        </span>
        <button className="secondary" onClick={() => downloadGameLog(session)}>
          <Download size={16} /> 导出游戏日志
        </button>
      </div>
      <ol className="log-list">
        {session.game.log.map((line, i) => (
          <li key={i}>
            {line}
            {session.aiLog
              ?.filter((d) => d.gameLogIndex === i)
              .map((decision, j) => (
                <details className="ai-decision" key={j}>
                  <summary>
                    AI 策略 · {decision.selected.label} ·{" "}
                    {decision.selected.scoreKind === "order" ? "ΔS" : "局部 ΔV"}
                    ={number(decision.selected.gain)}
                  </summary>
                  <p>
                    {decision.evaluationVersion?.startsWith("current-turn-v3")
                      ? "当前回合搜索 · 深度"
                      : "目标深度"}{" "}
                    {decision.targetDepth} / 完成 {decision.completedDepth} ·
                    本回合思考 {decision.elapsedMs} ms · {decision.nodes} 个节点
                    {decision.fallback ? " · 使用合法保底方案" : ""}
                    {decision.timedOut ? " · 已达搜索预算" : ""}
                  </p>
                  <p>
                    {decision.evaluationVersion?.startsWith("current-turn-v3")
                      ? `${decision.evaluationVersion === "current-turn-v3.1-monotone" ? "v3.1 · 同分优先空牌位更多" : "v3"} · 仅记录搜索已有评分；中间动作按 S、结束回合按 V，显示每类已评分候选的前 ${decision.config.topK} 个。未计算的差值显示 —。`
                      : decision.evaluationVersion === "local-projects-v2"
                        ? `v2 · 每类全部合法动作已估值，显示前 ${decision.config.topK} 个。`
                        : `v1 旧记录 · 每类 top-${decision.config.topK} 的范围为便宜评分剪枝后的候选。`}
                    完整方案搜索值 Q：{number(decision.searchValue)}。
                  </p>
                  {decision.scoreGain !== undefined && (
                    <p>
                      即时规则分变化 {number(decision.scoreGain)} · 潜力变化{" "}
                      {number(decision.potentialGain!)} · 评分变化{" "}
                      {number(decision.selected.gain)}
                      。评分变化包含潜力重新估计。
                      {decision.actualScoreAfter && (
                        <>
                          {" "}
                          实际执行后规则分：{decision.actualScoreAfter.total}。
                        </>
                      )}
                    </p>
                  )}
                  {decision.usesSampledRefill && (
                    <p>
                      补充资源按模拟抽样估计（种子 {decision.sampleSeed}），此处
                      分数是决策时的预测。
                    </p>
                  )}
                  {(() => {
                    const plan =
                      decision.planSummary ??
                      session.aiLog?.find(
                        (d) =>
                          decision.planId &&
                          d.planId === decision.planId &&
                          d.planSummary,
                      )?.planSummary;
                    return (
                      plan && (
                        <details className="plan-log">
                          <summary>
                            完整方案比较 · Q={number(plan.value)}
                            {plan.gain !== undefined && (
                              <> · 相对根局面 {number(plan.gain)}</>
                            )}
                          </summary>
                          <p>
                            {plan.rootValue !== undefined && (
                              <>根局面 V={number(plan.rootValue)} · </>
                            )}
                            {plan.evaluations} 次评分计算 ·{" "}
                            {plan.expandedActions} 个动作展开 · 停止原因：
                            {
                              {
                                completed: "完成",
                                time: "时间预算",
                                evaluations: "评分计算预算",
                                stepLimit: "动作层数上限",
                              }[plan.stopReason]
                            }
                          </p>
                          {plan.searchStats && (
                            <p>
                              完成束宽{" "}
                              {plan.searchStats.widthsCompleted.join(" / ") ||
                                "—"}{" "}
                              · 尝试束宽 {plan.searchStats.widthAttempted} ·{" "}
                              {plan.searchStats.completePlans} 个完整方案 ·{" "}
                              {plan.searchStats.orderEvaluations} 次便宜评分 /{" "}
                              {plan.searchStats.fullEvaluations} 次完整 V · 合并{" "}
                              {plan.searchStats.duplicateStates} 个重复局面 ·
                              日志新增评分 {plan.searchStats.logEvaluations}
                            </p>
                          )}
                          <ol>
                            {plan.topPlans.map((p, index) => (
                              <li key={index}>
                                Q={number(p.value)} · 场景分数{" "}
                                {p.scenarioValues.map(number).join(" / ")}
                                {p.freeCardSlots !== undefined && (
                                  <> · 空牌位 {p.freeCardSlots}</>
                                )}
                                {p.terminal ? " · 已到终局" : ""}
                                {p.evaluation && (
                                  <details>
                                    <summary>完整 V 分解</summary>
                                    <Estimate value={p.evaluation} />
                                  </details>
                                )}
                                <p>
                                  {p.labels?.join(" → ") ??
                                    JSON.stringify(p.actions)}
                                </p>
                              </li>
                            ))}
                          </ol>
                        </details>
                      )
                    );
                  })()}
                  {decision.before && (
                    <details>
                      <summary>操作前评分</summary>
                      <Estimate value={decision.before} />
                    </details>
                  )}
                  <p>
                    <b>实际执行</b>
                  </p>
                  <Candidate candidate={decision.selected} />
                  {Object.entries(decision.topByOperation).map(
                    ([type, list]) => (
                      <section key={type}>
                        <b>
                          {labels[type as keyof typeof labels]} · 合法{" "}
                          {decision.legalCounts[type as keyof typeof labels]} /
                          已评分{" "}
                          {
                            decision.evaluatedCounts[
                              type as keyof typeof labels
                            ]
                          }
                        </b>
                        {list.map((candidate, k) => (
                          <Candidate key={k} candidate={candidate} />
                        ))}
                      </section>
                    ),
                  )}
                </details>
              ))}
          </li>
        ))}
      </ol>
    </div>
  );
}
