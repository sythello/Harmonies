import { Download } from "lucide-react";
import type { Session } from "./game/engine";
import { downloadGameLog } from "./game/ai/log";
import type { CandidateLog, Evaluation } from "./game/ai/types";

const number = (n: number) => Number(n.toFixed(3));
const vector = (values: number[]) => values.map(number).join(" / ");
function Estimate({ value }: { value: Evaluation }) {
  return (
    <div className="estimate-detail">
      <p>
        当前 {number(value.actualScore.total)} + 潜力 {number(value.potential)}{" "}
        = <b>{number(value.value)}</b> · R={value.R} · B={value.B} · F={value.F}
      </p>
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
        {candidate.label} · V={number(candidate.value)} · ΔV=
        {number(candidate.gain)}
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
                    AI 策略 · {decision.selected.label} · ΔV=
                    {number(decision.selected.gain)}
                  </summary>
                  <p>
                    目标深度 {decision.targetDepth} / 完成{" "}
                    {decision.completedDepth} · 本回合思考 {decision.elapsedMs}{" "}
                    ms · {decision.nodes} 个节点
                    {decision.fallback ? " · 使用合法保底方案" : ""}
                    {decision.timedOut ? " · 已达搜索预算" : ""}
                  </p>
                  <p>
                    每类保留 top-{decision.config.topK}
                    ，范围为经过剪枝后完整估值的候选。完整回合搜索值：
                    {number(decision.searchValue)}。
                  </p>
                  <details>
                    <summary>操作前估值</summary>
                    <Estimate value={decision.before} />
                  </details>
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
                          完整估值{" "}
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
