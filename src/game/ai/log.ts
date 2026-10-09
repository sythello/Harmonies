import type { Session } from "../engine";
import { COLORS } from "../engine";
import { EVALUATION_VERSION } from "./config";
import { observe } from "./observation";
import type { AIDecisionLog } from "./types";

export function exportGameLog(session: Session) {
  const aiDecisions = (session.aiLog ?? []).map((d) => ({
    ...d,
    evaluationVersion: d.evaluationVersion ?? "local-projects-v1",
  }));
  return {
    format: "harmonies-game-log",
    version: 3,
    exportedAt: new Date().toISOString(),
    evaluationVersion: EVALUATION_VERSION,
    evaluationVersions: [
      ...new Set(aiDecisions.map((d) => d.evaluationVersion)),
    ],
    vectorOrder: Object.entries(COLORS).map(([id, c]) => ({
      id: Number(id),
      name: c.name,
    })),
    algorithm: {
      formula:
        "V = actualScore + sum(G * exp(-m/max(1,B) - 0.5*sum(d[c]^2/(1+F*p[c]))))",
      orderFormula:
        "S = actualScore + sum_over_cards(max_project(nextAnimalGain * q)); independent per-card projects, no terrain gain",
      maxProjects: 6,
      topKScope:
        "v3：中间动作按 S、结束回合按 V 记录每类已评分候选的 top-k，selected 独立保留；完整回合按 V 排序。缺失值不补算。v1/v2 旧记录保留原始口径。",
      search:
        "v3：保持 v1 评分与资源期限公式，仅搜索当前回合；整层去重后按便宜评分剪枝，完成回合使用六项目 V；束宽翻倍，所有轮次保留最佳完整方案。只使用同一公开信息抽样，不模拟对手。",
      variables: {
        G: "项目净增规则分",
        m: "追加地形总数",
        mByColor: "按颜色追加数量",
        d: "扣除手中资源后的缺口",
        p: "剩余颜色比例",
        H: "尚未分配的手中地形",
        A: "未来颜色池",
        B: "剩余放置预算",
        F: "未来可取地形预算",
        q: "折算系数",
        W: "G*q",
        R: "估计剩余轮数",
        L: "预计结束轮次",
        T: "该玩家剩余行动回合数",
        terrainGain: "项目地形净收益",
        animalGain: "项目动物净收益",
        spiritGain: "项目自然之灵净收益",
      },
    },
    sessionId: session.id,
    players: session.game.players.map((p) => ({
      id: p.id,
      name: p.name,
      type: p.type,
      ai: p.ai,
    })),
    currentPosition: observe(session.game),
    gameLog: [...session.game.log],
    events: session.events ?? [],
    aiDecisions,
  };
}
export function downloadGameLog(session: Session) {
  const blob = new Blob([JSON.stringify(exportGameLog(session), null, 2)], {
    type: "application/json;charset=utf-8",
  });
  const url = URL.createObjectURL(blob),
    link = document.createElement("a");
  link.href = url;
  link.download = `harmonies-log-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("harmonies-logs", 1);
    request.onupgradeneeded = () =>
      request.result.createObjectStore("sessions");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
export async function readAILog(id: string): Promise<AIDecisionLog[]> {
  const db = await database();
  try {
    return await new Promise((resolve, reject) => {
      const request = db
        .transaction("sessions", "readonly")
        .objectStore("sessions")
        .get(id);
      request.onsuccess = () => resolve(request.result ?? []);
      request.onerror = () => reject(request.error);
    });
  } finally {
    db.close();
  }
}
export async function saveAILog(
  id: string,
  logs: AIDecisionLog[],
): Promise<void> {
  const db = await database();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction("sessions", "readwrite");
      transaction.objectStore("sessions").put(logs, id);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  } finally {
    db.close();
  }
}
