import type { Session } from "../engine";
import { COLORS } from "../engine";
import { EVALUATION_VERSION } from "./config";
import { observe } from "./observation";
import type { AIDecisionLog } from "./types";

export function exportGameLog(session: Session) {
  return {
    format: "harmonies-game-log",
    version: 1,
    exportedAt: new Date().toISOString(),
    evaluationVersion: EVALUATION_VERSION,
    vectorOrder: Object.entries(COLORS).map(([id, c]) => ({
      id: Number(id),
      name: c.name,
    })),
    algorithm: {
      formula:
        "V = actualScore + sum(G * exp(-m/max(1,B) - 0.5*sum(d[c]^2/(1+F*p[c]))))",
      maxProjects: 6,
      topKScope:
        "每个已执行决策、每种操作在完整估值候选中保留 top-k；先用便宜的排序剪枝。legalCounts/evaluatedCounts 标明覆盖范围。selected 始终单独保存。",
      search:
        "完整自身回合束搜索；对手使用贪心；后续资源使用两个公开信息抽样；限时迭代加深。",
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
    aiDecisions: session.aiLog ?? [],
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
