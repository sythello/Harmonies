import { useEffect, useRef, useState } from "react";
import { applyAction, type Action, type Game } from "./game/engine";
import { normalizeAI } from "./game/ai/config";
import { positionKey } from "./game/ai/actions";
import { observe } from "./game/ai/observation";
import type { AIDecisionLog, TurnPlan } from "./game/ai/types";

export function useAI(
  game: Game | undefined,
  enabled: boolean,
  paused: boolean,
  onAction: (action: Action, decision: AIDecisionLog) => void,
) {
  const callback = useRef(onAction);
  callback.current = onAction;
  const queue = useRef<{
    key: string;
    actions: Action[];
    decisions: AIDecisionLog[];
  } | null>(null);
  const [revision, setRevision] = useState(0);
  const [status, setStatus] = useState<
    "idle" | "thinking" | "playing" | "error"
  >("idle");
  const [error, setError] = useState("");
  const [lastPlan, setLastPlan] = useState<Pick<
    TurnPlan,
    "completedDepth" | "elapsedMs"
  > | null>(null);
  useEffect(() => {
    if (
      !game ||
      !enabled ||
      paused ||
      game.phase !== "playing" ||
      game.players[game.active].type !== "ai"
    ) {
      setStatus("idle");
      if (!enabled) queue.current = null;
      return;
    }
    let cancelled = false;
    const key = positionKey(game);
    if (queue.current?.key === key && queue.current.actions.length) {
      setStatus("playing");
      const timer = setTimeout(() => {
        if (cancelled || !queue.current) return;
        try {
          const action = queue.current.actions.shift()!,
            decision = queue.current.decisions.shift()!;
          queue.current.key = positionKey(applyAction(game, action));
          callback.current(action, decision);
        } catch (e) {
          queue.current = null;
          setError((e as Error).message);
          setStatus("error");
        }
      }, 240);
      return () => {
        cancelled = true;
        clearTimeout(timer);
      };
    }
    queue.current = null;
    setStatus("thinking");
    setError("");
    const worker = new Worker(new URL("./game/ai/worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (
      event: MessageEvent<{ result?: TurnPlan; error?: string }>,
    ) => {
      if (cancelled) return;
      if (event.data.error || !event.data.result?.actions.length) {
        setError(event.data.error ?? "AI 未返回有效计划。");
        setStatus("error");
        return;
      }
      const result = event.data.result;
      queue.current = {
        key,
        actions: result.actions,
        decisions: result.decisions,
      };
      setLastPlan({
        completedDepth: result.completedDepth,
        elapsedMs: result.elapsedMs,
      });
      setRevision((v) => v + 1);
    };
    worker.onerror = (event) => {
      if (!cancelled) {
        setError(event.message || "AI 线程启动失败");
        setStatus("error");
      }
    };
    worker.postMessage({
      view: observe(game),
      config: normalizeAI(game.players[game.active].ai),
    });
    return () => {
      cancelled = true;
      worker.terminate();
    };
  }, [game, enabled, paused, revision]);
  return {
    status,
    error,
    lastPlan,
    retry: () => {
      queue.current = null;
      setRevision((v) => v + 1);
    },
  };
}
