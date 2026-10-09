import { writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import fixtures from "../tests/fixtures/ai-audit-2026-09-27.json" with { type: "json" };
import { searchTurn } from "../src/game/ai/search-v2";
import { evaluatePosition } from "../src/game/ai/evaluate";
import type { AIObservation } from "../src/game/ai/types";

const reference = process.env.AI_V1_REFERENCE
  ? await import(pathToFileURL(process.env.AI_V1_REFERENCE).href)
  : null;
const maxNodes = Number(process.env.AI_AUDIT_EVALUATIONS ?? 200);
const rows = [];
for (const f of fixtures.filter(
  (f) => !f.position.turn.tookTokens && !f.position.turn.tookCard,
)) {
  const view = f.position as unknown as AIObservation;
  const old = reference?.searchTurn(view, f.config, { now: () => 0, maxNodes });
  const result = searchTurn(view, f.config, { now: () => 0, maxNodes });
  const summarize = (p: ReturnType<typeof searchTurn>) => ({
    actions: p.actions,
    completedDepth: p.completedDepth,
    evaluations: p.evaluations,
    rootV: p.decisions[0].before.value,
    selectedActionV: p.decisions[0].selected.value,
    Q: p.decisions[0].searchValue,
  });
  rows.push({
    round: view.round,
    sourceIndex: f.index,
    maxEvaluations: maxNodes,
    legacyRecorded: f.legacy,
    legacyFixedBudget: old ? summarize(old) : undefined,
    v2: summarize(result),
    independentRootV: evaluatePosition(view).value,
  });
  console.log(JSON.stringify(rows.at(-1)));
  await writeFile(
    process.env.AI_AUDIT_OUTPUT ?? "/tmp/harmonies-ai-v2-audit.json",
    JSON.stringify(rows, null, 2) + "\n",
  );
}
