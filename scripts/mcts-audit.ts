import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import {
  auditMcts,
  collectCases,
  enumerateTurn,
  verifyPlans,
  type CaseResult,
} from "./audit/mcts-audit";
import { caseHtml, indexHtml } from "./audit/mcts-report";

const args = process.argv.slice(2);
const option = (key: string, fallback: string) =>
  args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = path.resolve(option("--output", "rl/runs/mcts-audit-20261010"));
const file = fileURLToPath(import.meta.url);
mkdirSync(output, { recursive: true });
const inputPath = path.join(output, "inputs.json");

if (args.includes("--case")) {
  const inputs = JSON.parse(readFileSync(inputPath, "utf8")) as ReturnType<
    typeof collectCases
  >;
  const index = Number(option("--case", "0")),
    position = inputs.cases[index];
  const result: CaseResult = {
    position,
    mcts: [128, 1024].map((budget) =>
      auditMcts(
        position.input,
        budget,
        position.seed * 2 + position.input.active,
      ),
    ),
    exact: enumerateTurn(position.input, 60000),
  };
  for (const method of [...result.mcts, result.exact])
    verifyPlans(position.input, method.plans);
  for (const m of result.mcts) {
    assert.equal(
      m.rootPolicy.reduce((n, e) => n + e.visits, 0),
      m.simulations,
    );
    if (result.exact.complete && m.plans.length)
      assert.ok(m.plans[0].score.total <= result.exact.plans[0].score.total);
  }
  writeFileSync(
    path.join(output, `${position.id}.json`),
    JSON.stringify(result),
  );
  writeFileSync(
    path.join(output, `${position.id}.html`),
    caseHtml(
      result,
      inputs.cases.map((c) => c.id),
    ),
  );
  console.log(
    JSON.stringify({
      id: position.id,
      exactComplete: result.exact.complete,
      exactMs: Math.round(result.exact.elapsedMs),
      exactBest: result.exact.plans[0]?.score.total,
      mcts: result.mcts.map((m) => ({
        simulations: m.simulations,
        plans: m.plans.length,
        best: m.plans[0]?.score.total,
      })),
    }),
  );
} else {
  const seed = Number(option("--seed", "71000")),
    games = Number(option("--games", "100"));
  const workers = Number(option("--workers", "4"));
  assert.ok(Number.isInteger(games) && games > 0);
  assert.ok(Number.isInteger(workers) && workers > 0 && workers <= 8);
  if (!existsSync(inputPath))
    writeFileSync(inputPath, JSON.stringify(collectCases(games, seed)));
  const inputs = JSON.parse(readFileSync(inputPath, "utf8")) as ReturnType<
    typeof collectCases
  >;
  assert.equal(inputs.games.length, games);
  assert.equal(inputs.games[0].seed, seed);
  const sources = [
    "src/game/ai/rl/mcts.ts",
    "src/game/engine.ts",
    "src/game/ai/actions.ts",
    "src/game/ai/observation.ts",
    "scripts/audit/mcts-audit.ts",
    "scripts/audit/mcts-report.ts",
    "scripts/mcts-audit.ts",
  ];
  const metadata = {
    createdAt: new Date().toISOString(),
    seed,
    games,
    positions: inputs.cases.length,
    workers,
    timeoutMsPerPosition: 60000,
    node: process.version,
    cpu: os.cpus()[0].model,
    logicalCpus: os.cpus().length,
    budget: "one tree per position; no extra completion search",
    policy: "uniform legal priors",
    value:
      "current absolute score / 100; absoluteWeight=1, rankWeight=relativeWeight=0",
    sources: Object.fromEntries(
      sources.map((p) => [
        p,
        createHash("sha256").update(readFileSync(p)).digest("hex"),
      ]),
    ),
  };
  const results: CaseResult[] = [];
  function save() {
    results.sort((a, b) => a.position.id.localeCompare(b.position.id));
    const exact = results.filter((r) => r.exact.complete);
    const mean = (values: number[]) =>
      values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
    const stats = [0, 1].map((i) => ({
      simulations: [128, 1024][i],
      zeroPlans: results.filter((r) => !r.mcts[i].plans.length).length,
      fewerThanThree: results.filter((r) => r.mcts[i].plans.length < 3).length,
      provenOptimalHits: exact.filter(
        (r) =>
          r.mcts[i].plans[0]?.score.total === r.exact.plans[0]?.score.total,
      ).length,
      meanRegretWhenPlanFound: mean(
        exact
          .filter((r) => r.mcts[i].plans.length)
          .map(
            (r) =>
              r.exact.plans[0].score.total - r.mcts[i].plans[0].score.total,
          ),
      ),
      meanMs: mean(results.map((r) => r.mcts[i].elapsedMs)),
    }));
    writeFileSync(
      path.join(output, "summary.json"),
      JSON.stringify(
        {
          metadata,
          completed: results.length,
          exactComplete: exact.length,
          exactMeanMs: mean(results.map((r) => r.exact.elapsedMs)),
          exactMaxMs: Math.max(0, ...results.map((r) => r.exact.elapsedMs)),
          mcts: stats,
          cases: results.map((r) => ({
            id: r.position.id,
            exactComplete: r.exact.complete,
            exactBest: r.exact.plans[0]?.score.total,
            exactMs: r.exact.elapsedMs,
            mcts: r.mcts.map((m) => ({
              simulations: m.simulations,
              plans: m.plans.length,
              best: m.plans[0]?.score.total,
              ms: m.elapsedMs,
            })),
          })),
        },
        null,
        2,
      ),
    );
    writeFileSync(
      path.join(output, "index.html"),
      indexHtml(results, metadata),
    );
  }
  const exec = promisify(execFile);
  let cursor = 0;
  await Promise.all(
    Array.from({ length: workers }, async () => {
      while (cursor < inputs.cases.length) {
        const index = cursor++,
          c = inputs.cases[index];
        // Always recompute: stale reports must never silently mix source versions.
        const { stdout } = await exec(
          process.execPath,
          [
            "--import",
            "tsx",
            file,
            "--output",
            output,
            "--case",
            String(index),
          ],
          { maxBuffer: 1024 * 1024, timeout: 120000 },
        );
        results.push(
          JSON.parse(readFileSync(path.join(output, `${c.id}.json`), "utf8")),
        );
        process.stdout.write(
          `${results.length}/${inputs.cases.length} ${stdout}`,
        );
        if (results.length % 10 === 0) save();
      }
    }),
  );
  save();
  console.log(`Report: ${path.join(output, "index.html")}`);
}
