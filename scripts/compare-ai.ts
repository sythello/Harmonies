import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  applyAction,
  createGame,
  ranking,
  scorePlayer,
  type Game,
  type Side,
} from "../src/game/engine";
import { compareActions, legalActions } from "../src/game/ai/actions";
import { normalizeAI } from "../src/game/ai/config";
import { observe } from "../src/game/ai/observation";
import { searchTurn } from "../src/game/ai/search";

type Version = "v1" | "v3";
type Match = {
  id: string;
  mode: "duel" | "solo";
  side: Side;
  spirits: boolean;
  seed: number;
  seats: Version[];
};
const cwd = fileURLToPath(new URL("..", import.meta.url));
const legacyRef = process.env.AI_COMPARE_REF ?? "f36c7cc";
const legacyCommit = execFileSync("git", ["rev-parse", legacyRef], {
  cwd,
  encoding: "utf8",
}).trim();
const config = normalizeAI({
  timeMs: Number(process.env.AI_COMPARE_MS ?? 3500),
  depth: 2,
  beamWidth: 6,
  topK: 3,
  seed: 1729,
});
const outputPath =
  process.env.AI_COMPARE_OUTPUT ?? "/tmp/harmonies-ai-time-comparison.json";
const snapshot = await mkdtemp(path.join(os.tmpdir(), "harmonies-v1-exact-"));
const archive = execFileSync(
  "git",
  ["archive", "--format=tar", legacyCommit, "src/game"],
  { cwd, maxBuffer: 16 * 1024 * 1024 },
);
execFileSync("tar", ["-xf", "-", "-C", snapshot], { input: archive });

async function sourceDigest(directory: string): Promise<string> {
  const hash = createHash("sha256");
  async function visit(relative: string) {
    for (const entry of (
      await readdir(path.join(directory, relative))
    ).sort()) {
      const name = path.join(relative, entry);
      const full = path.join(directory, name);
      const children = await readdir(full).catch(() => null);
      if (children) await visit(name);
      else
        hash
          .update(name)
          .update("\0")
          .update(await readFile(full));
    }
  }
  await visit("");
  return hash.digest("hex");
}
function checkGame(game: Game) {
  const tokens = [
    ...game.bag,
    ...game.supply.flat(),
    ...game.turn.pending,
    ...game.discardedTokens,
    ...game.players.flatMap((p) => p.board.flatMap((c) => c.tokens)),
  ];
  assert.deepEqual(
    [1, 2, 3, 4, 5, 6].map((c) => tokens.filter((t) => t === c).length),
    [23, 23, 21, 19, 19, 15],
  );
  const animals = [
    ...game.deck,
    ...game.market.filter((c): c is number => c !== null),
    ...game.discardedCards,
    ...game.players.flatMap((p) =>
      p.cards.map((c) => c.id).filter((c) => c <= 32),
    ),
  ];
  assert.equal(animals.length, 32);
  assert.equal(new Set(animals).size, 32);
}
function distribution(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    n: values.length,
    mean: values.reduce((a, b) => a + b, 0) / (values.length || 1),
    median: sorted[Math.floor(sorted.length * 0.5)] ?? 0,
    p95:
      sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] ??
      0,
    max: sorted.at(-1) ?? 0,
  };
}
const scenarios: { side: Side; spirits: boolean; seed: number }[] = [
  { side: "A", spirits: false, seed: 11 },
  { side: "B", spirits: false, seed: 13 },
  { side: "A", spirits: true, seed: 17 },
  { side: "B", spirits: true, seed: 19 },
];
const matches: Match[] = scenarios.flatMap((scenario, i) => {
  const order: Version[] = i % 2 ? ["v3", "v1"] : ["v1", "v3"];
  return [
    ...order.map((first): Match => ({
      ...scenario,
      mode: "duel",
      id: `${scenario.side}-${scenario.spirits}-${scenario.seed}-duel-${first}-first`,
      seats: [first, first === "v1" ? "v3" : "v1"],
    })),
    ...order.map((version): Match => ({
      ...scenario,
      mode: "solo",
      id: `${scenario.side}-${scenario.spirits}-${scenario.seed}-solo-${version}`,
      seats: [version],
    })),
  ];
});
type TurnSample = {
  version: Version;
  seat: number;
  round: number;
  wallMs: number;
  cpuMs: number;
  reportedMs: number;
  depth: number;
  fallback: boolean;
  stats?: ReturnType<typeof searchTurn>["decisions"][number]["planSummary"];
  actions: ReturnType<typeof searchTurn>["actions"];
};
type MatchResult = Match & {
  scores: ReturnType<typeof scorePlayer>[];
  winners: Version[];
  samples: TurnSample[];
  rounds: number;
  finalReason: string;
};
const results: MatchResult[] = [];
const metadata = {
  startedAt: new Date().toISOString(),
  legacyCommit,
  currentGameSourceSha256: await sourceDigest(path.join(cwd, "src/game")),
  node: process.version,
  platform: `${process.platform}/${process.arch}`,
  cpu: os.cpus()[0]?.model,
  config,
  effectiveDepth: { v1: 2, v3: 1 },
  method:
    "Unmodified legacy Git snapshot versus current implementation; common actual-game engine, public observations only; serial execution; identical soft wall-time limit including diagnostics; warm-up excluded; duel seats swapped per seed, Solo paired per seed. Actual wall and CPU times recorded, no claim of equal time if soft limits overrun.",
};
function summarize() {
  const duels = results.filter((r) => r.mode === "duel");
  const soloPairs = scenarios.flatMap((s) => {
    const pair = results.filter((r) => r.mode === "solo" && r.seed === s.seed);
    const v1 = pair.find((r) => r.seats[0] === "v1");
    const v3 = pair.find((r) => r.seats[0] === "v3");
    return v1 && v3
      ? [
          {
            ...s,
            v1: v1.scores[0].total,
            v3: v3.scores[0].total,
            delta: v3.scores[0].total - v1.scores[0].total,
          },
        ]
      : [];
  });
  return {
    duels: {
      games: duels.length,
      v3Wins: duels.filter(
        (r) => r.winners.length === 1 && r.winners[0] === "v3",
      ).length,
      v1Wins: duels.filter(
        (r) => r.winners.length === 1 && r.winners[0] === "v1",
      ).length,
      draws: duels.filter((r) => r.winners.length === 2).length,
      scoreDifference: distribution(
        duels.map(
          (r) =>
            r.scores[r.seats.indexOf("v3")].total -
            r.scores[r.seats.indexOf("v1")].total,
        ),
      ),
      pairedDifferences: scenarios.flatMap((s) => {
        const pair = duels.filter((r) => r.seed === s.seed);
        return pair.length === 2
          ? [
              {
                ...s,
                meanDelta:
                  pair.reduce(
                    (sum, r) =>
                      sum +
                      r.scores[r.seats.indexOf("v3")].total -
                      r.scores[r.seats.indexOf("v1")].total,
                    0,
                  ) / 2,
              },
            ]
          : [];
      }),
    },
    solo: {
      pairs: soloPairs,
      scoreDifference: distribution(soloPairs.map((p) => p.delta)),
    },
    versions: Object.fromEntries(
      (["v1", "v3"] as Version[]).map((version) => {
        const samples = results.flatMap((r) =>
          r.samples.filter((s) => s.version === version),
        );
        return [
          version,
          {
            wallMs: distribution(samples.map((s) => s.wallMs)),
            cpuMs: distribution(samples.map((s) => s.cpuMs)),
            overBudgetTurns: samples.filter((s) => s.wallMs > config.timeMs)
              .length,
            overBudgetByMoreThan100ms: samples.filter(
              (s) => s.wallMs > config.timeMs + 100,
            ).length,
            fallbackTurns: samples.filter((s) => s.fallback).length,
            completePlans: distribution(
              samples.flatMap((s) =>
                s.stats?.searchStats ? [s.stats.searchStats.completePlans] : [],
              ),
            ),
            logMs: distribution(
              samples.flatMap((s) =>
                s.stats?.searchStats ? [s.stats.searchStats.logMs] : [],
              ),
            ),
            logEvaluations:
              version === "v3"
                ? samples.reduce(
                    (n, s) => n + (s.stats?.searchStats?.logEvaluations ?? 0),
                    0,
                  )
                : null,
            depths: [0, 1, 2, 3].map(
              (d) => samples.filter((s) => s.depth === d).length,
            ),
          },
        ];
      }),
    ),
  };
}
async function save(complete: boolean) {
  await writeFile(
    outputPath,
    JSON.stringify(
      {
        ...metadata,
        updatedAt: new Date().toISOString(),
        complete,
        summary: summarize(),
        results,
      },
      null,
      2,
    ) + "\n",
  );
}
try {
  const legacy = await import(
    pathToFileURL(path.join(snapshot, "src/game/ai/search.ts")).href
  );
  const legacyEngine = await import(
    pathToFileURL(path.join(snapshot, "src/game/engine.ts")).href
  );
  const planners: Record<Version, typeof searchTurn> = {
    v1: legacy.searchTurn,
    v3: searchTurn,
  };
  const limit = Number(process.env.AI_COMPARE_LIMIT ?? matches.length);
  const selected = matches.slice(0, limit);
  for (const [index, match] of selected.entries()) {
    let game = createGame({
      names: match.seats,
      first: 0,
      side: match.side,
      spirits: match.spirits,
      seed: match.seed,
      controllers: match.seats.map(() => ({ type: "ai", ai: config })),
    });
    // Warm both implementations before each match; these plans are discarded.
    for (const version of ["v1", "v3"] as Version[])
      planners[version](observe(game), { ...config, timeMs: 100 });
    console.log(`START ${index + 1}/${selected.length} ${match.id}`);
    const samples: TurnSample[] = [];
    while (game.phase !== "ended" && samples.length < 120) {
      const seat = game.active,
        version = match.seats[seat],
        round = game.round;
      const view = observe(game);
      const startCPU = process.cpuUsage(),
        start = performance.now();
      const plan = planners[version](view, config);
      const wallMs = performance.now() - start,
        cpu = process.cpuUsage(startCPU);
      assert.ok(plan.actions.length > 0);
      assert.equal(plan.actions.at(-1)?.type, "endTurn");
      assert.equal(plan.actions.filter((a) => a.type === "endTurn").length, 1);
      for (const action of plan.actions) {
        assert.ok(
          legalActions(game).some((a) => compareActions(a, action) === 0),
          "Illegal planned action",
        );
        game = applyAction(game, action);
        checkGame(game);
      }
      for (const player of game.players)
        assert.deepEqual(
          scorePlayer(player, game.side),
          legacyEngine.scorePlayer(player, game.side),
        );
      samples.push({
        version,
        seat,
        round,
        wallMs,
        cpuMs: (cpu.user + cpu.system) / 1000,
        reportedMs: plan.elapsedMs,
        depth: plan.completedDepth,
        fallback: plan.decisions[0].fallback,
        actions: plan.actions,
        stats: version === "v3" ? plan.decisions[0].planSummary : undefined,
      });
      console.log(
        `TURN ${match.id} r=${round} ${version} ${Math.round(wallMs)}ms depth=${plan.completedDepth}`,
      );
    }
    assert.equal(game.phase, "ended");
    assert.equal(new Set(game.players.map((p) => p.turns)).size, 1);
    const result: MatchResult = {
      ...match,
      scores: game.players.map((p) => scorePlayer(p, game.side)),
      winners: ranking(game)
        .filter((r) => r.rank === 1)
        .map((r) => match.seats[r.player.id]),
      samples,
      rounds: game.round,
      finalReason: game.finalReason,
    };
    results.push(result);
    await save(index === selected.length - 1);
    console.log(
      `RESULT ${match.id} scores=${result.scores.map((s) => s.total).join("/")} winner=${result.winners.join("/")}`,
    );
  }
  console.log(JSON.stringify(summarize(), null, 2));
  console.log(`Saved ${outputPath}`);
} finally {
  await rm(snapshot, { recursive: true, force: true });
}
