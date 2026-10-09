// Diagnostic only: instrument a temporary Git snapshot, never the game runtime.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const cwd = fileURLToPath(new URL("..", import.meta.url));
const commit = execFileSync("git", ["rev-parse", "f36c7cc"], {
  cwd,
  encoding: "utf8",
}).trim();
const temp = await mkdtemp(path.join(os.tmpdir(), "harmonies-v1-profile-"));
const archive = execFileSync(
  "git",
  ["archive", "--format=tar", commit, "src/game"],
  { cwd, maxBuffer: 16 * 1024 * 1024 },
);
const { mkdir } = await import("node:fs/promises");
for (const variant of ["original", "profiled"]) {
  await mkdir(path.join(temp, variant));
  execFileSync("tar", ["-xf", "-", "-C", path.join(temp, variant)], {
    input: archive,
  });
}
const profRoot = path.join(temp, "profiled/src/game/ai");
const helper = `
export const P: any = {
  phase: 'setup', phases: {}, beams: [], fullDetails: [], fallbackSteps: [], deep: [],
  reset() { this.phase='setup'; this.phases={}; this.beams=[]; this.fullDetails=[]; this.fallbackSteps=[]; this.deep=[]; this.last=performance.now(); },
  bucket() { return this.phases[this.phase] ??= { counters:{}, times:{}, offered:{}, actions:{} }; },
  add(k: string,n=1) { const b=this.bucket().counters; b[k]=(b[k]??0)+n; },
  setPhase(s: string) { const t=performance.now(); this.bucket().times.phase=(this.bucket().times.phase??0)+t-this.last; this.phase=s; this.last=t; },
  time(k: string,fn: any) { const t=performance.now(); try { return fn(); } finally { const b=this.bucket().times; b[k]=(b[k]??0)+performance.now()-t; } },
  legal(actions: any[]) { this.add('uncachedExpansions'); this.add('legalActions',actions.length); for(const a of actions) { const b=this.bucket().offered; b[a.type]=(b[a.type]??0)+1; } return actions; },
  action(a: any) { const b=this.bucket().actions; b[a.type]=(b[a.type]??0)+1; },
  counts(actions: any[]) { const b:any={}; for(const a of actions) b[a.type]=(b[a.type]??0)+1; return b; },
  finish() { this.setPhase('done'); return { phases:this.phases,beams:this.beams,fullDetails:this.fullDetails,fallbackSteps:this.fallbackSteps,deep:this.deep }; }
};
P.reset();
`;
await writeFile(path.join(profRoot, "profile.ts"), helper);
function replaceOnce(source: string, needle: string, replacement: string) {
  assert.equal(
    source.split(needle).length,
    2,
    `Expected one instrumentation anchor: ${needle.slice(0, 90)}`,
  );
  return source.replace(needle, replacement);
}
let search = await readFile(path.join(profRoot, "search.ts"), "utf8");
search = `import { P } from './profile';\n` + search;
const patches: [string, string][] = [
  [
    "for (const project of projects(player, false)) {",
    'for (const project of projects(player, false)) {\n    P.add("cheapProjects");',
  ],
  [
    "if (terms) best.set(card.id,",
    'if (terms) P.add("cheapResourceFeasible");\n    if (terms) best.set(card.id,',
  ],
  [
    "const key = `${id}:${positionKey(game)}`;",
    'P.add("fullCalls");\n    const key = `${id}:${positionKey(game)}`;',
  ],
  [
    "value = evaluatePosition(observe(game, id), id);",
    'P.add("fullMisses");\n      value = P.time("fullV", () => evaluatePosition(observe(game, id), id));',
  ],
  [
    "const cached = transitions.get(transitionKey);",
    'P.add("expandCalls");\n    const cached = transitions.get(transitionKey);\n    if(cached) P.add("expandHits");',
  ],
  [
    "for (const action of legalActions(game)) {",
    'for (const action of P.legal(P.time("legalEnumeration", () => legalActions(game)))) {',
  ],
  [
    "const next = applyAction(game, action);",
    'P.action(action);\n      const next = P.time("applyAction", () => applyAction(game, action));',
  ],
  [
    "const actual = scorePlayer(\n        next.players.find((p) => p.id === id)!,\n        next.side,\n      ).total;",
    'const actual = P.time("immediateScore", () => scorePlayer(\n        next.players.find((p) => p.id === id)!,\n        next.side,\n      ).total);',
  ],
  [
    "value: orderValue(next, id, actual),",
    'value: P.time("cheapOrder", () => { P.add("cheapCalls"); return orderValue(next, id, actual); }),',
  ],
  [
    "const greedy = (initial: Game, bounded: boolean): Node => {",
    'const greedy = (initial: Game, bounded: boolean): Node => {\n    P.add("greedyCalls");',
  ],
  [
    "const action = chosen.actions[0];",
    "const action = chosen.actions[0];\n      if(!bounded) P.fallbackSteps.push({ action, label:actionLabel(game,action), legalCounts:P.counts(candidates.map(n=>n.actions[0])) });",
  ],
  [
    "const fallback = greedy(root, false);",
    'P.setPhase("fallback");\n  const fallback = greedy(root, false);',
  ],
  [
    "let bestValue = evaluate(fallback.game).value;",
    'let bestValue = evaluate(fallback.game).value;\n  P.setPhase("currentTurn");',
  ],
  [
    "const completeTurns = (initial: Game, width: number): Node[] => {",
    "const completeTurns = (initial: Game, width: number): Node[] => {\n    const beam:any={phase:P.phase, width, round:initial.round, actor:initial.players[initial.active].id, layers:[],uniqueComplete:0,returned:0,interrupted:false}; P.beams.push(beam);",
  ],
  [
    "check();\n        const next = new Map<string, Node>();",
    "check();\n        const layer:any={step:step+1,parents:frontier.length,parentsFinished:0,offered:0,localKept:0,terminalEdges:0,merged:null,kept:null,done:false}; beam.layers.push(layer);\n        const next = new Map<string, Node>();",
  ],
  [
    "const expanded = expand(node.game, id, true);",
    "const expanded = expand(node.game, id, true);\n          layer.parentsFinished++; layer.offered+=expanded.length;",
  ],
  [
    "for (const child of candidates) {",
    "layer.localKept+=candidates.length;\n          for (const child of candidates) {",
  ],
  [
    'if (child.actions[0].type === "endTurn") {',
    'if (child.actions[0].type === "endTurn") {\n              layer.terminalEdges++;',
  ],
  [
    "frontier = [...next.values()].sort(compareNodes).slice(0, width);",
    "layer.merged=next.size;\n        frontier = [...next.values()].sort(compareNodes).slice(0, width);\n        layer.kept=frontier.length; layer.done=true;",
  ],
  [
    "// A deadline must not discard complete legal turns already discovered.",
    "beam.interrupted=true; beam.uniqueComplete=complete.size;\n      // A deadline must not discard complete legal turns already discovered.",
  ],
  [
    "return [...complete.values()].sort(compareNodes).slice(0, width);",
    "beam.uniqueComplete=complete.size; beam.returned=Math.min(complete.size,width);\n    return [...complete.values()].sort(compareNodes).slice(0, width);",
  ],
  [
    "for (let depth = 2; depth <= config.depth; depth++) {",
    'for (let depth = 2; depth <= config.depth; depth++) {\n        P.setPhase("depth"+depth);\n        const deep:any={depth,rootCandidates:turns.length,scenariosStarted:0,scenariosCompleted:0,candidatesCompleted:0,committed:false}; P.deep.push(deep);',
  ],
  [
    "check();\n            let simulated = sampleGame(",
    "check();\n            deep.scenariosStarted++;\n            let simulated = sampleGame(",
  ],
  [
    "total += evaluate(simulated).value;",
    "total += evaluate(simulated).value;\n            deep.scenariosCompleted++;",
  ],
  [
    "values.push({ node: candidate, value: total / 2 });",
    "values.push({ node: candidate, value: total / 2 });\n          deep.candidatesCompleted++;",
  ],
  [
    "completedDepth = depth;",
    "completedDepth = depth;\n        deep.committed=true;",
  ],
  [
    "// Only executed decision points are logged, not the search tree. No hidden arrays enter the log.",
    'P.setPhase("logs");\n  // Only executed decision points are logged, not the search tree. No hidden arrays enter the log.',
  ],
];
for (const [needle, replacement] of patches)
  search = replaceOnce(search, needle, replacement);
await writeFile(path.join(profRoot, "search.ts"), search);
let evaluate = await readFile(path.join(profRoot, "evaluate.ts"), "utf8");
evaluate = `import { P } from './profile';\n` + evaluate;
evaluate = replaceOnce(
  evaluate,
  "const h = horizon(view, playerId);",
  "const h = horizon(view, playerId);\n  const detail:any={phase:P.phase,hands:0,passes:0,animalProjects:0,terrainProjects:0,feasible:0,positive:0}; P.fullDetails.push(detail);",
);
evaluate = replaceOnce(
  evaluate,
  "for (const hand of hands) {",
  'for (const hand of hands) {\n    detail.hands++; P.add("fullHands");',
);
evaluate = replaceOnce(
  evaluate,
  "let chosen: ProjectEstimate | undefined;",
  'detail.passes++; P.add("fullPasses");\n      let chosen: ProjectEstimate | undefined;',
);
evaluate = replaceOnce(
  evaluate,
  "for (const project of projects(virtual)) {",
  'for (const project of projects(virtual)) {\n        detail[project.kind+"Projects"]++; P.add("full"+(project.kind==="animal"?"Animal":"Terrain")+"Projects");',
);
evaluate = replaceOnce(
  evaluate,
  "if (!terms) continue;",
  'if (!terms) continue;\n        detail.feasible++; P.add("fullResourceFeasible");',
);
evaluate = replaceOnce(
  evaluate,
  "if (G <= 0) continue;",
  'if (G <= 0) continue;\n        detail.positive++; P.add("fullPositive");',
);
await writeFile(path.join(profRoot, "evaluate.ts"), evaluate);

const load = (relative: string) =>
  import(pathToFileURL(path.join(temp, relative)).href);
try {
  const original = await load("original/src/game/ai/search.ts");
  const profiled = await load("profiled/src/game/ai/search.ts");
  const { P } = await load("profiled/src/game/ai/profile.ts");
  const engine = await load("original/src/game/engine.ts");
  const obs = await load("original/src/game/ai/observation.ts");
  const actions = await load("original/src/game/ai/actions.ts");
  const projectLib = await load("original/src/game/ai/projects.ts");
  const fixtures = JSON.parse(
    await readFile(
      path.join(cwd, "tests/fixtures/ai-audit-2026-09-27.json"),
      "utf8",
    ),
  );
  const positions = fixtures
    .filter(
      (f: any) =>
        !f.position.turn.tookTokens &&
        !f.position.turn.tookCard &&
        [1, 4, 8, 9].includes(f.position.round),
    )
    .map((f: any) => ({
      id: `user-log-round-${f.position.round}`,
      sourceIndex: f.index,
      view: f.position,
    }));
  const comparison = JSON.parse(
    await readFile(path.join(cwd, "docs/ai-time-comparison.json"), "utf8"),
  );
  const solo = comparison.results.find(
    (g: any) => g.id === "B-true-19-solo-v1",
  );
  let game = engine.createGame({
    names: solo.seats,
    first: 0,
    side: solo.side,
    spirits: solo.spirits,
    seed: solo.seed,
    controllers: [{ type: "ai", ai: comparison.config }],
  });
  for (const sample of solo.samples) {
    if (game.round === 5) break;
    for (const action of sample.actions)
      game = engine.applyAction(game, action);
  }
  assert.equal(game.round, 5);
  positions.push({
    id: "solo-B-spirit-round-5",
    sourceIndex: null,
    view: obs.observe(game),
  });
  function terrainSpace(view: any) {
    const board = structuredClone(view.players[view.active].board);
    return view.supply.flatMap((group: any[], index: number) => {
      if (group.length !== 3) return [];
      const edges = [0, 0, 0];
      let leaves = 0;
      const finals = new Set<string>();
      function visit(pending: any[], depth: number) {
        if (!pending.length) {
          leaves++;
          finals.add(board.map((c: any) => c.tokens.join("")).join("|"));
          return;
        }
        const seen = new Set<number>();
        pending.forEach((color: number, i: number) => {
          if (seen.has(color)) return;
          seen.add(color);
          for (const cell of board)
            if (engine.canPlace(cell, color)) {
              edges[depth]++;
              cell.tokens.push(color);
              visit(
                pending.filter((_: any, j: number) => i !== j),
                depth + 1,
              );
              cell.tokens.pop();
            }
        });
      }
      visit(group, 0);
      return [
        {
          group: index,
          colors: group,
          edges,
          orderedPlacementSequences: leaves,
          distinctFinalBoards: finals.size,
        },
      ];
    });
  }
  const rows: any[] = [];
  for (const f of positions) {
    const config = {
      depth: 2,
      timeMs: 3500,
      beamWidth: 6,
      topK: 3,
      seed: 1729,
    };
    original.searchTurn(f.view, { ...config, timeMs: 100 });
    profiled.searchTurn(f.view, { ...config, timeMs: 100 });
    const times: number[] = [];
    const baseline = original.searchTurn(f.view, config, {
      now: () => {
        const t = performance.now();
        times.push(t);
        return t;
      },
    });
    let cursor = 0;
    P.reset();
    const t0 = performance.now();
    const prof = profiled.searchTurn(f.view, config, {
      now: () => {
        assert.ok(cursor < times.length, "Extra clock read");
        return times[cursor++];
      },
    });
    const externalMs = performance.now() - t0;
    const stats = structuredClone(P.finish());
    assert.equal(cursor, times.length);
    assert.deepEqual(prof, baseline);
    assert.equal(
      Object.values(stats.phases).reduce(
        (n: number, p: any) =>
          n + Object.values(p.actions).reduce((a: number, b: any) => a + b, 0),
        0,
      ),
      baseline.nodes,
    );
    const player = f.view.players[f.view.active];
    const templates = player.cards
      .filter((c: any) => !engine.isComplete(c))
      .map((owned: any) => {
        const card = engine.CARDS[owned.id];
        const ownedPlayer = { ...player, cards: [owned] };
        const ps = [...projectLib.projects(ownedPlayer, false)];
        const targets = new Set(ps.map((p: any) => p.targetCell));
        return {
          id: card.id,
          name: card.name,
          patternCells: card.pattern.length,
          buildings: card.pattern.filter(
            (p: any) => p.colors[0] === 6 && p.colors[1] === 7,
          ).length,
          geometricAttempts: player.board.length * 6,
          feasibleProjects: ps.length,
          distinctTargets: targets.size,
        };
      });
    rows.push({
      id: f.id,
      sourceIndex: f.sourceIndex,
      n: f.view.players.length,
      side: f.view.side,
      spirits: f.view.spirits,
      round: f.view.round,
      boardCells: player.board.length,
      emptyCells: engine.emptyCount(player),
      activeCards: engine.activeCards(player).length,
      config,
      baseline: {
        elapsedMs: baseline.elapsedMs,
        nodes: baseline.nodes,
        completedDepth: baseline.completedDepth,
        actions: baseline.actions,
        decisions: baseline.decisions.map((d: any) => ({
          label: d.selected.label,
          action: d.selected.action,
          legalCounts: d.legalCounts,
          evaluatedCounts: d.evaluatedCounts,
        })),
      },
      clockReads: times.length,
      profiledExternalMs: externalMs,
      stats,
      terrainSpace: terrainSpace(f.view),
      templates,
    });
    const result = {
      commit,
      createdAt: new Date().toISOString(),
      method:
        "Original v1 runs at 3500 ms; instrumented snapshot replays its exact clock reads and must return an identical entire TurnPlan. Counters include partial work; instrumented phase/component timings are diagnostic and contain instrumentation overhead. No runtime AI source is modified.",
      rows,
    };
    await writeFile(
      path.join(cwd, "docs/ai-v1-enumeration-profile.json"),
      JSON.stringify(result, null, 2) + "\n",
    );
    console.log(
      JSON.stringify({
        id: f.id,
        ms: baseline.elapsedMs,
        profileMs: externalMs,
        nodes: baseline.nodes,
        depth: baseline.completedDepth,
        phases: stats.phases,
      }),
    );
  }
} finally {
  await rm(temp, { recursive: true, force: true });
}
