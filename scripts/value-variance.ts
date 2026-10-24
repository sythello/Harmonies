import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { applyAction, ranking } from "../src/game/engine";
import { observe, sampleGame } from "../src/game/ai/observation";
import { Network, type Values } from "../src/game/ai/rl/network";
import { collect, newGame, type CollectionDecision, type Trajectory } from "../src/game/ai/rl/trajectory";
import { cheapRollout, rolloutSeeds, valueMoments, VALUE_HEADS } from "./audit/value-variance";

const { values: args } = parseArgs({ options: {
  out: { type: "string" }, worker: { type: "string", default: "0" }, verify: { type: "boolean" },
} });
if (!args.out) throw new Error("--out is required; start through rl/value_variance.py");
const out = resolve(args.out), worker = Number(args.worker);
const read = (path: string) => JSON.parse(readFileSync(path, "utf8"));
const hash = (value: string) => createHash("sha256").update(value).digest("hex");
const cfg: { games: number; repeats: number; seed: number; workers: number; simulations: number; stateLimit: number; modelHash: string } = read(join(out, "config.json"));
const modelText = readFileSync(join(out, "model.json"), "utf8");
assert.equal(hash(modelText), cfg.modelHash);
const model = new Network(JSON.parse(modelText));
const atomic = (path: string, value: unknown) => {
  const tmp = `${path}.${process.pid}.tmp`;
  writeFileSync(tmp, JSON.stringify(value) + "\n"); renameSync(tmp, path);
};
type Row = {
  gameIndex: number; step: number; round: number; seat: number; playerId: number;
  observationHash: string; baseMode: string; prediction: Values;
  rollouts: ReturnType<typeof cheapRollout>[]; statistics: ReturnType<typeof valueMoments>; seconds: number;
};
function records(path: string, repair = false): Row[] {
  if (!existsSync(path)) return [];
  let text = readFileSync(path, "utf8");
  if (text && !text.endsWith("\n")) {
    assert.ok(repair, "Partial journal encountered during verification");
    text = text.slice(0, text.lastIndexOf("\n") + 1); writeFileSync(path, text);
  }
  return text.split("\n").filter(Boolean).map((s) => JSON.parse(s));
}

if (args.verify) {
  const started = performance.now();
  let states = 0, continuations = 0, actions = 0;
  for (let i = 0; i < cfg.games; i++) {
    const dir = join(out, `game-${String(i).padStart(4, "0")}`);
    const base: Trajectory = read(join(dir, "base.json")), rows = records(join(dir, "states.jsonl"));
    assert.equal(base.model, cfg.modelHash); assert.equal(base.seed, cfg.seed + i);
    assert.equal(base.first, i % 2); assert.equal(base.search.simulations, 64);
    assert.equal(base.search.noiseWeight, .25); assert.equal(base.search.temperature, 1);
    assert.deepEqual(base.playoutCapRandomization, { cheapProbability: .75, cheapSimulations: 16 });
    assert.equal(base.treeReuse, "within-turn");
    assert.equal(base.decisions?.length, base.actions.length);
    assert.equal(rows.length, Math.min(base.actions.length, cfg.stateLimit || Infinity));
    let game = newGame(base.seed, base.first);
    for (const [step, action] of base.actions.entries()) {
      const decision: CollectionDecision = base.decisions![step];
      assert.equal(decision.step, step); assert.equal(decision.seat, game.active);
      assert.equal(decision.simulations, decision.mode === "cheap" ? 16 : 64);
      assert.equal(decision.noiseWeight, decision.mode === "cheap" ? 0 : .25);
      assert.equal(decision.rootVisits - decision.reusedVisits, decision.simulations);
      if (step < rows.length) {
        const row = rows[step], view = observe(game);
        assert.equal(row.baseMode, decision.mode);
        assert.equal(row.gameIndex, i); assert.equal(row.step, step);
        assert.equal(row.round, game.round); assert.equal(row.seat, game.active);
        assert.equal(row.playerId, game.players[game.active].id);
        assert.equal(row.observationHash, hash(JSON.stringify(view)));
        assert.equal(row.rollouts.length, cfg.repeats);
        const prediction = model.predict(view).value;
        for (const h of VALUE_HEADS) for (let s = 0; s < 2; s++) assert.ok(Math.abs(row.prediction[h][s] - prediction[h][s]) < 1e-12);
        for (const [repeat, r] of row.rollouts.entries()) {
          assert.deepEqual({ environmentSeed: r.environmentSeed, searchSeed: r.searchSeed }, rolloutSeeds(i, step, repeat, cfg.repeats));
          let terminal = sampleGame(view, r.environmentSeed);
          for (const a of r.actions) { terminal = applyAction(terminal, a); terminal.log = []; }
          assert.equal(terminal.phase, "ended");
          const ranked = ranking(terminal);
          const scores = terminal.players.map((p) => ranked.find((e) => e.player.id === p.id)!.score.total);
          const ranks = terminal.players.map((p) => ranked.find((e) => e.player.id === p.id)!.rank);
          assert.deepEqual(r.scores, scores); assert.deepEqual(r.ranks, ranks);
          const s = row.seat, o = 1 - s, rank = Math.sign(ranks[o] - ranks[s]);
          const expected = { rank: [rank, -rank], relative: [(scores[s] - scores[o]) / 100, (scores[o] - scores[s]) / 100], absolute: [scores[s] / 100, scores[o] / 100] };
          for (const h of VALUE_HEADS) for (let j = 0; j < 2; j++) assert.ok(Math.abs(r.value[h][j] - expected[h][j]) < 1e-12);
          assert.equal(r.addedSimulations, cfg.simulations * r.actions.length);
          continuations++; actions += r.actions.length;
        }
        states++;
      }
      game = applyAction(game, action); game.log = [];
    }
    assert.equal(game.phase, "ended");
    const ranked = ranking(game);
    assert.deepEqual(base.scores, game.players.map((p) => ranked.find((r) => r.player.id === p.id)!.score.total));
    assert.deepEqual(base.ranks, game.players.map((p) => ranked.find((r) => r.player.id === p.id)!.rank));
  }
  const result = { passed: true, baseGames: cfg.games, states, continuations, actions, seconds: (performance.now() - started) / 1000 };
  atomic(join(out, "replay-verification.json"), result); console.log(JSON.stringify(result));
} else {
  assert.ok(Number.isInteger(worker) && worker >= 0 && worker < cfg.workers);
  let statesDone = 0, gamesDone = 0, rolloutActions = 0;
  const started = performance.now();
  for (let i = worker; i < cfg.games; i += cfg.workers) {
    const dir = join(out, `game-${String(i).padStart(4, "0")}`);
    mkdirSync(dir, { recursive: true });
    const source = join(dir, "base.json");
    if (!existsSync(source)) {
      const begin = performance.now();
      const base = collect(model, { seed: cfg.seed + i, first: i % 2, model: cfg.modelHash,
        search: { simulations: 64, seed: cfg.seed + i * 1000003 },
        playoutCapRandomization: { cheapProbability: .75, cheapSimulations: 16 } });
      base.collectionSeconds = (performance.now() - begin) / 1000; atomic(source, base);
    }
    const base: Trajectory = read(source), path = join(dir, "states.jsonl"), saved = records(path, true);
    assert.equal(base.model, cfg.modelHash); assert.equal(base.seed, cfg.seed + i);
    const target = Math.min(base.actions.length, cfg.stateLimit || Infinity);
    assert.ok(saved.length <= target);
    let game = newGame(base.seed, base.first);
    for (let step = 0; step < target; step++) {
      const view = observe(game), observationHash = hash(JSON.stringify(view));
      let row = saved[step];
      if (row) {
        assert.equal(row.observationHash, observationHash); assert.equal(row.step, step);
        assert.equal(row.gameIndex, i); assert.equal(row.rollouts.length, cfg.repeats);
      } else {
        const begin = performance.now();
        const rollouts = Array.from({ length: cfg.repeats }, (_, repeat) => cheapRollout(view, model, rolloutSeeds(i, step, repeat, cfg.repeats), cfg.simulations));
        row = { gameIndex: i, step, round: game.round, seat: game.active, playerId: game.players[game.active].id,
          observationHash, baseMode: base.decisions![step].mode, prediction: model.predict(view).value,
          rollouts, statistics: valueMoments(rollouts.map((r) => r.value)), seconds: (performance.now() - begin) / 1000 };
        appendFileSync(path, JSON.stringify(row) + "\n");
      }
      statesDone++; rolloutActions += row.rollouts.reduce((s, r) => s + r.actions.length, 0);
      atomic(join(out, `worker-${worker}.json`), { worker, pid: process.pid, phase: "running", gameIndex: i,
        statesDone, gamesDone, rolloutActions, stateInGame: step + 1, statesInGame: target, updatedAt: new Date().toISOString() });
      game = applyAction(game, base.actions[step]); game.log = [];
    }
    gamesDone++;
    atomic(join(dir, "complete.json"), { gameIndex: i, states: target, continuations: target * cfg.repeats });
    console.log(JSON.stringify({ gameIndex: i, states: target, gamesDone, statesDone, worker }));
  }
  atomic(join(out, `worker-${worker}.json`), { worker, pid: process.pid, phase: "complete", gamesDone, statesDone, rolloutActions, seconds: (performance.now() - started) / 1000, updatedAt: new Date().toISOString() });
}
