import assert from "node:assert/strict";
import { fork } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { seededRandom } from "../src/game/engine";
import { Network } from "../src/game/ai/rl/network";
import { searchConfig } from "../src/game/ai/rl/mcts";
import { collectCases } from "./audit/mcts-audit";
import { enumerateTurn } from "./audit/mcts-audit";
import { scorePlayer as referenceScore } from "../tests/helpers/reference-score";
import { applyAction, scorePlayer } from "../src/game/engine";
import { actionId, encode } from "../src/game/ai/rl/encoding";
import { observe, sampleGame } from "../src/game/ai/observation";
import { legalActions } from "../src/game/ai/actions";
import { evaluateSample, optimizeGame, type FitGame } from "./audit/nn-fit";

const args = process.argv.slice(2),
  command = args[0];
const option = (name: string, fallback: string) =>
  args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const dir = resolve(option("--out", "rl/runs/nn-fit-20261012"));
const json = (path: string) => JSON.parse(readFileSync(path, "utf8"));
const write = (path: string, value: unknown) =>
  writeFileSync(path, JSON.stringify(value) + "\n");
const dataset = (): FitGame[] =>
  readFileSync(`${dir}/dataset.jsonl`, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));

async function pool(kind: string, jobs: number[], extra: string[] = []) {
  let cursor = 0,
    completed = 0;
  const count = Math.min(jobs.length, Number(option("--workers", "4")));
  assert.ok(count >= 1 && Number.isInteger(count));
  const workers = Array.from({ length: count }, () =>
    fork(
      fileURLToPath(import.meta.url),
      ["worker", "--kind", kind, "--out", dir, ...extra],
      {
        execArgv: ["--import", "tsx"],
        stdio: ["ignore", "inherit", "inherit", "ipc"],
      },
    ),
  );
  try {
    return await Promise.all(
      workers.map(
        (worker) =>
          new Promise<unknown[]>((done, fail) => {
            const rows: unknown[] = [];
            let finished = false;
            const send = () => {
              if (cursor === jobs.length) {
                finished = true;
                worker.disconnect();
                done(rows);
              } else worker.send(jobs[cursor++]);
            };
            worker.on("error", fail);
            worker.on("exit", (code) => {
              if (!finished) fail(new Error(`Worker exited early: ${code}`));
            });
            worker.on("message", (row: any) => {
              if (row.error) {
                fail(new Error(row.error));
                return;
              }
              rows.push(row);
              completed++;
              if (
                kind === "collect" &&
                (completed % 25 === 0 || completed === jobs.length)
              )
                console.log(
                  JSON.stringify({
                    event: "collection_progress",
                    completed,
                    total: jobs.length,
                  }),
                );
              send();
            });
            send();
          }),
      ),
    );
  } finally {
    for (const w of workers) w.kill();
  }
}

async function main() {
  mkdirSync(dir, { recursive: true });
  if (command === "worker") {
    const kind = option("--kind", "");
    const raw =
      kind === "collect"
        ? (json(`${dir}/random-games.json`) as ReturnType<typeof collectCases>)
        : undefined;
    const manifest =
      kind === "collect" ? json(`${dir}/manifest.json`) : undefined;
    const games = kind !== "collect" ? dataset() : undefined;
    const samples = games?.flatMap((g) =>
      g.samples.map((s) => ({ sample: s, split: g.split, seed: g.seed })),
    );
    const model =
      kind === "evaluate"
        ? new Network(json(option("--model", "")))
        : undefined;
    process.on("message", (index: number) => {
      try {
        if (kind === "collect") {
          const start = raw!.cases.find(
            (c) => c.gameIndex === index + 1 && c.input.active === 0,
          )!;
          const row = optimizeGame(
            raw!.games[index],
            start,
            manifest.testGameIndices.includes(index + 1) ? "test" : "train",
          );
          write(`${dir}/games/${String(index + 1).padStart(3, "0")}.json`, row);
          process.send!({
            index,
            seconds: row.samples.reduce((s, r) => s + r.exact.seconds, 0),
          });
        } else if (kind === "verify") {
          const sample = samples![index].sample;
          let states = 0;
          const proven: { id: number; score: number }[] = [];
          assert.deepEqual(encode(sample.input), {
            numeric: sample.numeric,
            cards: sample.cards,
          });
          assert.equal(
            sample.visits.reduce((a, b) => a + b, 0),
            1024,
          );
          assert.equal(sample.exact.prefixPaths.length, 512);
          const root = sampleGame(sample.input, 0);
          assert.deepEqual(
            legalActions(root).map((a) => actionId(sample.input, a)),
            sample.legal,
          );
          assert.deepEqual(
            sample.exact.root.map((e) => e.id),
            sample.legal,
          );
          for (const edge of sample.exact.root) {
            assert.equal(
              edge.prefixVisits,
              sample.exact.prefixPaths.filter((p) => p[0] === edge.id).length,
            );
            assert.equal(
              edge.targetVisits,
              edge.prefixVisits + (edge.id === sample.actionId ? 512 : 0),
            );
            // Independent DFS for every first action, including every tied optimum.
            {
              const a = legalActions(root).find(
                (a) => actionId(sample.input, a) === edge.id,
              )!;
              const result = enumerateTurn(
                observe(applyAction(root, a)),
                60_000,
              );
              assert.equal(result.complete, true);
              assert.equal(result.plans[0].score.total, edge.score);
              states += result.states;
              proven.push({ id: edge.id, score: result.plans[0].score.total });
            }
          }
          const best = Math.max(...proven.map((e) => e.score));
          assert.equal(best, sample.exact.score);
          assert.deepEqual(
            proven.filter((e) => e.score === best).map((e) => e.id),
            sample.exact.optimalIds,
          );
          let game = root;
          for (const a of sample.exact.plan) game = applyAction(game, a);
          assert.equal(
            referenceScore(game.players[sample.seat], "A").total,
            sample.exact.score,
          );
          assert.equal(
            scorePlayer(game.players[sample.seat], "A").total,
            sample.exact.score,
          );
          process.send!({
            index,
            id: sample.id,
            score: sample.exact.score,
            states,
            rootActions: proven.length,
          });
        } else {
          const row = samples![index];
          process.send!({
            index,
            split: row.split,
            ...evaluateSample(
              row.sample,
              model!,
              row.seed + 1_000_000,
              Number(option("--simulations", "128")),
            ),
          });
        }
      } catch (error) {
        process.send!({
          error: String(error instanceof Error ? error.stack : error),
        });
      }
    });
    return;
  }
  const start = performance.now();
  if (command === "collect") {
    const count = Number(option("--games", "500")),
      seed = Number(option("--seed", "102000"));
    const testCount = Number(option("--test-games", String(count / 5)));
    assert.ok(
      Number.isInteger(count) &&
        count > 0 &&
        Number.isInteger(testCount) &&
        testCount > 0 &&
        testCount < count,
    );
    let raw: ReturnType<typeof collectCases>;
    try {
      raw = json(`${dir}/random-games.json`);
    } catch {
      raw = collectCases(count, seed);
      write(`${dir}/random-games.json`, raw);
    }
    assert.equal(raw.games.length, count);
    assert.equal(raw.games[0].seed, seed);
    const indices = Array.from({ length: count }, (_, i) => i + 1),
      random = seededRandom(1729);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const manifest = {
      version: 1,
      games: count,
      seed,
      splitSeed: 1729,
      trainGames: count - testCount,
      testGames: testCount,
      testGameIndices: indices.slice(0, testCount).sort((a, b) => a - b),
      chronologicallyOptimized: true,
      prefixSimulations: 512,
      optimalPathVisits: 512,
      rootOnlySupervision: true,
      serving: searchConfig({ simulations: 128 }),
      treeReuse: "within-turn",
      targetScore: "own absolute score",
    };
    write(`${dir}/manifest.json`, manifest);
    mkdirSync(`${dir}/games`, { recursive: true });
    await pool(
      "collect",
      Array.from({ length: count }, (_, i) => i),
    );
    const rows: FitGame[] = Array.from({ length: count }, (_, i) =>
      json(`${dir}/games/${String(i + 1).padStart(3, "0")}.json`),
    );
    assert.equal(rows.flatMap((g) => g.samples).length, 2 * count);
    writeFileSync(
      `${dir}/dataset.jsonl`,
      rows.map((g) => JSON.stringify(g)).join("\n") + "\n",
    );
    const searches = rows.flatMap((g) => g.samples.map((s) => s.exact));
    const summary = {
      games: count,
      samples: 2 * count,
      trainSamples: 2 * (count - testCount),
      testSamples: 2 * testCount,
      wallSeconds: (performance.now() - start) / 1000,
      exactSeconds: searches.reduce((s, e) => s + e.seconds, 0),
      maxExactSeconds: Math.max(...searches.map((e) => e.seconds)),
      states: searches.reduce((s, e) => s + e.states, 0),
      transitions: searches.reduce((s, e) => s + e.transitions, 0),
      simulations: searches.reduce((s, e) => s + e.simulations, 0),
    };
    write(`${dir}/collection.json`, summary);
    console.log(JSON.stringify(summary));
  } else if (command === "verify") {
    const count = dataset().length * 2;
    const rows = (
      await pool(
        "verify",
        Array.from({ length: count }, (_, i) => i),
      )
    ).flat();
    const summary = {
      verified: rows.length,
      seconds: (performance.now() - start) / 1000,
      checks: [
        "independent DFS optimal score on every sample",
        "every root-action optimum and all tied optimal actions on every sample",
        "independent scorer on every optimal endpoint",
        "1024 policy visits and 512 prefix paths",
        "feature encoding",
      ],
    };
    write(`${dir}/verification.json`, summary);
    console.log(JSON.stringify(summary));
  } else if (command === "evaluate") {
    const count = dataset().length * 2,
      limit = Number(option("--limit", String(count)));
    const rows = (
      await pool(
        "evaluate",
        Array.from({ length: Math.min(count, limit) }, (_, i) => i),
        [
          "--model",
          option("--model", ""),
          "--simulations",
          option("--simulations", "128"),
        ],
      )
    ).flat() as any[];
    rows.sort((a, b) => a.index - b.index);
    const output = {
      seconds: (performance.now() - start) / 1000,
      simulations: Number(option("--simulations", "128")),
      rows,
    };
    write(option("--result", `${dir}/evaluation.json`), output);
    console.log(
      JSON.stringify({
        event: "evaluated",
        samples: rows.length,
        seconds: output.seconds,
      }),
    );
  } else
    throw new Error(
      "Usage: scripts/nn-fit.ts collect|evaluate --out DIR [--games 500 --test-games 100 --workers 4] [--model JSON --result JSON]",
    );
}
main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
