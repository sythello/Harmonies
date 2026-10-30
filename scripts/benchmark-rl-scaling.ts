/** Compare both architectures on identical recorded positions, then real serving games. */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import os from "node:os";
import { applyAction } from "../src/game/engine";
import { observe } from "../src/game/ai/observation";
import { Network, type Evaluator } from "../src/game/ai/rl/network";
import { encode } from "../src/game/ai/rl/encoding";
import { SearchTree } from "../src/game/ai/rl/mcts";
import { evaluateGame, newGame } from "../src/game/ai/rl/trajectory";

const directory = process.argv[2];
const source = readFileSync(join(directory, "data.jsonl"), "utf8")
  .trim()
  .split("\n")
  .map((s) => JSON.parse(s));
const views = [];
for (const record of source.slice(0, 3)) {
  let game = newGame(record.seed, record.first);
  const selected = new Set(
    [0, 0.25, 0.5, 0.75].map((f) => Math.floor(record.actions.length * f)),
  );
  for (const [step, action] of record.actions.entries()) {
    if (selected.has(step)) views.push(observe(game));
    game = applyAction(game, action);
    game.log = [];
  }
}
const features = views.map(encode);
writeFileSync(join(directory, "positions.json"), JSON.stringify(views));
const names = ["ffn-v2", "spatial-v3"];
const networks = names.map(
  (name) =>
    new Network(
      JSON.parse(readFileSync(join(directory, name + "-initial.json"), "utf8")),
    ),
);
const metrics = names.map((name) => ({
  name,
  forwardMs: [] as number[],
  predictMs: [] as number[],
  search: {
    "16": [] as { ms: number; nnCalls: number }[],
    "128": [] as { ms: number; nnCalls: number }[],
  },
  servingGames: [] as {
    seconds: number;
    decisions: number;
    simulations: number;
    reusedDecisions: number;
  }[],
}));
let checksum = 0;
// JIT warmup for both models; loading and model validation are outside timed blocks.
for (let i = 0; i < 120; i++)
  for (const network of networks)
    checksum += network.predict(views[i % views.length]).logits[0];
for (let repeat = 0; repeat < 3; repeat++) {
  for (const index of repeat % 2 ? [1, 0] : [0, 1]) {
    const network = networks[index],
      metric = metrics[index];
    for (const kind of ["forward", "predict"] as const) {
      const started = performance.now(),
        count = 240;
      for (let n = 0; n < count; n++)
        checksum += (
          kind === "forward"
            ? network.forward(features[n % views.length])
            : network.predict(views[n % views.length])
        ).logits[0];
      metric[kind === "forward" ? "forwardMs" : "predictMs"].push(
        (performance.now() - started) / count,
      );
    }
    for (const simulations of [16, 128] as const) {
      let nnCalls = 0;
      const counted: Evaluator = {
        predict: (view) => {
          nnCalls++;
          return network.predict(view);
        },
      };
      const started = performance.now();
      for (let v = 0; v < views.length; v++) {
        const result = new SearchTree(counted).search(views[v], {
          simulations,
          noiseWeight: 0,
          temperature: 0,
          seed: 925000 + v,
        });
        if (result.rootVisits - result.reusedVisits !== simulations)
          throw new Error("Search budget mismatch");
        checksum += result.actionId;
      }
      metric.search[String(simulations) as "16" | "128"].push({
        ms: (performance.now() - started) / views.length,
        nnCalls: nnCalls / views.length,
      });
    }
    console.log(
      JSON.stringify({
        event: "serving-benchmark",
        repeat,
        model: metric.name,
      }),
    );
  }
}
for (let first = 0; first < 2; first++)
  for (const index of first ? [1, 0] : [0, 1]) {
    const started = performance.now();
    const game = evaluateGame(networks[index], networks[index], {
      seed: 2026103100,
      first,
      simulations: 128,
      models: [names[index], names[index]],
    });
    metrics[index].servingGames.push({
      seconds: (performance.now() - started) / 1000,
      decisions: game.decisions.length,
      simulations: game.decisions.length * 128,
      reusedDecisions: game.decisions.filter((d) => d.reusedVisits > 0).length,
    });
    writeFileSync(
      join(directory, `${names[index]}-serving-${first}.json`),
      JSON.stringify(game),
    );
    console.log(
      JSON.stringify({ event: "serving-game", model: names[index], first }),
    );
  }
writeFileSync(
  join(directory, "serving.json"),
  JSON.stringify(
    {
      hardware: {
        node: process.version,
        cpu: os.cpus()[0].model,
        platform: os.platform(),
        arch: os.arch(),
      },
      positions: views.length,
      repeats: 3,
      inferenceCallsPerRepeat: 240,
      note: "Initial random weights; fixed identical positions for microbenchmarks. Fresh trees for isolated search, within-turn reuse for complete games.",
      checksum,
      metrics,
    },
    null,
    2,
  ) + "\n",
);
