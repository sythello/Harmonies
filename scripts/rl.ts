import { createHash } from "node:crypto";
import {
  appendFileSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";
import { createServer } from "node:http";
import { parseArgs } from "node:util";
import { schema } from "../src/game/ai/rl/encoding";
import {
  Network,
  uniformEvaluator,
  type Evaluator,
} from "../src/game/ai/rl/network";
import { defaults, search, SearchTree } from "../src/game/ai/rl/mcts";
import {
  collect,
  evaluate,
  randomAgent,
  mctsAgent,
  type Agent,
} from "../src/game/ai/rl/trajectory";
import { searchTurn } from "../src/game/ai/search";
import { normalizeAI } from "../src/game/ai/config";

const { positionals, values: args } = parseArgs({
  allowPositionals: true,
  options: {
    model: { type: "string" },
    opponent: { type: "string", default: "random" },
    output: { type: "string" },
    input: { type: "string" },
    games: { type: "string", default: "2" },
    simulations: { type: "string", default: "128" },
    seed: { type: "string", default: "1729" },
    port: { type: "string", default: "8765" },
    "heuristic-ms": { type: "string", default: "100" },
  },
});
const integer = (name: keyof typeof args, min = 1) => {
  const n = Number(args[name]);
  if (!Number.isSafeInteger(n) || n < min) throw new Error(`Invalid --${name}`);
  return n;
};
function load(path: string | undefined): { evaluator: Evaluator; id: string } {
  if (path === "uniform") return { evaluator: uniformEvaluator, id: "uniform" };
  if (!path)
    throw new Error(
      "Specify --model path.json (or --model uniform for bootstrap)",
    );
  const text = readFileSync(path, "utf8");
  return {
    evaluator: new Network(JSON.parse(text)),
    id: createHash("sha256").update(text).digest("hex"),
  };
}
function write(path: string, value: unknown) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value) + "\n");
}
async function main() {
  const command = positionals[0];
  if (command === "schema") {
    if (args.output) write(args.output, schema);
    else console.log(JSON.stringify(schema));
    return;
  }
  if (!["collect", "evaluate", "serve", "act", "predict"].includes(command))
    throw new Error(
      "Usage: npm run rl -- schema|collect|evaluate|serve|act|predict [--model model.json]",
    );
  const model = load(args.model),
    simulations = integer("simulations"),
    seed = integer("seed", 0);
  const choose = (evaluator: Evaluator) =>
    mctsAgent(evaluator, { simulations, seed });
  if (command === "collect") {
    if (!args.output)
      throw new Error("collect requires --output trajectories.jsonl");
    mkdirSync(dirname(args.output), { recursive: true });
    for (let i = 0; i < integer("games"); i++) {
      const started = performance.now();
      const trajectory = collect(model.evaluator, {
        seed: seed + i,
        first: i % 2,
        model: model.id,
        search: { simulations, seed: seed + i * 1000003 },
      });
      const collectSeconds = (performance.now() - started) / 1000;
      // Append only complete games. A stopped game cannot leave partially labeled rows.
      appendFileSync(args.output, JSON.stringify(trajectory) + "\n");
      console.log(
        JSON.stringify({
          game: i,
          samples: trajectory.samples.length,
          scores: trajectory.scores,
          collectSeconds,
          writeSeconds: (performance.now() - started) / 1000 - collectSeconds,
          samplesPerSecond: trajectory.samples.length / collectSeconds,
          simulationsPerSecond:
            (trajectory.samples.length * simulations) / collectSeconds,
          reusedVisits: trajectory.samples.reduce(
            (n, s) => n + s.reusedVisits,
            0,
          ),
        }),
      );
    }
  } else if (command === "evaluate") {
    let opponent: Agent;
    let opponentId = args.opponent;
    const heuristicConfig = normalizeAI({
      timeMs: integer("heuristic-ms"),
      beamWidth: 1,
      topK: 1,
      seed,
    });
    if (args.opponent === "random") opponent = randomAgent(seed + 1);
    else if (args.opponent === "heuristic")
      opponent = (view, step) =>
        searchTurn(
          view,
          {
            ...heuristicConfig,
            seed: seed + step,
          },
          { log: false },
        ).actions[0];
    else {
      const loaded = load(args.opponent);
      opponent = choose(loaded.evaluator);
      opponentId = loaded.id;
    }
    const report = {
      model: model.id,
      opponent: args.opponent,
      opponentId,
      ...(args.opponent === "heuristic" ? { heuristicConfig } : {}),
      simulations,
      seed,
      treeReuse: "within-turn",
      ...evaluate(
        choose(model.evaluator),
        opponent,
        Array.from({ length: integer("games") }, (_, i) => seed + i),
      ),
    };
    if (args.output) write(args.output, report);
    console.log(JSON.stringify(report, null, 2));
  } else if (command === "act" || command === "predict") {
    const view = JSON.parse(readFileSync(args.input ?? 0, "utf8"));
    console.log(
      JSON.stringify(
        command === "predict"
          ? model.evaluator.predict(view)
          : search(view, model.evaluator, { simulations, seed }),
      ),
    );
  } else {
    // Explicit game/session keys prevent unrelated clients from sharing a tree.
    // LRU-bound storage; requests without a session retain one-shot behavior.
    const sessions = new Map<string, SearchTree>();
    const server = createServer(async (req, res) => {
      res.setHeader("Content-Type", "application/json");
      if (req.method === "GET" && req.url === "/health") {
        res.end(
          JSON.stringify({
            model: model.id,
            version: schema.version,
            simulations,
          }),
        );
        return;
      }
      if (req.method !== "POST" || req.url !== "/action") {
        res.writeHead(404).end();
        return;
      }
      try {
        let body = "";
        for await (const chunk of req) {
          body += chunk;
          if (body.length > 1000000) {
            res.writeHead(413).end();
            return;
          }
        }
        const view = JSON.parse(body);
        const session = req.headers["x-mcts-session"];
        if (
          session !== undefined &&
          (typeof session !== "string" ||
            !session.length ||
            session.length > 128)
        )
          throw new Error("Invalid X-MCTS-Session");
        let tree = session ? sessions.get(session) : undefined;
        if (!tree) tree = new SearchTree(model.evaluator);
        if (req.headers["x-mcts-reset"] === "true") tree.reset();
        if (session) {
          sessions.delete(session);
          sessions.set(session, tree);
          if (sessions.size > 64)
            sessions.delete(sessions.keys().next().value!);
        }
        const result = tree.search(view, {
          ...defaults,
          simulations,
          seed,
        });
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(400).end(JSON.stringify({ error: String(error) }));
      }
    });
    server.on("error", (error) => {
      console.error(error);
      process.exitCode = 1;
    });
    server.listen(integer("port"), "127.0.0.1", () =>
      console.log(
        `Harmonies RL ${model.id} listening on http://127.0.0.1:${args.port}`,
      ),
    );
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
