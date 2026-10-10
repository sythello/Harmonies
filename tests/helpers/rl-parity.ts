import { readFileSync } from "node:fs";
import { applyAction } from "../../src/game/engine";
import { legalActions } from "../../src/game/ai/actions";
import { observe } from "../../src/game/ai/observation";
import { encode } from "../../src/game/ai/rl/encoding";
import { Network } from "../../src/game/ai/rl/network";
import { newGame } from "../../src/game/ai/rl/trajectory";

const model = new Network(JSON.parse(readFileSync(process.argv[2], "utf8")));
let game = newGame(913);
const rows = [];
for (let step = 0; step < 20 && game.phase === "playing"; step++) {
  const view = observe(game);
  rows.push({ features: encode(view), prediction: model.predict(view) });
  const actions = legalActions(game);
  game = applyAction(game, actions[(step * 17) % actions.length]);
}
console.log(JSON.stringify(rows));
