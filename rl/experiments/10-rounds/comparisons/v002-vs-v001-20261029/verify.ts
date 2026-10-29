/** Independently replay recorded moves; do not invoke search or change training runs. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { applyAction, CARDS, createGame, ranking } from '../../../../../src/game/engine.ts';
import { legalActions } from '../../../../../src/game/ai/actions.ts';

const here = dirname(fileURLToPath(import.meta.url));
const run = resolve(here, 'run');
const root = resolve(here, '../../../../..');
const config = JSON.parse(readFileSync(resolve(run, 'config.json'), 'utf8'));
const bytes = readFileSync(resolve(run, 'games.jsonl'));
const games = bytes.toString().trim().split('\n').map(line => JSON.parse(line));
const hashes = [config.candidate, config.opponent].map(model => {
  const hash = createHash('sha256').update(readFileSync(resolve(root, model.path))).digest('hex');
  assert.equal(hash, model.sha256);
  return hash;
});
assert.equal(games.length, 100);
let actions = 0, reusedDecisions = 0;
for (const [index, record] of games.entries()) {
  assert.equal(record.seed, config.seed + Math.floor(index / 2));
  assert.equal(record.first, index % 2);
  assert.equal(record.purpose, 'gatekeeper');
  assert.equal(record.phase, 'ended');
  assert.equal(record.treeReuse, 'within-turn');
  assert.deepEqual(record.models, hashes);
  assert.equal(record.search.simulations, 128);
  assert.equal(record.search.noiseWeight, 0);
  assert.equal(record.search.temperature, 0);
  assert.equal(record.search.seed, record.seed);
  assert.ok(!('samples' in record));
  assert.ok(record.actions.length > 0);
  assert.equal(record.actions.length, record.decisions.length);
  let game = createGame({ names: ['P0', 'P1'], side: 'A', spirits: false, seed: record.seed, first: record.first });
  for (const [step, action] of record.actions.entries()) {
    const decision = record.decisions[step];
    assert.equal(decision.seat, game.active);
    assert.equal(decision.playerId, game.players[game.active].id);
    assert.equal(decision.rootVisits - decision.reusedVisits, 128);
    assert.ok(decision.reusedVisits >= 0);
    if (decision.reusedVisits > 0) reusedDecisions++;
    assert.ok(legalActions(game).some(legal => isDeepStrictEqual(legal, action)),
      `Illegal action at game ${index}, step ${step}`);
    game = applyAction(game, action);
    game.log = [];
    actions++;
  }
  assert.equal(game.phase, 'ended');
  assert.equal(game.round, record.rounds);
  assert.deepEqual(game.players.map(p => p.turns), record.turns);
  const ranked = ranking(game);
  const entries = game.players.map(p => ranked.find(r => r.player.id === p.id)!);
  assert.deepEqual(entries.map(r => r.score.total), record.scores);
  assert.deepEqual(entries.map(r => r.rank), record.ranks);
  assert.deepEqual(entries.map(r => r.score.cubes), record.settledAnimals);
  assert.deepEqual(game.players.map(p => p.cards.filter(c => !CARDS[c.id].spirit && c.placed === CARDS[c.id].points.length).length), record.completedAnimalCards);
  const candidate = ranked.find(r => r.player.id === 0)!;
  const opponent = ranked.find(r => r.player.id === 1)!;
  assert.equal(candidate.score.total, record.candidateScore);
  assert.equal(opponent.score.total, record.incumbentScore);
  assert.equal(record.result, candidate.rank === opponent.rank ? 0.5 : Number(candidate.rank < opponent.rank));
}
for (const file of readdirSync(run).filter(f => f.endsWith('-errors.jsonl'))) {
  assert.equal(readFileSync(resolve(run, file), 'utf8').trim(), '', file);
}
const result = { passed: true, games: games.length, pairedSeeds: games.length / 2,
  actions, reusedDecisions, additionalSimulations: actions * 128,
  illegalMoves: 0, maskConflictRecords: 0, modelHashes: hashes,
  gamesSha256: createHash('sha256').update(bytes).digest('hex'),
  checks: ['seed pairs', 'model identities', 'serving settings', '128 new simulations per action',
    'active player mapping', 'legal action membership', 'complete engine replay',
    'terminal scores and ranks', 'animal and card totals', 'outcome', 'no training samples', 'no error records'] };
writeFileSync(resolve(run, 'replay-verification.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
