"""Small serial CPU training/serving comparison on the same real v2 samples."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from train import Model, batch, export, loss_components, save, validate_game

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--data', default='rl/experiments/10-rounds/v002-playout-cap/run/iteration-0010/games.jsonl')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    games, samples = [], []
    with Path(args.data).open() as stream, (out / 'data.jsonl').open('w') as target:
        for line in stream:
            game = json.loads(line)
            validate_game(game)
            games.append(game)
            samples.extend(game['samples'])
            target.write(line)
            if len(samples) >= 512:
                break
    samples = samples[:512]
    assert len(samples) == 512
    tensors = [batch(samples[i:i+64], 'cpu') for i in range(0, len(samples), 64)]
    results, models, optimizers = {}, {}, {}
    for name in ('ffn-v2', 'spatial-v3'):
        torch.manual_seed(1729)
        model = Model(architecture=name)
        models[name] = model
        optimizers[name] = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        export(model, out / (name + '-initial.json'))
        results[name] = {'parameters': sum(p.numel() for p in model.parameters()), 'config': model.config,
                         'trainMsPerStep': [], 'firstLoss': None, 'lastLoss': None}

    def step(name, index):
        model, optimizer = models[name], optimizers[name]
        numeric, cards, target, mask, values = tensors[index % len(tensors)]
        optimizer.zero_grad()
        losses = loss_components(model(numeric, cards), target, mask, values)
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        return {k: float(v.detach()) for k, v in losses.items()}

    for name in models:
        for i in range(10):
            loss = step(name, i)
            if not i: results[name]['firstLoss'] = loss
    for repeat in range(3):
        for name in (list(models) if repeat % 2 == 0 else list(reversed(models))):
            started = time.perf_counter()
            for i in range(args.steps):
                loss = step(name, 10 + repeat * args.steps + i)
            results[name]['trainMsPerStep'].append((time.perf_counter() - started) * 1000 / args.steps)
            results[name]['lastLoss'] = loss
            print(json.dumps({'event': 'train-benchmark', 'repeat': repeat, 'model': name,
                              'msPerStep': results[name]['trainMsPerStep'][-1]}), flush=True)
    for name, model in models.items():
        save(model, optimizers[name], out / (name + '-trained'), 10 + 3 * args.steps)
    subprocess.run(['node', '--import', 'tsx', 'scripts/benchmark-rl-scaling.ts', str(out)], cwd=ROOT, check=True)
    serving = json.loads((out / 'serving.json').read_text())
    for row in serving['metrics']:
        row = dict(row)
        name = row.pop('name')
        results[name].update(row)
        results[name]['medianTrainMsPerStep'] = statistics.median(results[name]['trainMsPerStep'])
        results[name]['trainRowsPerSecond'] = 64000 / results[name]['medianTrainMsPerStep']
        results[name]['medianPredictMs'] = statistics.median(row['predictMs'])
        results[name]['medianSearchMs'] = {cap: statistics.median(x['ms'] for x in trials) for cap, trials in row['search'].items()}
    old, new = results.values()
    ratios = {key: new[key] / old[key] for key in ['parameters', 'medianTrainMsPerStep', 'medianPredictMs']}
    ratios['search16'] = new['medianSearchMs']['16'] / old['medianSearchMs']['16']
    ratios['search128'] = new['medianSearchMs']['128'] / old['medianSearchMs']['128']
    report = {'hardware': serving['hardware'], 'torch': torch.__version__, 'threads': 1, 'device': 'cpu',
              'samples': len(samples), 'sourceGames': len(games), 'source': str(Path(args.data)),
              'sourceSubsetSha256': hashlib.sha256((out / 'data.jsonl').read_bytes()).hexdigest(),
              'batchSize': 64, 'warmupSteps': 10, 'stepsPerRepeat': args.steps, 'repeats': 3,
              'timingScope': 'Prebuilt tensors; forward + losses + backward + clip + AdamW. Excludes JSON/SQLite parsing and per-parameter monitoring.',
              'models': results, 'newOverOldLatencyRatio': ratios, 'servingPositions': serving['positions'],
              'sourceSha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [ROOT/'rl/train.py', ROOT/'rl/spatial.json', Path(__file__), ROOT/'scripts/benchmark-rl-scaling.ts',
                   *sorted((ROOT/'src/game/ai/rl').glob('*.ts'))]}}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'ratios': ratios, 'report': str(out / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
