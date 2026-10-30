"""Run 400 independent paired matches through the existing serving gate CLI."""
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'rl/gatekeeper.py').exists())
SOURCE = ROOT / 'rl/experiments/10-rounds/v003-spatial/run/source'
sys.path.insert(0, str(SOURCE / 'rl'))
from loop_metrics import atomic_json, repair_tail

RUN = HERE / 'run'
SEED = 3000001729
GAMES = 400
SHARDS = [(0, 100), (100, 100), (200, 100), (300, 100)]
BASE = ROOT / 'rl/experiments/10-rounds'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.endswith('\n'):
                yield json.loads(line)


def main():
    RUN.mkdir(exist_ok=True)
    with (RUN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        models = []
        data_hashes = {}
        for version in ('v003-spatial', 'v002-playout-cap'):
            folder = BASE / version
            manifest = json.loads((folder / 'manifest.json').read_text())
            path = folder / manifest['best']['json']
            assert digest(path) == manifest['best']['jsonSha256']
            models.append({'version': version, 'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
            database = folder / 'run/replay.sqlite3'
            with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
                assert db.execute('SELECT count(*) FROM games WHERE seed BETWEEN ? AND ?',
                                  (SEED, SEED + GAMES // 2 - 1)).fetchone()[0] == 0
            for f in (folder / 'run').glob('iteration-*/metrics.json'):
                gate = json.loads(f.read_text())['gatekeeper']
                assert not set(range(gate['seed'], gate['seed'] + gate['games'] // 2)) & set(range(SEED, SEED + GAMES // 2))
            data_hashes[str(database.relative_to(ROOT))] = digest(database)
        config = {'candidate': models[0], 'opponent': models[1], 'games': GAMES, 'pairedSeeds': GAMES // 2,
                  'seed': SEED, 'simulations': 128, 'noiseWeight': 0, 'temperature': 0,
                  'treeReuse': 'within-turn', 'workers': len(SHARDS), 'shards': SHARDS,
                  'ruleScope': 'A-side, two players, no spirits', 'trainingAndGateSeedOverlap': 0,
                  'replaySha256Before': data_hashes}
        # JSON round-trip normalizes the shard tuples for resume comparison.
        config = json.loads(json.dumps(config))
        saved = RUN / 'config.json'
        if saved.exists():
            assert json.loads(saved.read_text()) == config, 'Comparison config changed'
        else:
            atomic_json(saved, config)
        provenance = RUN / 'source.json'
        if not provenance.exists():
            atomic_json(provenance, {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                    'sourceSha256': {str(p.relative_to(ROOT)): digest(p) for p in
                                        [SOURCE / 'scripts/rl.ts', *sorted((SOURCE / 'src/game').rglob('*.ts')), Path(__file__)]}})
        children = []
        started = time.time()
        try:
            for worker, (offset, count) in enumerate(SHARDS):
                output = RUN / f'worker-{worker}.jsonl'
                repair_tail(output)
                existing = list(records(output))
                assert len(existing) <= count
                for j, game in enumerate(existing):
                    index = offset + j
                    assert game['seed'] == SEED + index // 2 and game['first'] == index % 2
                    assert game['models'] == [m['sha256'] for m in models]
                    assert game['search']['simulations'] == 128 and game['search']['noiseWeight'] == game['search']['temperature'] == 0
                if len(existing) == count:
                    continue
                command = ['node', '--import', 'tsx', 'scripts/rl.ts', 'gate',
                           '--model', str(ROOT / models[0]['path']), '--opponent', str(ROOT / models[1]['path']),
                           '--games', str(count - len(existing)), '--game-offset', str(offset + len(existing)),
                           '--simulations', '128', '--seed', str(SEED), '--output', str(output),
                           '--error-output', str(RUN / f'worker-{worker}-errors.jsonl')]
                log = (RUN / f'worker-{worker}.log').open('a')
                child = subprocess.Popen(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
                children.append((child, log, worker))
            atomic_json(RUN / 'process.json', {'pid': __import__('os').getpid(), 'startedAt': started, 'cwd': str(SOURCE), 'children': [{'worker': w, 'pid': c.pid} for c, _, w in children]})
            last = -1
            while True:
                failed = [(w, c.returncode) for c, _, w in children if c.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f'Workers failed: {failed}')
                counts = [sum(1 for _ in records(RUN / f'worker-{w}.jsonl')) for w in range(len(SHARDS))]
                status = {'phase': 'running', 'completedGames': sum(counts), 'requiredGames': GAMES,
                          'workerGames': counts, 'elapsedSeconds': time.time() - started}
                atomic_json(RUN / 'state.json', status)
                if sum(counts) != last:
                    print(json.dumps(status), flush=True)
                    last = sum(counts)
                if all(c.poll() is not None for c, _, _ in children):
                    break
                time.sleep(2)
            assert sum(counts) == GAMES
            temporary = RUN / 'games.jsonl.tmp'
            with temporary.open('w') as out:
                for w in range(len(SHARDS)):
                    with (RUN / f'worker-{w}.jsonl').open() as stream:
                        for line in stream:
                            out.write(line)
            temporary.replace(RUN / 'games.jsonl')
            for path, before in data_hashes.items():
                assert digest(ROOT / path) == before, 'Training replay changed'
            for m in models:
                assert digest(ROOT / m['path']) == m['sha256'], 'Model changed'
            atomic_json(RUN / 'state.json', {**status, 'phase': 'complete', 'completedAt': time.time()})
            print('400 matches complete; original models and replay unchanged', flush=True)
        except BaseException as error:
            atomic_json(RUN / 'state.json', {'phase': 'failed', 'error': str(error)})
            raise
        finally:
            for child, log, _ in children:
                if child.poll() is None:
                    child.terminate()
                child.wait()
                log.close()


if __name__ == '__main__':
    main()
