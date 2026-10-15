"""Disk-backed, append-only replay rows and KataGo's recent-window schedule."""
import hashlib
import json
import random
import sqlite3
import zlib

import numpy as np
import torch

from train import SCHEMA, VALUE_HEADS, validate_game


def window_size(total, min_rows=100_000, taper_scale=50_000,
                expand_window_per_row=0.4, taper_window_exponent=0.65):
    # KataGo python/shuffle.py: compute_desired_num_rows, add_to_data_rows=0.
    if total <= min_rows:
        return total
    x = total - min_rows + taper_scale
    growth = (x ** taper_window_exponent - taper_scale ** taper_window_exponent)
    growth /= taper_window_exponent * taper_scale ** (taper_window_exponent - 1)
    return min(total, max(min_rows, int(min_rows + expand_window_per_row * growth)))


def sampling_passes(start, stop, rows, seed):
    """Yield a fresh uniform sample without replacement, only after the prior pass."""
    if not 1 <= start < stop or rows < 1:
        raise ValueError("Cannot sample an empty window or nonpositive row count")
    rng = random.Random(seed)
    while rows:
        take = min(stop - start, rows)
        yield rng.sample(range(start, stop), take)
        rows -= take


def encode_row(sample):
    # Little-endian float32 is also exact for this schema's small integer IDs.
    vector = np.asarray([*sample["numeric"], *sample["cards"],
                         *(x for h in VALUE_HEADS for x in sample["value"][h]),
                         len(sample["legal"]), *sample["legal"], *sample["policy"]], dtype="<f4")
    return zlib.compress(vector.tobytes(), level=1)


def decode_row(blob):
    vector = np.frombuffer(zlib.decompress(blob), dtype="<f4")
    n, c = SCHEMA["featureSize"], SCHEMA["cardInputs"]
    count = int(vector[n + c + 6])
    if len(vector) != n + c + 7 + count * 2:
        raise ValueError("Corrupt replay row")
    return (vector[:n], vector[n:n+c], vector[n+c:n+c+6].reshape(3, 2),
            vector[n+c+7:n+c+7+count].astype(np.int64), vector[n+c+7+count:])


class ReplayBuffer:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY, iteration INTEGER NOT NULL, ordinal INTEGER NOT NULL,
                seed INTEGER UNIQUE NOT NULL, rows INTEGER NOT NULL, digest TEXT NOT NULL,
                metrics TEXT NOT NULL, UNIQUE(iteration, ordinal));
            CREATE TABLE IF NOT EXISTS rows (id INTEGER PRIMARY KEY, game INTEGER NOT NULL, data BLOB NOT NULL);
        """)
        schema = json.dumps(SCHEMA, sort_keys=True)
        saved = self.db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
        if saved and saved[0] != schema:
            raise ValueError("Incompatible replay schema")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('schema', ?)", (schema,))

    def close(self):
        self.db.close()

    def totals(self):
        games, rows = self.db.execute("SELECT count(*), coalesce(sum(rows),0) FROM games").fetchone()
        return {"games": games, "rows": rows}

    def add_game(self, game, iteration, ordinal, metrics):
        validate_game(game)
        digest = hashlib.sha256(json.dumps(game, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        old = self.db.execute("SELECT digest FROM games WHERE iteration=? AND ordinal=?", (iteration, ordinal)).fetchone()
        if old:
            if old[0] != digest:
                raise ValueError("Previously ingested game changed")
            return False
        # One transaction per complete game: a crash cannot leave orphan/partial rows.
        with self.db:
            cursor = self.db.execute("INSERT INTO games(iteration,ordinal,seed,rows,digest,metrics) VALUES(?,?,?,?,?,?)",
                                     (iteration, ordinal, game["seed"], len(game["samples"]), digest, json.dumps(metrics)))
            self.db.executemany("INSERT INTO rows(game,data) VALUES(?,?)",
                                ((cursor.lastrowid, encode_row(s)) for s in game["samples"]))
        return True

    def game_metrics(self, iteration):
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT metrics FROM games WHERE iteration=? ORDER BY ordinal", (iteration,))]

    def batch(self, ids, device="cpu"):
        # Bound SQL variables independently of user batch size; preserve sampled order.
        blobs = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start+500]
            blobs.update(self.db.execute(f"SELECT id,data FROM rows WHERE id IN ({','.join('?' for _ in chunk)})", chunk))
        if any(i not in blobs for i in ids):
            raise ValueError("Missing replay row")
        rows = [decode_row(blobs[i]) for i in ids]
        numeric, cards, values = (np.stack([row[i] for row in rows]) for i in range(3))
        target = np.zeros((len(ids), SCHEMA["actionSize"]), dtype=np.float32)
        mask = np.zeros_like(target, dtype=bool)
        for i, row in enumerate(rows):
            mask[i, row[3]] = True
            target[i, row[3]] = row[4]
        tensor = lambda data: torch.from_numpy(data).to(device)
        return (tensor(numeric), tensor(cards.astype(np.int64)), tensor(target), tensor(mask),
                {h: tensor(values[:, i, :].copy()) for i, h in enumerate(VALUE_HEADS)})
