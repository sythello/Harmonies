"""Sampling, persistence and real multi-iteration/resume integration tests."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from loop import batches, repair_tail
from loop_metrics import collection_metrics, game_metrics, merge_moments, moments
from replay import ReplayBuffer, sampling_passes, window_size
from train import SCHEMA, batch, load

ROOT = Path(__file__).resolve().parents[1]


class ReplayTest(unittest.TestCase):
    def test_katago_window_schedule_and_without_replacement_passes(self):
        expected = {0: 0, 10: 10, 100000: 100000, 150000: 117512, 200000: 132072,
                    1000000: 277827, 10000000: 1029451}
        for n, w in expected.items():
            self.assertEqual(window_size(n), w)
        passes = list(sampling_passes(20, 27, 20, 71))
        self.assertEqual(list(map(len, passes)), [7, 7, 6])
        for rows in passes:
            self.assertEqual(len(rows), len(set(rows)))
            self.assertTrue(set(rows) <= set(range(20, 27)))
        self.assertNotEqual(passes[0], passes[1])
        self.assertEqual(passes, list(sampling_passes(20, 27, 20, 71)))
        self.assertEqual([i for b in batches(20, 27, 20, 8, 71) for i in b], sum(passes, []))
        production = list(batches(1, 123458, 400000, 64, 17))
        self.assertEqual(len(production), 6250)
        self.assertTrue(all(len(b) == 64 for b in production))

    def test_replay_atomic_games_dedup_and_float32_batch_parity(self):
        sample = {"numeric": [0.] * SCHEMA["featureSize"], "cards": [0] * SCHEMA["cardInputs"],
                  "legal": [0, 1], "policy": [0.25, 0.75], "actionId": 1, "seat": 0,
                  "value": {"rank": [1, -1], "relative": [0.1, -0.1], "absolute": [0.3, 0.2]}}
        game = {"version": SCHEMA["version"], "phase": "ended", "samples": [sample], "actions": [{}],
                "seed": 5, "ranks": [1, 2], "scores": [30, 20]}
        with tempfile.TemporaryDirectory() as tmp:
            buffer = ReplayBuffer(Path(tmp) / "replay.sqlite3")
            self.assertTrue(buffer.add_game(game, 1, 0, {}))
            self.assertFalse(buffer.add_game(game, 1, 0, {}))
            self.assertEqual(buffer.totals(), {"games": 1, "rows": 1})
            actual, expected = buffer.batch([1, 1]), batch([sample, sample], "cpu")
            for a, e in zip(actual[:4], expected[:4]):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
            for h in actual[4]:
                torch.testing.assert_close(actual[4][h], expected[4][h], rtol=0, atol=0)
            bad = copy.deepcopy(game)
            bad["seed"] = 6
            bad["samples"][0]["policy"] = [1, 1]
            with self.assertRaises(ValueError):
                buffer.add_game(bad, 1, 1, {})
            self.assertEqual(buffer.totals(), {"games": 1, "rows": 1})
            with self.assertRaisesRegex(ValueError, "changed"):
                changed = copy.deepcopy(game)
                changed["seed"] = 7
                buffer.add_game(changed, 1, 0, {})
            buffer.close()
            buffer = ReplayBuffer(Path(tmp) / "replay.sqlite3")
            self.assertEqual(buffer.totals()["rows"], 1)
            buffer.close()

    def test_metrics_merge_and_partial_record_recovery(self):
        combined = merge_moments([moments([1, 2]), moments([9]), moments([])])
        expected = moments([1, 2, 9])
        for key in expected:
            self.assertAlmostEqual(combined[key], expected[key])
        self.assertIsNone(merge_moments([moments([])])["mean"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "games.jsonl"
            path.write_bytes(b'{"ok":1}\n{"partial":')
            repair_tail(path)
            self.assertEqual(path.read_bytes(), b'{"ok":1}\n')
            path.write_bytes(b'{invalid}\n')
            with self.assertRaises(json.JSONDecodeError):
                repair_tail(path)


class LoopIntegrationTest(unittest.TestCase):
    def command(self, directory, iterations, *options):
        if not (directory / "config.json").exists():
            options = ("--gate-games", "2", "--gate-simulations", "4", *options)
        return subprocess.check_output([sys.executable, "rl/loop.py", "--out", str(directory),
                                        "--iterations", str(iterations), *options], cwd=ROOT, text=True)

    def test_real_loop_model_refresh_sampling_and_interrupted_training_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.command(directory, 1, "--games-per-iteration", "2", "--simulations", "4",
                         "--train-rows", "513", "--min-rows", "50", "--taper-scale", "25", "--hidden", "16", "16")
            self.command(directory, 2)  # inherits original configuration
            reports = [json.loads((directory / f"iteration-{i:04d}" / "metrics.json").read_text()) for i in (1, 2)]
            self.assertEqual(reports[1]["totals"]["games"], 4)
            self.assertEqual(reports[1]["sourceModel"], "iteration-0001/model")
            self.assertEqual(reports[1]["training"]["totalSteps"], 18)
            self.assertEqual(reports[1]["selfplayModel"], reports[0]["bestModel"])
            for i, report in enumerate(reports, 1):
                path = directory / f"iteration-{i:04d}"
                ids = np.fromfile(path / "sampled-rows.i64", dtype="<i8")
                self.assertEqual(len(ids), 513)
                self.assertEqual(report["training"]["steps"], 9)
                window = report["window"]
                self.assertTrue((ids >= window["firstRowId"]).all() and (ids <= window["lastRowId"]).all())
                for start in range(0, len(ids), window["rows"]):
                    segment = ids[start:start+window["rows"]]
                    self.assertEqual(len(segment), len(set(segment)))
                self.assertEqual(report["selfplay"]["legality"]["illegalMoves"], 0)
                self.assertEqual(report["selfplay"]["legality"]["maskConflicts"], 0)
                games = [json.loads(line) for line in (path / "games.jsonl").read_text().splitlines()]
                merged = collection_metrics([game_metrics(g) for g in games])
                self.assertEqual(merged, report["selfplay"])
                self.assertEqual(report["selfplay"]["simulations"], report["selfplay"]["rows"] * 4)
                self.assertEqual(report["gatekeeper"]["games"], 2)
                self.assertTrue(report["gatekeeper"]["complete"])
                self.assertEqual(report["gatekeeper"]["promoted"], report["gatekeeper"]["winRate"] >= 0.5)
            initial, _ = load(directory / "initial.pt")
            trained, _ = load(directory / "iteration-0002/model.pt")
            self.assertFalse(torch.equal(initial.trunk[0].weight, trained.trunk[0].weight))
            # Simulate a crash after training wrote weights, before iteration commit.
            second = directory / "iteration-0002"
            (second / "metrics.json").unlink()
            (second / "training.jsonl").write_text("interrupted\n")
            (second / "model.pt").write_bytes(b"incomplete checkpoint")
            self.command(directory, 2)
            replayed, _ = load(second / "model.pt")
            for k, v in trained.state_dict().items():
                torch.testing.assert_close(v, replayed.state_dict()[k], rtol=0, atol=0)
            buffer = ReplayBuffer(directory / "replay.sqlite3")
            self.assertEqual(buffer.totals()["games"], 4)
            buffer.close()
            before = (second / "model.pt").read_bytes()
            self.command(directory, 2)  # completed target is a no-op
            self.assertEqual(before, (second / "model.pt").read_bytes())
            self.assertIn("gradnorm", (directory / "dashboard.html").read_text())

    def test_warmup_and_resume_partial_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.command(directory, 1, "--games-per-iteration", "2", "--simulations", "2", "--hidden", "8", "8", "--wait-for-min-rows")
            first = directory / "iteration-0001"
            report = json.loads((first / "metrics.json").read_text())
            self.assertTrue(report["warmup"])
            self.assertIsNone(report["training"])
            lines = (first / "games.jsonl").read_bytes().splitlines(keepends=True)
            (first / "metrics.json").unlink()
            # Remove the database too, modelling a crash before second game's ingestion.
            (directory / "replay.sqlite3").unlink()
            (first / "games.jsonl").write_bytes(lines[0] + b'{"partial":')
            self.command(directory, 1)
            resumed = [json.loads(line) for line in (first / "games.jsonl").read_text().splitlines()]
            original = [json.loads(line) for line in lines]
            for a, b in zip(resumed, original):
                a.pop("collectionSeconds")
                b.pop("collectionSeconds")
                self.assertEqual(a, b)

    def test_default_trains_immediately_below_min_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.command(directory, 1, "--games-per-iteration", "1", "--simulations", "2", "--train-rows", "513", "--hidden", "8", "8")
            report = json.loads((directory / "iteration-0001/metrics.json").read_text())
            self.assertLess(report["totals"]["rows"], 100000)
            self.assertEqual(report["window"]["rows"], report["totals"]["rows"])
            self.assertEqual(report["training"]["rows"], 513)
            self.assertGreater(report["training"]["sampling"]["passes"], 1)


if __name__ == "__main__":
    unittest.main()
