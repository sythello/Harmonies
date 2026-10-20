"""Gate boundary, separate model routing, and real serving-match resume tests."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gatekeeper import summarize_gate
from loop import DEFAULTS, run
from train import load

ROOT = Path(__file__).resolve().parents[1]


def fake_games(results):
    return [{"result": result, "candidateScore": 10, "incumbentScore": 10, "first": i % 2,
             "rounds": 1, "turns": [1, 1], "ranks": [1, 1], "decisions": []}
            for i, result in enumerate(results)]


class GatekeeperTest(unittest.TestCase):
    def test_exact_threshold_draw_credit_and_no_early_promotion(self):
        for results, expected in [([1]*50 + [0]*50, True), ([0.5]*100, True),
                                  ([1]*49 + [0.5]*2 + [0]*49, True),
                                  ([1]*49 + [0.5] + [0]*50, False), ([1]*99, None)]:
            result = summarize_gate(fake_games(results), 100)
            self.assertEqual(result["promoted"], expected)
            self.assertEqual(result["winRate"], sum(results)/len(results))
        self.assertIsNone(summarize_gate([], 100)["promoted"])

    def test_rejected_model_still_trains_while_best_collects_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = {**DEFAULTS, "games_per_iteration": 1, "simulations": 2, "train_rows": 65,
                      "cheap_search_probability": 0, "max_window_passes": 0,
                      "hidden": [8, 8], "gate_games": 2, "gate_simulations": 4}
            comparisons = []

            def controlled_gate(path, candidate, incumbent, cfg, iteration, callback):
                comparisons.append((candidate.relative_to(directory).as_posix(), incumbent.relative_to(directory).as_posix()))
                g = summarize_gate(fake_games([1, 1] if iteration == 2 else [0, 0]), 2)
                g.update(candidateModel=str(candidate.relative_to(directory)), incumbentModel=str(incumbent.relative_to(directory)),
                         simulations=4, noiseWeight=0, temperature=0)
                callback(g)
                return g

            with patch("loop.gatekeeper", controlled_gate), contextlib.redirect_stdout(io.StringIO()):
                run(directory, config, 1)  # rejected
                run(directory, config, 3)  # resume: pass, then reject
            reports = [json.loads((directory/f"iteration-{i:04d}/metrics.json").read_text()) for i in (1, 2, 3)]
            self.assertEqual([r["bestModel"] for r in reports], ["initial", "iteration-0002/model", "iteration-0002/model"])
            self.assertEqual([r["selfplayModel"] for r in reports], ["initial", "initial", "iteration-0002/model"])
            self.assertEqual([r["sourceModel"] for r in reports], ["initial", "iteration-0001/model", "iteration-0002/model"])
            self.assertEqual([r["latestModel"] for r in reports], [f"iteration-{i:04d}/model" for i in (1, 2, 3)])
            self.assertEqual(comparisons, [("iteration-0001/model", "initial"), ("iteration-0002/model", "initial"),
                                          ("iteration-0003/model", "iteration-0002/model")])
            self.assertEqual(reports[-1]["totals"]["games"], 3)  # no evaluation rows in replay
            self.assertEqual(load(directory/"iteration-0003/model.pt")[1]["steps"], 6)

    def test_real_gate_resumes_missing_game_without_retraining_or_duplicate_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            command = [sys.executable, "rl/loop.py", "--out", str(directory), "--iterations", "1"]
            subprocess.check_output([*command, "--games-per-iteration", "1", "--simulations", "2",
                                     "--cheap-search-probability", "0", "--max-window-passes", "0",
                                     "--train-rows", "65", "--hidden", "8", "8", "--gate-games", "4",
                                     "--gate-simulations", "4"], cwd=ROOT, text=True)
            current = directory/"iteration-0001"
            report = json.loads((current/"metrics.json").read_text())
            records = (current/"gate-games.jsonl").read_bytes().splitlines(keepends=True)
            checkpoint = current/"model.pt"
            before = checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns
            (current/"metrics.json").unlink()
            (current/"gate-games.jsonl").write_bytes(b"".join(records[:3]) + b'{"unfinished":')
            subprocess.check_output(command, cwd=ROOT, text=True)
            after = json.loads((current/"metrics.json").read_text())
            self.assertEqual(before, (checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns))
            self.assertEqual(after["training"], report["training"])
            self.assertEqual(after["totals"], report["totals"])
            self.assertEqual(after["gatekeeper"]["winRate"], report["gatekeeper"]["winRate"])
            self.assertEqual(after["bestModel"], report["bestModel"])
            resumed = (current/"gate-games.jsonl").read_bytes().splitlines()
            self.assertEqual(len(resumed), 4)
            for a, b in zip(records, resumed):
                x, y = json.loads(a), json.loads(b)
                x.pop("collectionSeconds")
                y.pop("collectionSeconds")
                self.assertEqual(x, y)
            self.assertEqual(after["gatekeeper"]["telemetry"]["rows"],
                             sum(len(json.loads(line)["decisions"]) for line in resumed))

    def test_legacy_run_config_is_upgraded_without_inventing_gate_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            command = [sys.executable, "rl/loop.py", "--out", str(directory), "--iterations", "1"]
            subprocess.check_output([*command, "--games-per-iteration", "1", "--simulations", "2", "--hidden", "8", "8",
                                     "--cheap-search-probability", "0",
                                     "--wait-for-min-rows"], cwd=ROOT, text=True)
            path = directory/"config.json"
            config = json.loads(path.read_text())
            config.pop("gate_games")
            config.pop("gate_simulations")
            for key in ("cheap_search_probability", "cheap_simulations", "max_window_passes"):
                config.pop(key)
            path.write_text(json.dumps(config))
            path = directory/"iteration-0001/metrics.json"
            report = json.loads(path.read_text())
            for key in ("bestModel", "latestModel", "gatekeeper", "selfplayModel"):
                report.pop(key)
            path.write_text(json.dumps(report))
            subprocess.check_output(command, cwd=ROOT, text=True)
            self.assertEqual(json.loads((directory/"config.json").read_text())["gate_games"], 100)
            self.assertEqual(json.loads((directory/"config.json").read_text())["cheap_search_probability"], 0)
            self.assertEqual(json.loads((directory/"config.json").read_text())["max_window_passes"], 0)
            self.assertNotIn("gatekeeper", json.loads(path.read_text()))
            status = json.loads((directory/"state.json").read_text())
            self.assertEqual(status["bestModel"], status["latestModel"])


if __name__ == "__main__":
    unittest.main()
