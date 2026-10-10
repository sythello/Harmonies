"""Integration tests require npm dependencies and rl/requirements.txt."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from train import Model, SCHEMA, export, load, losses, read_games

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True)


class PipelineTest(unittest.TestCase):
    def test_collect_train_resume_export_and_typescript_parity(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "initial")
            trained = str(Path(directory) / "trained")
            data = str(Path(directory) / "games.jsonl")
            command(sys.executable, "rl/train.py", "init", "--hidden", "16", "16", "--out", prefix)
            command("node", "--import", "tsx", "scripts/rl.ts", "collect", "--model", prefix + ".json",
                    "--games", "2", "--simulations", "4", "--output", data)
            games = read_games([data])
            self.assertEqual(len(games), 2)
            command(sys.executable, "rl/train.py", "train", "--resume", prefix + ".pt", "--data", data,
                    "--epochs", "2", "--out", trained)
            initial, before = load(prefix + ".pt")
            model, after = load(trained + ".pt")
            self.assertGreater(after["steps"], before["steps"])
            for head in ("policy", "rank", "relative", "absolute"):
                self.assertFalse(torch.equal(initial.heads[head].weight, model.heads[head].weight))
            self.assertFalse(torch.equal(initial.embedding.weight, model.embedding.weight))
            self.assertFalse(torch.equal(initial.trunk[0].weight, model.trunk[0].weight))
            rows = json.loads(command("node", "--import", "tsx", "tests/helpers/rl-parity.ts", trained + ".json"))
            numeric = torch.tensor([r["features"]["numeric"] for r in rows])
            cards = torch.tensor([r["features"]["cards"] for r in rows])
            with torch.no_grad():
                prediction = model(numeric, cards)
            for name in ("policy", "rank", "relative", "absolute"):
                actual = torch.tensor([r["prediction"]["logits"] if name == "policy" else
                                       r["prediction"]["value"][name] for r in rows])
                torch.testing.assert_close(actual, prediction[name], rtol=2e-5, atol=2e-6)
            # A tiny real batch should be learnable, including the shared embedding.
            samples = games[0]["samples"][:8]
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            start = sum(losses(model, samples, "cpu").values()).item()
            for _ in range(30):
                optimizer.zero_grad()
                loss = sum(losses(model, samples, "cpu").values())
                loss.backward()
                optimizer.step()
            self.assertLess(sum(losses(model, samples, "cpu").values()).item(), start)
            # Invalid/truncated games and bootstrapped values must never enter training.
            invalid = copy.deepcopy(games[0])
            invalid["samples"][0]["value"]["absolute"][0] += 1
            Path(data).write_text(json.dumps(invalid) + "\n")
            with self.assertRaisesRegex(ValueError, "terminal"):
                read_games([data])
            invalid["phase"] = "playing"
            Path(data).write_text(json.dumps(invalid) + "\n")
            with self.assertRaisesRegex(ValueError, "complete"):
                read_games([data])

    def test_illegal_policy_logits_have_no_loss_or_gradient(self):
        model = Model(hidden=(4,), embedding_dim=2)
        sample = {"numeric": [0.0] * SCHEMA["featureSize"], "cards": [0] * SCHEMA["cardInputs"],
                  "legal": [2, 7], "policy": [0.25, 0.75],
                  "value": {"rank": [1.0, -1.0], "relative": [0.2, -0.2], "absolute": [0.8, 0.6]}}
        before = losses(model, [sample], "cpu")["policy"].item()
        with torch.no_grad():
            model.heads["policy"].bias[0] = 1e6
        loss = losses(model, [sample], "cpu")["policy"]
        self.assertEqual(before, loss.item())
        loss.backward()
        self.assertEqual(model.heads["policy"].bias.grad[0].item(), 0)
        self.assertNotEqual(model.heads["policy"].bias.grad[2].item(), 0)


if __name__ == "__main__":
    unittest.main()
