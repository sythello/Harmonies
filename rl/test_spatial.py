"""Hex geometry/padding, spatial gradients, checkpoint and cross-runtime parity."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from test_pipeline import command
from train import Model, SPATIAL, SCHEMA, export, load, losses, read_games, save


class SpatialTest(unittest.TestCase):
    def test_boundary_is_not_empty_and_weights_shared_across_players(self):
        torch.set_num_threads(1)
        model = Model(architecture="spatial-v3")
        self.assertEqual(model.config['hidden'], [512, 128])
        self.assertEqual(model.trunk[0].in_features, 5334)
        numeric = torch.zeros(1, SCHEMA['featureSize'])
        for start in SPATIAL['playerOffsets']:
            numeric[0, start:start + 667:29] = 1
        with torch.no_grad():
            for layer in model.spatial.filters.values():
                layer.weight.zero_()
                layer.bias.zero_()
                layer.weight[:, 29::30] = 1
            actual = model.spatial(numeric).reshape(5, 2, 23, 16)
        for k, kernel in enumerate(SPATIAL['kernels']):
            expected = torch.tensor([patch.count(23) for patch in kernel['indices']], dtype=torch.float32)
            for player in (0, 1):
                torch.testing.assert_close(actual[k, player], expected[:, None].expand(-1, 16))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2900467)

    def test_spatial_collect_train_checkpoint_and_typescript_parity(self):
        torch.set_num_threads(1)
        torch.manual_seed(410)
        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp) / 'model')
            data = str(Path(tmp) / 'games.jsonl')
            model = Model(hidden=[16, 8], architecture='spatial-v3')
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            export(model, prefix + '.json')
            command('node', '--import', 'tsx', 'scripts/rl.ts', 'collect', '--model', prefix + '.json',
                    '--games', '1', '--simulations', '2', '--output', data)
            samples = read_games([data])[0]['samples'][:16]
            initial = {name: p.detach().clone() for name, p in model.named_parameters()}
            first_loss = sum(losses(model, samples, 'cpu').values()).item()
            for _ in range(15):
                optimizer.zero_grad()
                loss = sum(losses(model, samples, 'cpu').values())
                loss.backward()
                optimizer.step()
            self.assertLess(sum(losses(model, samples, 'cpu').values()).item(), first_loss)
            for name, p in model.named_parameters():
                if name.endswith('weight'):
                    self.assertFalse(torch.equal(initial[name], p), name)
            save(model, optimizer, prefix, 15)
            restored, checkpoint = load(prefix + '.pt')
            self.assertEqual(checkpoint['steps'], 15)
            rows = json.loads(command('node', '--import', 'tsx', 'tests/helpers/rl-parity.ts', prefix + '.json'))
            numeric = torch.tensor([r['features']['numeric'] for r in rows])
            cards = torch.tensor([r['features']['cards'] for r in rows])
            with torch.no_grad():
                expected = restored(numeric, cards)
                original = model(numeric, cards)
            for name in expected:
                actual = torch.tensor([r['prediction']['logits'] if name == 'policy' else r['prediction']['value'][name] for r in rows])
                torch.testing.assert_close(actual, expected[name], rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(original[name], expected[name], rtol=0, atol=0)
            # Missing/mismatched architecture metadata must fail before inference.
            weights = json.loads(Path(prefix + '.json').read_text())
            weights.pop('architecture')
            Path(prefix + '.json').write_text(json.dumps(weights))
            rejected = subprocess.run(['node', '--import', 'tsx', 'tests/helpers/rl-parity.ts', prefix + '.json'],
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('requires filters and target towers', rejected.stderr)

    def test_loop_records_spatial_architecture_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            command(sys.executable, 'rl/loop.py', '--out', tmp, '--iterations', '1',
                    '--architecture', 'spatial-v3', '--hidden', '16', '8',
                    '--games-per-iteration', '2', '--simulations', '2', '--cheap-search-probability', '0',
                    '--train-rows', '64', '--gate-games', '2', '--gate-simulations', '2')
            command(sys.executable, 'rl/loop.py', '--out', tmp, '--iterations', '2')
            config = json.loads((Path(tmp) / 'config.json').read_text())
            self.assertEqual(config['architecture'], 'spatial-v3')
            model, checkpoint = load(Path(tmp) / 'iteration-0002/model.pt')
            self.assertEqual(model.config['architecture'], 'spatial-v3')
            self.assertEqual(checkpoint['steps'], 2)


if __name__ == '__main__':
    unittest.main()
