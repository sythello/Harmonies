"""Checks for the experiment's exact/MSE definitions, independent of training."""
import unittest
import torch
from fit_audit import value_metrics


class MetricTest(unittest.TestCase):
    def test_score_rounding_and_scaled_mse(self):
        prediction = torch.tensor([[.334, -.336], [.421, .408]])
        target = torch.tensor([[.33, -.33], [.42, .41]])
        result = value_metrics(prediction, target, "relative")
        self.assertAlmostEqual(result["exact"], .75)
        self.assertAlmostEqual(result["exactOwn"], 1)
        self.assertAlmostEqual(result["exactOpponent"], .5)
        self.assertAlmostEqual(result["exactBoth"], .5)
        self.assertAlmostEqual(result["mse"], (.004**2+.006**2+.001**2+.002**2)/4, places=8)
        self.assertAlmostEqual(result["msePoints"], .1425, places=5)

    def test_rank_uses_nearest_legal_value_including_draw(self):
        prediction = torch.tensor([[.8, -.7], [.1, -.3], [.49, -.49]])
        target = torch.tensor([[1., -1.], [0., 0.], [1., -1.]])
        result = value_metrics(prediction, target, "rank")
        self.assertAlmostEqual(result["exact"], 2/3, places=6)
        self.assertAlmostEqual(result["exactBoth"], 2/3, places=6)


if __name__ == "__main__":
    unittest.main()
