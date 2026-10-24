import unittest

from variance_report import summarize


def row(game, values, prediction):
    return {
        'gameIndex': game,
        'rollouts': [{'value': {h: [v, -v] for h in ('rank', 'relative', 'absolute')}} for v in values],
        'prediction': {h: [prediction, -prediction] for h in ('rank', 'relative', 'absolute')},
        'statistics': {
            h: [{'mean': sign * sum(values) / len(values),
                 'variance': sum((v - sum(values) / len(values))**2 for v in values) / (len(values)-1)}
                for sign in (1, -1)] for h in ('rank', 'relative', 'absolute')
        },
    }


class VarianceReportTest(unittest.TestCase):
    def test_different_state_means_are_not_conditional_noise(self):
        result = summarize([row(0, [0]*10, 0), row(1, [1]*10, 1)])
        rank = result['heads']['rank'][0]
        self.assertEqual(rank['meanWithinVariance'], 0)
        self.assertEqual(rank['pooledPopulationVariance'], .25)
        self.assertEqual(rank['mseAgainstRolloutMean'], 0)
        self.assertEqual(result['heads']['absolute'][0]['pooledPopulationVariance'], 2500)

    def test_sample_variance_score_units_and_mse_decomposition(self):
        result = summarize([row(0, [-1]*5 + [1]*5, .5)])
        rank = result['heads']['rank'][0]
        self.assertAlmostEqual(rank['meanWithinVariance'], 10/9)
        self.assertAlmostEqual(rank['modelMSE'], 1.25)
        self.assertAlmostEqual(rank['mseAgainstRolloutMean'], .25)
        self.assertAlmostEqual(result['heads']['relative'][0]['meanWithinVariance'], 100000/9)
        self.assertEqual(rank['outcomeFrequency'], {'win': .5, 'draw': 0, 'loss': .5})


if __name__ == '__main__':
    unittest.main()
