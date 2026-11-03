import unittest

import numpy as np

from sdla_hasard.dual import DualConfig, objective, solve


class DualTests(unittest.TestCase):
    def test_constant_cost_has_known_solution(self):
        cfg = DualConfig(beta=.2, upper=10)
        result = solve([1, 7, 3], [20, 20, 20], 25, cfg)
        self.assertAlmostEqual(result, .2 * (2*20-25), places=3)
        self.assertLess(solve([2], [3], 25, cfg), cfg.tolerance)
        self.assertAlmostEqual(solve([2], [100], 25, cfg), 10, places=3)

    def test_heterogeneous_cost_stationarity(self):
        cfg = DualConfig(beta=.15)
        rewards, costs = np.array([1., 3., 2.]), np.array([14., 22., 35.])
        multiplier = solve(rewards, costs, 25, cfg)
        x = (rewards-multiplier*costs)/cfg.temperature
        weights = np.exp(x-x.max()); weights /= weights.sum()
        derivative = 25-costs.mean()-weights@costs+multiplier/cfg.beta
        self.assertLess(abs(derivative), .01)

    def test_reward_translation_and_large_inputs(self):
        cfg = DualConfig()
        rewards, costs = [1., 2., 3.], [10., 20., 30.]
        initial = solve(rewards, costs, 25, cfg)
        shifted = solve(np.array(rewards)+1e6, costs, 25, cfg)
        self.assertAlmostEqual(initial, shifted, places=3)
        self.assertTrue(np.isfinite(objective(2, np.array(rewards)+1e6, costs, 25, cfg)))

    def test_invalid_inputs(self):
        for rewards, costs in [([], []), ([1], [1, 2]), ([np.nan], [1])]:
            with self.assertRaises(ValueError):
                solve(rewards, costs, 25)
        with self.assertRaises(ValueError):
            DualConfig(beta=0)


if __name__ == "__main__":
    unittest.main()
