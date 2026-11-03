"""Strongly convex scalar objective over paired history statistics."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DualConfig:
    temperature: float = 1.0
    beta: float = 0.1
    lower: float = 0.0
    upper: float = 10.0
    iterations: int = 50
    tolerance: float = 1e-4

    def __post_init__(self):
        values = (self.temperature, self.beta, self.lower, self.upper, self.tolerance)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Dual settings must be finite")
        if self.temperature <= 0 or self.beta <= 0 or self.tolerance <= 0:
            raise ValueError("Temperature, beta and tolerance must be positive")
        if self.lower < 0 or self.upper <= self.lower or self.iterations < 1:
            raise ValueError("Invalid multiplier interval or iteration count")


def paired_arrays(rewards, costs):
    rewards, costs = np.asarray(rewards, dtype=float), np.asarray(costs, dtype=float)
    if rewards.ndim != 1 or rewards.shape != costs.shape or not rewards.size:
        raise ValueError("Rewards/costs must be nonempty paired 1-D arrays")
    if not np.isfinite(rewards).all() or not np.isfinite(costs).all():
        raise ValueError("History statistics must be finite")
    return rewards, costs


def objective(multiplier, rewards, costs, budget, config=DualConfig()):
    rewards, costs = paired_arrays(rewards, costs)
    if not np.isfinite(multiplier) or not np.isfinite(budget):
        raise ValueError("Multiplier and budget must be finite")
    x = (rewards - multiplier * costs) / config.temperature
    shift = x.max()
    log_mean_exp = shift + np.log(np.exp(x - shift).mean())
    return float(config.temperature * log_mean_exp
                 + multiplier * (budget - costs.mean())
                 + multiplier**2 / (2 * config.beta))


def solve(rewards, costs, budget, config=DualConfig()):
    rewards, costs = paired_arrays(rewards, costs)
    lo, hi = config.lower, config.upper
    for _ in range(config.iterations):
        if hi - lo < config.tolerance:
            break
        left, right = lo + (hi-lo)/3, hi - (hi-lo)/3
        if objective(left, rewards, costs, budget, config) < objective(right, rewards, costs, budget, config):
            hi = right
        else:
            lo = left
    return float(np.clip((lo+hi)/2, config.lower, config.upper))
