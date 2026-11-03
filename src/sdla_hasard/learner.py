"""HASARD adapter using rolling mini-batch summaries and dual-first updates."""
from collections import deque

import numpy as np
import torch
from sample_factory.algo.learning.ppo_lag_learner import PPOLagLearner

from sdla_hasard.dual import DualConfig, solve


class SDLALearner(PPOLagLearner):
    def __init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self.dual_config = DualConfig(
            temperature=cfg.sdla_temperature, beta=cfg.sdla_beta,
            upper=cfg.sdla_lambda_max, iterations=cfg.sdla_iterations,
            tolerance=cfg.sdla_tolerance)
        if cfg.sdla_history < cfg.sdla_warmup or cfg.sdla_warmup < 1:
            raise ValueError("Require history >= warmup >= 1")
        if self.safety_bound <= 0 or cfg.sdla_reference_budget <= 0:
            raise ValueError("Cost budgets must be positive")
        self.history = deque(maxlen=cfg.sdla_history)
        self.warmup = cfg.sdla_warmup
        self.reference_budget = cfg.sdla_reference_budget
        self.cost_scale = self.reference_budget / float(self.safety_bound)
        self.current_batch_return = None

    def _calculate_losses(self, mb, num_invalids):
        with torch.no_grad():
            self.current_batch_return = float(mb["returns"].mean().item())
        return super()._calculate_losses(mb, num_invalids)

    def _update_lagrange(self, mean_cost):
        violation = (mean_cost - self.safety_bound).detach()
        if self.current_batch_return is not None:
            self.history.append((self.current_batch_return, float(mean_cost.detach().item())))
        if len(self.history) >= self.warmup:
            statistics = np.asarray(self.history) * self.cost_scale
            value = solve(statistics[:, 0], statistics[:, 1], self.reference_budget, self.dual_config)
            self.lambda_lagr = torch.tensor(value, device=self.device, dtype=torch.float32)
        return violation

    def _record_summaries(self, train_loop_vars):
        stats = super()._record_summaries(train_loop_vars)
        stats["sdla_history_size"] = len(self.history)
        stats["sdla_beta"] = self.dual_config.beta
        stats["sdla_temperature"] = self.dual_config.temperature
        return stats
