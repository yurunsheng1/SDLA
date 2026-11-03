# Structural Dual Adaptation for Safe Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2611.00636-b31b1b.svg)](https://arxiv.org/abs/2611.00636)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Runsheng Yu, Zhuoyang Song, Zengde Deng, Qixin Zhang, Li Shen, Jiaxing Zhang**

[Paper](https://arxiv.org/abs/2611.00636) · [PDF](https://arxiv.org/pdf/2611.00636)

This repository provides a minimal **PPO-SDLA** implementation on HASARD, with **Armament Burden, Level 1** as the default, built on the official [HASARD](https://github.com/TTomilin/HASARD) PPO implementation.

## Overview

For a fixed policy, the exact bounded Lagrange multiplier response switches between its endpoints at the constraint boundary. **Structural DuaL Adaptation (SDLA)** introduces a surrogate dual objective combining current-cost centering, trajectory weighting, and quadratic regularization to obtain a unique continuous response. A bounded scalar search computes the multiplier without a dual gradient stepsize.

The paper characterizes multiplier activation and feedback sensitivity, analyzes entropy-regularized tabular policies, and evaluates neural PPO/SAC variants on OmniSafe and HASARD. This repository contains the HASARD PPO adapter, a standalone scalar solver, and runnable training and validation commands.

## Install

Use Linux, Python 3.10 or 3.11, and a suitable PyTorch installation. A CUDA-enabled PyTorch build is needed for GPU training. The training entry point provides a minimal training configuration.

From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
mkdir -p .deps
git clone https://github.com/TTomilin/HASARD.git .deps/HASARD
git -C .deps/HASARD checkout af9165f72ec6da96ffb60f0a9d3baebb8068c3a8
python -m pip install -e '.deps/HASARD[sample-factory]'
python -m pip install -e .
python -m sdla_hasard.install --hasard-root .deps/HASARD
```

The installer makes three small registrations: learner selection, the cost-value model head, and cost-value rollout buffers. It rejects an unexpected revision or unrelated modifications and is safe to run again. The dependency checkout and generated files are gitignored.

## Check and run

Numerical solver checks:

```bash
python -m unittest discover -s tests -v
```

Environment smoke test (random actions; not a trained policy):

```bash
python -m sdla_hasard.env_smoke --steps 128
```

Short CPU training check:

```bash
python -m sdla_hasard.train \
  --device=cpu --serial_mode=True \
  --num_workers=1 --num_envs_per_worker=1 \
  --rollout=64 --batch_size=64 --recurrence=16 --num_epochs=1 \
  --train_for_env_steps=128 --sdla_warmup=1 --experiment=sdla_armament_l1_smoke
```

Longer run:

```bash
python -m sdla_hasard.train --device=gpu --seed=0 \
  --train_for_env_steps=1000000 --experiment=sdla_armament_l1_seed0
```

To run another HASARD task, set `--env`, `--level`, and its episode cost budget with `--safety_bound`. For example, Remedy Rush Level 1 with a budget of 5:

```bash
python -m sdla_hasard.train --device=gpu --seed=0 \
  --env=remedy_rush --level=1 --safety_bound=5 \
  --train_for_env_steps=1000000 --experiment=sdla_remedy_l1_seed0
```

Available tasks are `armament_burden`, `collateral_damage`, `detonators_dilemma`, `precipice_plunge`, `remedy_rush`, and `volcanic_venture`; levels are 1, 2, or 3.

Results and checkpoints go to `train_dir/`. W&B and video recording are disabled by default. Use a new experiment name for a fresh run; HASARD may otherwise resume an existing run.

## Implementation

The dual uses a rolling history of paired **mini-batch reward-target means and step-cost means**. The history statistics are scaled by the ratio of the reference budget to HASARD's step-level safety bound. The objective is

```text
g(lambda) = tau * log mean(exp((R - lambda*C)/tau))
            + lambda * (d - mean(C)) + lambda**2/(2*beta)
```

`R` and `C` here denote scaled mini-batch summaries, not full-episode returns. This follows the existing HASARD implementation. The PPO parent updates lambda before forming the cost-adjusted advantage, once per mini-batch after history warm-up. The scalar solve holds those statistics fixed and uses stabilized log-mean-exp and bounded ternary search.

Files:

- `dual.py`: NumPy objective and solver.
- `learner.py`: adapter to HASARD's `PPOLagLearner`.
- `install.py`: small, idempotent learner/safety-model registration patches for one pinned HASARD revision.
- `train.py`: HASARD training entry point.
- `env_smoke.py`: headless environment check.

## Citation

If you use SDLA in your research, please cite:

```bibtex
@misc{yu2026structuraldualadaptation,
  title         = {Structural Dual Adaptation for Safe Reinforcement Learning},
  author        = {Runsheng Yu and Zhuoyang Song and Zengde Deng and Qixin Zhang and Li Shen and Jiaxing Zhang},
  year          = {2026},
  eprint        = {2611.00636},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.00636}
}
```

## Attribution

Built on [HASARD](https://github.com/TTomilin/HASARD) by Tristan Tomilin et al. and its bundled Sample Factory implementation. See the upstream repository for environment assets, baseline implementations, citations, and dependency licenses. The upstream MIT notice is retained in `LICENSE`.
