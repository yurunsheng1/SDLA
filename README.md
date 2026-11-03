# PPO-SDLA on one HASARD task

A minimal review version for **Armament Burden, Level 1**. It adds a scalar SDLA solver and a learner adapter to the official HASARD PPO implementation. No training logs, checkpoints, manuscript files, or benchmark-result claims are included.

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
- `train.py`: single-task training entry point.
- `env_smoke.py`: headless environment check.

## Install

Use Linux, Python 3.10 or 3.11, and a suitable PyTorch installation. A CUDA-enabled PyTorch build is needed for GPU training. The default settings are examples for reviewing the code, not a claim to reproduce a paper result.

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

The task and level are deliberately fixed by the entry point. Results and checkpoints go to `train_dir/`. W&B and video recording are disabled by default. Use a new experiment name for a fresh run; HASARD may otherwise resume an existing run.

## Validation

Checked on Linux / Python 3.10 on 2026-11-03:

- Four numerical solver tests passed.
- Pinned-upstream registration passed twice (idempotency checked).
- Headless Armament Burden Level 1 ran for 128 environment steps.
- The documented CPU serial training check exited successfully and performed two learner updates, including dual updates. HASARD reported 512 simulator frames because of rollout/frame-skip granularity.

This is a functional smoke check, not a full training or performance evaluation. GPU and long-run performance have not been validated in this preview.

## Attribution

Built on [HASARD](https://github.com/TTomilin/HASARD) by Tristan Tomilin et al. and its bundled Sample Factory implementation. See the upstream repository for environment assets, baseline implementations, citations, and dependency licenses. The upstream MIT notice is retained in `LICENSE`.
