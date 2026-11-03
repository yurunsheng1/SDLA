"""Train one HASARD task: Armament Burden Level 1."""
import sys


def parse_config(argv=None):
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sample_factory.doom.doom_params import add_doom_env_args, doom_override_defaults
    parser, _ = parse_sf_args(argv=argv)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    parser.add_argument("--sdla_temperature", type=float, default=1.0)
    parser.add_argument("--sdla_beta", type=float, default=0.1)
    parser.add_argument("--sdla_lambda_max", type=float, default=10.0)
    parser.add_argument("--sdla_history", type=int, default=20)
    parser.add_argument("--sdla_warmup", type=int, default=5)
    parser.add_argument("--sdla_reference_budget", type=float, default=25.0)
    parser.add_argument("--sdla_iterations", type=int, default=50)
    parser.add_argument("--sdla_tolerance", type=float, default=1e-4)
    parser.set_defaults(algo="PPOCvx", env="armament_burden", level=1,
                        experiment="sdla_armament_l1", num_workers=1, worker_num_splits=1,
                        num_envs_per_worker=4, rollout=128, batch_size=512,
                        recurrence=32, with_vtrace=False, record=False,
                        with_wandb=False, async_rl=False,
                        train_for_env_steps=1_000_000)
    cfg = parse_full_cfg(parser, argv)
    if cfg.algo != "PPOCvx" or cfg.env != "armament_burden" or cfg.level != 1:
        raise ValueError("This minimal entry point supports PPOCvx / Armament Burden Level 1 only")
    if cfg.with_vtrace:
        raise ValueError("This adapter expects GAE returns; use --with_vtrace=False")
    return cfg


def main(argv=None):
    from sample_factory.algo.learning import learner_worker
    from sdla_hasard.learner import SDLALearner
    if getattr(learner_worker, "SDLALearner", None) is not SDLALearner:
        raise RuntimeError("Run python -m sdla_hasard.install --hasard-root .deps/HASARD first")
    from sample_factory.train import register_vizdoom_components, run_rl
    register_vizdoom_components()
    return run_rl(parse_config(argv))


if __name__ == "__main__":
    sys.exit(main())
