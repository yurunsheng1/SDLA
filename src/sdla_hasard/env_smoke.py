"""Run a short headless random-policy check on Armament Burden Level 1."""
import argparse
import json


def main():
    import hasard
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    env = hasard.make("ArmamentBurdenLevel1-v0", render_mode="rgb_array",
                      resolution="160X120", seed=args.seed)
    reward_sum = cost_sum = 0.0
    completed = 0
    try:
        env.action_space.seed(args.seed)
        env.reset(seed=args.seed)
        for _ in range(args.steps):
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            reward_sum += float(reward)
            cost_sum += float(info["cost"])
            if terminated or truncated:
                completed += 1
                env.reset()
        print(json.dumps({"steps": args.steps, "completed_episodes": completed,
                          "accumulated_reward": reward_sum, "accumulated_cost": cost_sum}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
