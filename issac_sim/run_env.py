import argparse

from stable_baselines3 import PPO
from issac_sim.envs.sofa_env import SoftSofaEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO with SOFA-backed environment.")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--model-path", type=str, default="ppo_soft_sofa")
    return parser.parse_args()


def main():
    args = parse_args()
    env = SoftSofaEnv()
    try:
        # Dict observation requires MultiInputPolicy.
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_sofa_tensorboard/",
        )

        print("Start training...")
        model.learn(total_timesteps=args.total_timesteps)

        model.save(args.model_path)
        print(f"Model saved to {args.model_path}")

        print("Run deterministic evaluation rollout...")
        obs, _info = env.reset()
        for _ in range(args.eval_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()