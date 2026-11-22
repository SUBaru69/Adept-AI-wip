import argparse
import os
from core.trainer import train_ppo, evaluate, plot_scores
from envs.mario import MarioEnv
from agents.ppo import PPOAgent


def main():
    parser = argparse.ArgumentParser(description="RL Mario Runner")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    env = MarioEnv()
    agent = PPOAgent(state_shape=env.state_shape, action_size=env.action_size)

    ckpt_path = os.path.join("checkpoints", "mario_ppo.pt")
    best_path = os.path.join("checkpoints", "mario_ppo_best.pt")

    if args.ckpt or args.load:
        target_path = args.ckpt if args.ckpt else (best_path if args.best else ckpt_path)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"No checkpoint found at {target_path}")
        agent.load(target_path)
        print(f"Loaded checkpoint from {target_path}")

    if args.total_steps > 0:
        print(
            f"Training [ppo] on [mario] with {args.num_envs} parallel envs "
            f"(state_shape={env.state_shape}, action_size={env.action_size}) "
            f"for {args.total_steps:,} steps..."
        )
        scores = train_ppo(
            env_fn=MarioEnv,
            agent=agent,
            total_steps=args.total_steps,
            num_envs=args.num_envs,
            log_every=args.log_every,
            save_path=ckpt_path if args.save else None,
        )

        if args.save:
            agent.save(ckpt_path)
            print(f"Saved latest checkpoint to {ckpt_path}")

        if args.plot and scores:
            plot_path = os.path.join("checkpoints", "mario_ppo_curve.png")
            plot_scores(scores, title="ppo on mario", window=20, save_path=plot_path)

    if args.eval_episodes > 0:
        evaluate(
            env=env,
            agent=agent,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
            render=not args.no_render,
            render_delay=0.02,
        )

    env.close()


if __name__ == "__main__":
    main()