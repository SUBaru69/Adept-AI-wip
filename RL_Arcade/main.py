import argparse
import os
from core.trainer import train, train_ppo, evaluate, plot_scores
from envs.gridworld import GridWorldEnv
from envs.snake import SnakeEnv
from envs.pong import PongEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent


ENV_REGISTRY = {
    "gridworld": GridWorldEnv,
    "snake": lambda: SnakeEnv(obs_mode="features"),
    "snake_grid": lambda: SnakeEnv(obs_mode="grid"),
    "pong": PongEnv,
}


def build_agent(agent_name: str, state_shape: tuple[int, ...], action_size: int):
    if agent_name == "q_learning":
        return QLearningAgent(action_size=action_size)
    if agent_name == "dqn":
        return DQNAgent(state_shape=state_shape, action_size=action_size)
    if agent_name == "ppo":
        return PPOAgent(state_shape=state_shape, action_size=action_size)
    raise ValueError(f"Unknown agent '{agent_name}'")


def main():
    parser = argparse.ArgumentParser(description="RL Arcade Runner")
    parser.add_argument("--env", choices=list(ENV_REGISTRY.keys()), default="gridworld")
    parser.add_argument("--agent", choices=["q_learning", "dqn", "ppo"], default="q_learning")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    env_fn = ENV_REGISTRY[args.env]
    env = env_fn()
    agent = build_agent(args.agent, env.state_shape, env.action_size)

    ext = "pt" if args.agent in ("dqn", "ppo") else "pkl"
    ckpt_path = os.path.join("checkpoints", f"{args.env}_{args.agent}.{ext}")
    best_path = os.path.join("checkpoints", f"{args.env}_{args.agent}_best.{ext}")

    if args.load:
        target_path = best_path if args.best else ckpt_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"No checkpoint found at {target_path}")
        agent.load(target_path)
        print(f"Loaded checkpoint from {target_path}")

    should_train = (args.total_steps > 0 and args.episodes > 0) if args.agent == "ppo" else (args.episodes > 0)

    if should_train:
        if args.agent == "ppo":
            print(
                f"Training [ppo] on [{args.env}] with {args.num_envs} parallel envs "
                f"(state_shape={env.state_shape}, action_size={env.action_size}) "
                f"for {args.total_steps:,} steps..."
            )
            scores = train_ppo(
                env_fn=env_fn,
                agent=agent,
                total_steps=args.total_steps,
                num_envs=args.num_envs,
                log_every=args.log_every,
                save_path=ckpt_path if args.save else None,
            )
        else:
            print(
                f"Training [{args.agent}] on [{args.env}] "
                f"(state_shape={env.state_shape}, action_size={env.action_size}) "
                f"for {args.episodes} episodes..."
            )
            scores = train(
                env=env,
                agent=agent,
                episodes=args.episodes,
                max_steps=args.max_steps,
                log_every=args.log_every,
            )

        if args.save:
            agent.save(ckpt_path)
            print(f"Saved latest checkpoint to {ckpt_path}")

        if args.plot and scores:
            plot_path = os.path.join("checkpoints", f"{args.env}_{args.agent}_curve.png")
            plot_scores(scores, title=f"{args.agent} on {args.env}", window=100, save_path=plot_path)

    if args.eval_episodes > 0:
        evaluate(
            env=env,
            agent=agent,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
            render=not args.no_render,
        )

    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()