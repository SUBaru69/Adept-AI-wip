import os
import time
import numpy as np
import matplotlib.pyplot as plt
from core.base_env import BaseEnv
from core.base_agent import BaseAgent


def train(
    env: BaseEnv,
    agent: BaseAgent,
    episodes: int = 500,
    max_steps: int = 300,
    log_every: int = 50,
    render_every: int = 0,
    render_delay: float = 0.08,
) -> list[float]:
    scores: list[float] = []

    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0
            should_render = bool(render_every and ep % render_every == 0)

            for _ in range(max_steps):
                action = agent.select_action(state, training=True)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if should_render:
                    env.render()
                    time.sleep(render_delay)

                if done:
                    break

            scores.append(total_reward)

            if log_every and ep % log_every == 0:
                window = scores[-log_every:]
                avg_score = float(np.mean(window))
                eps_info = f" | eps: {agent.epsilon:.3f}" if hasattr(agent, "epsilon") else ""
                print(
                    f"Episode {ep:>5}/{episodes} | "
                    f"Avg Reward (last {len(window)}): {avg_score:>7.2f} | "
                    f"Best: {max(window):>6.2f}{eps_info}"
                )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user — keeping progress so far.")

    return scores


def train_ppo(
    env_fn,
    agent,
    total_steps: int = 5_000_000,
    num_envs: int = 32,
    rollout_steps: int = 128,
    log_every: int = 25,
    save_path: str | None = None,
) -> list[float]:
    envs = [env_fn() for _ in range(num_envs)]
    states = np.stack([env.reset() for env in envs], axis=0)
    state_shape = envs[0].state_shape
    track_apples = hasattr(envs[0], "score")

    ep_rewards = np.zeros(num_envs, dtype=np.float32)
    completed_scores: list[float] = []
    completed_apples: list[int] = []

    steps_per_update = num_envs * rollout_steps
    num_updates = max(1, total_steps // steps_per_update)
    start_time = time.perf_counter()

    obs_buf = np.zeros((rollout_steps, num_envs, *state_shape), dtype=np.float32)
    act_buf = np.zeros((rollout_steps, num_envs), dtype=np.int64)
    logp_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    rew_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    done_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    val_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)

    best_avg_reward = -float("inf")
    best_path = None
    if save_path:
        root, ext = os.path.splitext(save_path)
        best_path = f"{root}_best{ext}"

    try:
        for update in range(1, num_updates + 1):
            if hasattr(agent, "set_progress"):
                agent.set_progress((update - 1) / max(1, num_updates - 1))

            for t in range(rollout_steps):
                actions, log_probs, values = agent.get_action_and_value(states)

                obs_buf[t] = states
                act_buf[t] = actions
                logp_buf[t] = log_probs
                val_buf[t] = values

                next_states = np.empty_like(states)
                for i, env in enumerate(envs):
                    ns, reward, done = env.step(int(actions[i]))
                    rew_buf[t, i] = reward
                    done_buf[t, i] = float(done)
                    ep_rewards[i] += reward

                    if done:
                        completed_scores.append(float(ep_rewards[i]))
                        if track_apples:
                            completed_apples.append(int(env.score))
                        ep_rewards[i] = 0.0
                        ns = env.reset()

                    next_states[i] = ns

                states = next_states

            last_values = agent.get_values(states)
            agent.update(obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, last_values)

            if log_every and (update % log_every == 0 or update == num_updates):
                global_step = update * steps_per_update
                sps = int(global_step / max(1e-6, time.perf_counter() - start_time))
                recent_r = completed_scores[-100:]
                avg_r = float(np.mean(recent_r)) if recent_r else 0.0

                apple_str = ""
                if track_apples and completed_apples:
                    recent_a = completed_apples[-100:]
                    apple_str = f" | Apples(100): {np.mean(recent_a):>5.1f} (Best: {max(recent_a):>2})"

                best_tag = ""
                if best_path and len(completed_scores) >= 100 and avg_r > best_avg_reward:
                    best_avg_reward = avg_r
                    agent.save(best_path)
                    best_tag = " [NEW BEST]"

                print(
                    f"Step {global_step:>9}/{num_updates * steps_per_update} | "
                    f"Eps: {len(completed_scores):>6} | "
                    f"AvgR(100): {avg_r:>7.2f}{apple_str} | "
                    f"SPS: {sps:>5}{best_tag}"
                )
    except KeyboardInterrupt:
        print("\nTraining stopped early by user (Ctrl+C) — saving progress...")

    return completed_scores


def evaluate(
    env: BaseEnv,
    agent: BaseAgent,
    episodes: int = 5,
    max_steps: int = 5000,
    render: bool = True,
    render_delay: float = 0.05,
) -> list[float]:
    scores: list[float] = []

    for ep in range(1, episodes + 1):
        if getattr(env, "closed", False):
            break

        state = env.reset()
        total_reward = 0.0
        steps = 0

        if render:
            print(f"\n--- Evaluation Episode {ep} ---")
            env.render()
            time.sleep(render_delay)

        for _ in range(max_steps):
            if getattr(env, "closed", False):
                break

            action = agent.select_action(state, training=False)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()
                time.sleep(render_delay)

            if done:
                break

        scores.append(total_reward)
        apple_info = f" | Apples: {env.score}" if hasattr(env, "score") else ""
        print(f"Eval Episode {ep} finished in {steps} steps | Total Reward: {total_reward:.2f}{apple_info}")

    return scores


def plot_scores(
    scores: list[float],
    title: str = "Training Progress",
    window: int = 100,
    save_path: str | None = None,
) -> None:
    episodes = np.arange(1, len(scores) + 1)
    plt.figure(figsize=(9, 4.5))
    plt.plot(episodes, scores, alpha=0.2, label="Episode Reward")

    if len(scores) >= window:
        kernel = np.ones(window) / window
        moving_avg = np.convolve(scores, kernel, mode="valid")
        plt.plot(episodes[window - 1 :], moving_avg, linewidth=2, label=f"{window}-Ep Moving Avg")

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved learning curve to {save_path}")

    plt.show()