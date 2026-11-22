# Adept-AI — Modular PPO Reinforcement Learning Suite

> **Status:** Work in Progress — actively tuning Mario reward shaping and UI polish.

A modular PyTorch Reinforcement Learning framework implementing **Proximal Policy Optimization (PPO)** with Generalized Advantage Estimation (GAE) across three progressively complex environments: state-based **Pong**, grid-vision **Snake**, and raw-pixel **Super Mario Bros (World 1-1)**, all controlled through a unified dark-mode desktop dashboard.

<!-- Drag and drop your demo video (.mp4) or dashboard screenshot directly into the GitHub README editor below -->

---

## Environments & Benchmarks

| Environment | Observation Space | Policy Architecture | Parallel Envs | Key Result |
| :--- | :--- | :--- | :---: | :--- |
| **Pong** | 6D Physics Vector | 2-Layer MLP (128 units) | 32 | **+19.72** avg reward (4.5/5 wins, ~8,600 SPS) |
| **Snake** | 12×12 4-Channel Grid | Residual CNN | 32 | **32 Apples** peak evaluation score |
| **Super Mario Bros** | 4×84×84 Grayscale Stack | Nature CNN | 8 | **3,161 `x_pos`** (Flagpole cleared), 2,328 avg |

---

## Key Features

- **Unified Dark-Mode Dashboard (`RL_Arcade/gui.py`):** Built with `customtkinter`, featuring live KPI cards, dark-themed learning curve rendering, checkpoint switching (`Untrained` / `Latest` / `Best`), and a real-time subprocess execution console.
- **Interactive Human vs. AI Pong:** Play directly against the trained PPO Pong agent using `W`/`S` or `Up`/`Down` arrow keys—powered by the exact same `PongEnv` and `.pt` checkpoint with zero retraining.
- **In-Game Translucent HUD:** Real-time Pygame overlay displaying the active model mode and live score/`x_pos` metrics.
- **Reward Shaping (Super Mario Bros):** Extends the default progress/time/death reward (`Δx + Δt + d`) with coin and power-up status bonuses to address PPO's default speedrunner convergence.

---

## Project Structure

```text
Adept-AI-wip/
├── RL_Arcade/
│   ├── agents/          # PPO (MLP & Residual CNN), DQN, Q-Learning
│   ├── checkpoints/     # Saved .pt weights and learning curves
│   ├── core/            # BaseEnv, BaseAgent, vectorized Trainer
│   ├── envs/            # Snake and Pong environments with HUD & Human vs. AI
│   ├── gui.py           # Unified dark-mode presentation dashboard
│   └── main.py          # CLI training and evaluation entry point
├── RL_Mario/
│   ├── agents/          # Nature CNN Actor-Critic PPO
│   ├── checkpoints/     # Saved Mario .pt weights and learning curves
│   ├── core/            # Base interfaces and parallel rollout trainer
│   ├── envs/            # 4-frame skip/stack 84x84 Mario wrapper
│   └── main.py          # Mario training and evaluation entry point
├── requirements.txt
├── .gitignore
└── README.md
