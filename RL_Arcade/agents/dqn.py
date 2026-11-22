import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from core.base_agent import BaseAgent


class QNetwork(nn.Module):
    def __init__(self, state_shape: tuple[int, ...], action_size: int):
        super().__init__()
        self.is_spatial = len(state_shape) == 3

        if self.is_spatial:
            c, h, w = state_shape
            self.conv = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            with torch.no_grad():
                flat_size = self.conv(torch.zeros(1, c, h, w)).reshape(1, -1).size(1)

            self.head = nn.Sequential(
                nn.Linear(flat_size, 256),
                nn.ReLU(),
                nn.Linear(256, action_size),
            )
        else:
            in_features = int(np.prod(state_shape))
            self.net = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_spatial:
            x = self.conv(x)
            x = x.reshape(x.size(0), -1)
            return self.head(x)
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_shape: tuple[int, ...],
        action_size: int,
        lr: float = 5e-4,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9985,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        min_buffer_size: int = 1_000,
        train_every: int = 4,
        target_update_every: int = 250,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.train_every = train_every
        self.target_update_every = target_update_every

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        arch = "CNN" if len(state_shape) == 3 else "MLP"
        print(f"[DQN] Using {arch} on {self.device} ({dev_name})")

        self.policy_net = QNetwork(state_shape, action_size).to(self.device)
        self.target_net = QNetwork(state_shape, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory: deque = deque(maxlen=buffer_size)

        self.step_count = 0
        self.learn_steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, float(done)))
        self.step_count += 1

        if done and self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if len(self.memory) < self.min_buffer_size or self.step_count % self.train_every != 0:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.from_numpy(np.asarray(next_states, dtype=np.float32)).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, best_next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)