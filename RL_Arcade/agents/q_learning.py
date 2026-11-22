import os
import pickle
from collections import defaultdict
import numpy as np
from core.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        action_size: int,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_size, dtype=np.float32)
        )

    @staticmethod
    def _state_key(state: np.ndarray) -> tuple:
        return tuple(np.round(state, 2).tolist())

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))
        q_vals = self.q_table[self._state_key(state)]
        max_q = np.max(q_vals)
        best_actions = np.flatnonzero(q_vals == max_q)
        return int(np.random.choice(best_actions))

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        s_key = self._state_key(state)
        ns_key = self._state_key(next_state)

        current_q = self.q_table[s_key][action]
        next_max_q = 0.0 if done else float(np.max(self.q_table[ns_key]))
        target_q = reward + self.gamma * next_max_q

        self.q_table[s_key][action] = current_q + self.lr * (target_q - current_q)

        if done and self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q_table": dict(self.q_table), "epsilon": self.epsilon}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(
            lambda: np.zeros(self.action_size, dtype=np.float32),
            data["q_table"],
        )
        self.epsilon = data.get("epsilon", self.epsilon_min)