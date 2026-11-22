import numpy as np
from core.base_env import BaseEnv


class GridWorldEnv(BaseEnv):
    def __init__(self, size: int = 5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.pits = {(1, 1), (2, 3), (3, 1)}
        self.agent_pos = self.start

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (2,)  # (row, col)

    @property
    def action_size(self) -> int:
        return 4  # 0=Up, 1=Right, 2=Down, 3=Left

    def reset(self) -> np.ndarray:
        self.agent_pos = self.start
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        r, c = self.agent_pos
        if action == 0:    # Up
            r = max(0, r - 1)
        elif action == 1:  # Right
            c = min(self.size - 1, c + 1)
        elif action == 2:  # Down
            r = min(self.size - 1, r + 1)
        elif action == 3:  # Left
            c = max(0, c - 1)

        self.agent_pos = (r, c)

        if self.agent_pos == self.goal:
            return self._get_state(), 10.0, True
        if self.agent_pos in self.pits:
            return self._get_state(), -10.0, True

        return self._get_state(), -0.1, False

    def render(self) -> None:
        lines = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                pos = (r, c)
                if pos == self.agent_pos:
                    row.append("A")
                elif pos == self.goal:
                    row.append("G")
                elif pos in self.pits:
                    row.append("X")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        print("\n" + "\n".join(lines) + "\n")