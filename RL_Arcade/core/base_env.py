from abc import ABC, abstractmethod
import numpy as np


class BaseEnv(ABC):
    @property
    @abstractmethod
    def state_shape(self) -> tuple[int, ...]:
        """Shape of the observation array, e.g. (2,) or (4, 84, 84)."""
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Number of discrete actions available."""
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset game and return initial state."""
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Run one step and return (next_state, reward, done)."""
        pass

    @abstractmethod
    def render(self) -> None:
        """Display the current game state."""
        pass