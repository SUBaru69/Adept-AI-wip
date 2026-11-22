from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        pass

    @abstractmethod
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass