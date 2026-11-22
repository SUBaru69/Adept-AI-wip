import os
import random
import numpy as np
import pygame
from core.base_env import BaseEnv


class SnakeEnv(BaseEnv):
    DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    BG_COLOR = (20, 24, 32)
    GRID_COLOR = (34, 40, 52)
    HUD_COLOR = (11, 15, 25)
    HEAD_COLOR = (80, 220, 120)
    BODY_COLOR = (45, 165, 90)
    FOOD_COLOR = (235, 75, 75)
    TEXT_COLOR = (248, 250, 252)
    ACCENT_COLOR = (16, 185, 129)

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        cell_size: int = 40,
        obs_mode: str = "features",
    ):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.hud_height = 44
        self.obs_mode = obs_mode

        self.snake: list[tuple[int, int]] = []
        self.dir_idx = 0
        self.food = (0, 0)
        self.score = 0
        self.steps_since_food = 0

        self._screen = None
        self._font = None
        self.closed = False

    @property
    def state_shape(self) -> tuple[int, ...]:
        if self.obs_mode == "grid":
            return (4, self.height + 2, self.width + 2)
        return (11,)

    @property
    def action_size(self) -> int:
        return 4 if self.obs_mode == "grid" else 3

    def reset(self) -> np.ndarray:
        mid_r, mid_c = self.height // 2, self.width // 2
        self.snake = [(mid_r, mid_c), (mid_r, mid_c - 1), (mid_r, mid_c - 2)]
        self.dir_idx = 0
        self.score = 0
        self.steps_since_food = 0
        self._place_food()
        return self._get_state()

    def _place_food(self) -> bool:
        occupied = set(self.snake)
        empty = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in occupied
        ]
        if not empty:
            return False
        self.food = random.choice(empty)
        return True

    def _is_collision(self, pt: tuple[int, int]) -> bool:
        r, c = pt
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return True
        return pt in self.snake[1:]

    def _next_point(self, head: tuple[int, int], d_idx: int) -> tuple[int, int]:
        dr, dc = self.DIRS[d_idx]
        return (head[0] + dr, head[1] + dc)

    def _get_feature_state(self) -> np.ndarray:
        head = self.snake[0]
        straight_d = self.dir_idx
        right_d = (self.dir_idx + 1) % 4
        left_d = (self.dir_idx - 1) % 4

        return np.array(
            [
                int(self._is_collision(self._next_point(head, straight_d))),
                int(self._is_collision(self._next_point(head, right_d))),
                int(self._is_collision(self._next_point(head, left_d))),
                int(self.dir_idx == 0),
                int(self.dir_idx == 1),
                int(self.dir_idx == 2),
                int(self.dir_idx == 3),
                int(self.food[0] < head[0]),
                int(self.food[0] > head[0]),
                int(self.food[1] < head[1]),
                int(self.food[1] > head[1]),
            ],
            dtype=np.float32,
        )

    def _get_grid_state(self) -> np.ndarray:
        grid = np.zeros((4, self.height + 2, self.width + 2), dtype=np.float32)

        grid[3, 0, :] = 1.0
        grid[3, -1, :] = 1.0
        grid[3, :, 0] = 1.0
        grid[3, :, -1] = 1.0

        hr, hc = self.snake[0]
        if 0 <= hr < self.height and 0 <= hc < self.width:
            grid[0, hr + 1, hc + 1] = 1.0

        body = self.snake[1:]
        body_len = len(body)
        for idx, (r, c) in enumerate(body):
            if 0 <= r < self.height and 0 <= c < self.width:
                val = 1.0 - 0.75 * (idx / max(1, body_len - 1)) if body_len > 1 else 1.0
                grid[1, r + 1, c + 1] = val

        fr, fc = self.food
        grid[2, fr + 1, fc + 1] = 1.0
        return grid

    def _get_state(self) -> np.ndarray:
        if self.obs_mode == "grid":
            return self._get_grid_state()
        return self._get_feature_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if self.obs_mode == "grid":
            if (action + 2) % 4 != self.dir_idx:
                self.dir_idx = action
        else:
            if action == 1:
                self.dir_idx = (self.dir_idx + 1) % 4
            elif action == 2:
                self.dir_idx = (self.dir_idx - 1) % 4

        new_head = self._next_point(self.snake[0], self.dir_idx)
        self.steps_since_food += 1

        max_idle = min(250, max(60, 6 * len(self.snake)))
        if self._is_collision(new_head) or self.steps_since_food > max_idle:
            return self._get_state(), -10.0, True

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            if not self._place_food():
                return self._get_state(), 50.0, True
            reward = 10.0
        else:
            self.snake.pop()
            reward = -0.02

        return self._get_state(), reward, False

    def _init_pygame(self) -> None:
        pygame.init()
        win_w = self.width * self.cell_size
        win_h = self.height * self.cell_size + self.hud_height
        self._screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption(f"RL Arcade - Snake ({self.obs_mode})")
        self._font = pygame.font.SysFont("consolas", 15, bold=True)

    def render(self) -> None:
        if self.closed:
            return
        if self._screen is None:
            self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                self.close()
                return

        self._screen.fill(self.BG_COLOR)

        win_w = self.width * self.cell_size
        pygame.draw.rect(self._screen, self.HUD_COLOR, (0, 0, win_w, self.hud_height))

        mode_txt = os.environ.get("RL_HUD_MODE", "Snake  |  PPO")
        left = self._font.render(mode_txt, True, self.TEXT_COLOR)
        right = self._font.render(f"APPLES: {self.score}", True, self.ACCENT_COLOR)
        self._screen.blit(left, (12, 14))
        self._screen.blit(right, (win_w - right.get_width() - 12, 14))

        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(
                    c * self.cell_size,
                    self.hud_height + r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self._screen, self.GRID_COLOR, rect, 1)

        fr, fc = self.food
        food_rect = pygame.Rect(
            fc * self.cell_size + 5,
            self.hud_height + fr * self.cell_size + 5,
            self.cell_size - 10,
            self.cell_size - 10,
        )
        pygame.draw.rect(self._screen, self.FOOD_COLOR, food_rect, border_radius=8)

        for idx, (r, c) in enumerate(self.snake):
            color = self.HEAD_COLOR if idx == 0 else self.BODY_COLOR
            pad = 2 if idx == 0 else 4
            seg_rect = pygame.Rect(
                c * self.cell_size + pad,
                self.hud_height + r * self.cell_size + pad,
                self.cell_size - pad * 2,
                self.cell_size - pad * 2,
            )
            pygame.draw.rect(self._screen, color, seg_rect, border_radius=6)

        pygame.display.flip()

    def close(self) -> None:
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._font = None