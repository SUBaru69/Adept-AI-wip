from collections import deque
import os
import warnings
import numpy as np
import pygame
from PIL import Image
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from core.base_env import BaseEnv

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

STATUS_RANK = {"small": 0, "tall": 1, "fireball": 2}


class MarioEnv(BaseEnv):
    def __init__(
        self,
        env_id: str = "SuperMarioBros-1-1-v0",
        frame_skip: int = 4,
        stack_size: int = 4,
        obs_size: int = 84,
    ):
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        self.obs_size = obs_size

        try:
            raw_env = gym_super_mario_bros.make(
                env_id, apply_api_compatibility=True, render_mode="rgb_array"
            )
        except TypeError:
            raw_env = gym_super_mario_bros.make(env_id)

        self.env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
        self.frames: deque[np.ndarray] = deque(maxlen=stack_size)
        self.last_rgb: np.ndarray | None = None
        self.score = 0
        self._prev_coins = 0
        self._prev_status = "small"

        self._screen = None
        self._hud_font = None
        self.closed = False

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.stack_size, self.obs_size, self.obs_size)

    @property
    def action_size(self) -> int:
        return int(self.env.action_space.n)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        self.last_rgb = frame.copy()
        img = Image.fromarray(self.last_rgb).convert("L")
        img = img.resize((self.obs_size, self.obs_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def reset(self) -> np.ndarray:
        out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        processed = self._preprocess(obs)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(processed)
        self.score = 0
        self._prev_coins = 0
        self._prev_status = "small"
        return np.stack(self.frames, axis=0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        total_reward = 0.0
        bonus_reward = 0.0
        done = False
        obs = None

        for _ in range(self.frame_skip):
            out = self.env.step(action)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = out

            total_reward += float(reward)
            if isinstance(info, dict):
                if "x_pos" in info:
                    self.score = max(self.score, int(info["x_pos"]))
                coins = int(info.get("coins", self._prev_coins))
                if coins > self._prev_coins:
                    bonus_reward += 0.5 * (coins - self._prev_coins)
                self._prev_coins = coins

                status = str(info.get("status", self._prev_status))
                if STATUS_RANK.get(status, 0) > STATUS_RANK.get(self._prev_status, 0):
                    bonus_reward += 2.0
                self._prev_status = status

            if done:
                break

        self.frames.append(self._preprocess(obs))
        return np.stack(self.frames, axis=0), (total_reward / 15.0) + bonus_reward, done

    def render(self) -> None:
        if self.closed or self.last_rgb is None:
            return

        if self._screen is None:
            pygame.init()
            h, w = self.last_rgb.shape[:2]
            self._screen = pygame.display.set_mode((w * 2, h * 2))
            pygame.display.set_caption("RL Mario - World 1-1")
            self._hud_font = pygame.font.SysFont("consolas", 15, bold=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                self.close()
                return

        surf = pygame.surfarray.make_surface(np.transpose(self.last_rgb, (1, 0, 2)))
        scaled = pygame.transform.scale(surf, self._screen.get_size())
        self._screen.blit(scaled, (0, 0))

        hud_bar = pygame.Surface((self._screen.get_width(), 28), pygame.SRCALPHA)
        hud_bar.fill((11, 15, 25, 195))
        self._screen.blit(hud_bar, (0, 0))

        mode_txt = os.environ.get("RL_HUD_MODE", "Super Mario Bros  |  PPO")
        left = self._hud_font.render(mode_txt, True, (248, 250, 252))
        right = self._hud_font.render(f"X-POS: {self.score} / 3161", True, (16, 185, 129))
        self._screen.blit(left, (10, 6))
        self._screen.blit(right, (self._screen.get_width() - right.get_width() - 10, 6))

        pygame.display.flip()

    def close(self) -> None:
        if self._screen is not None:
            pygame.quit()
            self._screen = None
        try:
            self.env.close()
        except Exception:
            pass