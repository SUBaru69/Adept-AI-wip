import os
import random
import numpy as np
import pygame
from core.base_env import BaseEnv


class PongEnv(BaseEnv):
    BG_COLOR = (18, 22, 30)
    FG_COLOR = (230, 235, 245)
    PLAYER_COLOR = (80, 220, 120)
    CPU_COLOR = (235, 85, 85)

    def __init__(self, width: int = 600, height: int = 400):
        self.width = width
        self.height = height
        self.paddle_w = 12
        self.paddle_h = 70
        self.ball_size = 12
        self.paddle_speed = 6.5
        self.cpu_speed = 5.2
        self.win_score = 5
        self.human_mode = os.environ.get("RL_HUMAN_PONG") == "1"

        self.player_y = 0.0
        self.cpu_y = 0.0
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.player_score = 0
        self.cpu_score = 0
        self.score = 0

        self._screen = None
        self._font = None
        self._hud_font = None
        self.closed = False

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (6,)

    @property
    def action_size(self) -> int:
        return 3

    def _serve_ball(self, towards_player: bool | None = None) -> None:
        self.ball_x = self.width / 2.0
        self.ball_y = self.height / 2.0
        if towards_player is None:
            towards_player = random.choice([True, False])
        speed = 5.5
        angle = random.uniform(-0.6, 0.6)
        self.ball_vx = speed * np.cos(angle) * (1 if towards_player else -1)
        self.ball_vy = speed * np.sin(angle)

    def reset(self) -> np.ndarray:
        self.player_y = (self.height - self.paddle_h) / 2.0
        self.cpu_y = (self.height - self.paddle_h) / 2.0
        self.player_score = 0
        self.cpu_score = 0
        self.score = 0
        self._serve_ball()
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.array(
            [
                self.ball_x / self.width,
                self.ball_y / self.height,
                self.ball_vx / 10.0,
                self.ball_vy / 10.0,
                (self.player_y + self.paddle_h / 2.0) / self.height,
                (self.cpu_y + self.paddle_h / 2.0) / self.height,
            ],
            dtype=np.float32,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if action == 1:
            self.player_y -= self.paddle_speed
        elif action == 2:
            self.player_y += self.paddle_speed
        self.player_y = float(np.clip(self.player_y, 0, self.height - self.paddle_h))

        if self.human_mode:
            if pygame.get_init():
                keys = pygame.key.get_pressed()
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    self.cpu_y -= self.paddle_speed
                if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                    self.cpu_y += self.paddle_speed
        else:
            cpu_center = self.cpu_y + self.paddle_h / 2.0
            if cpu_center < self.ball_y - 6:
                self.cpu_y += self.cpu_speed
            elif cpu_center > self.ball_y + 6:
                self.cpu_y -= self.cpu_speed
        self.cpu_y = float(np.clip(self.cpu_y, 0, self.height - self.paddle_h))

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y <= 0:
            self.ball_y = 0.0
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y >= self.height - self.ball_size:
            self.ball_y = float(self.height - self.ball_size)
            self.ball_vy = -abs(self.ball_vy)

        reward = 0.0
        done = False

        cpu_x = 24
        if (
            self.ball_vx < 0
            and cpu_x <= self.ball_x <= cpu_x + self.paddle_w
            and self.cpu_y - self.ball_size <= self.ball_y <= self.cpu_y + self.paddle_h
        ):
            self.ball_x = float(cpu_x + self.paddle_w)
            rel = ((self.ball_y + self.ball_size / 2) - (self.cpu_y + self.paddle_h / 2)) / (self.paddle_h / 2)
            self.ball_vx = min(10.0, abs(self.ball_vx) * 1.04)
            self.ball_vy = float(np.clip(rel * 6.5, -7.5, 7.5))

        player_x = self.width - 24 - self.paddle_w
        if (
            self.ball_vx > 0
            and player_x - self.ball_size <= self.ball_x <= player_x + self.paddle_w
            and self.player_y - self.ball_size <= self.ball_y <= self.player_y + self.paddle_h
        ):
            self.ball_x = float(player_x - self.ball_size)
            rel = ((self.ball_y + self.ball_size / 2) - (self.player_y + self.paddle_h / 2)) / (self.paddle_h / 2)
            self.ball_vx = -min(10.0, abs(self.ball_vx) * 1.04)
            self.ball_vy = float(np.clip(rel * 6.5, -7.5, 7.5))
            reward = 0.2

        if self.ball_x < 0:
            self.player_score += 1
            self.score = self.player_score
            reward = 1.0
            if self.player_score >= self.win_score:
                done = True
            else:
                self._serve_ball(towards_player=False)
        elif self.ball_x > self.width:
            self.cpu_score += 1
            reward = -1.0
            if self.cpu_score >= self.win_score:
                done = True
            else:
                self._serve_ball(towards_player=True)

        return self._get_state(), reward, done

    def render(self) -> None:
        if self.closed:
            return
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((self.width, self.height))
            caption = (
                "RL Arcade - Pong (Red = You, Green = AI)"
                if self.human_mode
                else "RL Arcade - Pong (Green = AI)"
            )
            pygame.display.set_caption(caption)
            self._font = pygame.font.SysFont("consolas", 28, bold=True)
            self._hud_font = pygame.font.SysFont("consolas", 14, bold=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                self.close()
                return

        self._screen.fill(self.BG_COLOR)

        for y in range(32, self.height, 24):
            pygame.draw.rect(self._screen, (50, 58, 72), (self.width // 2 - 2, y, 4, 12))

        txt = self._font.render(f"{self.cpu_score}   :   {self.player_score}", True, self.FG_COLOR)
        self._screen.blit(txt, (self.width // 2 - txt.get_width() // 2, 36))

        pygame.draw.rect(self._screen, self.CPU_COLOR, (24, int(self.cpu_y), self.paddle_w, self.paddle_h), border_radius=4)
        pygame.draw.rect(
            self._screen,
            self.PLAYER_COLOR,
            (self.width - 24 - self.paddle_w, int(self.player_y), self.paddle_w, self.paddle_h),
            border_radius=4,
        )
        pygame.draw.rect(
            self._screen,
            self.FG_COLOR,
            (int(self.ball_x), int(self.ball_y), self.ball_size, self.ball_size),
            border_radius=6,
        )

        hud_bar = pygame.Surface((self.width, 28), pygame.SRCALPHA)
        hud_bar.fill((11, 15, 25, 210))
        self._screen.blit(hud_bar, (0, 0))

        mode_txt = os.environ.get("RL_HUD_MODE", "Pong  |  PPO")
        right_txt = "RED: YOU (W/S)  |  GREEN: AI" if self.human_mode else "RED: CPU  |  GREEN: AI"
        left = self._hud_font.render(mode_txt, True, (248, 250, 252))
        right = self._hud_font.render(right_txt, True, (16, 185, 129))
        self._screen.blit(left, (10, 6))
        self._screen.blit(right, (self.width - right.get_width() - 10, 6))

        pygame.display.flip()

    def close(self) -> None:
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._font = None
            self._hud_font = None