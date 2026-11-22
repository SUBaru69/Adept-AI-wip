import os
import subprocess
import sys
import threading
import customtkinter as ctk
import numpy as np
from PIL import Image

ctk.set_appearance_mode("dark")

PALETTE = {
    "bg": "#0B0F19",
    "sidebar": "#0F1624",
    "card": "#131B2E",
    "card_alt": "#182238",
    "border": "#1F2A44",
    "accent": "#6366F1",
    "accent_hover": "#4F46E5",
    "accent_soft": "#1E2547",
    "danger": "#DC2626",
    "danger_hover": "#B91C1C",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "green": "#10B981",
    "green_hover": "#059669",
    "amber": "#F59E0B",
    "cyan": "#06B6D4",
}

GAMES = {
    "Snake": {
        "subtitle": "12×12 Spatial Grid  •  Residual CNN + PPO",
        "dir": r"C:\dev\RL_Arcade",
        "base_cmd": ["main.py", "--env", "snake_grid", "--agent", "ppo", "--episodes", "0"],
        "curve": r"C:\dev\RL_Arcade\checkpoints\snake_grid_ppo_curve.png",
        "kpis": [
            ("Architecture", "Residual CNN", "Dilated 12×12", PALETTE["accent"]),
            ("Peak Score", "32 Apples", "Best Eval Run", PALETTE["green"]),
            ("Parallel Envs", "32 Workers", "Vectorized CPU", PALETTE["cyan"]),
            ("Algorithm", "PPO + GAE", "Clipped Policy", PALETTE["amber"]),
        ],
    },
    "Pong": {
        "subtitle": "6D Physics State  •  MLP Actor-Critic + PPO",
        "dir": r"C:\dev\RL_Arcade",
        "base_cmd": ["main.py", "--env", "pong", "--agent", "ppo", "--episodes", "0"],
        "curve": r"C:\dev\RL_Arcade\checkpoints\pong_ppo_curve.png",
        "kpis": [
            ("Architecture", "2-Layer MLP", "128 Hidden Units", PALETTE["accent"]),
            ("Avg Reward", "+19.72", "4.5 / 5 Avg Score", PALETTE["green"]),
            ("Throughput", "~8,600 SPS", "32 Parallel Envs", PALETTE["cyan"]),
            ("Algorithm", "PPO + GAE", "Zero Hardcoding", PALETTE["amber"]),
        ],
    },
    "Super Mario Bros": {
        "subtitle": "84×84 Stacked Pixels  •  Nature CNN + PPO",
        "dir": r"C:\dev\RL_Mario",
        "base_cmd": ["main.py", "--total-steps", "0"],
        "curve": r"C:\dev\RL_Mario\checkpoints\mario_ppo_curve.png",
        "kpis": [
            ("Architecture", "Nature CNN", "4-Frame Stack", PALETTE["accent"]),
            ("Best Distance", "3,161 x_pos", "Flagpole Cleared", PALETTE["green"]),
            ("Avg Distance", "2,328 x_pos", "74% of World 1-1", PALETTE["cyan"]),
            ("Training", "2.0M Steps", "8 Parallel Envs", PALETTE["amber"]),
        ],
    },
}


class RLLauncher(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("RL Arcade — PPO Dashboard")
        self.geometry("1160x740")
        self.minsize(1020, 660)
        self.configure(fg_color=PALETTE["bg"])

        self.selected_game = "Snake"
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self.kpi_widgets: list[tuple[ctk.CTkLabel, ctk.CTkLabel, ctk.CTkLabel, ctk.CTkFrame]] = []
        self.proc: subprocess.Popen | None = None
        self._curve_img: ctk.CTkImage | None = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_workspace()
        self._select_game("Snake")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(
            self,
            width=230,
            corner_radius=0,
            fg_color=PALETTE["sidebar"],
            border_width=1,
            border_color=PALETTE["border"],
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(6, weight=1)

        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, padx=18, pady=(22, 24), sticky="ew")

        logo = ctk.CTkLabel(
            brand,
            text="RL",
            width=36,
            height=36,
            corner_radius=10,
            fg_color=PALETTE["accent"],
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        logo.pack(side="left", padx=(0, 10))

        title_col = ctk.CTkFrame(brand, fg_color="transparent")
        title_col.pack(side="left")
        ctk.CTkLabel(
            title_col,
            text="RL Arcade",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=PALETTE["text"],
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_col,
            text="PPO Suite",
            font=ctk.CTkFont(size=11),
            text_color=PALETTE["muted"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            sidebar,
            text="ENVIRONMENTS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=PALETTE["muted"],
        ).grid(row=1, column=0, padx=20, pady=(0, 8), sticky="w")

        for idx, name in enumerate(GAMES.keys(), start=2):
            btn = ctk.CTkButton(
                sidebar,
                text=f"   {name}",
                height=42,
                corner_radius=10,
                anchor="w",
                fg_color="transparent",
                hover_color=PALETTE["card_alt"],
                text_color=PALETTE["muted"],
                font=ctk.CTkFont(size=13, weight="bold"),
                command=lambda n=name: self._select_game(n),
            )
            btn.grid(row=idx, column=0, padx=14, pady=3, sticky="ew")
            self.nav_buttons[name] = btn

        gpu_card = ctk.CTkFrame(
            sidebar,
            corner_radius=12,
            fg_color=PALETTE["card"],
            border_width=1,
            border_color=PALETTE["border"],
        )
        gpu_card.grid(row=7, column=0, padx=14, pady=18, sticky="ew")

        ctk.CTkLabel(
            gpu_card,
            text="Hardware Accelerated",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=PALETTE["text"],
        ).pack(anchor="w", padx=14, pady=(12, 2))
        ctk.CTkLabel(
            gpu_card,
            text="NVIDIA RTX 3050 A\nPyTorch CUDA Active",
            justify="left",
            font=ctk.CTkFont(size=11),
            text_color=PALETTE["muted"],
        ).pack(anchor="w", padx=14, pady=(0, 12))

    def _build_workspace(self):
        ws = ctk.CTkFrame(self, fg_color="transparent")
        ws.grid(row=0, column=1, sticky="nsew", padx=24, pady=20)
        ws.grid_columnconfigure(0, weight=1)
        ws.grid_rowconfigure(2, weight=3)
        ws.grid_rowconfigure(3, weight=2)

        header = ctk.CTkFrame(ws, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.grid_columnconfigure(0, weight=1)

        title_wrap = ctk.CTkFrame(header, fg_color="transparent")
        title_wrap.grid(row=0, column=0, sticky="w")

        self.title_label = ctk.CTkLabel(
            title_wrap,
            text="",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=PALETTE["text"],
        )
        self.title_label.pack(anchor="w")
        self.subtitle_label = ctk.CTkLabel(
            title_wrap,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=PALETTE["muted"],
        )
        self.subtitle_label.pack(anchor="w")

        self.status_pill = ctk.CTkLabel(
            header,
            text="●  Ready",
            corner_radius=14,
            fg_color=PALETTE["card"],
            text_color=PALETTE["green"],
            font=ctk.CTkFont(size=12, weight="bold"),
            padx=14,
            pady=6,
        )
        self.status_pill.grid(row=0, column=1, sticky="e")

        kpi_row = ctk.CTkFrame(ws, fg_color="transparent")
        kpi_row.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        for i in range(4):
            kpi_row.grid_columnconfigure(i, weight=1)
            card = ctk.CTkFrame(
                kpi_row,
                corner_radius=12,
                fg_color=PALETTE["card"],
                border_width=1,
                border_color=PALETTE["border"],
            )
            pad_r = 0 if i == 3 else 12
            card.grid(row=0, column=i, sticky="ew", padx=(0, pad_r))

            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=14, pady=(12, 4))

            dot = ctk.CTkFrame(top, width=10, height=10, corner_radius=5, fg_color=PALETTE["accent"])
            dot.pack(side="left", padx=(0, 8))

            lbl = ctk.CTkLabel(
                top, text="", font=ctk.CTkFont(size=11), text_color=PALETTE["muted"]
            )
            lbl.pack(side="left")

            val = ctk.CTkLabel(
                card, text="", font=ctk.CTkFont(size=20, weight="bold"), text_color=PALETTE["text"]
            )
            val.pack(anchor="w", padx=14, pady=(0, 2))

            sub = ctk.CTkLabel(
                card, text="", font=ctk.CTkFont(size=11), text_color=PALETTE["green"]
            )
            sub.pack(anchor="w", padx=14, pady=(0, 12))

            self.kpi_widgets.append((lbl, val, sub, dot))

        mid = ctk.CTkFrame(ws, fg_color="transparent")
        mid.grid(row=2, column=0, sticky="nsew", pady=(0, 16))
        mid.grid_columnconfigure(0, weight=3)
        mid.grid_columnconfigure(1, weight=2)
        mid.grid_rowconfigure(0, weight=1)

        curve_card = ctk.CTkFrame(
            mid,
            corner_radius=12,
            fg_color=PALETTE["card"],
            border_width=1,
            border_color=PALETTE["border"],
        )
        curve_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        curve_card.grid_columnconfigure(0, weight=1)
        curve_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            curve_card,
            text="Training Curve",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=PALETTE["text"],
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 6))

        self.curve_label = ctk.CTkLabel(
            curve_card,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=PALETTE["muted"],
        )
        self.curve_label.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        ctrl_card = ctk.CTkFrame(
            mid,
            corner_radius=12,
            fg_color=PALETTE["card"],
            border_width=1,
            border_color=PALETTE["border"],
        )
        ctrl_card.grid(row=0, column=1, sticky="nsew")
        ctrl_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            ctrl_card,
            text="Launch Configuration",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=PALETTE["text"],
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(14, 10))

        ctk.CTkLabel(
            ctrl_card,
            text="Checkpoint Mode",
            font=ctk.CTkFont(size=12),
            text_color=PALETTE["muted"],
        ).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 6))

        self.mode_var = ctk.StringVar(value="Best")
        ctk.CTkSegmentedButton(
            ctrl_card,
            values=["Untrained", "Latest", "Best"],
            variable=self.mode_var,
            fg_color=PALETTE["bg"],
            selected_color=PALETTE["accent"],
            selected_hover_color=PALETTE["accent_hover"],
            unselected_color=PALETTE["bg"],
            unselected_hover_color=PALETTE["card_alt"],
        ).grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))

        ctk.CTkLabel(
            ctrl_card,
            text="Evaluation Episodes",
            font=ctk.CTkFont(size=12),
            text_color=PALETTE["muted"],
        ).grid(row=3, column=0, sticky="w", padx=18, pady=(0, 6))

        self.ep_var = ctk.StringVar(value="3")
        ctk.CTkSegmentedButton(
            ctrl_card,
            values=["1", "2", "3", "5"],
            variable=self.ep_var,
            fg_color=PALETTE["bg"],
            selected_color=PALETTE["accent"],
            selected_hover_color=PALETTE["accent_hover"],
            unselected_color=PALETTE["bg"],
            unselected_hover_color=PALETTE["card_alt"],
        ).grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 16))

        btn_row = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        btn_row.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 10))
        btn_row.grid_columnconfigure(0, weight=2)
        btn_row.grid_columnconfigure(1, weight=1)

        self.launch_btn = ctk.CTkButton(
            btn_row,
            text="Launch Demo",
            height=40,
            corner_radius=10,
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            font=ctk.CTkFont(size=14, weight="bold"),
            command=lambda: self._on_launch(human=False),
        )
        self.launch_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.stop_btn = ctk.CTkButton(
            btn_row,
            text="Stop",
            height=40,
            corner_radius=10,
            fg_color=PALETTE["danger"],
            hover_color=PALETTE["danger_hover"],
            font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled",
            command=self._on_stop,
        )
        self.stop_btn.grid(row=0, column=1, sticky="ew")

        self.human_btn = ctk.CTkButton(
            ctrl_card,
            text="Play vs AI  (W / S or Arrows)",
            height=38,
            corner_radius=10,
            fg_color=PALETTE["green"],
            hover_color=PALETTE["green_hover"],
            text_color="#052E16",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: self._on_launch(human=True),
        )
        self.human_btn.grid(row=6, column=0, sticky="ew", padx=18, pady=(0, 14))

        console_card = ctk.CTkFrame(
            ws,
            corner_radius=12,
            fg_color=PALETTE["card"],
            border_width=1,
            border_color=PALETTE["border"],
        )
        console_card.grid(row=3, column=0, sticky="nsew")
        console_card.grid_columnconfigure(0, weight=1)
        console_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            console_card,
            text="Live Execution Console",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=PALETTE["text"],
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(10, 4))

        self.log_box = ctk.CTkTextbox(
            console_card,
            fg_color=PALETTE["bg"],
            text_color="#CBD5E1",
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=8,
            border_width=1,
            border_color=PALETTE["border"],
            state="disabled",
        )
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 12))

    def _select_game(self, name: str):
        self.selected_game = name
        info = GAMES[name]

        for game_name, btn in self.nav_buttons.items():
            active = game_name == name
            btn.configure(
                fg_color=PALETTE["accent_soft"] if active else "transparent",
                text_color=PALETTE["text"] if active else PALETTE["muted"],
            )

        self.title_label.configure(text=name)
        self.subtitle_label.configure(text=info["subtitle"])

        for (lbl, val, sub, dot), (k_lbl, k_val, k_sub, k_col) in zip(
            self.kpi_widgets, info["kpis"]
        ):
            lbl.configure(text=k_lbl)
            val.configure(text=k_val)
            sub.configure(text=k_sub, text_color=k_col)
            dot.configure(fg_color=k_col)

        if name == "Pong":
            self.human_btn.grid()
        else:
            self.human_btn.grid_remove()

        self._load_curve(info["curve"])

    def _load_curve(self, path: str):
        if not os.path.exists(path):
            self._curve_img = None
            self.curve_label.configure(image=None, text=f"No learning curve found at:\n{path}")
            return

        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)

        sat = arr.max(axis=2) - arr.min(axis=2)
        is_neutral = sat < 25.0

        lum = arr.mean(axis=2) / 255.0
        inv = 1.0 - lum
        bg = np.array([19.0, 27.0, 46.0], dtype=np.float32)
        fg = np.array([226.0, 232.0, 240.0], dtype=np.float32)
        dark_neutral = bg + inv[..., None] * (fg - bg)

        arr[is_neutral] = dark_neutral[is_neutral]
        dark_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        w, h = dark_img.size
        max_w, max_h = 480, 260
        scale = min(max_w / w, max_h / h)
        new_size = (int(w * scale), int(h * scale))

        self._curve_img = ctk.CTkImage(light_image=dark_img, dark_image=dark_img, size=new_size)
        self.curve_label.configure(image=self._curve_img, text="")

    def _append_log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _build_command(self) -> tuple[list[str], str]:
        info = GAMES[self.selected_game]
        cmd = [sys.executable, "-u", *info["base_cmd"], "--eval-episodes", self.ep_var.get()]

        mode = self.mode_var.get()
        if mode == "Latest":
            cmd.append("--load")
        elif mode == "Best":
            cmd.extend(["--load", "--best"])

        return cmd, info["dir"]

    def _on_launch(self, human: bool = False):
        if self.proc is not None and self.proc.poll() is None:
            return

        cmd, cwd = self._build_command()
        self._clear_log()
        tag = " [Human vs AI]" if human else ""
        self._append_log(f"> [{cwd}]{tag} {' '.join(cmd[2:])}\n\n")

        self.launch_btn.configure(state="disabled", text="Running...")
        self.human_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_pill.configure(text="●  Running", text_color=PALETTE["amber"])

        threading.Thread(
            target=self._run_process, args=(cmd, cwd, human), daemon=True
        ).start()

    def _run_process(self, cmd: list[str], cwd: str, human: bool):
        try:
            run_env = os.environ.copy()
            mode_label = f"Human vs AI ({self.mode_var.get()})" if human else f"{self.mode_var.get()} Model"
            run_env["RL_HUD_MODE"] = f"{self.selected_game}  |  {mode_label}"
            run_env["RL_HUMAN_PONG"] = "1" if human else "0"

            self.proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.after(0, self._append_log, line)
            self.proc.wait()
            code = self.proc.returncode
            self.after(0, self._append_log, f"\n[Process exited with code {code}]\n")
        except Exception as e:
            self.after(0, self._append_log, f"\n[Error] {e}\n")
        finally:
            self.proc = None
            self.after(0, self._reset_buttons)

    def _reset_buttons(self):
        self.launch_btn.configure(state="normal", text="Launch Demo")
        self.human_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_pill.configure(text="●  Ready", text_color=PALETTE["green"])

    def _on_stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self._append_log("\n[Stopped by user]\n")

    def _on_close(self):
        self._on_stop()
        self.destroy()


if __name__ == "__main__":
    app = RLLauncher()
    app.mainloop()