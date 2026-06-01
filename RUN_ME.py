"""
navigator_gui.py — Legend of Turtlebot: Breath of the Forest
GUI flow: SplashScreen → HomePage → NavigatorGUI (dashboard)
"""

import os
import sys
import math
import time
import subprocess
import webbrowser

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import pybullet as p

from goal_detector import GoalDetector
from maze_grid import (
    START_POS, GOAL_POS,
    WORLD_ORIG, CELL_SIZE, GRID_ROWS, GRID_COLS,
)

_DIR = os.path.dirname(os.path.abspath(__file__))

# Zelda website teal palette, literally extracted using a online eyedropper tool.
_BG           = "#071e18"   
_PANEL        = "#0c2820"   
_PANEL_LIGHT  = "#112e24"   
_BORDER       = "#1e5038"   
_BORDER_DIM   = "#112a1e"   
_GOLD         = "#c8a040"   
_GOLD_BRIGHT  = "#e8c858"  
_TEXT         = "#f0e8d0"  
_TEXT_DIM     = "#7a9e8a"   
_ACCENT       = "#3aaa78"   
_ACCENT_HI    = "#5ac898"   
_DANGER       = "#e05040"   
_SKY          = "#4a88a8"   

_FT  = "Georgia"            
_FU  = "Segoe UI"           
_FM  = "Courier New"       


# ══════════════════════════════════════════════════════════════════════════════
# Image helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_image_safe(filename):
    """Load an image from the project directory; return None if missing."""
    path = os.path.join(_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        return None


def fit_image_cover(img, w, h):
    """Scale img to fill (w × h) exactly, cropping edges — no distortion."""
    iw, ih = img.size
    scale  = max(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img    = img.resize((nw, nh), Image.LANCZOS)
    left   = (nw - w) // 2
    top    = (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


def fit_image_contain(img, w, h):
    """Scale img to fit inside (w × h) — letterbox, no distortion."""
    iw, ih = img.size
    scale  = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    return img.resize((nw, nh), Image.LANCZOS)


def blend_to_black(img, alpha):
    """Blend img toward black; alpha 0 = full image, 1 = black."""
    black = Image.new("RGBA", img.size, (0, 0, 0, 255))
    return Image.blend(img.convert("RGBA"), black, alpha)


# ══════════════════════════════════════════════════════════════════════════════
# Background music
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_music_playing():
    """
    Start the Zelda main theme looping if it isn't already playing.
    Silent no-op when pygame is unavailable or the file is missing.
    Safe to call multiple times — only starts the track once.
    """
    path = os.path.join(_DIR, "assets", "Zelda Main Theme Song.mp3")
    if not os.path.exists(path):
        print("Music file not found — skipping audio.")
        return
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(0.25)
            pygame.mixer.music.play(loops=-1)   # loop forever
    except Exception as e:
        print(f"Background music unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SplashScreen
# ══════════════════════════════════════════════════════════════════════════════

class SplashScreen:
    """
    Full-window opening splash.
    Flow: fade-in (1.5 s) → hold (3.5 s) → fade-out (0.8 s) → on_done().
    Click or any key skips straight to fade-out.
    Falls back to a dark themed canvas if splash_background.png is missing.
    """

    _FADE_IN_MS  = 1500
    _HOLD_MS     = 3500
    _FADE_OUT_MS = 800
    _FRAME_MS    = 40   # ~25 fps

    def __init__(self, root, on_done):
        self.root    = root
        self.on_done = on_done
        self._skip   = False

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        # Windowed splash: 85 % of screen, centred, with title bar
        self._sw = int(sw * 0.85)
        self._sh = int(sh * 0.85)
        sx = (sw - self._sw) // 2
        sy = (sh - self._sh) // 2
        root.geometry(f"{self._sw}x{self._sh}+{sx}+{sy}")
        root.title("Legend of Turtlebot: Breath of the Forest")
        root.configure(bg="black")
        root.resizable(False, False)

        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Load & preprocess background
        bg_src = load_image_safe("assets/splash_background.png")
        if bg_src:
            self._bg = fit_image_cover(bg_src, self._sw, self._sh)
        else:
            self._bg = Image.new("RGBA", (self._sw, self._sh), (7, 30, 22, 255))

        # Load & preprocess title text (offset to the right)
        txt_src = load_image_safe("assets/title_text.png")
        if txt_src:
            th = int(self._sh * 0.84)
            tw = int(self._sw * 0.98)
            self._txt = fit_image_contain(txt_src, tw, th)
        else:
            self._txt = None

        # Animation state
        self._phase      = "fade_in"
        self._tick       = 0
        self._total      = self._FADE_IN_MS // self._FRAME_MS

        root.bind("<Button-1>", self._on_skip)
        root.bind("<Key>",      self._on_skip)
        root.focus_force()

        # Hold a black screen for 1 second (MP3 has a brief lead-in delay),
        # then begin the fade-in so image and audio arrive together.
        self.root.after(2000, self._animate)

    # ── skip ──────────────────────────────────────────────────────
    def _on_skip(self, _event=None):
        if not self._skip:
            self._skip = True
            self._start_phase("fade_out", self._FADE_OUT_MS)

    # ── animation loop ────────────────────────────────────────────
    def _animate(self):
        if self._phase == "fade_in":
            t     = self._tick / max(self._total, 1)
            alpha = 1.0 - t          # black → clear
            self._draw(alpha)
            if self._advance():
                self._start_phase("hold", self._HOLD_MS)

        elif self._phase == "hold":
            self._draw(0.0)
            if self._advance():
                self._start_phase("fade_out", self._FADE_OUT_MS)

        elif self._phase == "fade_out":
            t     = self._tick / max(self._total, 1)
            alpha = t                # clear → black
            self._draw(alpha)
            if self._advance():
                self.on_done()
                return

        self.root.after(self._FRAME_MS, self._animate)

    def _advance(self):
        self._tick += 1
        return self._tick >= self._total

    def _start_phase(self, phase, duration_ms):
        self._phase = phase
        self._tick  = 0
        self._total = duration_ms // self._FRAME_MS

    # ── render one frame ─────────────────────────────────────────
    def _draw(self, black_alpha):
        frame = self._bg.copy()

        if self._txt:
            tw, th = self._txt.size
            # Centred-right: ~60 % from left edge, vertically centred
            tx = int(self._sw * 0.54) - tw // 2 + int(self._sw * 0.12)
            ty = (self._sh - th) // 2
            frame.paste(self._txt, (tx, ty), self._txt)

        if black_alpha > 0.0:
            frame = blend_to_black(frame, black_alpha)

        tk_img = ImageTk.PhotoImage(frame.convert("RGB"))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=tk_img)
        self.canvas.image = tk_img

        # "skip" hint — appears 2 s into the hold phase (50 ticks × 40 ms)
        if self._phase == "hold" and self._tick >= 50:
            self.canvas.create_text(
                self._sw // 2, self._sh - 60,
                text="PRESS ANY KEY OR CLICK TO CONTINUE",
                fill="#ffffff",
                font=(_FU, 20),
                stipple="gray50",
            )


# ══════════════════════════════════════════════════════════════════════════════
# HomePage
# ══════════════════════════════════════════════════════════════════════════════

class HomePage:
    """
    Zelda / BotW-themed launcher homepage.
    Hero banner (landscape image) + three action cards.
    """

    def __init__(self, root):
        self.root = root
        root.overrideredirect(False)
        root.title("Legend of Turtlebot: Breath of the Forest")
        root.configure(bg=_BG)
        root.resizable(True, True)

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        self._WIN_W = int(sw * 0.85)
        self._WIN_H = int(sh * 0.85)
        # Window is already at this size from the splash — no geometry change needed

        self._bg_src  = load_image_safe("assets/home_background.png")
        self._txt_src = load_image_safe("assets/title_text.png")
        self._build()

    def _build(self):
        banner_h = int(self._WIN_H * 0.44)

        # ── Full-page background canvas (bottom layer) ──────────────
        # Draws home_background.png at 78 % darkness behind all content.
        # The inner content frame is placed as a window on this canvas so
        # the background is visible in any uncovered margins / below footer.
        page_bg = tk.Canvas(self.root, bg=_BG, highlightthickness=0)
        page_bg.pack(fill="both", expand=True)

        inner = tk.Frame(page_bg, bg=_BG)
        _iid  = page_bg.create_window(0, 0, anchor="nw", window=inner)

        def _on_page_resize(e):
            page_bg.itemconfig(_iid, width=e.width)
            if not self._bg_src or e.width < 1 or e.height < 1:
                return
            img  = fit_image_cover(self._bg_src.convert("RGB"), e.width, e.height)
            img  = Image.blend(img.convert("RGBA"),
                               Image.new("RGBA", img.size, (7, 30, 22, 255)), 0.78)
            tk_i = ImageTk.PhotoImage(img.convert("RGB"))
            page_bg.delete("home_bg")
            page_bg.create_image(0, 0, anchor="nw", image=tk_i, tags="home_bg")
            page_bg.home_bg_img = tk_i       # keep reference
            page_bg.tag_lower("home_bg")     # behind the inner frame window

        page_bg.bind("<Configure>", _on_page_resize)

        # All content goes inside 'inner' — treat it as the page root from here
        r = inner

        # ── Hero banner (home_background.png at 38 % dark) ──────
        banner = tk.Canvas(r, height=banner_h, bg=_BG, highlightthickness=0)
        banner.pack(fill="x")
        banner.bind("<Configure>",
            lambda e: self._redraw_banner(banner, e.width, e.height))
        self._banner_canvas = banner
        self._banner_h      = banner_h
        self._redraw_banner(banner, self._WIN_W, banner_h)

        # ── Gold rule ────────────────────────────────────────────
        tk.Frame(r, bg=_GOLD, height=1).pack(fill="x")

        # ── Subtitle row ─────────────────────────────────────────
        sub = tk.Frame(r, bg=_BG)
        sub.pack(fill="x", padx=32, pady=(12, 6))
        tk.Label(sub, text="SELECT AN ACTION",
                 font=(_FT, 11, "bold"), fg=_GOLD, bg=_BG,
                 anchor="w").pack(side="left")
        tk.Label(sub, text="By Abisha Nasim & William Sklibosios",
                 font=(_FU, 9), fg=_TEXT_DIM, bg=_BG,
                 anchor="e").pack(side="right")

        # ── Cards row ────────────────────────────────────────────
        cards = tk.Frame(r, bg=_BG)
        cards.pack(fill="both", expand=True, padx=28, pady=(4, 10))
        for col in range(3):
            cards.grid_columnconfigure(col, weight=1)
        cards.grid_rowconfigure(0, weight=1)

        self._card(cards, 0,
                   label="TRAIN MODEL",
                   icon=">>",
                   desc=(
                       "Run PPO reinforcement learning to train\n"
                       "the TurtleBot3 maze navigation policy."
                   ),
                   btn="BEGIN TRAINING",
                   cmd=self._launch_train)

        self._card(cards, 1,
                   label="TEST MODEL",
                   icon="|>",
                   desc=(
                       "Load the trained policy and watch the\n"
                       "robot navigate three maze scenarios."
                   ),
                   btn="RUN NAVIGATOR",
                   cmd=self._launch_test)

        self._card(cards, 2,
                   label="OPEN WANDB",
                   icon="[~]",
                   desc=(
                       "View training runs, rewards, metrics,\n"
                       "and experiment logs on Weights & Biases."
                   ),
                   btn="OPEN DASHBOARD",
                   cmd=self._launch_wandb)

        # ── Footer ───────────────────────────────────────────────
        tk.Frame(r, bg=_BORDER_DIM, height=1).pack(fill="x")
        footer_row = tk.Frame(r, bg=_BG)
        footer_row.pack(fill="x", padx=16, pady=4)
        tk.Label(footer_row,
                 text="Legend of Turtlebot: Breath of the Forest  ·  v1.0",
                 font=(_FU, 8), fg=_TEXT_DIM, bg=_BG).pack(side="left")

        self._muted = False
        self._mute_btn = tk.Button(
            footer_row,
            text="♪  MUTE",
            font=(_FU, 8, "bold"),
            fg=_TEXT_DIM, bg=_BG,
            activeforeground=_GOLD, activebackground=_BG,
            relief="flat", bd=0, cursor="hand2",
            command=self._toggle_music,
        )
        self._mute_btn.pack(side="right")
        self._mute_btn.bind("<Enter>", lambda e: self._mute_btn.config(fg=_GOLD))
        self._mute_btn.bind("<Leave>", lambda e: self._mute_btn.config(fg=_TEXT_DIM))

    def _redraw_banner(self, canvas, w, h):
        if w < 1 or h < 1:
            return
        if self._bg_src:
            img = fit_image_cover(self._bg_src.convert("RGB"), w, h)
            # Subtle darkening so text stays readable
            img = Image.blend(
                img.convert("RGBA"),
                Image.new("RGBA", img.size, (0, 0, 0, 180)),
                0.38,
            )
            tk_img = ImageTk.PhotoImage(img.convert("RGB"))
        else:
            tk_img = None

        canvas.delete("all")
        if tk_img:
            canvas.create_image(0, 0, anchor="nw", image=tk_img)
            canvas.image = tk_img   # prevent GC

        cx = w // 2
        if self._txt_src:
            max_tw  = min(int(w * 0.62 * 2), w)
            max_th  = min(int(h * 0.58 * 2), h)
            timg    = fit_image_contain(self._txt_src, max_tw, max_th)
            tw, th  = timg.size
            tx, ty  = cx - tw // 2, h // 2 - th // 2 - 10
            tk_t    = ImageTk.PhotoImage(timg)
            canvas.create_image(tx, ty, anchor="nw", image=tk_t)
            canvas.title_img = tk_t                # keep reference per canvas
        else:
            canvas.create_text(cx, h // 2 - 14,
                text="LEGEND OF TURTLEBOT",
                font=(_FT, 28, "bold"), fill=_GOLD_BRIGHT)
            canvas.create_text(cx, h // 2 + 24,
                text="Breath of the Forest",
                font=(_FT, 14, "italic"), fill=_TEXT)

    def _card(self, parent, col, label, icon, desc, btn, cmd):
        pad_r = 12 if col < 2 else 0
        card  = tk.Frame(parent, bg=_PANEL,
                         highlightbackground=_BORDER, highlightthickness=1)
        card.grid(row=0, column=col, sticky="nsew", padx=(0, pad_r), pady=4)

        tk.Label(card, text=icon,
                 font=(_FM, 20, "bold"), fg=_GOLD, bg=_PANEL).pack(pady=(20, 2))

        tk.Label(card, text=label,
                 font=(_FT, 13, "bold"), fg=_GOLD_BRIGHT, bg=_PANEL).pack(pady=(0, 8))

        tk.Frame(card, bg=_BORDER_DIM, height=1).pack(fill="x", padx=18, pady=(0, 10))

        tk.Label(card, text=desc,
                 font=(_FU, 9), fg=_TEXT, bg=_PANEL,
                 justify="center").pack(padx=14, pady=(0, 16))

        b = tk.Button(card,
                      text=btn,
                      font=(_FU, 9, "bold"),
                      fg=_BG, bg=_GOLD,
                      activeforeground=_BG, activebackground=_GOLD_BRIGHT,
                      relief="flat", bd=0,
                      padx=18, pady=9,
                      cursor="hand2",
                      command=cmd)
        b.pack(pady=(0, 20))
        b.bind("<Enter>", lambda e: b.config(bg=_GOLD_BRIGHT))
        b.bind("<Leave>", lambda e: b.config(bg=_GOLD))

    def _toggle_music(self):
        try:
            import pygame
            if self._muted:
                pygame.mixer.music.set_volume(0.25)
                self._muted = False
                self._mute_btn.config(text="♪  MUTE")
            else:
                pygame.mixer.music.set_volume(0.0)
                self._muted = True
                self._mute_btn.config(text="♪  UNMUTE")
        except Exception:
            pass

    def _launch_train(self):
        flags = subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0
        subprocess.Popen(
            [sys.executable, os.path.join(_DIR, "train.py")],
            creationflags=flags,
        )

    def _launch_test(self):
        flags = subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0
        subprocess.Popen(
            [sys.executable, os.path.join(_DIR, "test.py")],
            creationflags=flags,
        )

    def _launch_wandb(self):
        webbrowser.open("https://wandb.ai")


# ══════════════════════════════════════════════════════════════════════════════
# RobotGUI — live navigation dashboard
# ══════════════════════════════════════════════════════════════════════════════

class NavigatorGUI:
    """
    Dashboard for live robot navigation.
      - Robot POV + CNN detection        (top-left camera panel)
      - PyBullet 3D overhead view        (top-right camera panel)
      - 2D bird's-eye path trace         (lower-left canvas)
      - LiDAR 360° bar scan              (lower-right canvas)
      - Stats bar, progress bar, episode result panel
    Public API is unchanged from the original implementation.
    """

    def __init__(self, env):
        self.env            = env
        self._path          = []
        self._next_callback = None
        self._waiting       = False

        try:
            self._goal_detector = GoalDetector()
            print("Goal detector initialised")
        except Exception as e:
            self._goal_detector = None
            print(f"CNN not available: {e}")

        # ── Window ─────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Legend of Turtlebot: Breath of the Forest — Navigator")
        self.root.configure(bg=_BG)
        self.root.resizable(True, True)
        _sw = self.root.winfo_screenwidth()
        _sh = self.root.winfo_screenheight()
        _ww = int(_sw * 0.85)
        _wh = int(_sh * 0.85)
        _wx = (_sw - _ww) // 2
        _wy = (_sh - _wh) // 2
        self.root.geometry(f"{_ww}x{_wh}+{_wx}+{_wy}")

        # ── TTK styles ─────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("default")
        style.configure("LT.Horizontal.TProgressbar",
                        troughcolor=_PANEL,
                        background=_GOLD,
                        bordercolor=_BORDER_DIM,
                        darkcolor=_GOLD,
                        lightcolor=_GOLD_BRIGHT)
        style.configure("LT.Vertical.TScrollbar",
                        background=_PANEL,
                        troughcolor=_BG,
                        arrowcolor=_BORDER)

        # ── Scrollable wrapper ─────────────────────────────────────
        outer = tk.Frame(self.root, bg=_BG)
        outer.pack(fill="both", expand=True)

        _sc = tk.Canvas(outer, bg=_BG, highlightthickness=0)
        _sb = ttk.Scrollbar(outer, orient="vertical", command=_sc.yview,
                             style="LT.Vertical.TScrollbar")
        _sc.configure(yscrollcommand=_sb.set)
        _sb.pack(side="right", fill="y")
        _sc.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(_sc, bg=_BG)
        _win_id = _sc.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>",
            lambda e: _sc.configure(scrollregion=_sc.bbox("all")))

        # Draw navigator_background.png (90 % dark) on the scroll canvas so it
        # shows in any area not covered by the content frame (below footer, wide gaps).
        _mf_src = load_image_safe("assets/navigator_background.png")

        def _on_sc_resize(e, _src=_mf_src):
            _sc.itemconfig(_win_id, width=e.width)
            if _src and e.width > 1 and e.height > 1:
                img  = fit_image_cover(_src.convert("RGB"), e.width, e.height)
                img  = Image.blend(img.convert("RGBA"),
                                   Image.new("RGBA", img.size, (7, 30, 22, 255)), 0.90)
                tk_i = ImageTk.PhotoImage(img.convert("RGB"))
                _sc.delete("sc_bg")
                _sc.create_image(0, 0, anchor="nw", image=tk_i, tags="sc_bg")
                _sc.sc_bg_img = tk_i     # prevent GC
                _sc.tag_lower("sc_bg")   # behind _inner window

        _sc.bind("<Configure>", _on_sc_resize)
        _sc.bind_all("<MouseWheel>",
            lambda e: _sc.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────

    def _build_ui(self):
        i = self._inner

        # ── Atmosphere strip — navigator_background.png + title image ─
        self._atm_src     = load_image_safe("assets/navigator_background.png")
        self._atm_txt_src = load_image_safe("assets/title_text.png")
        self._atm_canvas = tk.Canvas(i, height=150, bg=_BG, highlightthickness=0)
        self._atm_canvas.pack(fill="x")
        self._atm_canvas.bind("<Configure>",
            lambda e: self._draw_atmosphere(e.width))
        self._draw_atmosphere(820)

        self._sep(i)

        # ── Current scenario / seed label ──────────────────────────
        self.var_scenario = tk.StringVar(value="")
        tk.Label(i, textvariable=self.var_scenario,
                 font=(_FT, 12, "bold"), fg=_GOLD_BRIGHT, bg=_BG
                 ).pack(pady=(6, 2))

        # ── Top row: POV | OVERHEAD | BIRD'S EYE (equal thirds) ────
        top_row = tk.Frame(i, bg=_BG)
        top_row.pack(fill="x", padx=20, pady=8)
        top_row.grid_columnconfigure(0, weight=2, minsize=160)   # cameras get more weight
        top_row.grid_columnconfigure(1, weight=2, minsize=160)
        top_row.grid_columnconfigure(2, weight=1, minsize=140)   # map stays readable

        self.pov_label      = self._cam_panel(top_row, "ROBOT POV  ·  CNN DETECTION",  0)
        self.overhead_label = self._cam_panel(top_row, "OVERHEAD VIEW  ·  3D PHYSICS", 1)

        map_box = self._panel_frame(top_row, "BIRD'S EYE PATH", 2)
        self.map_canvas = tk.Canvas(map_box, bg=_BG, highlightthickness=0)
        self.map_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.map_canvas.bind("<Configure>", lambda _e: self._on_map_resize())

        self._sep(i)

        # ── LiDAR full-width below the three panels ─────────────────
        lidar_outer = tk.Frame(i, bg=_PANEL,
                               highlightbackground=_BORDER, highlightthickness=1)
        lidar_outer.pack(fill="x", padx=20, pady=(4, 0))
        tk.Label(lidar_outer, text="◆ LIDAR SCAN  ·  360°",
                 font=(_FU, 8, "bold"), fg=_GOLD, bg=_PANEL,
                 anchor="w").pack(pady=(6, 2), padx=8, fill="x")
        self.lidar_canvas = tk.Canvas(lidar_outer, height=140,
                                      bg=_BG, highlightthickness=0)
        self.lidar_canvas.pack(fill="x", padx=8, pady=(0, 8))
        self.lidar_canvas.bind("<Configure>", lambda _e: self._render_lidar())

        self._sep(i)

        # ── Stats bar ──────────────────────────────────────────────
        stats = tk.Frame(i, bg=_PANEL,
                         highlightbackground=_BORDER, highlightthickness=1)
        stats.pack(fill="x", padx=20)
        for col in range(5):
            stats.grid_columnconfigure(col, weight=1)

        self.var_episode = self._stat_col(stats, "EPISODE",      0)
        self.var_steps   = self._stat_col(stats, "STEPS",        1)
        self.var_reward  = self._stat_col(stats, "TOTAL REWARD", 2)
        self.var_dist    = self._stat_col(stats, "DIST TO GOAL", 3)
        self.var_status  = self._stat_col(stats, "STATUS",       4)

        self._sep(i)

        # ── Progress bar ───────────────────────────────────────────
        pf = tk.Frame(i, bg=_BG)
        pf.pack(fill="x", padx=28, pady=(6, 10))
        tk.Label(pf, text="EPISODE PROGRESS",
                 font=(_FU, 7), fg=_TEXT_DIM, bg=_BG).pack(anchor="w")
        self.progress = ttk.Progressbar(
            pf, length=700, mode="determinate", maximum=100,
            style="LT.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(2, 4))

        self._sep(i)

        # ── Result panel (hidden until episode ends) ────────────────
        self.result_frame = tk.Frame(i, bg=_BG)
        self.result_frame.pack(fill="x", padx=20, pady=(8, 14))
        self.result_frame.pack_forget()

        self.result_title = tk.Label(self.result_frame, text="",
                                     font=(_FT, 13, "bold"), bg=_BG)
        self.result_title.pack(pady=(14, 2))

        self.result_reason = tk.Label(self.result_frame, text="",
                                      font=(_FU, 9), fg=_TEXT_DIM, bg=_BG,
                                      wraplength=700, justify="center")
        self.result_reason.pack(pady=(0, 8))

        score_row = tk.Frame(self.result_frame, bg=_PANEL,
                             highlightbackground=_BORDER, highlightthickness=1)
        score_row.pack(fill="x", padx=40, pady=(4, 8))
        self.score_reward  = self._score_col(score_row, "FINAL REWARD", 0)
        self.score_steps   = self._score_col(score_row, "STEPS TAKEN",  1)
        self.score_dist    = self._score_col(score_row, "FINAL DIST",   2)
        self.score_outcome = self._score_col(score_row, "OUTCOME",      3)

        self._next_btn = tk.Button(
            self.result_frame,
            text="▶  NEXT SCENARIO",
            font=(_FU, 10, "bold"),
            fg=_BG, bg=_GOLD,
            activeforeground=_BG, activebackground=_GOLD_BRIGHT,
            relief="flat", bd=0, padx=24, pady=10,
            cursor="hand2", command=self._on_next,
        )
        self._next_btn.pack(pady=(4, 16))
        self._next_btn.bind("<Enter>", lambda e: self._next_btn.config(bg=_GOLD_BRIGHT))
        self._next_btn.bind("<Leave>", lambda e: self._next_btn.config(bg=_GOLD))

        # ── Initial blank cameras ───────────────────────────────────
        self._episode    = 0
        self._max_steps  = 4000
        self._MAP_W      = 280
        self._MAP_H      = 220
        self._path_world = []   # world-coord path, safe to redraw after resize
        self._visit_map  = {}   # cached visit_map for resize redraws
        self._last_lidar = None # cached lidar data for resize redraws

        self._ph  = self._make_blank(320, 240)
        self._oph = self._make_blank(320, 240)
        self.pov_label.configure(image=self._ph)
        self.overhead_label.configure(image=self._oph)

        self._draw_map_base()
        self._overlay_visited_cells({})

    # ── Widget builders ────────────────────────────────────────────

    def _sep(self, parent):
        tk.Frame(parent, bg=_BORDER_DIM, height=1).pack(fill="x", padx=20, pady=1)

    def _panel_frame(self, parent, title, col):
        box = tk.Frame(parent, bg=_PANEL,
                       highlightbackground=_BORDER, highlightthickness=1)
        box.grid(row=0, column=col, sticky="nsew", padx=5)
        tk.Label(box, text=f"◆ {title}",
                 font=(_FU, 8, "bold"), fg=_GOLD, bg=_PANEL,
                 anchor="w").pack(pady=(6, 2), padx=8, fill="x")
        return box

    def _cam_panel(self, parent, title, col):
        box = self._panel_frame(parent, title, col)
        lbl = tk.Label(box, bg=_PANEL)
        lbl.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        return lbl

    def _stat_col(self, parent, label, col):
        f = tk.Frame(parent, bg=_PANEL)
        f.grid(row=0, column=col, padx=16, pady=10, sticky="nsew")
        tk.Label(f, text=label, font=(_FU, 7), fg=_TEXT_DIM, bg=_PANEL).pack()
        var = tk.StringVar(value="—")
        tk.Label(f, textvariable=var, font=(_FM, 14, "bold"),
                 fg=_GOLD_BRIGHT, bg=_PANEL).pack()
        return var

    def _score_col(self, parent, label, col):
        f = tk.Frame(parent, bg=_PANEL)
        f.grid(row=0, column=col, padx=30, pady=10)
        tk.Label(f, text=label, font=(_FU, 7), fg=_TEXT_DIM, bg=_PANEL).pack()
        var = tk.StringVar(value="—")
        tk.Label(f, textvariable=var, font=(_FM, 15, "bold"),
                 fg=_GOLD_BRIGHT, bg=_PANEL).pack()
        return var

    # ── Public API (unchanged signatures) ─────────────────────────

    def update(self, steps, reward, dist, status, lidar=None, visit_map=None):
        self.var_steps.set(str(steps))
        self.var_reward.set(f"{reward:.1f}")
        self.var_dist.set(f"{dist:.2f}m")
        self.var_episode.set(str(self._episode))
        self.var_status.set(status)
        self.progress["value"] = min(100, steps / self._max_steps * 100)

        self._refresh_pov()
        self._refresh_overhead()
        if visit_map is not None:
            self._overlay_visited_cells(visit_map)
        self._update_path()
        if lidar is not None:
            self._draw_lidar(lidar)
        self.root.update()

    def new_episode(self, episode_num, scenario_name, max_steps=6000):
        self._episode    = episode_num
        self._max_steps  = max_steps
        self._path_world = []
        self.progress["value"] = 0
        self.result_frame.pack_forget()
        self._draw_map_base()
        self._overlay_visited_cells({})
        self.var_episode.set(str(episode_num))
        self.var_status.set("RUNNING")
        self.var_scenario.set(scenario_name)
        self.root.update()

    def show_result(self, status, reward, steps, dist,
                    scenario_name, is_last=False):
        colour_map  = {"REACHED GOAL": _ACCENT_HI, "COLLISION": _DANGER}
        outcome_map = {"REACHED GOAL": "SUCCESS",   "COLLISION": "COLLISION"}
        reason_map  = {
            "REACHED GOAL": (
                f"The TurtleBot navigated '{scenario_name}' and reached the goal. "
                "The PPO policy found a clear path through the forest maze."
            ),
            "COLLISION": (
                f"The TurtleBot collided in '{scenario_name}'. "
                "LiDAR detected the obstacle too late. Further training will help."
            ),
        }
        colour  = colour_map.get(status, "#ffaa40")
        outcome = outcome_map.get(status, "TIMEOUT")
        reason  = reason_map.get(
            status,
            f"TurtleBot ran out of steps in '{scenario_name}'. More training needed.")

        self.result_title.configure(
            text=f"[ {outcome} ]  —  {scenario_name}", fg=colour)
        self.result_reason.configure(text=reason)
        self.score_reward.set(f"{reward:.1f}")
        self.score_steps.set(str(steps))
        self.score_dist.set(f"{dist:.2f}m")
        self.score_outcome.set(outcome)

        self._next_btn.configure(
            text="[ ALL SCENARIOS COMPLETE ]" if is_last else "▶  NEXT SCENARIO",
            bg=_GOLD, state="normal")

        self.result_frame.pack(fill="x", padx=20, pady=(8, 14))
        self.root.update()

    def set_next_callback(self, callback):
        self._next_callback = callback

    def wait_for_next(self):
        self._next_btn.configure(state="normal")
        self._waiting = True
        while self._waiting:
            self.root.update()
            time.sleep(0.05)

    def close(self):
        self.root.destroy()

    def _on_next(self):
        self._waiting = False
        self.result_frame.pack_forget()
        if self._next_callback:
            self._next_callback()

    # ── POV camera ────────────────────────────────────────────────

    def _refresh_pov(self):
        phys = self.env._physics
        if phys is None:
            return
        pos, orn = p.getBasePositionAndOrientation(
            self.env._robot, physicsClientId=phys)
        yaw = p.getEulerFromQuaternion(orn)[2]

        eye    = [pos[0], pos[1], pos[2] + 0.15]
        target = [pos[0] + math.cos(yaw), pos[1] + math.sin(yaw), pos[2] + 0.15]
        view   = p.computeViewMatrix(eye, target, [0, 0, 1], physicsClientId=phys)
        proj   = p.computeProjectionMatrixFOV(
            fov=90, aspect=4/3, nearVal=0.05, farVal=10, physicsClientId=phys)
        _, _, rgb, _, _ = p.getCameraImage(320, 240, view, proj,
                                            physicsClientId=phys)

        arr  = np.array(rgb, dtype=np.uint8).reshape(240, 320, 4)
        img  = Image.fromarray(arr, "RGBA").convert("RGB")
        draw = ImageDraw.Draw(img)

        if self._goal_detector:
            detected, conf, bbox = self._goal_detector.detect(img)
            if detected and bbox is not None:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline=_GOLD, width=3)
                lbl = f"GOAL {conf*100:.0f}%"
                draw.rectangle([x1, y1-16, x1+len(lbl)*8, y1], fill=_GOLD)
                draw.text((x1+2, y1-15), lbl, fill="#000000")
            status_t   = "CNN: GOAL DETECTED" if detected else "CNN: SEARCHING..."
            status_col = _GOLD if detected else _DANGER
        else:
            status_t   = "CNN: UNAVAILABLE"
            status_col = _TEXT_DIM

        draw.rectangle([0, 222, 320, 240], fill=(7, 24, 18))
        draw.text((4, 224), status_t, fill=status_col)

        # Scale up to fill the label's current width, height fixed at 4:3.
        # Never read winfo_height() — that creates a growth feedback loop.
        dw = max(self.pov_label.winfo_width(), 320)
        dh = int(dw * 3 / 4)
        if (dw, dh) != (320, 240):
            img = img.resize((dw, dh), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(img)
        self.pov_label.configure(image=tk_img)
        self.pov_label.image = tk_img

    # ── Overhead 3D camera ────────────────────────────────────────

    _OVH_HALF_H = 6.14
    _OVH_HALF_W = 8.19

    def _refresh_overhead(self):
        phys = self.env._physics
        if phys is None:
            return
        view = p.computeViewMatrix(
            cameraEyePosition    = [0, 0, 16],
            cameraTargetPosition = [0, 0,  0],
            cameraUpVector       = [0, 1,  0],
            physicsClientId      = phys)
        proj = p.computeProjectionMatrixFOV(
            fov=42, aspect=4/3, nearVal=0.1, farVal=25,
            physicsClientId=phys)
        _, _, rgb, _, _ = p.getCameraImage(320, 240, view, proj,
                                            physicsClientId=phys)

        arr = np.array(rgb, dtype=np.uint8).reshape(240, 320, 4)
        img = Image.fromarray(arr, "RGBA").convert("RGB")

        pos, orn = p.getBasePositionAndOrientation(
            self.env._robot, physicsClientId=phys)
        rx, ry = self._ovh_to_px(pos[0], pos[1])
        yaw    = p.getEulerFromQuaternion(orn)[2]
        dx, dy = int(14 * math.cos(yaw)), int(-14 * math.sin(yaw))

        draw = ImageDraw.Draw(img)
        draw.ellipse([rx-5, ry-5, rx+5, ry+5],
                     fill=_GOLD, outline="#ffffff", width=1)
        draw.line([rx, ry, rx+dx, ry+dy], fill="#ffffff", width=2)

        # Scale up to fill the label's current width, height fixed at 4:3.
        # Never read winfo_height() — that creates a growth feedback loop.
        dw = max(self.overhead_label.winfo_width(), 320)
        dh = int(dw * 3 / 4)
        if (dw, dh) != (320, 240):
            img = img.resize((dw, dh), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(img)
        self.overhead_label.configure(image=tk_img)
        self.overhead_label.image = tk_img

    def _ovh_to_px(self, wx, wy):
        px = int((wx + self._OVH_HALF_W) / (2 * self._OVH_HALF_W) * 320)
        py = int((self._OVH_HALF_H - wy) / (2 * self._OVH_HALF_H) * 240)
        return px, py

    # ── Bird's eye 2D path map ────────────────────────────────────

    _WORLD_MIN = -5.0
    _WORLD_MAX =  5.0

    def _map_uniform_scale(self):
        """Single scale + centred offsets so the maze never stretches."""
        span = self._WORLD_MAX - self._WORLD_MIN
        pad  = 10
        uw   = max(self._MAP_W - 2 * pad, 1)
        uh   = max(self._MAP_H - 2 * pad, 1)
        s    = min(uw, uh) / span
        ox   = pad + (uw - span * s) / 2
        oy   = pad + (uh - span * s) / 2
        return s, ox, oy

    def _world_to_canvas(self, wx, wy):
        s, ox, oy = self._map_uniform_scale()
        cx = int((wx - self._WORLD_MIN) * s + ox)
        cy = int((self._WORLD_MAX - wy) * s + oy)
        return cx, cy

    def _cell_canvas_rect(self, row, col):
        x0w   = WORLD_ORIG + col * CELL_SIZE
        x1w   = WORLD_ORIG + (col + 1) * CELL_SIZE
        y_top = (-WORLD_ORIG) - row * CELL_SIZE
        y_bot = (-WORLD_ORIG) - (row + 1) * CELL_SIZE
        cx0, cy0 = self._world_to_canvas(x0w, y_top)
        cx1, cy1 = self._world_to_canvas(x1w, y_bot)
        return cx0, cy0, cx1, cy1

    def _sync_map_size(self):
        w = self.map_canvas.winfo_width()
        h = self.map_canvas.winfo_height()
        if w > 1:
            self._MAP_W = w
        if h > 1:
            self._MAP_H = h

    def _on_map_resize(self):
        """Debounced handler — fires 80 ms after the last Configure event."""
        if hasattr(self, "_map_resize_job"):
            self.root.after_cancel(self._map_resize_job)
        self._map_resize_job = self.root.after(80, self._do_map_resize)

    def _do_map_resize(self):
        self._sync_map_size()
        self._draw_map_base()
        self._overlay_visited_cells(self._visit_map)
        self._redraw_full_path()

    def _redraw_full_path(self):
        """Redraw the entire path from stored world coordinates."""
        c = self.map_canvas
        c.delete("path_line")
        c.delete("robot_dot")
        path = self._path_world
        n    = len(path)
        for idx in range(1, n):
            t   = idx / max(n - 1, 1)
            r   = int(45  + 155 * t)
            g   = int(120 + 48  * t)
            b   = int(20  + 35  * t)
            col = f"#{r:02x}{g:02x}{b:02x}"
            x0, y0 = self._world_to_canvas(*path[idx - 1])
            x1, y1 = self._world_to_canvas(*path[idx])
            c.create_line(x0, y0, x1, y1,
                          fill=col, width=2, capstyle=tk.ROUND,
                          tags="path_line")
            if idx % 30 == 0:
                c.create_oval(x1-2, y1-2, x1+2, y1+2,
                              fill=col, outline="", tags="path_line")
        c.tag_raise("path_line")
        c.tag_raise("wall_seg")
        c.tag_raise("marker")
        if path:
            rx, ry = self._world_to_canvas(*path[-1])
            c.create_oval(rx-4, ry-4, rx+4, ry+4,
                          fill=_GOLD, outline=_BG, width=2, tags="robot_dot")

    def _draw_map_base(self):
        self._sync_map_size()
        c = self.map_canvas
        c.delete("all")
        c.create_rectangle(0, 0, self._MAP_W, self._MAP_H,
                           fill=_BG, outline="", tags="bg")

        if hasattr(self.env, "_walls") and self.env._walls:
            s, _, _ = self._map_uniform_scale()
            for seg in self.env._walls:
                if len(seg) == 4:
                    wx, wy, hx, hy = seg
                    px, py = self._world_to_canvas(wx, wy)
                    rw = max(int(hx * s), 2)
                    rh = max(int(hy * s), 2)
                    c.create_rectangle(px-rw, py-rh, px+rw, py+rh,
                                       fill=_BORDER_DIM, outline=_BORDER,
                                       width=1, tags="wall_seg")

        gx, gy = self._world_to_canvas(GOAL_POS[0], GOAL_POS[1])
        c.create_rectangle(gx-6, gy-6, gx+6, gy+6,
                           fill=_GOLD, outline=_BORDER, width=1, tags="marker")
        c.create_text(gx, gy, text="G",
                      fill=_BG, font=(_FM, 7, "bold"), tags="marker")

        sx, sy = self._world_to_canvas(START_POS[0], START_POS[1])
        c.create_oval(sx-5, sy-5, sx+5, sy+5,
                      fill=_BORDER, outline=_GOLD, tags="marker")
        c.create_text(sx, sy, text="S",
                      fill=_GOLD, font=(_FM, 7, "bold"), tags="marker")

    def _overlay_visited_cells(self, visit_map):
        self._visit_map = visit_map   # cache for resize redraws
        c = self.map_canvas
        c.delete("cell_fill")
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                cx0, cy0, cx1, cy1 = self._cell_canvas_rect(row, col)
                fill = "#0f2e22" if (row, col) in visit_map else "#091e14"
                c.create_rectangle(cx0, cy0, cx1, cy1,
                                   fill=fill, outline=_BORDER_DIM,
                                   width=1, tags="cell_fill")
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                cx0, cy0, cx1, cy1 = self._cell_canvas_rect(row, col)
                lx, ly = (cx0+cx1)//2, (cy0+cy1)//2
                c.create_text(
                    lx, ly,
                    text=chr(ord("A") + col) + str(row + 1),
                    fill=_TEXT_DIM if (row, col) in visit_map else _BORDER_DIM,
                    font=(_FM, 6), tags="cell_fill")
        c.tag_raise("path_line")
        c.tag_raise("wall_seg")
        c.tag_raise("marker")

    def _update_path(self):
        self._sync_map_size()
        phys = self.env._physics
        if phys is None:
            return
        pos, _ = p.getBasePositionAndOrientation(
            self.env._robot, physicsClientId=phys)
        self._path_world.append((pos[0], pos[1]))  # store world coords
        c   = self.map_canvas
        n   = len(self._path_world)

        if n >= 2:
            t   = min((n - 1) / max(n - 1, 1), 1.0)
            r   = int(45  + 155 * t)
            g   = int(120 + 48  * t)
            b   = int(20  + 35  * t)
            col = f"#{r:02x}{g:02x}{b:02x}"
            x0, y0 = self._world_to_canvas(*self._path_world[-2])
            x1, y1 = self._world_to_canvas(*self._path_world[-1])
            c.create_line(x0, y0, x1, y1,
                          fill=col, width=2, capstyle=tk.ROUND,
                          tags="path_line")
            if (n - 1) % 30 == 0:
                c.create_oval(x1-2, y1-2, x1+2, y1+2,
                              fill=col, outline="", tags="path_line")

        c.tag_raise("path_line")
        c.tag_raise("wall_seg")
        c.tag_raise("marker")

        if self._path_world:
            rx, ry = self._world_to_canvas(*self._path_world[-1])
            c.delete("robot_dot")
            c.create_oval(rx-4, ry-4, rx+4, ry+4,
                          fill=_GOLD, outline=_BG, width=2, tags="robot_dot")

    # ── LiDAR bar chart ───────────────────────────────────────────

    def _draw_lidar(self, lidar):
        """Cache latest lidar reading and render it."""
        self._last_lidar = lidar
        self._render_lidar()

    def _render_lidar(self):
        """Render LiDAR bars fully from the cached reading and current canvas size."""
        lidar = self._last_lidar
        if lidar is None:
            return

        c = self.lidar_canvas
        c.delete("all")

        W = c.winfo_width()
        H = c.winfo_height()
        if W < 2 or H < 2:
            return

        n = len(lidar)
        if n == 0:
            return

        # Padding: top reserved for labels, bottom margin, left/right edges
        PAD_L = 6
        PAD_R = 6
        PAD_T = 18   # label row height
        PAD_B = 4

        plot_w = W - PAD_L - PAD_R
        plot_h = H - PAD_T - PAD_B

        bw = plot_w / n   # bar width — scales with canvas, no fixed values

        for idx, val in enumerate(lidar):
            x0 = PAD_L + idx * bw
            x1 = PAD_L + (idx + 1) * bw - 0.5   # tiny gap between bars
            bar_h = max(int(val * plot_h), 1)
            y0 = PAD_T + plot_h - bar_h
            y1 = PAD_T + plot_h

            # Close (low val) → red/amber; far (high val) → forest green
            r   = int(160 * (1 - val))
            g   = int(100 + 68 * val)
            col = f"#{r:02x}{g:02x}14"
            c.create_rectangle(x0, y0, x1, y1, fill=col, outline="")

        # Centre divider at FRONT (index n//2)
        front_x = PAD_L + (n / 2) * bw
        c.create_line(front_x, PAD_T, front_x, H - PAD_B,
                      fill=_BORDER_DIM, dash=(3, 3))

        # Labels — anchored to plot edges, never outside canvas
        label_y = PAD_T // 2
        c.create_text(PAD_L,           label_y, text="LEFT",
                      fill=_TEXT_DIM, font=(_FU, 7), anchor="w")
        c.create_text(front_x,         label_y, text="FRONT",
                      fill=_TEXT_DIM, font=(_FU, 7))
        c.create_text(W - PAD_R,       label_y, text="RIGHT",
                      fill=_TEXT_DIM, font=(_FU, 7), anchor="e")

    # ── Helpers ───────────────────────────────────────────────────

    def _make_blank(self, w, h):
        arr = np.array([[[7, 24, 18]]], dtype=np.uint8)
        arr = np.broadcast_to(arr, (h, w, 3)).copy()
        return ImageTk.PhotoImage(Image.fromarray(arr, "RGB"))

    def _draw_atmosphere(self, w):
        """Render the landscape atmosphere strip — a darkened echo of the splash."""
        c = self._atm_canvas
        h = c.winfo_height()
        h = h if h > 1 else 150
        if w < 1:
            return
        c.delete("all")
        if self._atm_src:
            img = fit_image_cover(self._atm_src.convert("RGB"), w, h)
            # 58 % dark so the mystic forest is clearly visible as a background
            img = Image.blend(
                img.convert("RGBA"),
                Image.new("RGBA", img.size, (7, 24, 18, 255)),
                0.58,
            )
            # Overlay title text at 2.5× the homepage-banner proportion,
            # capped to the canvas bounds so it never overflows.
            if self._atm_txt_src:
                max_tw = min(int(w * 0.62 * 4.5), w)
                max_th = min(int(h * 0.58 * 4.5), h)
                timg   = fit_image_contain(self._atm_txt_src, max_tw, max_th)
                tw, th = timg.size
                tx     = (w - tw) // 2
                ty     = (h - th) // 2
                img.paste(timg, (tx, ty), timg)
            tk_img = ImageTk.PhotoImage(img.convert("RGB"))
            c.create_image(0, 0, anchor="nw", image=tk_img)
            c.atm_img = tk_img   # prevent GC


# ══════════════════════════════════════════════════════════════════════════════
# App entry point  (python project_gui.py)
# ══════════════════════════════════════════════════════════════════════════════

def launch_app():
    """Splash → animated shrink → HomePage."""
    _ensure_music_playing()   # start music before the first frame appears
    root = tk.Tk()
    root.withdraw()

    def show_homepage():
        # Tear down splash content and bindings
        for widget in root.winfo_children():
            widget.destroy()
        root.unbind("<Button-1>")
        root.unbind("<Key>")
        root.resizable(True, True)
        HomePage(root)

    root.deiconify()
    SplashScreen(root, on_done=show_homepage)
    root.mainloop()


if __name__ == "__main__":
    launch_app()
    