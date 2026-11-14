#!/usr/bin/env python3
"""GPU Burn GUI v1.0 for Linux.

A Tkinter front-end for the unchanged gpu_burn backend.  It exposes the
same options as the Windows GUI, streams gpu_burn stdout/stderr live, turns
CR progress redraws into a progress bar, and shows nvidia-smi telemetry while
a burn runs.  The backend is never modified by this GUI.

The script also works on Windows when a gpu_burn.exe is supplied for the
repository's smoke test (see GPU_BURN_GUI_BACKEND).

Example normal use on Linux:
    make
    python3 linux/GPU-Burn-GUI.py

Test-only environment variables:
    GPU_BURN_GUI_TEST=1        auto-refresh GPUs and auto-start a burn
    GPU_BURN_TEST_SECONDS=N    duration for that burn (default 2)
    GPU_BURN_GUI_TEST_KEEP=1   keep the window open after the burn
    GPU_BURN_GUI_LANG=zh-TW    force the UI language (en or zh-TW)
    GPU_BURN_GUI_BACKEND=...   backend path override
    GPU_BURN_GUI_COMPARE=...   compare.fatbin path override
"""

from __future__ import annotations

import locale
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from tkinter import font as tkfont
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk

APP_TITLE = "GPU Burn GUI v1.0"
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

C_BG = "#111318"
C_PANEL = "#17191f"
C_PANEL2 = "#20242b"
C_INPUT = "#1d2129"
C_BORDER = "#2c3038"
C_TEXT = "#e8edf4"
C_MUTED = "#9aa7b8"
C_ORANGE = "#ff8a00"
C_ORANGE2 = "#c66f00"
C_BLUE = "#2e6fba"
C_GREY = "#2b3038"
C_GREEN = "#65c17a"
C_RED = "#ff6b6b"

ZH = {
    "GPU Burn GUI": "GPU Burn 圖形介面",
    "Ready. Set options and click Start burn.": "就緒：設定選項後按下「開始燒機」。",
    "Burning GPU stress test...": "GPU 燒機測試中…",
    "Finished: GPU test OK.": "完成：GPU 測試結果良好。",
    "Finished: FAULTY GPU detected.": "完成：偵測到 GPU 錯誤。",
    "Finished with errors.": "結束，但有錯誤。",
    "Stopped by user.": "使用者已中止。",
    "Stop requested.": "已要求中止。",
    "Backend missing.": "找不到後端程式。",
    "Refreshing GPU list...": "正在更新 GPU 清單…",
    "GPU list refreshed.": "GPU 清單已更新。",
    "No GPU list output.": "GPU 清單沒有輸出。",
    "Burn options": "燒機選項",
    "Advanced": "進階選項",
    "GPU list": "GPU 清單",
    "Live monitor": "nvidia-smi 即時監控",
    "Memory to use": "記憶體用量",
    "Burn duration (seconds)": "燒機時間（秒）",
    "Use doubles (-d)": "使用雙精度（-d）",
    "Try Tensor cores (-tc)": "嘗試使用 Tensor Cores（-tc）",
    "GPU selection": "GPU 選擇",
    "Compare kernel (-c)": "比較核心（-c）",
    "SIGTERM timeout (s) (-stts)": "SIGTERM 逾時秒數（-stts）",
    "seconds": "秒",
    "Refresh GPUs (-l)": "重新整理 GPU（-l）",
    "Start burn": "開始燒機",
    "Stop burn": "中止燒機",
    "Open log folder": "開啟記錄資料夾",
    "Help (-h)": "顯示說明（-h）",
    "Browse...": "瀏覽…",
    "Backend default (90% available VRAM)": "後端預設（可用 VRAM 的 90%）",
    "Percent of available VRAM": "可用 VRAM 百分比",
    "Fixed amount (MB)": "固定容量（MB）",
    "All GPUs (backend default)": "所有 GPU（後端預設）",
    "Selected GPU": "選擇的 GPU",
    "Backend executable": "後端程式",
    "Compare kernel": "比較核心",
    "Last burn": "最近燒機",
    "GPUs found": "找到的 GPU",
    "Working directory": "工作目錄",
    "Memory mode": "記憶體模式",
    "Burn duration": "燒機時間",
    "Not run": "尚未執行",
    "No GPU selected.": "未選擇 GPU。",
    "No compare kernel selected.": "未選擇比較核心。",
    "Waiting for a burn.": "等待燒機。",
    "nvidia-smi live monitor": "nvidia-smi 即時監控",
    "The monitor refreshes while a burn runs; values come from nvidia-smi --query-gpu.":
        "燒機期間會持續更新；數值來自 nvidia-smi --query-gpu。",
    "nvidia-smi was not found on PATH.": "在 PATH 找不到 nvidia-smi。",
    "Could not read nvidia-smi output.": "無法讀取 nvidia-smi 輸出。",
    "GPU {id}: {name}": "GPU {id}: {name}",
    "Temp (C)": "溫度（C）",
    "Utilization (%)": "使用率（%）",
    "VRAM (used / total MB)": "VRAM（已用／總量 MB）",
    "Power (W) / limit": "功耗（W）／上限",
    "Fan (%)": "風扇（%）",
    "Choose compare kernel": "選擇比較核心檔案",
    "Confirm burn": "確認燒機",
    "Start burn now with these options?\n\n{cmd}": "要用這些選項開始燒機嗎？\n\n{cmd}",
    "GPU selection uses CUDA_VISIBLE_DEVICES to match the native backend option.":
        "GPU 選擇使用 CUDA_VISIBLE_DEVICES，對應原生後端的行為。",
    "The backend remains the original gpu_burn (Linux).": "後端仍是原版 gpu_burn（Linux）。",
    "The GUI only orchestrates options; gpu_burn itself is untouched.":
        "GUI 只負責選項與顯示；原始 gpu_burn 不變。",
    "Backend found.": "已找到後端。",
    "Backend missing: {path}": "找不到後端：{path}",
    "Build gpu_burn first.": "請先建置 gpu_burn。",
    "GPU list output written to the log.": "GPU 清單已寫入記錄。",
    "Help output written to the log.": "說明已寫入記錄。",
    "Failed to start backend: {error}": "無法啟動後端：{error}",
    "Could not open log folder: {error}": "無法開啟記錄資料夾：{error}",
    "Failed to stop backend cleanly: {error}": "無法乾淨地中止後端：{error}",
    "Error while reading GPU list: {error}": "讀取 GPU 清單時發生錯誤：{error}",
    "Backend is missing; set GPU_BURN_GUI_BACKEND or build gpu_burn.":
        "找不到後端；請設定 GPU_BURN_GUI_BACKEND 或先建置 gpu_burn。",
    "OK": "OK",
    "FAULTY": "FAULTY",
    "Stopped": "已停止",
    "Error": "錯誤",
    "Auto-test finished. Keeping the window open: {keep}":
        "自動測試完成。保留視窗：{keep}",
    "Burning GPU tests...": "GPU 燒機測試中…",
}

UI_LANG = "en"


def detect_language() -> str:
    want = os.environ.get("GPU_BURN_GUI_LANG", "").strip().lower()
    if want in ("zh-tw", "zh", "zh-hk", "zh-mo", "cht", "chinese"):
        return "zh-TW"
    if want in ("en", "en-us", "english"):
        return "en"
    try:
        raw = locale.getlocale()[0] or ""
        if raw.lower().startswith(("zh_tw", "zh-hk", "zh-mo", "chinese", "zh-tw")):
            return "zh-TW"
        if raw.lower().startswith("zh"):
            # Prefer Traditional Chinese for this GUI.
            return "zh-TW"
    except Exception:
        pass
    return "en"


UI_LANG = detect_language()


def tr(key: str) -> str:
    if UI_LANG == "zh-TW":
        return ZH.get(key, key)
    return key


def decode_bytes(data) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    if data is None:
        return ""
    return str(data)


def state_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "GPU-Burn-GUI"
    base = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")))
    return base / "gpu-burn-gui"


def find_backend() -> Path | None:
    env_path = os.environ.get("GPU_BURN_GUI_BACKEND")
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return path.resolve()

    candidates = []
    if os.name == "nt":
        candidates.extend([
            BASE_DIR / "win" / "build" / "Release" / "gpu_burn.exe",
            BASE_DIR / "win" / "package" / "gpu-burn-win-v0.3" / "gpu_burn.exe",
            APP_DIR / "gpu_burn.exe",
            BASE_DIR / "gpu_burn.exe",
        ])
    else:
        candidates.extend([
            BASE_DIR / "gpu_burn",
            APP_DIR / "gpu_burn",
            Path.cwd() / "gpu_burn",
            BASE_DIR / "build" / "gpu_burn",
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    path = shutil.which("gpu_burn.exe" if os.name == "nt" else "gpu_burn")
    if path:
        return Path(path).resolve()

    # Windows smoke-test fallback: the packaged build is outside the repo root.
    if os.name == "nt":
        fallback = BASE_DIR / "win" / "build" / "Release" / "gpu_burn.exe"
        if fallback.is_file():
            return fallback.resolve()
    return None


def find_compare(backend: Path | None) -> str:
    env_path = os.environ.get("GPU_BURN_GUI_COMPARE")
    if env_path and Path(env_path).expanduser().is_file():
        return str(Path(env_path).expanduser().resolve())

    candidates = [BASE_DIR / "compare.fatbin", Path.cwd() / "compare.fatbin"]
    if backend:
        candidates.insert(0, backend.parent / "compare.fatbin")
    if os.name == "nt":
        candidates.append(BASE_DIR / "win" / "build" / "Release" / "compare.fatbin")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return ""


def find_nvidia_smi() -> str:
    env_path = os.environ.get("GPU_BURN_GUI_NVIDIA_SMI")
    if env_path and Path(env_path).expanduser().is_file():
        return str(Path(env_path).expanduser().resolve())

    path = shutil.which("nvidia-smi")
    if path:
        return path

    candidates = []
    if os.name == "nt":
        candidates.extend([
            Path("C:\\Windows\\System32\\nvidia-smi.exe"),
            Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
            Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi",
            Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "NVIDIA Corporation" / "nvidia-smi.exe",
        ])
    else:
        candidates.extend([
            Path("/usr/bin/nvidia-smi"),
            Path("/usr/local/bin/nvidia-smi"),
            Path("/usr/local/cuda/bin/nvidia-smi"),
            Path("/opt/nvidia/bin/nvidia-smi"),
        ])
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def run_capture(args, timeout=120, env=None) -> dict:
    """Run a helper command and return decoded stdout/stderr."""
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            env=env or os.environ.copy(),
        )
        return {
            "returncode": proc.returncode,
            "stdout": decode_bytes(proc.stdout),
            "stderr": decode_bytes(proc.stderr),
        }
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": f"{exc}"}


def reader_thread(stream, tag: str, output_queue: queue.Queue) -> None:
    """Read a binary pipe in the background and queue chunks for the UI thread."""
    try:
        while True:
            try:
                chunk = stream.read(4096)
            except Exception as exc:
                output_queue.put((tag, f"[reader] {exc}".encode("utf-8", errors="replace")))
                break
            if not chunk:
                break
            output_queue.put((tag, chunk))
    finally:
        try:
            stream.close()
        except Exception:
            pass


def shorten_text(value: str, length: int = 76) -> str:
    if len(value) <= length:
        return value
    tail = value[-max(12, length - 12):]
    return "…" + tail


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.backend_path = find_backend()
        self.compare_file = find_compare(self.backend_path)
        self.smi_path = find_nvidia_smi()
        self.state_dir = state_dir()
        self.log_dir = self.state_dir
        self.session_log = None
        self.log_file = None

        self.gpus = []
        self.proc = None
        self.read_threads = []
        self.output_queue = queue.Queue()
        self.stdout_text = ""
        self.stderr_text = ""
        self.stop_requested = False
        self.status_key = "Ready. Set options and click Start burn."
        self.last_result_key = "Not run"
        self.progress_value = 0.0

        self.testing = os.environ.get("GPU_BURN_GUI_TEST") == "1"
        self.test_keep = os.environ.get("GPU_BURN_GUI_TEST_KEEP") == "1"

        self.label_widgets = []
        self.button_widgets = []
        self.check_widgets = []
        self.tab_pages = []
        self.tab_keys = []

        self._poll_after = None
        self._process_after = None
        self._monitor_after = None

        self._open_log()

        self._setup_ui()
        self._update_language()
        self.update_summary()

        # Startup timing: let the window paint before the GPU list runs.
        self.root.after(180, lambda: self.refresh_gpu_list())
        self.root.after(900, self.auto_start_if_enabled)

    # ------------------------------------------------------------------ setup

    def _open_log(self) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.log_dir = Path.cwd()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = self.log_dir / f"GPU-Burn-GUI_{stamp}.log"
        try:
            self.session_log.write_text("", encoding="utf-8")
        except OSError:
            self.session_log = None

    def _choose_ui_font(self) -> str:
        families = set(tkfont.families(self.root))
        if os.name == "nt":
            candidates = ["Segoe UI", "Microsoft YaHei UI", "Arial"]
        else:
            candidates = [
                "Noto Sans CJK TC", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
                "AR PL UMing CN", "DejaVu Sans", "Liberation Sans", "Arial",
            ]
        for family in candidates:
            if family in families:
                return family
        return "TkDefaultFont"

    def _choose_mono_font(self) -> str:
        families = set(tkfont.families(self.root))
        candidates = ["Cascadia Mono", "Consolas", "DejaVu Sans Mono",
                      "Liberation Mono", "Noto Sans Mono", "Courier New", "Courier"]
        for family in candidates:
            if family in families:
                return family
        return "TkFixedFont"

    def _setup_ui(self) -> None:
        self.root.title(APP_TITLE)
        self.root.geometry("1160x920")
        self.root.minsize(1000, 760)
        self.root.configure(bg=C_BG)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(3, weight=1)  # tabs
        self.root.grid_rowconfigure(6, weight=1)  # log

        self.ui_family = self._choose_ui_font()
        self.mono_family = self._choose_mono_font()
        self.font_title = tkfont.Font(family=self.ui_family, size=16, weight="bold")
        self.font_subtitle = tkfont.Font(family=self.ui_family, size=11, weight="bold")
        self.font_ui = tkfont.Font(family=self.ui_family, size=10)
        self.font_small = tkfont.Font(family=self.ui_family, size=9)
        self.font_mono = tkfont.Font(family=self.mono_family, size=9)
        self.font_big = tkfont.Font(family=self.ui_family, size=12, weight="bold")
        self.font_sec = tkfont.Font(family=self.ui_family, size=10, weight="bold")

        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure("Gpu.TNotebook", background=C_BG, borderwidth=0, tabmargin=0)
        self.style.configure("Gpu.TNotebook.Tab", background=C_PANEL, foreground=C_MUTED,
                             padding=[15, 7], font=self.font_ui.name)
        self.style.map("Gpu.TNotebook.Tab",
                       background=[("selected", C_PANEL2)],
                       foreground=[("selected", C_TEXT), ("active", C_ORANGE)])
        self.style.configure("Gpu.Horizontal.TProgressbar",
                             troughcolor=C_PANEL, background=C_ORANGE, thickness=12,
                             darkcolor=C_ORANGE, lightcolor=C_ORANGE)
        self.style.configure("Gpu.TCombobox",
                             fieldbackground=C_INPUT, background=C_INPUT,
                             foreground=C_TEXT, selectbackground=C_PANEL2,
                             selectforeground=C_TEXT, bordercolor=C_BORDER,
                             arrowcolor=C_MUTED)
        self.style.map("Gpu.TCombobox",
                       fieldbackground=[("readonly", C_INPUT)],
                       foreground=[("readonly", C_TEXT)],
                       background=[("readonly", C_INPUT)])
        self.root.option_add("*TCombobox*Listbox.background", C_PANEL2)
        self.root.option_add("*TCombobox*Listbox.foreground", C_TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", C_ORANGE)
        self.root.option_add("*TCombobox*Listbox.selectForeground", C_BG)

        header = tk.Frame(self.root, bg=C_BG)
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 2))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(header, text=APP_TITLE, bg=C_BG, fg=C_TEXT, font=self.font_title,
                 anchor="w").grid(row=0, column=0, sticky="w")
        self.subtitle_label = self._label(header, "GPU Burn GUI", font=self.font_subtitle,
                                          bg=C_BG, fg=C_MUTED)
        self.subtitle_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.lang_button = tk.Button(
            header, text="English" if UI_LANG == "zh-TW" else "中文",
            bg=C_GREY, fg=C_TEXT, activebackground=C_PANEL2, activeforeground=C_ORANGE,
            font=self.font_ui, bd=0, relief="flat", padx=10, pady=4, cursor="hand2",
            highlightthickness=0, command=self.toggle_language,
        )
        self.lang_button.grid(row=0, column=1, rowspan=2, sticky="ne", padx=(16, 0))

        self.status_label = tk.Label(
            self.root, text=tr(self.status_key), bg=C_BG, fg=C_ORANGE,
            font=self.font_subtitle, anchor="w",
        )
        self.status_label.grid(row=1, column=0, sticky="ew", padx=18, pady=(4, 4))

        summary = tk.Frame(self.root, bg=C_BG)
        summary.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 2))

        left = tk.Frame(summary, bg=C_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 52))
        left.columnconfigure(1, weight=1)
        right = tk.Frame(summary, bg=C_BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(1, weight=1)

        self.sel_value = self._summary_field(left, 0, "Selected GPU")
        self.backend_value = self._summary_field(left, 1, "Backend executable")
        self.compare_value = self._summary_field(left, 2, "Compare kernel")
        self.last_value = self._summary_field(left, 3, "Last burn")
        self.gpus_value = self._summary_field(right, 0, "GPUs found")
        self.workdir_value = self._summary_field(right, 1, "Working directory")
        self.memmode_value = self._summary_field(right, 2, "Memory mode")
        self.duration_value = self._summary_field(right, 3, "Burn duration")

        self.tabs = ttk.Notebook(self.root, style="Gpu.TNotebook")
        self.tabs.grid(row=3, column=0, sticky="nsew", padx=18, pady=(6, 0))

        burn_page = self._new_tab("Burn options")
        self._build_burn_page(burn_page)
        adv_page = self._new_tab("Advanced")
        self._build_advanced_page(adv_page)
        gpu_page = self._new_tab("GPU list")
        self._build_gpu_page(gpu_page)
        monitor_page = self._new_tab("Live monitor")
        self._build_monitor_page(monitor_page)
        self.monitor_page = monitor_page

        actions = tk.Frame(self.root, bg=C_BG)
        actions.grid(row=4, column=0, sticky="ew", padx=18, pady=(8, 2))
        actions.grid_columnconfigure(0, weight=0)
        self.btn_refresh = self._button(actions, "Refresh GPUs (-l)", self.refresh_gpu_list,
                                        bg=C_GREY)
        self.btn_refresh.grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.btn_start = self._button(actions, "Start burn",
                                      lambda: self.start_burn(skip_confirm=False),
                                      bg=C_ORANGE, fg=C_TEXT, active_bg=C_ORANGE2,
                                      width=13)
        self.btn_start.grid(row=0, column=1, padx=(0, 8), sticky="w")
        self.btn_stop = self._button(actions, "Stop burn", self.stop_burn,
                                     bg=C_BLUE, fg=C_TEXT, active_bg="#24568d",
                                     width=12)
        self.btn_stop.grid(row=0, column=2, padx=(0, 8), sticky="w")
        self.btn_stop.configure(state="disabled", disabledforeground=C_MUTED)
        self.btn_logs = self._button(actions, "Open log folder", self.open_log_folder,
                                     bg=C_GREY, width=16)
        self.btn_logs.grid(row=0, column=3, padx=(0, 8), sticky="w")
        self.btn_help = self._button(actions, "Help (-h)", self.show_help,
                                     bg=C_GREY, width=10)
        self.btn_help.grid(row=0, column=4, padx=(0, 8), sticky="w")

        progress_frame = tk.Frame(self.root, bg=C_BG)
        progress_frame.grid(row=5, column=0, sticky="ew", padx=18, pady=(2, 4))
        progress_frame.columnconfigure(1, weight=1)
        self.progress_label = tk.Label(progress_frame, text="0%", bg=C_BG, fg=C_ORANGE,
                                       font=self.font_ui, anchor="w", width=7)
        self.progress_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.progress_bar = ttk.Progressbar(
            progress_frame, orient="horizontal", length=900,
            mode="determinate", maximum=100.0, value=0.0,
            style="Gpu.Horizontal.TProgressbar",
        )
        self.progress_bar.grid(row=0, column=1, sticky="ew", pady=1)

        log_frame = tk.Frame(self.root, bg=C_BG)
        log_frame.grid(row=6, column=0, sticky="nsew", padx=18, pady=(2, 2))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        self.log_text = self._make_text(log_frame, self.font_mono, height=10)
        self.log_text.configure(state="normal")

        footer = self._label(self.root, "The GUI only orchestrates options; gpu_burn itself is untouched.",
                             bg=C_BG, fg=C_MUTED, font=self.font_small)
        footer.grid(row=7, column=0, sticky="ew", padx=18, pady=(2, 10))

        self.append_log(f"{APP_TITLE}")
        self.append_log("=== GPU Burn GUI for Linux ===")
        self.append_log(f"Backend: {self.backend_path if self.backend_path else 'not found'}")
        self.append_log(f"Compare: {self.compare_file if self.compare_file else 'not found'}")
        self.append_log(f"Log file: {self.session_log}")
        self.append_log(f"Python: {sys.version.splitlines()[0]}")
        self.append_log(f"nvidia-smi: {self.smi_path if self.smi_path else 'not found'}")
        self.append_log("-----------------------------------")
        self.append_log("Backend found." if self.backend_path else "Backend missing.")
        self.append_log(str(self.backend_path) if self.backend_path else "Build gpu_burn first.")

    def _summary_field(self, parent, row, key):
        label = self._label(parent, key, font=self.font_small, fg=C_MUTED, bg=C_BG)
        label.grid(row=row, column=0, sticky="w", pady=1)
        value = tk.Label(parent, text="", bg=C_BG, fg=C_TEXT, font=self.font_small,
                         anchor="w", justify="left")
        value.grid(row=row, column=1, sticky="w", padx=(14, 0), pady=1)
        return value

    def _label(self, parent, key, **kwargs):
        defaults = dict(bg=parent.cget("bg") if "bg" not in kwargs else kwargs.get("bg"),
                        fg=C_TEXT, font=self.font_ui, anchor="w", justify="left")
        defaults.update(kwargs)
        label = tk.Label(parent, text=tr(key), **defaults)
        self.label_widgets.append((label, key))
        return label

    def _button(self, parent, key, command, bg=C_GREY, fg=C_TEXT,
                active_bg=C_PANEL2, active_fg=C_ORANGE, width=16, font=None):
        button = tk.Button(
            parent, text=tr(key), command=command, bg=bg, fg=fg,
            activebackground=active_bg, activeforeground=active_fg,
            font=font or self.font_ui, bd=0, relief="flat", padx=10, pady=5,
            cursor="hand2", highlightthickness=0, width=width,
        )
        self.button_widgets.append((button, key))
        return button

    def _check(self, parent, key, variable, font=None):
        checkbox = tk.Checkbutton(
            parent, text=tr(key), variable=variable, bg=parent.cget("bg"),
            fg=C_TEXT, activebackground=C_PANEL, activeforeground=C_TEXT,
            selectcolor=C_PANEL2, font=font or self.font_ui, bd=0, relief="flat",
            highlightthickness=0, anchor="w", padx=0, pady=2,
        )
        self.check_widgets.append((checkbox, key))
        return checkbox

    def _new_tab(self, key: str) -> tk.Frame:
        page = tk.Frame(self.tabs, bg=C_PANEL)
        self.tabs.add(page, text=tr(key))
        self.tab_pages.append(page)
        self.tab_keys.append(key)
        return page

    def _make_text(self, parent, font, height=8, width=80, row=0) -> tk.Text:
        frame = tk.Frame(parent, bg=C_PANEL)
        frame.grid(row=row, column=0, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        text = tk.Text(
            frame, bg=C_INPUT, fg=C_TEXT, font=font, height=height, width=width,
            insertbackground=C_ORANGE, selectbackground=C_ORANGE,
            selectforeground=C_BG, wrap="none", bd=0, relief="flat",
            padx=6, pady=4, highlightthickness=1, highlightbackground=C_BORDER,
            state="normal",
        )
        vsb = tk.Scrollbar(frame, orient="vertical", command=text.yview)
        hsb = tk.Scrollbar(frame, orient="horizontal", command=text.xview)
        vsb.configure(troughcolor=C_PANEL2, bg=C_PANEL, activebackground=C_MUTED,
                      highlightthickness=0, bd=0)
        hsb.configure(troughcolor=C_PANEL2, bg=C_PANEL, activebackground=C_MUTED,
                      highlightthickness=0, bd=0)
        text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        text.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        return text

    def _build_burn_page(self, page: tk.Frame) -> None:
        body = tk.Frame(page, bg=C_PANEL)
        body.pack(fill="both", expand=True, padx=16, pady=12)
        body.columnconfigure(1, weight=1)

        self._label(body, "Memory to use", font=self.font_subtitle).grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        mem_frame = tk.Frame(body, bg=C_PANEL)
        mem_frame.grid(row=1, column=0, columnspan=2, sticky="w")
        self.mem_mode = ttk.Combobox(
            mem_frame, style="Gpu.TCombobox", state="readonly", width=50,
            values=[
                tr("Backend default (90% available VRAM)"),
                tr("Percent of available VRAM"),
                tr("Fixed amount (MB)"),
            ],
        )
        self.mem_mode.grid(row=0, column=0, sticky="w", padx=(0, 12), pady=2)
        self.mem_value = tk.StringVar(value="90")
        self.spin_mem = tk.Spinbox(
            mem_frame, from_=1, to=99999, increment=1, width=10,
            textvariable=self.mem_value, bg=C_INPUT, fg=C_TEXT,
            insertbackground=C_ORANGE, buttonbackground=C_GREY,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=C_BORDER,
            font=self.font_ui,
        )
        self.spin_mem.grid(row=0, column=1, padx=(0, 8), pady=2)
        self.mem_unit_label = tk.Label(mem_frame, text="", bg=C_PANEL, fg=C_MUTED,
                                       font=self.font_ui)
        self.mem_unit_label.grid(row=0, column=2, sticky="w", padx=(2, 0))
        self.mem_mode.bind("<<ComboboxSelected>>", self.on_mem_mode_changed)
        self.mem_mode.current(0)
        self.mem_mode.set(tr("Backend default (90% available VRAM)"))
        self.spin_mem.configure(state="disabled")

        self._label(body, "Burn duration (seconds)", font=self.font_subtitle).grid(
            row=2, column=0, sticky="w", pady=(14, 4))
        dur_frame = tk.Frame(body, bg=C_PANEL)
        dur_frame.grid(row=3, column=0, columnspan=2, sticky="w")
        self.duration_value_var = tk.StringVar(value="3600")
        self.spin_duration = tk.Spinbox(
            dur_frame, from_=1, to=864000, increment=60, width=12,
            textvariable=self.duration_value_var, bg=C_INPUT, fg=C_TEXT,
            insertbackground=C_ORANGE, buttonbackground=C_GREY,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=C_BORDER,
            font=self.font_ui,
        )
        self.spin_duration.grid(row=0, column=0, padx=(0, 8), pady=2)
        tk.Label(dur_frame, text=tr("seconds"), bg=C_PANEL, fg=C_MUTED,
                 font=self.font_ui).grid(row=0, column=1, sticky="w")

        opts_frame = tk.Frame(body, bg=C_PANEL)
        opts_frame.grid(row=4, column=0, columnspan=2, sticky="w", pady=(16, 4))
        self.var_doubles = tk.BooleanVar(value=False)
        self.var_tensors = tk.BooleanVar(value=True)
        self._check(opts_frame, "Use doubles (-d)", self.var_doubles).grid(
            row=0, column=0, sticky="w")
        self._check(opts_frame, "Try Tensor cores (-tc)", self.var_tensors).grid(
            row=0, column=1, sticky="w", padx=(18, 0))

        self._label(body, "GPU selection", font=self.font_subtitle).grid(
            row=5, column=0, sticky="w", pady=(14, 4))
        gpu_frame = tk.Frame(body, bg=C_PANEL)
        gpu_frame.grid(row=6, column=0, columnspan=2, sticky="w")
        self.gpu_combo = ttk.Combobox(
            gpu_frame, style="Gpu.TCombobox", state="normal",
            values=[tr("All GPUs (backend default)")], width=60,
        )
        self.gpu_combo.grid(row=0, column=0, sticky="w", padx=(0, 12), pady=2)
        self.gpu_combo.current(0)
        self.gpu_combo.set(tr("All GPUs (backend default)"))
        self.gpu_combo.bind("<<ComboboxSelected>>", self.on_gpu_selection_changed)

        info = self._label(
            body, "GPU selection uses CUDA_VISIBLE_DEVICES to match the native backend option.",
            font=self.font_small, fg=C_MUTED,
        )
        info.grid(row=7, column=0, columnspan=2, sticky="w", pady=(16, 2))

    def _build_advanced_page(self, page: tk.Frame) -> None:
        body = tk.Frame(page, bg=C_PANEL)
        body.pack(fill="both", expand=True, padx=16, pady=12)
        body.columnconfigure(1, weight=1)

        self._label(body, "Compare kernel (-c)", font=self.font_subtitle).grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        compare_frame = tk.Frame(body, bg=C_PANEL)
        compare_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        compare_frame.columnconfigure(0, weight=1)
        self.compare_var = tk.StringVar(value=self.compare_file)
        compare_entry = tk.Entry(
            compare_frame, textvariable=self.compare_var, bg=C_INPUT, fg=C_TEXT,
            insertbackground=C_ORANGE, relief="flat", bd=0, highlightthickness=1,
            highlightbackground=C_BORDER, font=self.font_ui, width=60,
        )
        compare_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=2)
        self.browse_button = tk.Button(
            compare_frame, text=tr("Browse..."), command=self.browse_compare,
            bg=C_GREY, fg=C_TEXT, activebackground=C_PANEL2, activeforeground=C_ORANGE,
            font=self.font_ui, bd=0, relief="flat", padx=10, pady=2,
            cursor="hand2", highlightthickness=0, width=10,
        )
        self.browse_button.grid(row=0, column=1, sticky="w", pady=2)

        self._label(body, "SIGTERM timeout (s) (-stts)", font=self.font_subtitle).grid(
            row=2, column=0, sticky="w", pady=(14, 4))
        stts_frame = tk.Frame(body, bg=C_PANEL)
        stts_frame.grid(row=3, column=0, columnspan=2, sticky="w")
        self.stts_var = tk.StringVar(value="30")
        self.spin_stts = tk.Spinbox(
            stts_frame, from_=0, to=3600, increment=5, width=10,
            textvariable=self.stts_var, bg=C_INPUT, fg=C_TEXT,
            insertbackground=C_ORANGE, buttonbackground=C_GREY,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=C_BORDER,
            font=self.font_ui,
        )
        self.spin_stts.grid(row=0, column=0, padx=(0, 8), pady=2)
        tk.Label(stts_frame, text=tr("seconds"), bg=C_PANEL, fg=C_MUTED,
                 font=self.font_ui).grid(row=0, column=1, sticky="w")

        self._label(body, "The backend remains the original gpu_burn (Linux).",
                    font=self.font_small, fg=C_MUTED).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(18, 2))
        self._label(body, "Help output written to the log.", fg=C_MUTED,
                    font=self.font_small).grid(row=5, column=0, columnspan=2,
                                                sticky="w", pady=(4, 2))

    def _build_gpu_page(self, page: tk.Frame) -> None:
        body = tk.Frame(page, bg=C_PANEL)
        body.pack(fill="both", expand=True, padx=16, pady=12)
        body.grid_rowconfigure(1, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self._label(body, "GPU list", font=self.font_subtitle).grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        self.gpu_text = self._make_text(body, self.font_mono, height=8, width=94, row=1)
        self.gpu_text.configure(state="disabled")

    def _build_monitor_page(self, page: tk.Frame) -> None:
        body = tk.Frame(page, bg=C_PANEL)
        body.pack(fill="both", expand=True, padx=16, pady=12)
        body.grid_rowconfigure(2, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self.monitor_title_label = self._label(
            body, "nvidia-smi live monitor", font=self.font_subtitle)
        self.monitor_title_label.grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.monitor_note_label = self._label(
            body, "The monitor refreshes while a burn runs; values come from nvidia-smi --query-gpu.",
            font=self.font_small, fg=C_MUTED,
        )
        self.monitor_note_label.grid(row=1, column=0, sticky="w", pady=(0, 4))
        self.monitor_text = self._make_text(body, self.font_mono, height=7, width=94, row=2)
        self.monitor_text.configure(state="disabled")

    # ------------------------------------------------------------------ logs

    def append_log(self, message: str) -> None:
        if self.log_text is None:
            return
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        if self.session_log:
            try:
                with open(self.session_log, "a", encoding="utf-8", errors="replace") as handle:
                    handle.write(message + "\n")
            except Exception as exc:
                self.log_text.insert("end", f"[log write error] {exc}\n")

    def _log_text(self, ok: bool, line: str, extra: str = "") -> None:
        self.append_log(line)
        if extra:
            self.append_log(extra)

    def _set_status(self, key: str, color=C_ORANGE) -> None:
        self.status_key = key
        self.status_label.config(text=tr(key), fg=color)
        self.append_log(tr(key))

    def _set_progress(self, value: float) -> None:
        value = max(0.0, min(100.0, value))
        self.progress_value = value
        self.progress_bar.configure(value=value)
        self.progress_label.configure(text=f"{value:.0f}%")

    def update_summary(self) -> None:
        selected = self._selected_gpu()
        if selected:
            selected_name = selected["name"]
            selected_text = tr("GPU {id}: {name}").format(id=selected["id"], name=selected_name)
        else:
            selected_text = tr("All GPUs (backend default)")

        backend_text = str(self.backend_path) if self.backend_path else tr("Backend missing.")
        compare_text = self.compare_var.get().strip() if hasattr(self, "compare_var") else self.compare_file
        if not compare_text:
            compare_text = tr("No compare kernel selected.")
        workdir = str(self.backend_path.parent) if self.backend_path else str(Path.cwd())

        duration = "0"
        try:
            duration = str(int(float(self.duration_value_var.get())))
        except Exception:
            pass
        mode = self.mem_mode.current()
        mem_key = [
            "Backend default (90% available VRAM)",
            "Percent of available VRAM",
            "Fixed amount (MB)",
        ][mode]
        mem_text = tr(mem_key)

        self.sel_value.config(text=selected_text)
        self.backend_value.config(text=shorten_text(backend_text))
        self.compare_value.config(text=shorten_text(compare_text))
        self.last_value.config(text=tr(self.last_result_key))
        self.gpus_value.config(text=str(len(self.gpus)))
        self.workdir_value.config(text=shorten_text(workdir))
        self.memmode_value.config(text=mem_text)
        self.duration_value.config(text=f"{duration} " + tr("seconds"))

    # ------------------------------------------------------------------ GPU list

    def refresh_gpu_list(self, select_first: bool = False) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self._set_status("Refreshing GPU list...", C_MUTED)
        if not self.backend_path:
            self._set_status("Backend missing.", C_RED)
            self.append_log("Backend missing: " + os.environ.get("GPU_BURN_GUI_BACKEND", "not set"))
            self.update_summary()
            return
        cwd = str(self.backend_path.parent)
        capture_env = os.environ.copy()
        capture_env.pop("CUDA_VISIBLE_DEVICES", None)
        result = run_capture([str(self.backend_path), "-l"], timeout=90, env=capture_env)
        output = result["stdout"] + result["stderr"]
        self.append_log("=== GPU list ===")
        if result["stdout"]:
            for line in result["stdout"].splitlines():
                if line.strip():
                    self.append_log(line)
        if result["stderr"]:
            for line in result["stderr"].splitlines():
                if line.strip():
                    self.append_log("[stderr] " + line)
        if not result["stdout"].strip() and not result["stderr"].strip():
            self.append_log("No GPU list output.")

        old_selected = self._selected_gpu_id()
        self.gpus = []
        pattern = re.compile(r"^ID\s+(\d+):\s+(.+?),\s*(\d+)\s*MB\s*$", re.IGNORECASE)
        for line in result["stdout"].splitlines():
            match = pattern.match(line.strip())
            if match:
                self.gpus.append({
                    "id": int(match.group(1)),
                    "name": match.group(2).strip(),
                    "memory": int(match.group(3)),
                })

        values = [tr("All GPUs (backend default)")]
        for gpu in self.gpus:
            values.append(tr("GPU {id}: {name}").format(id=gpu["id"], name=gpu["name"]))
        self.gpu_combo.configure(values=values)

        index = 0
        if select_first and self.gpus:
            index = 1
        elif old_selected is not None:
            for i, gpu in enumerate(self.gpus):
                if gpu["id"] == old_selected:
                    index = i + 1
                    break
        self.gpu_combo.current(index)
        if 0 <= index < len(values):
            self.gpu_combo.set(values[index])

        self.gpu_text.configure(state="normal")
        self.gpu_text.delete("1.0", "end")
        self.gpu_text.insert("end", output if output.strip() else tr("No GPU list output."))
        self.gpu_text.configure(state="disabled")
        self.gpus_value.config(text=str(len(self.gpus)))
        self._set_status("GPU list refreshed.", C_GREEN)
        self.append_log(f"GPUs found: {len(self.gpus)}")
        self.update_summary()
        self._update_idle_monitor()

    def _selected_gpu(self) -> dict | None:
        index = self.gpu_combo.current() if hasattr(self, "gpu_combo") else 0
        if index <= 0:
            return None
        idx = index - 1
        if 0 <= idx < len(self.gpus):
            return self.gpus[idx]
        return None

    def _selected_gpu_id(self) -> int | None:
        gpu = self._selected_gpu()
        return gpu["id"] if gpu else None

    def on_gpu_selection_changed(self, _event=None) -> None:
        self.update_summary()
        self._update_idle_monitor()

    def on_mem_mode_changed(self, _event=None) -> None:
        mode = self.mem_mode.current()
        if mode == 2:
            self.spin_mem.configure(state="normal")
            self.mem_value.set("7700")
            self.mem_unit_label.configure(text="MB")
        elif mode == 1:
            self.spin_mem.configure(state="normal")
            self.mem_value.set("90")
            self.mem_unit_label.configure(text="%")
        else:
            self.spin_mem.configure(state="disabled")
            self.mem_unit_label.configure(text="")
        self.update_summary()

    def browse_compare(self) -> None:
        current = self.compare_var.get().strip()
        initialdir = str(Path(current).parent) if current else str(BASE_DIR)
        chosen = filedialog.askopenfilename(
            parent=self.root, title=tr("Choose compare kernel"),
            initialdir=initialdir,
            filetypes=[("Compare kernel", "*.fatbin *.btcubin *.bin"), ("All files", "*")],
        )
        if chosen:
            self.compare_var.set(chosen)
            self.update_summary()
        self._update_idle_monitor()

    # ------------------------------------------------------------------ burn

    def _build_args(self) -> list:
        args = []
        mode = self.mem_mode.current()
        if mode == 1:
            raw = self.mem_value.get().strip()
            args.extend(["-m", f"{int(float(raw))}%"])
        elif mode == 2:
            raw = self.mem_value.get().strip()
            args.extend(["-m", str(int(float(raw)))])

        if self.var_doubles.get():
            args.append("-d")
        if self.var_tensors.get():
            args.append("-tc")

        compare = self.compare_var.get().strip() if hasattr(self, "compare_var") else ""
        if compare:
            args.extend(["-c", compare])

        stts = self._safe_int(self.stts_var.get(), 0)
        if stts > 0:
            args.extend(["-stts", str(stts)])

        duration = self._safe_int(self.duration_value_var.get(), 0)
        if duration <= 0:
            raise ValueError("Duration must be greater than 0.")
        args.append(str(duration))
        return args

    @staticmethod
    def _safe_int(value: str, default: int) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    def _command_display(self, args: list) -> str:
        gpu_id = self._selected_gpu_id()
        backend = str(self.backend_path) if self.backend_path else "gpu_burn"
        command = f"{backend} " + " ".join(args)
        if gpu_id is not None:
            return f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
        return f"CUDA_VISIBLE_DEVICES=<unset> {command}"

    def start_burn(self, skip_confirm: bool = False) -> None:
        if self.proc and self.proc.poll() is None:
            self._set_status("Burning GPU stress test...", C_ORANGE)
            return

        if not self.backend_path:
            self._set_status("Backend missing.", C_RED)
            if not self.testing:
                messagebox.showerror(tr("Backend missing."), tr("Backend is missing; set GPU_BURN_GUI_BACKEND or build gpu_burn."), parent=self.root)
            return

        try:
            args = self._build_args()
        except Exception as exc:
            self.append_log(f"[error] {exc}")
            messagebox.showerror(tr("Finished with errors."), str(exc), parent=self.root)
            return

        command_display = self._command_display(args)
        if not skip_confirm:
            prompt = tr("Start burn now with these options?\n\n{cmd}").format(cmd=command_display)
            if not messagebox.askyesno(tr("Confirm burn"), prompt, parent=self.root):
                return

        env = os.environ.copy()
        gpu_id = self._selected_gpu_id()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)

        kwargs = dict(
            cwd=str(self.backend_path.parent),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["preexec_fn"] = os.setsid

        self.append_log(f"Command: {command_display}")
        self._set_status("Burning GPU stress test...", C_ORANGE)
        self.stop_requested = False
        self.stdout_text = ""
        self.stderr_text = ""
        self.read_threads = []
        self.output_queue = queue.Queue()
        self.progress_value = 0.0
        self._set_progress(0)

        try:
            self.proc = subprocess.Popen(
                [str(self.backend_path)] + args,
                **kwargs,
            )
        except Exception as exc:
            self.proc = None
            self.append_log(f"[error] {exc}")
            self._set_status("Finished with errors.", C_RED)
            if not self.testing:
                messagebox.showerror(tr("Finished with errors."), tr("Failed to start backend: {error}").format(error=str(exc)), parent=self.root)
            return

        self.read_threads = [
            threading.Thread(target=reader_thread, args=(self.proc.stdout, "out", self.output_queue), daemon=True),
            threading.Thread(target=reader_thread, args=(self.proc.stderr, "err", self.output_queue), daemon=True),
        ]
        for thread in self.read_threads:
            thread.start()

        self._cancel_timers()
        self._poll_after = self.root.after(100, self._poll_output)
        self._process_after = self.root.after(250, self._poll_process)
        self._monitor_after = self.root.after(1000, self.update_monitor)

        self.tabs.select(self.monitor_page)
        self.btn_start.configure(state="disabled")
        self.btn_refresh.configure(state="disabled")
        self.btn_stop.configure(state="normal", disabledforeground=C_MUTED)

    def auto_start_if_enabled(self) -> None:
        if not self.testing:
            return
        try:
            seconds = int(os.environ.get("GPU_BURN_TEST_SECONDS", "2"))
        except ValueError:
            seconds = 2
        self.duration_value_var.set(str(seconds))
        self.var_doubles.set(False)
        self.var_tensors.set(True)
        if self.gpus:
            values = [tr("All GPUs (backend default)")]
            for gpu in self.gpus:
                values.append(tr("GPU {id}: {name}").format(id=gpu["id"], name=gpu["name"]))
            self.gpu_combo.configure(values=values)
            self.gpu_combo.current(1)
            self.gpu_combo.set(values[1])
        self.update_summary()
        self.append_log("[auto-test] Starting burn automatically.")
        self.start_burn(skip_confirm=True)

    def _poll_output(self) -> None:
        if self.proc is None:
            return
        self._drain_output()
        self._poll_after = self.root.after(100, self._poll_output)

    def _drain_output(self) -> None:
        while True:
            try:
                tag, raw = self.output_queue.get_nowait()
            except queue.Empty:
                break
            text = decode_bytes(raw)
            if not text:
                continue
            if tag == "out":
                self.stdout_text += text
            else:
                self.stderr_text += text

            normalized = text.replace("\r\n", "\n").replace("\r", "\n")
            for line in normalized.splitlines():
                if not line.strip():
                    continue
                if tag != "out":
                    self.append_log("[stderr] " + line)
                else:
                    self.append_log(line)
                    percent_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", line)
                    if percent_matches:
                        try:
                            self._set_progress(float(percent_matches[-1]))
                        except ValueError:
                            pass

    def _poll_process(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            self._finish_burn()
            return
        self._process_after = self.root.after(250, self._poll_process)

    def _cancel_timers(self) -> None:
        for attribute in ("_poll_after", "_process_after", "_monitor_after"):
            after_id = getattr(self, attribute)
            if after_id:
                try:
                    self.root.after_cancel(after_id)
                except Exception:
                    pass
                setattr(self, attribute, None)

    def _finish_burn(self) -> None:
        if self.proc is None:
            return
        returncode = self.proc.poll()
        for thread in self.read_threads:
            thread.join(2.0)
        self._drain_output()
        self._set_progress(100)
        self._cancel_timers()
        self._monitor_after = None
        self.btn_start.configure(state="normal")
        self.btn_refresh.configure(state="normal")
        self.btn_stop.configure(state="disabled")

        result_key, status_key = self._classify_result(returncode, self.stop_requested)
        self.last_result_key = result_key
        self._set_status(status_key, C_GREEN if result_key == "OK" else C_RED if result_key in ("FAULTY", "Error") else C_ORANGE)
        self.append_log(f"=== Exit code: {returncode} ===")
        self.append_log(f"Last burn: {tr(result_key)}")
        self.append_log("=== GPU Burn GUI stopped ===")
        self.update_summary()
        self._update_idle_monitor()
        self.proc = None
        self._poll_after = None
        self._process_after = None

        if self.testing:
            self.append_log(tr("Auto-test finished. Keeping the window open: {keep}").format(keep=str(self.test_keep)))
            if not self.test_keep:
                self.root.after(800, self.on_close)

    def _classify_result(self, returncode, stopped: bool):
        if stopped:
            return "Stopped", "Stopped by user."
        text = self.stdout_text
        if "Tested" in text:
            if "FAULTY" in text:
                return "FAULTY", "Finished: FAULTY GPU detected."
            if re.search(r"GPU\s+\d+:\s*OK", text):
                return "OK", "Finished: GPU test OK."
            return "FAULTY", "Finished: FAULTY GPU detected."
        if "FAULTY" in text:
            return "FAULTY", "Finished: FAULTY GPU detected."
        return "Error", "Finished with errors."

    def stop_burn(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.stop_requested = True
        self._set_status("Stop requested.", C_ORANGE)
        self.append_log("Stop requested. Sending SIGTERM/taskkill.")
        self.btn_stop.configure(state="disabled")
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                    capture_output=True, timeout=15,
                )
            else:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                timer = threading.Timer(3.0, self._force_kill_linux)
                timer.daemon = True
                timer.start()
        except Exception as exc:
            self.append_log(tr("Failed to stop backend cleanly: {error}").format(error=str(exc)))

    def _force_kill_linux(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass

    # ------------------------------------------------------------------ monitor

    def _update_idle_monitor(self) -> None:
        if self.proc is None:
            self._show_waiting_monitor()

    def _show_waiting_monitor(self) -> None:
        self.monitor_text.configure(state="normal")
        self.monitor_text.delete("1.0", "end")
        self.monitor_text.insert("end", tr("Waiting for a burn."))
        self.monitor_text.configure(state="disabled")

    def update_monitor(self) -> None:
        if not self.proc:
            self._show_waiting_monitor()
            return
        self._monitor_after = None
        self._query_nvidia_smi()
        if self.proc and self.proc.poll() is None:
            self._monitor_after = self.root.after(1000, self.update_monitor)

    def _query_nvidia_smi(self) -> None:
        if not self.smi_path:
            self.monitor_text.configure(state="normal")
            self.monitor_text.delete("1.0", "end")
            self.monitor_text.insert("end", tr("nvidia-smi was not found on PATH."))
            self.monitor_text.configure(state="disabled")
            return
        fields = (
            "index,name,temperature.gpu,utilization.gpu,memory.used,"
            "memory.total,power.draw,power.limit,fan.speed"
        )
        query = [self.smi_path, "--query-gpu=" + fields, "--format=csv,noheader,nounits"]
        gpu_id = self._selected_gpu_id()
        if gpu_id is not None:
            query = [self.smi_path, "-i", str(gpu_id)] + query[1:]
        try:
            measured = subprocess.run(
                query, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=12, env=os.environ.copy(),
            )
            text = decode_bytes(measured.stdout).strip()
            if not text:
                raise RuntimeError(decode_bytes(measured.stderr).strip() or "empty output")
        except Exception as exc:
            self.monitor_text.configure(state="normal")
            self.monitor_text.delete("1.0", "end")
            self.monitor_text.insert("end", tr("Could not read nvidia-smi output.") + "\n" + str(exc))
            self.monitor_text.configure(state="disabled")
            return

        lines = [""]
        for row in text.splitlines():
            parts = [part.strip() for part in row.split(",")]
            if len(parts) < 9:
                continue
            gpu_index, name, temp, util, used, total, power, limit, fan = parts[:9]
            lines.append(tr("GPU {id}: {name}").format(id=gpu_index, name=name))
            lines.append(f"{tr('Temp (C)')}: {temp}")
            lines.append(f"{tr('Utilization (%)')}: {util}")
            lines.append(f"{tr('VRAM (used / total MB)')}: {used} / {total}")
            lines.append(f"{tr('Power (W) / limit')}: {power} / {limit}")
            lines.append(f"{tr('Fan (%)')}: {fan}")
            lines.append("")

        self.monitor_text.configure(state="normal")
        self.monitor_text.delete("1.0", "end")
        self.monitor_text.insert("end", "\n".join(lines))
        self.monitor_text.configure(state="disabled")

    # ------------------------------------------------------------------ actions

    def show_help(self) -> None:
        if not self.backend_path:
            self._set_status("Backend missing.", C_RED)
            return
        result = run_capture([str(self.backend_path), "-h"], timeout=30)
        self.append_log("=== gpu_burn help ===")
        for out in (result["stdout"], result["stderr"]):
            if out.strip():
                for line in out.splitlines():
                    if line.strip():
                        self.append_log(line)
        self._set_status("Help output written to the log.", C_GREEN)

    def open_log_folder(self) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(self.log_dir))
            else:
                opener = shutil.which("xdg-open")
                if opener:
                    subprocess.Popen([opener, str(self.log_dir)])
                else:
                    self.append_log(tr("Could not open log folder: {error}").format(error="xdg-open missing"))
        except Exception as exc:
            self.append_log(tr("Could not open log folder: {error}").format(error=str(exc)))

    def toggle_language(self) -> None:
        global UI_LANG
        UI_LANG = "en" if UI_LANG == "zh-TW" else "zh-TW"
        self._update_language()

    def _update_language(self) -> None:
        for widget, key in self.label_widgets:
            widget.configure(text=tr(key))
        for widget, key in self.button_widgets:
            widget.configure(text=tr(key))
        for widget, key in self.check_widgets:
            widget.configure(text=tr(key))
        for page, key in zip(self.tab_pages, self.tab_keys):
            self.tabs.tab(page, text=tr(key))

        self.status_label.configure(text=tr(self.status_key))
        self.lang_button.configure(text="English" if UI_LANG == "zh-TW" else "中文")

        current_mode = self.mem_mode.current() if hasattr(self, "mem_mode") else 0
        if hasattr(self, "mem_mode"):
            self.mem_mode.configure(values=[
                tr("Backend default (90% available VRAM)"),
                tr("Percent of available VRAM"),
                tr("Fixed amount (MB)"),
            ])
            self.mem_mode.current(current_mode)
            mode_values = [
                tr("Backend default (90% available VRAM)"),
                tr("Percent of available VRAM"),
                tr("Fixed amount (MB)"),
            ]
            if 0 <= current_mode < len(mode_values):
                self.mem_mode.set(mode_values[current_mode])

        if hasattr(self, "gpu_combo"):
            current_gpu = self.gpu_combo.current()
            values = [tr("All GPUs (backend default)")]
            for gpu in self.gpus:
                values.append(tr("GPU {id}: {name}").format(id=gpu["id"], name=gpu["name"]))
            self.gpu_combo.configure(values=values)
            self.gpu_combo.current(current_gpu)
            if 0 <= current_gpu < len(values):
                self.gpu_combo.set(values[current_gpu])

        self.update_summary()
        if self.proc is None:
            self._show_waiting_monitor()

    def on_close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.stop_burn()
            for thread in self.read_threads:
                thread.join(1.5)
        self.root.destroy()

    # ------------------------------------------------------------------ lifecycle helpers

    def _log_startup(self) -> None:
        self.append_log(f"{APP_TITLE}")
        self.append_log("-----------------------------------")
        self.append_log(f"Backend: {self.backend_path or 'not found'}")
        self.append_log(f"Compare: {self.compare_file or 'not found'}")
        self.append_log(f"nvidia-smi: {self.smi_path or 'not found'}")


def main() -> None:
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
