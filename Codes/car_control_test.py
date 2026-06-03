"""
NeuroLink — Manual Car Control Test
واجهة اختبار يدوي للسيارة قبل ربطها بالـ BCI
تشغيل: py -3.12 car_control_test.py
"""

import tkinter as tk
from tkinter import font as tkfont
import requests
import threading
import time

# ============================================================
# SETTINGS
# ============================================================
CAR_IP   = "192.168.4.1"   # IP ثابت للـ ESP32 Access Point
CAR_PORT = 80
BASE_URL = f"http://{CAR_IP}:{CAR_PORT}"

# مدة تحريك كل أمر (ثواني)
DURATION = {
    "forward":  1,    
    "backward": 1,    
    "left":     0.8,  
    "right":    0.8,  
    "stop":     0,
}

# ============================================================
# HTTP — إرسال الأمر للسيارة
# ============================================================
def send_command(command, status_var, timer_var, moving_flag):
    """
    يرسل الأمر للسيارة وبعد المدة المحددة يرسل stop تلقائياً.
    يشتغل في thread منفصل عشان ما يجمّد الواجهة.
    """
    if moving_flag[0] and command != "stop":
        return  # تجاهل الضغطة إذا السيارة شغّالة

    moving_flag[0] = True
    duration = DURATION.get(command, 4)

    try:
        # --- أرسل الأمر ---
        url = f"{BASE_URL}/{command}"
        resp = requests.get(url, timeout=3)

        if resp.status_code == 200:
            status_var.set(f"✓  {command.upper()}")
        else:
            status_var.set(f"✗  خطأ: {resp.status_code}")
            moving_flag[0] = False
            return

    except requests.exceptions.ConnectionError:
        status_var.set("✗  تعذّر الاتصال — تأكد من WiFi")
        moving_flag[0] = False
        return
    except requests.exceptions.Timeout:
        status_var.set("✗  انتهت مهلة الاتصال")
        moving_flag[0] = False
        return

    # --- إذا stop، انتهى العمل ---
    if command == "stop" or duration == 0:
        moving_flag[0] = False
        timer_var.set("")
        return

    # --- عدّاد تنازلي ---
    timer_var.set(f"{duration}s")
    time.sleep(duration)

    # --- أوقف السيارة تلقائياً ---
    try:
        requests.get(f"{BASE_URL}/stop", timeout=3)
        status_var.set("■  STOP (auto)")
    except:
        status_var.set("■  STOP (timeout)")

    timer_var.set("")
    moving_flag[0] = False


def send_in_thread(command, status_var, timer_var, moving_flag):
    t = threading.Thread(
        target=send_command,
        args=(command, status_var, timer_var, moving_flag),
        daemon=True
    )
    t.start()


# ============================================================
# CONNECTION TEST
# ============================================================
def test_connection(status_var):
    status_var.set("⟳  جاري الاتصال...")
    try:
        resp = requests.get(f"{BASE_URL}/status", timeout=3)
        if resp.status_code == 200:
            status_var.set(f"✓  متصل — {resp.text.strip()}")
        else:
            status_var.set(f"✗  استجابة غير متوقعة: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        status_var.set("✗  لا يوجد اتصال — تأكد من WiFi NeuroLink_Car")
    except requests.exceptions.Timeout:
        status_var.set("✗  انتهت مهلة الاتصال")


def test_in_thread(status_var):
    t = threading.Thread(target=test_connection, args=(status_var,), daemon=True)
    t.start()


# ============================================================
# GUI
# ============================================================
def build_gui():
    root = tk.Tk()
    root.title("NeuroLink — Car Control Test")
    root.configure(bg="#0d0d0d")
    root.resizable(False, False)

    # ── Fonts ──
    title_font  = tkfont.Font(family="Courier New", size=16, weight="bold")
    label_font  = tkfont.Font(family="Courier New", size=11)
    btn_font    = tkfont.Font(family="Courier New", size=13, weight="bold")
    status_font = tkfont.Font(family="Courier New", size=10)
    timer_font  = tkfont.Font(family="Courier New", size=22, weight="bold")

    # ── Colors ──
    BG        = "#0d0d0d"
    PANEL     = "#1a1a1a"
    ACCENT    = "#00ff88"
    BTN_BG    = "#1e1e1e"
    BTN_FG    = "#e0e0e0"
    BTN_ACT   = "#2a2a2a"
    FWD_CLR   = "#00cc66"
    BWD_CLR   = "#ff4444"
    LFT_CLR   = "#4488ff"
    RGT_CLR   = "#ff9900"
    STP_CLR   = "#888888"
    WARN      = "#ffaa00"

    # ── State ──
    status_var  = tk.StringVar(value="○  غير متصل")
    timer_var   = tk.StringVar(value="")
    moving_flag = [False]   # قائمة عشان نعدّل القيمة داخل الـ threads

    pad = dict(padx=12, pady=8)

    # ────────────────────────────────────────────
    # HEADER
    # ────────────────────────────────────────────
    header = tk.Frame(root, bg=BG)
    header.pack(fill="x", padx=20, pady=(20, 5))

    tk.Label(header, text="◈ NEUROLINK", font=title_font,
             bg=BG, fg=ACCENT).pack(side="left")
    tk.Label(header, text="CAR CONTROL TEST", font=label_font,
             bg=BG, fg="#555555").pack(side="left", padx=10)

    # ────────────────────────────────────────────
    # CONNECTION PANEL
    # ────────────────────────────────────────────
    conn_frame = tk.Frame(root, bg=PANEL, bd=0, relief="flat")
    conn_frame.pack(fill="x", padx=20, pady=(0, 10))

    tk.Label(conn_frame, text=f"ESP32 → {CAR_IP}",
             font=status_font, bg=PANEL, fg="#666666").pack(side="left", padx=12, pady=8)

    tk.Label(conn_frame, textvariable=status_var,
             font=status_font, bg=PANEL, fg=ACCENT, width=40,
             anchor="w").pack(side="left", padx=5)

    tk.Button(conn_frame, text="TEST", font=status_font,
              bg=BTN_BG, fg=WARN, activebackground=BTN_ACT,
              activeforeground=WARN, bd=0, padx=10, pady=4,
              cursor="hand2",
              command=lambda: test_in_thread(status_var)).pack(side="right", padx=12)

    # ────────────────────────────────────────────
    # TIMER DISPLAY
    # ────────────────────────────────────────────
    tk.Label(root, textvariable=timer_var,
             font=timer_font, bg=BG, fg=ACCENT, height=2).pack()

    # ────────────────────────────────────────────
    # CONTROL BUTTONS (D-pad layout)
    # ────────────────────────────────────────────
    ctrl = tk.Frame(root, bg=BG)
    ctrl.pack(padx=20, pady=10)

    def make_btn(parent, text, color, cmd, row, col,
                 rowspan=1, colspan=1, w=10, h=3):
        b = tk.Button(
            parent, text=text, font=btn_font,
            bg=BTN_BG, fg=color, activebackground=BTN_ACT,
            activeforeground=color, bd=0, width=w, height=h,
            relief="flat", cursor="hand2",
            command=lambda: send_in_thread(cmd, status_var, timer_var, moving_flag)
        )
        b.grid(row=row, column=col, rowspan=rowspan, columnspan=colspan,
               padx=6, pady=6)
        # hover effect
        b.bind("<Enter>", lambda e, b=b, c=color: b.config(bg="#2a2a2a"))
        b.bind("<Leave>", lambda e, b=b: b.config(bg=BTN_BG))
        return b

    #         col0        col1        col2
    # row0              FORWARD
    # row1    LEFT       STOP        RIGHT
    # row2              BACKWARD

    make_btn(ctrl, "▲\nFORWARD\n1s",  FWD_CLR, "forward",  row=0, col=1)
    make_btn(ctrl, "◄\nLEFT\n0.8s",     LFT_CLR, "left",     row=1, col=0)
    make_btn(ctrl, "■\nSTOP",          STP_CLR, "stop",     row=1, col=1, w=10, h=3)
    make_btn(ctrl, "►\nRIGHT\n0.8s",    RGT_CLR, "right",    row=1, col=2)
    make_btn(ctrl, "▼\nBACKWARD\n1s", BWD_CLR, "backward", row=2, col=1)

    # ────────────────────────────────────────────
    # KEYBOARD SHORTCUTS
    # ────────────────────────────────────────────
    root.bind("<Up>",    lambda e: send_in_thread("forward",  status_var, timer_var, moving_flag))
    root.bind("<Down>",  lambda e: send_in_thread("backward", status_var, timer_var, moving_flag))
    root.bind("<Left>",  lambda e: send_in_thread("left",     status_var, timer_var, moving_flag))
    root.bind("<Right>", lambda e: send_in_thread("right",    status_var, timer_var, moving_flag))
    root.bind("<space>", lambda e: send_in_thread("stop",     status_var, timer_var, moving_flag))

    # ────────────────────────────────────────────
    # KEYBOARD HINT
    # ────────────────────────────────────────────
    hint = tk.Frame(root, bg=PANEL)
    hint.pack(fill="x", padx=20, pady=(0, 20))

    hints = [
        ("↑", "Forward"), ("↓", "Backward"),
        ("←", "Left"),    ("→", "Right"),
        ("Space", "Stop")
    ]
    for key, lbl in hints:
        f = tk.Frame(hint, bg=PANEL)
        f.pack(side="left", padx=8, pady=6)
        tk.Label(f, text=key, font=status_font, bg="#2a2a2a", fg=ACCENT,
                 padx=6, pady=2).pack(side="left")
        tk.Label(f, text=f" {lbl}", font=status_font, bg=PANEL,
                 fg="#555555").pack(side="left")

    # ────────────────────────────────────────────
    # DURATION CONTROLS
    # ────────────────────────────────────────────
    dur_frame = tk.Frame(root, bg=PANEL)
    dur_frame.pack(fill="x", padx=20, pady=(0, 20))

    tk.Label(dur_frame, text="مدة الحركة:",
             font=status_font, bg=PANEL, fg="#666666").pack(side="left", padx=12, pady=6)

    def set_duration(secs):
        DURATION["forward"]  = secs
        DURATION["backward"] = secs
        DURATION["left"]     = max(2, secs - 1)
        DURATION["right"]    = max(2, secs - 1)
        # تحديث نص الأزرار
        for widget in ctrl.winfo_children():
            t = widget.cget("text")
            if "FORWARD" in t:
                widget.config(text=f"▲\nFORWARD\n{secs}s")
            elif "BACKWARD" in t:
                widget.config(text=f"▼\nBACKWARD\n{secs}s")
            elif "LEFT" in t:
                widget.config(text=f"◄\nLEFT\n{max(2, secs-1)}s")
            elif "RIGHT" in t:
                widget.config(text=f"►\nRIGHT\n{max(2, secs-1)}s")

    for s in [3, 4, 5, 7]:
        tk.Button(dur_frame, text=f"{s}s",
                  font=status_font, bg=BTN_BG, fg=WARN,
                  activebackground=BTN_ACT, bd=0, padx=10, pady=4,
                  cursor="hand2",
                  command=lambda s=s: set_duration(s)).pack(side="left", padx=4)

    root.mainloop()


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    build_gui()
