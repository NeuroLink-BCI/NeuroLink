import brainflow as bf
import time
import csv
import os
import winsound
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import tkinter as tk
import threading


# ============================================================
# AUTO-RECORDING SETTINGS
# ============================================================
TRIALS_PER_MOVEMENT = 10  # عدد الملفات لكل حركة
INTER_TRIAL_REST = 1       # ثانية واحدة فاصلة بين كل ملف وملف


# ============================================================
# FULL-SCREEN WINDOW HELPER
# ============================================================
def make_fullscreen(window):
    window.attributes("-fullscreen", True)
    window.configure(bg="black")


# ============================================================
# GUI: Movement Selection Window (FULL SCREEN)
# ============================================================
def choose_movement_gui():
    result = {"choice": None}

    def set_choice(val):
        result["choice"] = val
        window.destroy()

    window = tk.Tk()
    window.title("Select Movement")
    make_fullscreen(window)

    tk.Label(window, text="Select Movement", font=("Arial", 40),
             bg="black", fg="white").pack(pady=50)

    tk.Label(window, text=f"({TRIALS_PER_MOVEMENT} trials per movement)",
             font=("Arial", 18), bg="black", fg="gray").pack(pady=5)

    btn_style = {"font": ("Arial", 20), "width": 10, "height": 1,
                 "bg": "black", "fg": "white", "activebackground": "gray20"}

    tk.Button(window, text="Right", **btn_style,
              command=lambda: set_choice("right")).pack(pady=10)
    tk.Button(window, text="Left", **btn_style,
              command=lambda: set_choice("left")).pack(pady=10)
    tk.Button(window, text="Forward", **btn_style,
              command=lambda: set_choice("forward")).pack(pady=10)
    tk.Button(window, text="Backward", **btn_style,
              command=lambda: set_choice("backward")).pack(pady=10)
    tk.Button(window, text="Stop", **btn_style,
              command=lambda: set_choice("stop")).pack(pady=10)

    tk.Button(window, text="FINISH SESSION", font=("Arial", 20),
              bg="red", fg="white", width=15, height=1,
              command=lambda: set_choice("finish_session")).pack(pady=40)

    window.mainloop()
    return result["choice"]


# ============================================================
# GUI: Session phases DURING the 8-second recording (FULL SCREEN)
# مع دعم abort عند أي ضغطة
# ============================================================
def show_recording_phases(movement, trial_num, total_trials, abort_flag):
    """
    abort_flag: قائمة [False] — لو صارت [True] معناها المستخدم ضغط شي
                و التسجيل لازم يلغى.
    """
    window = tk.Tk()
    window.title("Session Phases")
    make_fullscreen(window)

    # --- مهمة جداً: أي ضغطة (مفتاح أو ماوس) ترفع abort_flag وتقفل النافذة ---
    def trigger_abort(event=None):
        abort_flag[0] = True
        try:
            window.destroy()
        except:
            pass

    window.bind("<Key>", trigger_abort)
    window.bind("<Button-1>", trigger_abort)
    window.bind("<Button-2>", trigger_abort)
    window.bind("<Button-3>", trigger_abort)

    # --- شريط علوي يبين رقم الـ trial ---
    counter_label = tk.Label(
        window,
        text=f"{movement.upper()}   |   Trial {trial_num} / {total_trials}",
        font=("Arial", 24), bg="black", fg="gray"
    )
    counter_label.pack(pady=20)

    label = tk.Label(window, text="", font=("Arial", 60),
                     bg="black", fg="white")
    label.pack(expand=True)

    # دالة مساعدة: تنفّذ time.sleep بشكل تدريجي وتفحص abort_flag باستمرار
    def sleep_with_abort(seconds):
        end = time.time() + seconds
        while time.time() < end:
            if abort_flag[0]:
                return
            window.update()
            time.sleep(0.05)

    if movement == "stop":
        # شاشة سوداء في كل المراحل
        label.config(text="")
        window.update()
        sleep_with_abort(2)   # 2 ثواني راحة
        if abort_flag[0]:
            return

        sleep_with_abort(4)   # 4 ثواني ستوب
        if abort_flag[0]:
            return

        sleep_with_abort(2)   # 2 ثواني راحة

    else:
        # الحركات العادية
        for i in range(2, 0, -1):
            if abort_flag[0]:
                break
            label.config(text=f"Get Ready...\n{i}")
            window.update()
            sleep_with_abort(1)

        for i in range(4, 0, -1):
            if abort_flag[0]:
                break
            label.config(text=f"{movement}\n{i}")
            window.update()
            sleep_with_abort(1)

        for i in range(2, 0, -1):
            if abort_flag[0]:
                break
            label.config(text=f"Relax...\n{i}")
            window.update()
            sleep_with_abort(1)

    try:
        window.destroy()
    except:
        pass


# ============================================================
# GUI: شاشة فاصلة بين الـ trials (1 ثانية)
# ============================================================
def show_inter_trial_rest(next_trial_num, total_trials, abort_flag):
    """شاشة بسيطة 1 ثانية بين كل ملف والملف اللي بعده."""
    window = tk.Tk()
    window.title("Rest")
    make_fullscreen(window)

    def trigger_abort(event=None):
        abort_flag[0] = True
        try:
            window.destroy()
        except:
            pass

    window.bind("<Key>", trigger_abort)
    window.bind("<Button-1>", trigger_abort)
    window.bind("<Button-2>", trigger_abort)
    window.bind("<Button-3>", trigger_abort)

    tk.Label(
        window,
        text=f"Next: Trial {next_trial_num} / {total_trials}",
        font=("Arial", 36), bg="black", fg="gray"
    ).pack(expand=True)

    end = time.time() + INTER_TRIAL_REST
    while time.time() < end and not abort_flag[0]:
        window.update()
        time.sleep(0.05)

    try:
        window.destroy()
    except:
        pass


# ============================================================
# KnightBoard Class
# ============================================================
class KnightBoard:
    def __init__(self, serial_port: str, num_channels: int):
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.num_channels = num_channels
        
        self.board_shim = BoardShim(BoardIds.NEUROPAWN_KNIGHT_BOARD.value, self.params)
        self.board_id = self.board_shim.get_board_id()
        self.eeg_channels = self.board_shim.get_exg_channels(self.board_id)
        self.sampling_rate = self.board_shim.get_sampling_rate(self.board_id)

    def start_stream(self, buffer_size: int = 450000):
        self.board_shim.prepare_session()
        self.board_shim.start_stream(buffer_size)
        print("Stream started.")
        time.sleep(2)
        for x in range(1, self.num_channels + 1):
            time.sleep(0.5)
            cmd = f"chon_{x}_12"
            self.board_shim.config_board(cmd)
            print(f"sending {cmd}")
            time.sleep(1)
            rld = f"rldadd_{x}"
            self.board_shim.config_board(rld)
            print(f"sending {rld}")
            time.sleep(0.5)

    def stop_stream(self):
        self.board_shim.stop_stream()
        self.board_shim.release_session()
        print("Stream stopped and session released.")


# ============================================================
# Session Folder
# ============================================================
session_folder = "session_1"
os.makedirs(session_folder, exist_ok=True)

Knight_board = KnightBoard("COM3", 8)
Knight_board.start_stream()

print("\n=== Session Started ===\n")


# ============================================================
# دالة تسجيل ملف واحد (نفس بروتوكول الكود الأصلي تماماً)
# ============================================================
def record_one_trial(movement, trial_num, total_trials):
    """
    يسجل ملف واحد بنفس البروتوكول الأصلي:
      - 8 ثواني تسجيل
      - GUI يعرض المراحل في thread منفصل
      - أي ضغطة تلغي التسجيل
    
    Returns:
        True  → التسجيل تم بنجاح وتم الحفظ
        False → التسجيل اتلغى (المستخدم ضغط شي)
    """
    abort_flag = [False]

    # تشغيل الواجهة في Thread منفصل
    phase_thread = threading.Thread(
        target=show_recording_phases,
        args=(movement, trial_num, total_trials, abort_flag)
    )
    phase_thread.start()

    # تجهيز اسم الملف
    existing = [f for f in os.listdir(session_folder) if f.startswith(movement + "_")]
    count = len(existing) + 1
    filename = f"{session_folder}/{movement}_{count}.csv"

    csv_file = open(filename, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    winsound.Beep(1000, 500)

    print(f"\n[{trial_num}/{total_trials}] Recording {movement}_{count} for 8 seconds...")

    # تفريغ البفر قبل بدء التسجيل
    Knight_board.board_shim.get_board_data()

    start_time = time.time()
    aborted = False

    while True:
        # فحص الـ abort flag
        if abort_flag[0]:
            aborted = True
            break

        data = Knight_board.board_shim.get_board_data()

        if data.size > 0:
            for row in data.T:
                writer.writerow(row)

        if time.time() - start_time >= 8:
            break

        # نوم قصير عشان ما نأكل CPU
        time.sleep(0.01)

    csv_file.close()

    # انتظار انتهاء thread الواجهة
    phase_thread.join(timeout=2)

    if aborted:
        # حذف الملف الملوث
        try:
            os.remove(filename)
            print(f"  ⚠ ABORTED — deleted {filename} (user input detected)")
        except Exception as e:
            print(f"  ⚠ ABORTED — failed to delete {filename}: {e}")
        # صوت تنبيه مختلف
        winsound.Beep(400, 300)
        return False
    else:
        print(f"  ✓ Saved: {filename}")
        return True


# ============================================================
# MAIN LOOP — تسجيل أوتوماتيكي 40 ملف لكل حركة
# ============================================================
while True:

    movement = choose_movement_gui()

    if movement == "finish_session":
        print("Finishing session...")
        break

    if movement is None:
        print("No movement selected.")
        continue

    # حساب كم ملف موجود مسبقاً لهذه الحركة
    existing_count = len(
        [f for f in os.listdir(session_folder) if f.startswith(movement + "_")]
    )

    print("\n" + "=" * 60)
    print(f"  STARTING AUTO-RECORDING: {movement.upper()}")
    print(f"  Target: {TRIALS_PER_MOVEMENT} clean trials")
    print(f"  Existing files for this movement: {existing_count}")
    print("=" * 60)

    successful = 0
    aborted_count = 0

    # نسجل لين نوصل العدد المطلوب من الملفات الناجحة
    while successful < TRIALS_PER_MOVEMENT:

        trial_num = successful + 1
        ok = record_one_trial(movement, trial_num, TRIALS_PER_MOVEMENT)

        if ok:
            successful += 1
        else:
            aborted_count += 1
            print(f"  Re-trying trial {trial_num}...")
            # شوية وقت قبل ما نعيد عشان ما نشتغل بسرعة
            time.sleep(1)

        # لو خلصنا، طلع من اللوب
        if successful >= TRIALS_PER_MOVEMENT:
            break

        # شاشة راحة قصيرة بين الـ trials
        rest_abort = [False]
        show_inter_trial_rest(successful + 1, TRIALS_PER_MOVEMENT, rest_abort)

    print("\n" + "=" * 60)
    print(f"  ✓ FINISHED: {movement.upper()}")
    print(f"  Successful trials: {successful}")
    print(f"  Aborted/retried:   {aborted_count}")
    print("=" * 60)
    print("\nReturning to movement selection...\n")


Knight_board.stop_stream()
print("\n=== Session Complete ===")
