"""
╔══════════════════════════════════════════════════════════════════╗
║         NeuroLink — Session Calibration BCI System (v5)          ║
║                                                                  ║
║  Option 1: Full Pipeline (record + fine-tune + semi real-time)   ║
║  Option 2: Existing Calibration (fine-tune + semi real-time)     ║
║  Option 3: Direct Prediction (no calibration, no fine-tuning)    ║
║                                                                  ║
║  Run:  py -3.12 neurolink_calibration_system_v5.py               ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import sys
import csv
import time
import shutil
import random
import threading
import warnings
import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog
from datetime import datetime
from PIL import Image, ImageTk

import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import requests

warnings.filterwarnings('ignore')

# ================================================================
# SECTION 1: CONFIGURATION
# ================================================================

SERIAL_PORT = "COM3"
NUM_CHANNELS = 8
SAMPLING_RATE = 125

EEG_COL_START = 1
EEG_COL_END = 9

DISCARD_SECONDS = 2
DISCARD_SAMPLES = int(DISCARD_SECONDS * SAMPLING_RATE)
MI_FIXED_LENGTH = 625
RECORDING_SECONDS = 8

LOW_FREQ = 8
HIGH_FREQ = 30
FILTER_ORDER = 4

WINDOW_SIZE = 250
STRIDE = 62

LABEL_NAMES = ['backward', 'forward', 'left', 'right', 'stop']
LABEL_MAP = {'backward': 0, 'forward': 1, 'left': 2, 'right': 3, 'stop': 4}
NUM_CLASSES = 5

# --- Calibration: 20 per class -> 10 train + 5 holdout + 5 semi-test ---
TRIALS_PER_CLASS = 20
SEMI_TEST_PER_CLASS = 5
HOLDOUT_PER_CLASS = 5
TRAIN_PER_CLASS = TRIALS_PER_CLASS - SEMI_TEST_PER_CLASS - HOLDOUT_PER_CLASS
INTER_TRIAL_REST = 1

# --- Fine-tuning ---
FT_EPOCHS = 30
FT_PATIENCE = 8
FT_LEARNING_RATE = 1e-4
FT_BATCH_SIZE = 16

# --- Model ---
BASE_MODEL_PATH = "neurolink_final_v25.h5"

# --- Car ---
CAR_IP = "192.168.4.1"
CAR_PORT = 80
CAR_BASE_URL = f"http://{CAR_IP}:{CAR_PORT}"
CAR_TIMEOUT = 3
CAR_ENABLED = False

DURATION = {
    "forward": 1.5, "backward": 1.5,
    "left": 0.8, "right": 0.8, "stop": 0.0,
}

# --- Filter coefficients ---
nyquist = 0.5 * SAMPLING_RATE
b_filt, a_filt = sig.butter(FILTER_ORDER, [LOW_FREQ/nyquist, HIGH_FREQ/nyquist], btype='band')

SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CALIB_FOLDER = f"calibration_{SESSION_TIMESTAMP}"
SEMI_TEST_FOLDER = f"semi_test_{SESSION_TIMESTAMP}"
DEBUG_FOLDER = f"debug_{SESSION_TIMESTAMP}"

# --- Logo ---
LOGO_PATH = "NEUROLINK_.png"

# --- GUI Colors (NeuroLink branding) ---
BG_WHITE = "#FFFFFF"
NAVY = "#1B1464"
NEON_BLUE = "#00D4FF"
NEON_FUCHSIA = "#FF00FF"
NEON_NAVY = "#4040FF"
TEXT_DARK = "#1B1464"
TEXT_GRAY = "#888888"
PANEL_BG = "#F5F5FA"
SUCCESS_GREEN = "#00C853"
ERROR_RED = "#FF1744"

CMD_COLORS = {
    'forward': '#00C853', 'backward': '#FF1744',
    'left': '#2979FF', 'right': '#FF9100', 'stop': '#9E9E9E'
}


# ================================================================
# SECTION 2: PREPROCESSING (identical to training)
# ================================================================

def bandpass_filter(data):
    filtered = np.zeros_like(data, dtype=np.float64)
    for ch in range(data.shape[0]):
        filtered[ch] = sig.filtfilt(b_filt, a_filt, data[ch].astype(np.float64))
    return filtered

def apply_car(data):
    return data - np.mean(data, axis=0, keepdims=True)

def normalize_channels(data):
    normalized = np.zeros_like(data)
    for ch in range(data.shape[0]):
        mu = np.mean(data[ch])
        sigma = np.std(data[ch])
        if sigma > 1e-8:
            normalized[ch] = (data[ch] - mu) / sigma
        else:
            normalized[ch] = data[ch] - mu
    return normalized

def preprocess_trial(raw_data):
    data = bandpass_filter(raw_data)
    data = apply_car(data)
    data = normalize_channels(data)
    return data

def preprocess_and_predict_csv(file_path, model):
    """Load CSV -> preprocess -> predict. Returns (command, confidence, probs) or Nones."""
    try:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < EEG_COL_END:
            return None, None, None
        eeg = df.iloc[:, EEG_COL_START:EEG_COL_END].values.T.astype(np.float64)
        if eeg.shape[1] < DISCARD_SAMPLES + WINDOW_SIZE:
            return None, None, None
        eeg = eeg[:, DISCARD_SAMPLES:]
        if eeg.shape[1] < MI_FIXED_LENGTH:
            return None, None, None
        eeg = eeg[:, -MI_FIXED_LENGTH:]
        data = preprocess_trial(eeg)
        window = data[:, -WINDOW_SIZE:]
        X = window[np.newaxis, ..., np.newaxis]
        probs = model.predict(X, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        return LABEL_NAMES[pred_idx], float(probs[pred_idx]), probs
    except Exception as e:
        print(f"  Error: {e}")
        return None, None, None

def create_windows(X, y, window_size=WINDOW_SIZE, stride=STRIDE):
    windows, labels = [], []
    for i in range(len(X)):
        trial = X[i]
        n_win = (trial.shape[1] - window_size) // stride + 1
        for w in range(n_win):
            s = w * stride
            windows.append(trial[:, s:s+window_size])
            labels.append(y[i])
    return np.array(windows), np.array(labels)

def augment_data(X_windows, y_labels):
    augmented_X = [X_windows.copy()]
    augmented_y = [y_labels.copy()]
    n = len(X_windows)
    ws = X_windows.shape[2]
    # Scaling
    scales = np.random.uniform(0.8, 1.2, size=(n, 1, 1))
    augmented_X.append(X_windows * scales)
    augmented_y.append(y_labels.copy())
    # Noise
    noise = np.random.randn(*X_windows.shape) * 0.1
    augmented_X.append(X_windows + noise)
    augmented_y.append(y_labels.copy())
    # Time shift
    X_shifted = np.zeros_like(X_windows)
    for i in range(n):
        shift = np.random.randint(-15, 16)
        if shift > 0:
            X_shifted[i, :, shift:] = X_windows[i, :, :ws-shift]
            X_shifted[i, :, :shift] = X_windows[i, :, 0:1]
        elif shift < 0:
            X_shifted[i, :, :shift] = X_windows[i, :, -shift:]
            X_shifted[i, :, shift:] = X_windows[i, :, -1:]
        else:
            X_shifted[i] = X_windows[i]
    augmented_X.append(X_shifted)
    augmented_y.append(y_labels.copy())
    X_f = np.concatenate(augmented_X)
    y_f = np.concatenate(augmented_y)
    idx = np.random.permutation(len(X_f))
    return X_f[idx], y_f[idx]

def load_csv_trial(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < EEG_COL_END:
            return None
        eeg = df.iloc[:, EEG_COL_START:EEG_COL_END].values.T.astype(np.float64)
        if eeg.shape[1] < DISCARD_SAMPLES + WINDOW_SIZE:
            return None
        eeg = eeg[:, DISCARD_SAMPLES:]
        if eeg.shape[1] < MI_FIXED_LENGTH:
            return None
        return eeg[:, -MI_FIXED_LENGTH:]
    except:
        return None


# ================================================================
# SECTION 3: KNIGHTBOARD CLASS
# ================================================================

class KnightBoard:
    def __init__(self, serial_port, num_channels):
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.num_channels = num_channels
        self.board_shim = BoardShim(BoardIds.NEUROPAWN_KNIGHT_BOARD.value, self.params)
        self.board_id = self.board_shim.get_board_id()
        self.eeg_channels = self.board_shim.get_exg_channels(self.board_id)
        self.sampling_rate = self.board_shim.get_sampling_rate(self.board_id)

    def start_stream(self, buffer_size=450000):
        self.board_shim.prepare_session()
        self.board_shim.start_stream(buffer_size)
        print("  Stream started.")
        time.sleep(2)
        for x in range(1, self.num_channels + 1):
            time.sleep(0.5)
            self.board_shim.config_board(f"chon_{x}_12")
            time.sleep(1)
            self.board_shim.config_board(f"rldadd_{x}")
            time.sleep(0.5)
        print(f"  All {self.num_channels} channels configured.")

    def stop_stream(self):
        try:
            self.board_shim.stop_stream()
            self.board_shim.release_session()
            print("  Stream stopped.")
        except Exception as e:
            print(f"  Warning: {e}")

    def flush_buffer(self):
        self.board_shim.get_board_data()

    def get_data(self):
        return self.board_shim.get_board_data()


# ================================================================
# SECTION 4: RECORDING GUI (same as newRec_Auto.py)
# ================================================================

def make_fullscreen(window):
    window.attributes("-fullscreen", True)
    window.configure(bg="black")

def beep(freq=1000, duration=500):
    if HAS_WINSOUND:
        winsound.Beep(freq, duration)

def show_recording_phases(movement, trial_num, total_trials, abort_flag):
    window = tk.Tk()
    window.title("Session Phases")
    make_fullscreen(window)
    def trigger_abort(event=None):
        abort_flag[0] = True
        try: window.destroy()
        except: pass
    window.bind("<Key>", trigger_abort)
    window.bind("<Button-1>", trigger_abort)
    window.bind("<Button-2>", trigger_abort)
    window.bind("<Button-3>", trigger_abort)
    counter_label = tk.Label(window,
        text=f"CALIBRATION  |  {movement.upper()}  |  Trial {trial_num} / {total_trials}",
        font=("Arial", 24), bg="black", fg="gray")
    counter_label.pack(pady=20)
    label = tk.Label(window, text="", font=("Arial", 60), bg="black", fg="white")
    label.pack(expand=True)
    def sleep_with_abort(seconds):
        end = time.time() + seconds
        while time.time() < end:
            if abort_flag[0]: return
            window.update()
            time.sleep(0.05)
    if movement == "stop":
        label.config(text=""); window.update()
        sleep_with_abort(2)
        if not abort_flag[0]: sleep_with_abort(4)
        if not abort_flag[0]: sleep_with_abort(2)
    else:
        for i in range(2, 0, -1):
            if abort_flag[0]: break
            label.config(text=f"Get Ready...\n{i}"); window.update(); sleep_with_abort(1)
        for i in range(4, 0, -1):
            if abort_flag[0]: break
            label.config(text=f"{movement.upper()}\n{i}"); window.update(); sleep_with_abort(1)
        for i in range(2, 0, -1):
            if abort_flag[0]: break
            label.config(text=f"Relax...\n{i}"); window.update(); sleep_with_abort(1)
    try: window.destroy()
    except: pass

def show_inter_trial_rest(next_num, total, abort_flag):
    window = tk.Tk(); window.title("Rest"); make_fullscreen(window)
    def trigger_abort(event=None):
        abort_flag[0] = True
        try: window.destroy()
        except: pass
    window.bind("<Key>", trigger_abort); window.bind("<Button-1>", trigger_abort)
    tk.Label(window, text=f"Next: Trial {next_num} / {total}",
             font=("Arial", 36), bg="black", fg="gray").pack(expand=True)
    end = time.time() + INTER_TRIAL_REST
    while time.time() < end and not abort_flag[0]:
        window.update(); time.sleep(0.05)
    try: window.destroy()
    except: pass

def record_one_trial(board, movement, trial_num, total_trials, save_folder):
    abort_flag = [False]
    phase_thread = threading.Thread(target=show_recording_phases,
        args=(movement, trial_num, total_trials, abort_flag))
    phase_thread.start()
    existing = [f for f in os.listdir(save_folder) if f.startswith(movement + "_")]
    count = len(existing) + 1
    filename = os.path.join(save_folder, f"{movement}_{count}.csv")
    csv_file = open(filename, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    beep(1000, 500)
    board.flush_buffer()
    start_time = time.time()
    aborted = False
    while True:
        if abort_flag[0]: aborted = True; break
        data = board.get_data()
        if data.size > 0:
            for row in data.T: writer.writerow(row)
        if time.time() - start_time >= RECORDING_SECONDS: break
        time.sleep(0.01)
    csv_file.close()
    phase_thread.join(timeout=2)
    if aborted:
        try: os.remove(filename)
        except: pass
        beep(400, 300); return False
    else:
        print(f"    Saved: {filename}"); return True


# ================================================================
# SECTION 5: CALIBRATION RECORDING
# ================================================================

def run_calibration(board):
    os.makedirs(CALIB_FOLDER, exist_ok=True)
    movements = ['right', 'left', 'forward', 'backward', 'stop']
    total = TRIALS_PER_CLASS * NUM_CLASSES
    print(f"\n  CALIBRATION: {TRIALS_PER_CLASS} trials/class × {NUM_CLASSES} = {total}")
    print(f"  Split: {TRAIN_PER_CLASS} train + {HOLDOUT_PER_CLASS} holdout + {SEMI_TEST_PER_CLASS} semi-test")

    for movement in movements:
        print(f"\n  --- {movement.upper()} ({TRIALS_PER_CLASS} trials) ---")
        # Countdown
        w = tk.Tk(); make_fullscreen(w)
        l = tk.Label(w, text="", font=("Arial", 50), bg="black", fg="white"); l.pack(expand=True)
        for i in range(3, 0, -1):
            l.config(text=f"Next: {movement.upper()}\n\nStarting in {i}..."); w.update(); time.sleep(1)
        w.destroy()

        successful = 0
        while successful < TRIALS_PER_CLASS:
            ok = record_one_trial(board, movement, successful+1, TRIALS_PER_CLASS, CALIB_FOLDER)
            if ok: successful += 1
            else: time.sleep(1)
            if successful < TRIALS_PER_CLASS:
                rest_abort = [False]
                show_inter_trial_rest(successful+1, TRIALS_PER_CLASS, rest_abort)
        print(f"  Completed {movement.upper()}: {successful} saved")

    print(f"\n  CALIBRATION COMPLETE -> {CALIB_FOLDER}/")
    return CALIB_FOLDER


# ================================================================
# SECTION 6: SPLIT + FINE-TUNE
# ================================================================

def split_and_prepare(calib_folder, semi_test_folder):
    """Split calibration data: semi-test files separated, rest -> train + holdout."""
    os.makedirs(semi_test_folder, exist_ok=True)

    # Group files by class
    files_by_class = {name: [] for name in LABEL_NAMES}
    for f in sorted(os.listdir(calib_folder)):
        if not f.lower().endswith('.csv'): continue
        movement = f.lower().replace('.csv', '').split('_')[0]
        if movement in LABEL_MAP:
            files_by_class[movement].append(f)

    X_train_list, y_train_list = [], []
    X_hold_list, y_hold_list = [], []
    semi_test_files = []

    for cls_name in LABEL_NAMES:
        cls_files = files_by_class[cls_name]
        cls_idx = LABEL_MAP[cls_name]
        print(f"    {cls_name:>10s}: {len(cls_files)} files")

        if len(cls_files) < 3:
            print(f"      WARNING: too few files for {cls_name}")
            continue

        # Random shuffle
        random.seed(42)
        indices = list(range(len(cls_files)))
        random.shuffle(indices)

        # Determine split sizes
        n_semi = min(SEMI_TEST_PER_CLASS, len(cls_files) // 3)
        n_hold = min(HOLDOUT_PER_CLASS, (len(cls_files) - n_semi) // 3)
        n_train = len(cls_files) - n_semi - n_hold

        semi_idx = indices[:n_semi]
        hold_idx = indices[n_semi:n_semi+n_hold]
        train_idx = indices[n_semi+n_hold:]

        # Move semi-test files
        for idx in semi_idx:
            fname = cls_files[idx]
            src = os.path.join(calib_folder, fname)
            dst = os.path.join(semi_test_folder, fname)
            shutil.copy2(src, dst)
            semi_test_files.append(fname)

        # Load holdout trials
        for idx in hold_idx:
            eeg = load_csv_trial(os.path.join(calib_folder, cls_files[idx]))
            if eeg is not None:
                X_hold_list.append(eeg)
                y_hold_list.append(cls_idx)

        # Load train trials
        for idx in train_idx:
            eeg = load_csv_trial(os.path.join(calib_folder, cls_files[idx]))
            if eeg is not None:
                X_train_list.append(eeg)
                y_train_list.append(cls_idx)

    X_train = np.array(X_train_list) if X_train_list else None
    y_train = np.array(y_train_list) if y_train_list else None
    X_hold = np.array(X_hold_list) if X_hold_list else None
    y_hold = np.array(y_hold_list) if y_hold_list else None

    print(f"\n    Train: {len(X_train_list)} trials")
    print(f"    Holdout: {len(X_hold_list)} trials")
    print(f"    Semi-test: {len(semi_test_files)} files -> {semi_test_folder}/")

    return X_train, y_train, X_hold, y_hold, semi_test_files


def fine_tune_model(base_model_path, X_train_raw, y_train_raw, X_hold_raw, y_hold_raw):
    """Fine-tune pretrained model on calibration data."""
    print(f"\n  Loading: {base_model_path}")
    model = tf.keras.models.load_model(base_model_path)
    print(f"    Params: {model.count_params():,}")

    # Print layers
    print(f"\n  Layer map:")
    for i, layer in enumerate(model.layers):
        print(f"    [{i:2d}] {layer.name:30s} {'FREEZE' if i < 7 else 'TRAIN'}")

    # Preprocess
    X_train_proc = np.array([preprocess_trial(t) for t in X_train_raw])
    X_hold_proc = np.array([preprocess_trial(t) for t in X_hold_raw]) if X_hold_raw is not None and len(X_hold_raw) > 0 else None

    # Window
    X_train_win, y_train_win = create_windows(X_train_proc, y_train_raw)
    X_hold_win, y_hold_win = (create_windows(X_hold_proc, y_hold_raw) if X_hold_proc is not None else (None, None))

    print(f"    Train windows: {len(X_train_win)}")
    if X_hold_win is not None:
        print(f"    Holdout windows: {len(X_hold_win)}")

    # Augment train only
    X_train_aug, y_train_aug = augment_data(X_train_win, y_train_win)
    print(f"    After augmentation: {len(X_train_aug)} (4x)")

    # Debug save
    os.makedirs(DEBUG_FOLDER, exist_ok=True)
    np.save(os.path.join(DEBUG_FOLDER, "train_windows.npy"), X_train_win)
    np.save(os.path.join(DEBUG_FOLDER, "train_labels.npy"), y_train_win)
    if X_hold_win is not None:
        np.save(os.path.join(DEBUG_FOLDER, "holdout_windows.npy"), X_hold_win)
        np.save(os.path.join(DEBUG_FOLDER, "holdout_labels.npy"), y_hold_win)

    # Prepare TF
    X_tf = X_train_aug[..., np.newaxis]
    y_tf = to_categorical(y_train_aug, NUM_CLASSES)
    X_h_tf = X_hold_win[..., np.newaxis] if X_hold_win is not None else None
    y_h_tf = to_categorical(y_hold_win, NUM_CLASSES) if y_hold_win is not None else None

    # Freeze Block 1
    freeze_until = 7
    for i, layer in enumerate(model.layers):
        if 'dropout' in layer.name.lower():
            freeze_until = i + 1; break
    for i, layer in enumerate(model.layers):
        layer.trainable = (i >= freeze_until)

    trainable_p = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"    Trainable: {trainable_p:,} / {model.count_params():,}")

    model.compile(optimizer=tf.keras.optimizers.Adam(FT_LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    classes = np.unique(y_train_aug)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_aug)
    class_weights = dict(zip(classes.astype(int), weights))

    val_data = (X_h_tf, y_h_tf) if X_h_tf is not None else None
    monitor = 'val_loss' if val_data else 'loss'

    history = model.fit(X_tf, y_tf, batch_size=FT_BATCH_SIZE, epochs=FT_EPOCHS,
                        validation_data=val_data, class_weight=class_weights,
                        callbacks=[EarlyStopping(monitor=monitor, patience=FT_PATIENCE,
                                                restore_best_weights=True)], verbose=2)

    # Holdout results
    holdout_results = None
    if X_h_tf is not None:
        h_loss, h_acc = model.evaluate(X_h_tf, y_h_tf, verbose=0)
        h_preds = np.argmax(model.predict(X_h_tf, verbose=0), axis=1)
        h_true = np.argmax(y_h_tf, axis=1)
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(h_true, h_preds): cm[t][p] += 1
        np.save(os.path.join(DEBUG_FOLDER, "confusion_matrix.npy"), cm)

        per_class = {}
        for cls in range(NUM_CLASSES):
            mask = h_true == cls
            if np.sum(mask) > 0:
                per_class[LABEL_NAMES[cls]] = float(np.mean(h_preds[mask] == cls))

        holdout_results = {
            'accuracy': float(h_acc), 'loss': float(h_loss),
            'confusion_matrix': cm, 'per_class': per_class
        }
        print(f"\n  HOLDOUT: {h_acc:.1%}")
        for name, acc in per_class.items():
            print(f"    {name:>10s}: {acc:.0%}")

    # Save
    session_path = f"neurolink_session_{SESSION_TIMESTAMP}.h5"
    model.save(session_path)
    print(f"\n  Saved: {session_path}")

    return model, session_path, holdout_results


# ================================================================
# SECTION 7: CAR CONTROLLER
# ================================================================

class CarController:
    def __init__(self):
        self.is_busy = False

    def test_connection(self):
        if not CAR_ENABLED:
            return True, "Car control DISABLED"
        try:
            resp = requests.get(f"{CAR_BASE_URL}/status", timeout=CAR_TIMEOUT)
            return resp.status_code == 200, resp.text.strip()
        except:
            return False, "Cannot reach ESP32"

    def send_command(self, command, callback=None):
        if not CAR_ENABLED:
            if callback: callback(f"[SIM] {command.upper()}")
            return
        self.is_busy = True
        duration = DURATION.get(command, 1.0)
        def _worker():
            try:
                requests.get(f"{CAR_BASE_URL}/{command}", timeout=CAR_TIMEOUT)
                if callback: callback(f"✓ {command.upper()}")
            except Exception as e:
                if callback: callback(f"✗ Error: {e}")
            if command != "stop" and duration > 0:
                time.sleep(duration)
                try: requests.get(f"{CAR_BASE_URL}/stop", timeout=CAR_TIMEOUT)
                except: pass
            self.is_busy = False
        threading.Thread(target=_worker, daemon=True).start()


# ================================================================
# SECTION 8: RESULTS GUI (NeuroLink branded)
# ================================================================

def build_results_gui(session_model, session_model_path, semi_test_folder,
                      holdout_results, base_model_path_used):
    """
    The main results GUI with NeuroLink branding.
    Shows holdout results, semi-test predictions, file-by-file testing, car control.
    """
    root = tk.Tk()
    root.title("NeuroLink — BCI Prediction Dashboard")
    root.configure(bg=BG_WHITE)
    root.geometry("900x750")
    root.minsize(800, 600)

    # Make window resizable with scrollbar
    main_canvas = tk.Canvas(root, bg=BG_WHITE, highlightthickness=0)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
    scrollable_frame = tk.Frame(main_canvas, bg=BG_WHITE)

    scrollable_frame.bind("<Configure>",
        lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
    canvas_window = main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    main_canvas.configure(yscrollcommand=scrollbar.set)

    # Make scrollable_frame fill canvas width
    def _on_canvas_configure(event):
        main_canvas.itemconfig(canvas_window, width=event.width)
    main_canvas.bind("<Configure>", _on_canvas_configure)

    # Mouse wheel scroll
    def _on_mousewheel(event):
        main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    root.bind_all("<MouseWheel>", _on_mousewheel)

    scrollbar.pack(side="right", fill="y")
    main_canvas.pack(side="left", fill="both", expand=True)

    frame = scrollable_frame  # alias

    # ── HEADER with Logo ──
    header = tk.Frame(frame, bg=BG_WHITE)
    header.pack(fill="x", padx=30, pady=(15, 5))

    try:
        logo_img = Image.open(LOGO_PATH)
        logo_img = logo_img.resize((80, 80), Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_img)
        logo_label = tk.Label(header, image=logo_photo, bg=BG_WHITE)
        logo_label.image = logo_photo
        logo_label.pack(side="left", padx=(0, 15))
    except Exception as e:
        print(f"  Logo not found ({LOGO_PATH}): {e}")
        nl_label = tk.Label(header, text="NL", font=("Helvetica", 36, "bold"),
                           bg=NEON_BLUE, fg="white", width=3, height=1)
        nl_label.pack(side="left", padx=(0, 15))

    title_frame = tk.Frame(header, bg=BG_WHITE)
    title_frame.pack(side="left")
    tk.Label(title_frame, text="NeuroLink", font=("Helvetica", 28, "bold"),
             bg=BG_WHITE, fg=NAVY).pack(anchor="w")
    tk.Label(title_frame, text="Session Calibration BCI System",
             font=("Helvetica", 12), bg=BG_WHITE, fg=TEXT_GRAY).pack(anchor="w")

    # Separator
    tk.Frame(frame, bg=NEON_BLUE, height=2).pack(fill="x", padx=30, pady=(5, 15))

    # ── MODEL INFO ──
    info_frame = tk.Frame(frame, bg=PANEL_BG, bd=0, relief="flat")
    info_frame.pack(fill="x", padx=30, pady=(0, 10))

    tk.Label(info_frame, text="Session Info", font=("Helvetica", 14, "bold"),
             bg=PANEL_BG, fg=NAVY).pack(anchor="w", padx=15, pady=(10, 5))
    tk.Label(info_frame, text=f"Base Model:  {base_model_path_used}",
             font=("Helvetica", 10), bg=PANEL_BG, fg=TEXT_DARK).pack(anchor="w", padx=15)
    tk.Label(info_frame, text=f"Session Model:  {session_model_path}",
             font=("Helvetica", 10), bg=PANEL_BG, fg=TEXT_DARK).pack(anchor="w", padx=15)
    tk.Label(info_frame, text=f"Test Folder:  {semi_test_folder}",
             font=("Helvetica", 10), bg=PANEL_BG, fg=TEXT_DARK).pack(anchor="w", padx=15, pady=(0, 10))

    # ── HOLDOUT RESULTS ──
    if holdout_results:
        hold_frame = tk.Frame(frame, bg=PANEL_BG)
        hold_frame.pack(fill="x", padx=30, pady=(0, 10))

        acc = holdout_results['accuracy']
        acc_color = SUCCESS_GREEN if acc >= 0.6 else ERROR_RED
        tk.Label(hold_frame, text="Holdout Validation", font=("Helvetica", 14, "bold"),
                 bg=PANEL_BG, fg=NAVY).pack(anchor="w", padx=15, pady=(10, 5))
        tk.Label(hold_frame, text=f"Accuracy: {acc:.1%}",
                 font=("Helvetica", 18, "bold"), bg=PANEL_BG, fg=acc_color).pack(anchor="w", padx=15)

        for name, cls_acc in holdout_results['per_class'].items():
            c = CMD_COLORS.get(name, TEXT_DARK)
            tk.Label(hold_frame, text=f"  {name}: {cls_acc:.0%}",
                     font=("Helvetica", 11), bg=PANEL_BG, fg=c).pack(anchor="w", padx=15)

        # Confusion matrix as text
        cm = holdout_results['confusion_matrix']
        cm_text = "Confusion Matrix (rows=True, cols=Predicted):\n"
        cm_text += f"{'':>12s}" + "".join(f" {n[:5]:>6s}" for n in LABEL_NAMES) + "\n"
        for i in range(NUM_CLASSES):
            row = f"{LABEL_NAMES[i]:>12s}"
            for j in range(NUM_CLASSES):
                if i == j: row += f"  [{cm[i][j]:3d}]"
                else: row += f"   {cm[i][j]:3d} "
            total = np.sum(cm[i])
            row += f"   {cm[i][i]/total:.0%}" if total > 0 else ""
            cm_text += row + "\n"

        cm_label = tk.Label(hold_frame, text=cm_text, font=("Courier", 9),
                           bg=PANEL_BG, fg=TEXT_DARK, justify="left")
        cm_label.pack(anchor="w", padx=15, pady=(5, 10))

    # ── SEPARATOR ──
    tk.Frame(frame, bg=NEON_FUCHSIA, height=2).pack(fill="x", padx=30, pady=10)

    # ── SEMI REAL-TIME TEST ──
    test_frame = tk.Frame(frame, bg=BG_WHITE)
    test_frame.pack(fill="x", padx=30, pady=(0, 10))

    tk.Label(test_frame, text="Semi Real-Time Test", font=("Helvetica", 16, "bold"),
             bg=BG_WHITE, fg=NAVY).pack(anchor="w", pady=(0, 10))

    # --- Car connection ---
    car_frame = tk.Frame(test_frame, bg=PANEL_BG)
    car_frame.pack(fill="x", pady=(0, 10))

    car = CarController()
    car_status_var = tk.StringVar(value="Not connected")

    tk.Label(car_frame, text="Car Control:", font=("Helvetica", 11, "bold"),
             bg=PANEL_BG, fg=NAVY).pack(side="left", padx=10, pady=8)
    car_status_label = tk.Label(car_frame, textvariable=car_status_var,
                                font=("Helvetica", 10), bg=PANEL_BG, fg=TEXT_GRAY)
    car_status_label.pack(side="left", padx=5)

    def toggle_car():
        global CAR_ENABLED
        CAR_ENABLED = not CAR_ENABLED
        if CAR_ENABLED:
            ok, msg = car.test_connection()
            car_status_var.set(f"{'✓' if ok else '✗'} {msg}")
            car_btn.config(text="Disconnect Car", bg=ERROR_RED)
        else:
            car_status_var.set("Disabled")
            car_btn.config(text="Connect Car", bg=NEON_BLUE)

    car_btn = tk.Button(car_frame, text="Connect Car", font=("Helvetica", 10, "bold"),
                        bg=NEON_BLUE, fg="white", bd=0, padx=15, pady=5,
                        cursor="hand2", command=toggle_car)
    car_btn.pack(side="right", padx=10, pady=5)

    # --- Run All Button ---
    all_results_frame = tk.Frame(test_frame, bg=BG_WHITE)
    all_results_scroll = tk.Scrollbar(all_results_frame)
    all_results_text = tk.Text(all_results_frame, height=15, font=("Courier", 10),
                               bg=PANEL_BG, fg=TEXT_DARK, bd=0, wrap="none",
                               yscrollcommand=all_results_scroll.set)
    all_results_scroll.config(command=all_results_text.yview)

    def run_all_predictions():
        if not os.path.exists(semi_test_folder):
            all_results_text.insert("end", "ERROR: Test folder not found!\n")
            return

        all_results_text.delete("1.0", "end")
        csv_files = sorted([f for f in os.listdir(semi_test_folder) if f.endswith('.csv')])

        if not csv_files:
            all_results_text.insert("end", "No CSV files found in test folder.\n")
            return

        correct, total = 0, 0
        cm_all = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        results_log = []

        for fname in csv_files:
            fpath = os.path.join(semi_test_folder, fname)
            true_label = fname.lower().replace('.csv', '').split('_')[0]
            if true_label not in LABEL_MAP: continue

            cmd, conf, probs = preprocess_and_predict_csv(fpath, session_model)
            if cmd is None: continue

            true_idx = LABEL_MAP[true_label]
            pred_idx = LABEL_MAP[cmd]
            cm_all[true_idx][pred_idx] += 1
            is_correct = cmd == true_label
            if is_correct: correct += 1
            total += 1

            mark = "✓" if is_correct else "✗"
            line = f"  {mark} {fname:<25s} True: {true_label:<10s} Pred: {cmd:<10s} ({conf:.0%})\n"
            all_results_text.insert("end", line)
            results_log.append(line)

        # Summary
        if total > 0:
            acc = correct / total
            summary = f"\n  ═══ ACCURACY: {correct}/{total} = {acc:.1%} ═══\n\n"
            all_results_text.insert("end", summary)

            # CM text
            cm_text = "  Confusion Matrix:\n"
            cm_text += f"  {'':>12s}" + "".join(f" {n[:5]:>6s}" for n in LABEL_NAMES) + "\n"
            for i in range(NUM_CLASSES):
                row = f"  {LABEL_NAMES[i]:>12s}"
                for j in range(NUM_CLASSES):
                    if i == j: row += f"  [{cm_all[i][j]:3d}]"
                    else: row += f"   {cm_all[i][j]:3d} "
                t = np.sum(cm_all[i])
                row += f"   {cm_all[i][i]/t:.0%}" if t > 0 else ""
                cm_text += row + "\n"
            all_results_text.insert("end", cm_text)

            # Save report
            report_path = os.path.join(DEBUG_FOLDER, "semi_test_report.txt")
            os.makedirs(DEBUG_FOLDER, exist_ok=True)
            with open(report_path, 'w') as f:
                f.writelines(results_log)
                f.write(summary)
                f.write(cm_text)
            all_results_text.insert("end", f"\n  Report saved: {report_path}\n")

        all_results_text.see("end")
        root.update()

    run_all_btn = tk.Button(test_frame, text="▶  Run All Predictions",
                            font=("Helvetica", 13, "bold"), bg=NAVY, fg="white",
                            activebackground=NEON_NAVY, activeforeground="white",
                            bd=0, padx=20, pady=10, cursor="hand2",
                            command=run_all_predictions)
    run_all_btn.pack(fill="x", pady=(0, 5))
    all_results_frame.pack(fill="x", pady=(0, 10))
    all_results_scroll.pack(side="right", fill="y")
    all_results_text.pack(side="left", fill="x", expand=True)

    # ── SEPARATOR ──
    tk.Frame(frame, bg=NEON_BLUE, height=1).pack(fill="x", padx=30, pady=10)

    # --- File by File ---
    manual_frame = tk.Frame(frame, bg=BG_WHITE)
    manual_frame.pack(fill="x", padx=30, pady=(0, 10))

    tk.Label(manual_frame, text="Test File by File", font=("Helvetica", 14, "bold"),
             bg=BG_WHITE, fg=NAVY).pack(anchor="w", pady=(0, 5))

    # File list
    file_list_frame = tk.Frame(manual_frame, bg=BG_WHITE)
    file_list_frame.pack(fill="x")

    file_scrollbar = tk.Scrollbar(file_list_frame)
    file_scrollbar.pack(side="right", fill="y")

    file_listbox = tk.Listbox(file_list_frame, height=8, font=("Courier", 10),
                               bg=PANEL_BG, fg=TEXT_DARK, selectbackground=NEON_BLUE,
                               selectforeground="white", bd=0, highlightthickness=1,
                               highlightcolor=NEON_BLUE, yscrollcommand=file_scrollbar.set)
    file_listbox.pack(side="left", fill="x", expand=True)
    file_scrollbar.config(command=file_listbox.yview)

    # Populate file list
    test_csv_files = []
    if os.path.exists(semi_test_folder):
        test_csv_files = sorted([f for f in os.listdir(semi_test_folder) if f.endswith('.csv')])
        for f in test_csv_files:
            file_listbox.insert(tk.END, f"  {f}")
    if not test_csv_files:
        file_listbox.insert(tk.END, "  No test files found")

    # Result display
    result_display = tk.Frame(manual_frame, bg=PANEL_BG)
    result_display.pack(fill="x", pady=(10, 0))

    pred_var = tk.StringVar(value="—")
    conf_var = tk.StringVar(value="")
    file_var = tk.StringVar(value="")

    tk.Label(result_display, textvariable=pred_var, font=("Helvetica", 24, "bold"),
             bg=PANEL_BG, fg=NAVY).pack(pady=(10, 0))
    tk.Label(result_display, textvariable=conf_var, font=("Helvetica", 14),
             bg=PANEL_BG, fg=TEXT_GRAY).pack()
    tk.Label(result_display, textvariable=file_var, font=("Helvetica", 10),
             bg=PANEL_BG, fg=TEXT_GRAY).pack(pady=(0, 5))

    # Probability bars
    bar_frame = tk.Frame(result_display, bg=PANEL_BG)
    bar_frame.pack(fill="x", padx=20, pady=(0, 10))
    bar_widgets = {}
    for name in LABEL_NAMES:
        row = tk.Frame(bar_frame, bg=PANEL_BG)
        row.pack(fill="x", pady=2)
        lbl = tk.Label(row, text=f"{name:>10s}", font=("Courier", 10),
                       bg=PANEL_BG, fg=TEXT_GRAY, width=10, anchor="e")
        lbl.pack(side="left")
        bar_bg = tk.Frame(row, bg="#E0E0E0", height=16, width=300)
        bar_bg.pack(side="left", padx=(8, 4))
        bar_bg.pack_propagate(False)
        bar_fill = tk.Frame(bar_bg, bg=TEXT_GRAY, height=16, width=0)
        bar_fill.place(x=0, y=0)
        pct = tk.Label(row, text="  0%", font=("Courier", 10),
                       bg=PANEL_BG, fg=TEXT_GRAY, width=5, anchor="w")
        pct.pack(side="left")
        bar_widgets[name] = (lbl, bar_bg, bar_fill, pct)

    car_action_var = tk.StringVar(value="")
    tk.Label(result_display, textvariable=car_action_var, font=("Helvetica", 11),
             bg=PANEL_BG, fg=TEXT_GRAY).pack(pady=(0, 10))

    def predict_selected():
        sel = file_listbox.curselection()
        if not sel or not test_csv_files: return
        fname = test_csv_files[sel[0]]
        fpath = os.path.join(semi_test_folder, fname)

        true_label = fname.lower().replace('.csv', '').split('_')[0]
        cmd, conf, probs = preprocess_and_predict_csv(fpath, session_model)

        if cmd is None:
            pred_var.set("FILE ERROR")
            return

        color = CMD_COLORS.get(cmd, NAVY)
        is_correct = cmd == true_label
        mark = "✓ CORRECT" if is_correct else "✗ WRONG"
        pred_var.set(f"{cmd.upper()}  ({mark})")
        conf_var.set(f"Confidence: {conf:.1%}")
        file_var.set(f"File: {fname}  |  True: {true_label}")

        for i, name in enumerate(LABEL_NAMES):
            lbl, bar_bg, bar_fill, pct = bar_widgets[name]
            p = probs[i]
            w = int(p * 300)
            c = CMD_COLORS.get(name, TEXT_GRAY)
            if name == cmd:
                bar_fill.config(bg=c); lbl.config(fg=c); pct.config(text=f"{p:.0%}", fg=c)
            else:
                bar_fill.config(bg="#CCCCCC"); lbl.config(fg=TEXT_GRAY); pct.config(text=f"{p:.0%}", fg=TEXT_GRAY)
            bar_fill.place(x=0, y=0, width=max(1, w), height=16)

        # Send to car
        if CAR_ENABLED:
            car_action_var.set(f"Sending: {cmd.upper()}...")
            def car_cb(msg): car_action_var.set(msg)
            car.send_command(cmd, callback=car_cb)
        else:
            car_action_var.set(f"[CAR DISABLED] Would send: {cmd.upper()}")

        root.update()

    predict_btn = tk.Button(manual_frame, text="⚡  Predict Selected File",
                            font=("Helvetica", 12, "bold"), bg=NEON_FUCHSIA, fg="white",
                            activebackground="#CC00CC", bd=0, padx=15, pady=8,
                            cursor="hand2", command=predict_selected)
    predict_btn.pack(fill="x", pady=(10, 5))

    # Enter shortcut
    root.bind("<Return>", lambda e: predict_selected())

    # ── FOOTER ──
    tk.Frame(frame, bg=NAVY, height=2).pack(fill="x", padx=30, pady=(15, 0))
    tk.Label(frame, text="NeuroLink — Brain-Computer Interface  |  Graduation Project 2026",
             font=("Helvetica", 9), bg=BG_WHITE, fg=TEXT_GRAY).pack(pady=(5, 15))

    root.mainloop()


# ================================================================
# SECTION 9: MAIN
# ================================================================

def main():
    print("\n" + "=" * 60)
    print("  NEUROLINK — Session Calibration BCI System (v5)")
    print("=" * 60)

    if not os.path.exists(BASE_MODEL_PATH):
        print(f"\n  ERROR: {BASE_MODEL_PATH} not found!")
        print(f"  Place the pretrained model in this folder.")
        return

    print(f"\n  Base model: {BASE_MODEL_PATH}")
    print(f"""
  Options:
    [1] Full Pipeline    (record calibration -> fine-tune -> test)
    [2] Existing Calib   (load folder -> fine-tune -> test)
    [3] Direct Predict   (no calibration, use base model on CSV files)
    [0] Exit
    """)

    choice = input("  Enter choice (1/2/3/0): ").strip()

    if choice == '0':
        return

    elif choice == '1':
        # === FULL PIPELINE ===
        print("\n  Connecting to NeuroPawn Knight...")
        board = KnightBoard(SERIAL_PORT, NUM_CHANNELS)
        board.start_stream()
        time.sleep(5)

        calib_folder = run_calibration(board)
        board.stop_stream()

        print(f"\n  Splitting calibration data...")
        X_train, y_train, X_hold, y_hold, semi_files = split_and_prepare(calib_folder, SEMI_TEST_FOLDER)

        if X_train is None or len(X_train) == 0:
            print("  ERROR: No training data!"); return

        model, model_path, holdout_res = fine_tune_model(
            BASE_MODEL_PATH, X_train, y_train, X_hold, y_hold)

        print(f"\n  Launching GUI...")
        build_results_gui(model, model_path, SEMI_TEST_FOLDER, holdout_res, BASE_MODEL_PATH)

    elif choice == '2':
        # === EXISTING CALIBRATION ===
        # Model selection
        models = [f for f in os.listdir('.') if f.endswith('.h5') and 'neurolink' in f.lower()]
        if models:
            print("  Available models:")
            for m in sorted(models): print(f"    {m}")

        model_input = input(f"\n  Base model [{BASE_MODEL_PATH}]: ").strip()
        base_model = model_input if model_input else BASE_MODEL_PATH
        if not os.path.exists(base_model):
            print(f"  ERROR: {base_model} not found!"); return

        # Calibration folder
        calib_dirs = sorted([d for d in os.listdir('.') if d.startswith('calibration_') and os.path.isdir(d)])
        if calib_dirs:
            print("\n  Available calibration folders:")
            for d in calib_dirs:
                n = len([f for f in os.listdir(d) if f.endswith('.csv')])
                print(f"    {d} ({n} files)")

        calib_folder = input("\n  Calibration folder: ").strip()
        if not os.path.exists(calib_folder):
            print(f"  ERROR: {calib_folder} not found!"); return

        semi_folder = f"semi_test_{SESSION_TIMESTAMP}"
        print(f"\n  Splitting calibration data...")
        X_train, y_train, X_hold, y_hold, semi_files = split_and_prepare(calib_folder, semi_folder)

        if X_train is None or len(X_train) == 0:
            print("  ERROR: No training data!"); return

        model, model_path, holdout_res = fine_tune_model(
            base_model, X_train, y_train, X_hold, y_hold)

        print(f"\n  Launching GUI...")
        build_results_gui(model, model_path, semi_folder, holdout_res, base_model)

    elif choice == '3':
        # === DIRECT PREDICTION (no calibration) ===
        models = [f for f in os.listdir('.') if f.endswith('.h5') and 'neurolink' in f.lower()]
        if models:
            print("  Available models:")
            for m in sorted(models): print(f"    {m}")

        model_input = input(f"\n  Model to use [{BASE_MODEL_PATH}]: ").strip()
        model_path = model_input if model_input else BASE_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"  ERROR: {model_path} not found!"); return

        csv_folder = input("  CSV folder to test: ").strip()
        if not os.path.exists(csv_folder):
            print(f"  ERROR: {csv_folder} not found!"); return

        print(f"\n  Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"    Loaded. Params: {model.count_params():,}")

        build_results_gui(model, model_path, csv_folder, None, model_path)

    else:
        print("  Invalid choice.")


if __name__ == "__main__":
    main()
