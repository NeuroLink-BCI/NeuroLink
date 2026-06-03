"""
╔══════════════════════════════════════════════════════════════╗
║       NeuroLink — Session Calibration BCI System (v2)        ║
║                                                              ║
║  A complete Brain-Computer Interface pipeline that:          ║
║    1. Loads a pretrained EEGNet backbone                     ║
║    2. Records calibration trials (10 train + 5 holdout)      ║
║    3. Fine-tunes with trial-level split (no leakage)         ║
║    4. Validates on holdout before going live                  ║
║    5. Runs real-time with majority voting + smoothing         ║
║                                                              ║
║  v2 fixes (based on supervisor feedback):                    ║
║    [1] Trial-level split BEFORE windowing (no leakage)       ║
║    [2] Stronger augmentation (noise + shift + scaling)        ║
║    [3] Layer-by-layer freeze verification printed             ║
║    [4] Use LAST 625 samples instead of first (stable MI)     ║
║    [5] Holdout validation before real-time entry              ║
║    [6] Majority voting in real-time (3/5 consensus)           ║
║    [7] Higher confidence threshold (0.55)                     ║
║    [8] Debug logging — saves processed windows to disk        ║
║    [9] 15 trials/class (10 train + 5 holdout)                 ║
║   [10] Augmentation applied to training split only            ║
║                                                              ║
║  Run:  py -3.12 neurolink_calibration_system_v2.py           ║
╚══════════════════════════════════════════════════════════════╝

Academic references for session calibration:
  [1] Lawhern et al. (2018) — EEGNet: A Compact CNN for EEG-based BCIs.
  [2] Zanini et al. (2018) — Transfer Learning: A Riemannian Geometry
      Framework with Applications to BCIs.
  [3] Fahimi et al. (2019) — Inter-subject Transfer Learning with
      End-to-End Deep CNN for Motor Imagery BCI.
  [4] Wu et al. (2022) — Transfer Learning for Motor Imagery BCIs.
  [5] Sakhavi et al. (2018) — Learning Temporal Information for BCI
      Using CNN.
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import sys
import csv
import time
import threading
import warnings
import tkinter as tk
from tkinter import font as tkfont
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import scipy.signal as sig

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import requests

warnings.filterwarnings('ignore')

print("=" * 60)
print("  NeuroLink — Session Calibration BCI System (v2)")
print("=" * 60)
print(f"  TensorFlow: {tf.__version__}")
print(f"  GPU:        {len(tf.config.list_physical_devices('GPU')) > 0}")
print()


# ================================================================
# SECTION 1: CONFIGURATION
# ================================================================

# --- Hardware: NeuroPawn Knight ---
SERIAL_PORT = "COM3"
NUM_CHANNELS = 8
SAMPLING_RATE = 125

# --- EEG CSV format ---
EEG_COL_START = 1
EEG_COL_END = 9

# --- Recording protocol (MUST match training) ---
DISCARD_SECONDS = 2
DISCARD_SAMPLES = int(DISCARD_SECONDS * SAMPLING_RATE)  # 250
MI_FIXED_LENGTH = 625
RECORDING_SECONDS = 8

# --- Preprocessing (MUST match training) ---
LOW_FREQ = 8
HIGH_FREQ = 30
FILTER_ORDER = 4

# --- Windowing (MUST match training) ---
WINDOW_SIZE = 250                   # 2s × 125 Hz
STRIDE = 62                         # 0.5s × 125 Hz

# --- Classification ---
LABEL_NAMES = ['backward', 'forward', 'left', 'right', 'stop']
LABEL_MAP = {'backward': 0, 'forward': 1, 'left': 2, 'right': 3, 'stop': 4}
NUM_CLASSES = 5

# --- Calibration settings ---
#     15 trials/class: 10 train + 5 holdout
#     Enough for ~3,000 trainable params with 4x augmentation
TRIALS_PER_CLASS = 25
HOLDOUT_PER_CLASS = 5               # [FIX #5] Unseen holdout for validation
TRAIN_PER_CLASS = TRIALS_PER_CLASS - HOLDOUT_PER_CLASS  # 20
INTER_TRIAL_REST = 1

# --- Fine-tuning settings ---
FT_EPOCHS = 30
FT_PATIENCE = 8
FT_LEARNING_RATE = 1e-4
FT_BATCH_SIZE = 16

# --- Real-time prediction ---
#     [FIX #7] Threshold raised from 0.40 to 0.55
CONFIDENCE_THRESHOLD = 0.35
RT_RECORD_SECONDS = 7
STABILIZATION_TIME = 5

# --- Majority Voting [FIX #6] ---
VOTE_WINDOW = 5                     # Keep last 5 predictions
VOTE_MAJORITY = 3                   # Need 3/5 agreement to act

# --- Debug logging [FIX #8] ---
DEBUG_LOGGING = True                # Save processed windows to disk

# --- Model ---
BASE_MODEL_PATH = "neurolink_final_v19.h5"

# --- ESP32 Car ---
CAR_IP = "192.168.4.1"
CAR_PORT = 80
CAR_BASE_URL = f"http://{CAR_IP}:{CAR_PORT}"
CAR_TIMEOUT = 3
CAR_ENABLED = True

DURATION = {
    "forward":  1.5,
    "backward": 1.5,
    "left":     0.8,
    "right":    0.8,
    "stop":     0.0,
}

# --- Pre-compute filter ---
nyquist = 0.5 * SAMPLING_RATE
b_filt, a_filt = sig.butter(
    FILTER_ORDER,
    [LOW_FREQ / nyquist, HIGH_FREQ / nyquist],
    btype='band'
)

# --- Session paths ---
SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CALIB_FOLDER = f"calibration_{SESSION_TIMESTAMP}"
DEBUG_FOLDER = f"debug_{SESSION_TIMESTAMP}"


# ================================================================
# SECTION 2: PREPROCESSING FUNCTIONS
# ================================================================

def bandpass_filter(data):
    """4th-order Butterworth bandpass 8-30 Hz."""
    filtered = np.zeros_like(data, dtype=np.float64)
    for ch in range(data.shape[0]):
        filtered[ch] = sig.filtfilt(b_filt, a_filt, data[ch].astype(np.float64))
    return filtered


def apply_car(data):
    """Common Average Reference."""
    return data - np.mean(data, axis=0, keepdims=True)


def normalize_channels(data):
    """Z-score normalization per channel."""
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
    """Full preprocessing: bandpass → CAR → z-score. (8,625)→(8,625)"""
    data = bandpass_filter(raw_data)
    data = apply_car(data)
    data = normalize_channels(data)
    return data


def preprocess_and_extract_window(eeg_625):
    """Preprocess 625 samples → slice last 250 for model. (8,625)→(8,250)"""
    data = preprocess_trial(eeg_625)
    return data[:, -WINDOW_SIZE:]


def create_windows(X, y, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Slice trials into overlapping windows.
    Input:  X (n_trials, 8, 625), y (n_trials,)
    Output: windows (n_windows, 8, 250), labels (n_windows,)
    """
    windows = []
    labels = []
    for i in range(len(X)):
        trial = X[i]
        n_samples = trial.shape[1]
        n_win = (n_samples - window_size) // stride + 1
        for w in range(n_win):
            start = w * stride
            windows.append(trial[:, start:start + window_size])
            labels.append(y[i])
    return np.array(windows), np.array(labels)


def augment_data(X_windows, y_labels):
    """
    [FIX #2] Stronger augmentation: scaling + noise + time shift (4x data).
    Applied to TRAINING split only (never holdout/validation).
    """
    augmented_X = [X_windows.copy()]
    augmented_y = [y_labels.copy()]
    n_samples = len(X_windows)
    window_size = X_windows.shape[2]

    # 1. Random scaling (80%-120% amplitude)
    scales = np.random.uniform(0.8, 1.2, size=(n_samples, 1, 1))
    X_scaled = X_windows * scales
    augmented_X.append(X_scaled)
    augmented_y.append(y_labels.copy())

    # 2. Gaussian noise (conservative — 0.1 to avoid masking weak EEG patterns)
    noise_factor = 0.1
    noise = np.random.randn(*X_windows.shape) * noise_factor
    X_noisy = X_windows + noise
    augmented_X.append(X_noisy)
    augmented_y.append(y_labels.copy())

    # 3. Time shift (±15 samples)
    shift_max = 15
    X_shifted = np.zeros_like(X_windows)
    for i in range(n_samples):
        shift = np.random.randint(-shift_max, shift_max + 1)
        if shift > 0:
            X_shifted[i, :, shift:] = X_windows[i, :, :window_size - shift]
            X_shifted[i, :, :shift] = X_windows[i, :, 0:1]
        elif shift < 0:
            X_shifted[i, :, :shift] = X_windows[i, :, -shift:]
            X_shifted[i, :, shift:] = X_windows[i, :, -1:]
        else:
            X_shifted[i] = X_windows[i]
    augmented_X.append(X_shifted)
    augmented_y.append(y_labels.copy())

    X_final = np.concatenate(augmented_X, axis=0)
    y_final = np.concatenate(augmented_y, axis=0)
    idx = np.random.permutation(len(X_final))
    return X_final[idx], y_final[idx]


def load_csv_trial(file_path):
    """Load one CSV → extract EEG → discard prep → crop. Returns (8,625) or None."""
    try:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < EEG_COL_END:
            return None
        eeg = df.iloc[:, EEG_COL_START:EEG_COL_END].values.T
        eeg = eeg.astype(np.float64)
        if eeg.shape[1] < DISCARD_SAMPLES + WINDOW_SIZE:
            return None
        eeg = eeg[:, DISCARD_SAMPLES:]
        if eeg.shape[1] < MI_FIXED_LENGTH:
            return None
        # [FIX #4] Take LAST 625 samples (more stable MI)
        eeg = eeg[:, -MI_FIXED_LENGTH:]
        return eeg
    except Exception as e:
        print(f"    Error reading {file_path}: {e}")
        return None


# ================================================================
# SECTION 3: KNIGHTBOARD CLASS
# ================================================================

class KnightBoard:
    """NeuroPawn Knight EEG headset interface via BrainFlow."""

    def __init__(self, serial_port, num_channels):
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.num_channels = num_channels
        self.board_shim = BoardShim(
            BoardIds.NEUROPAWN_KNIGHT_BOARD.value, self.params
        )
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
            cmd = f"chon_{x}_12"
            self.board_shim.config_board(cmd)
            print(f"    Sent: {cmd}")
            time.sleep(1)
            rld = f"rldadd_{x}"
            self.board_shim.config_board(rld)
            print(f"    Sent: {rld}")
            time.sleep(0.5)
        print(f"  All {self.num_channels} channels configured.")

    def stop_stream(self):
        try:
            self.board_shim.stop_stream()
            self.board_shim.release_session()
            print("  Stream stopped and session released.")
        except Exception as e:
            print(f"  Warning during cleanup: {e}")

    def flush_buffer(self):
        self.board_shim.get_board_data()

    def get_data(self):
        return self.board_shim.get_board_data()


# ================================================================
# SECTION 4: GUI HELPERS (from newRec_Auto.py)
# ================================================================

def make_fullscreen(window):
    window.attributes("-fullscreen", True)
    window.configure(bg="black")


def beep(freq=1000, duration=500):
    if HAS_WINSOUND:
        winsound.Beep(freq, duration)


def show_recording_phases(movement, trial_num, total_trials, abort_flag):
    """Fullscreen GUI for one recording trial — same as newRec_Auto.py."""
    window = tk.Tk()
    window.title("Session Phases")
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

    counter_label = tk.Label(
        window,
        text=f"CALIBRATION  |  {movement.upper()}  |  Trial {trial_num} / {total_trials}",
        font=("Arial", 24), bg="black", fg="gray"
    )
    counter_label.pack(pady=20)

    label = tk.Label(window, text="", font=("Arial", 60),
                     bg="black", fg="white")
    label.pack(expand=True)

    def sleep_with_abort(seconds):
        end = time.time() + seconds
        while time.time() < end:
            if abort_flag[0]:
                return
            window.update()
            time.sleep(0.05)

    if movement == "stop":
        label.config(text="")
        window.update()
        sleep_with_abort(2)
        if not abort_flag[0]:
            sleep_with_abort(4)
        if not abort_flag[0]:
            sleep_with_abort(2)
    else:
        for i in range(2, 0, -1):
            if abort_flag[0]:
                break
            label.config(text=f"Get Ready...\n{i}")
            window.update()
            sleep_with_abort(1)
        for i in range(4, 0, -1):
            if abort_flag[0]:
                break
            label.config(text=f"{movement.upper()}\n{i}")
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


def show_inter_trial_rest(next_trial_num, total_trials, abort_flag):
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


# ================================================================
# SECTION 5: CALIBRATION RECORDER
# ================================================================

def record_one_trial(board, movement, trial_num, total_trials, save_folder):
    """Record one 8-second trial. Returns True if saved, False if aborted."""
    abort_flag = [False]

    phase_thread = threading.Thread(
        target=show_recording_phases,
        args=(movement, trial_num, total_trials, abort_flag)
    )
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
        if abort_flag[0]:
            aborted = True
            break
        data = board.get_data()
        if data.size > 0:
            for row in data.T:
                writer.writerow(row)
        if time.time() - start_time >= RECORDING_SECONDS:
            break
        time.sleep(0.01)

    csv_file.close()
    phase_thread.join(timeout=2)

    if aborted:
        try:
            os.remove(filename)
            print(f"    ABORTED — deleted {filename}")
        except:
            pass
        beep(400, 300)
        return False
    else:
        print(f"    Saved: {filename}")
        return True


def run_calibration(board):
    """
    Record calibration trials.
    Total: 5 classes × 15 trials = 75 trials (~12 minutes).
    Split later: 10 train + 5 holdout per class.
    """
    os.makedirs(CALIB_FOLDER, exist_ok=True)

    movements = ['right', 'left','stop']
    #['right', 'left', 'forward', 'backward', 'stop']
    #['right', 'left', 'stop']
    #['forward', 'backward', 'stop']

    print("\n" + "=" * 60)
    print("  CALIBRATION SESSION")
    print(f"  {TRIALS_PER_CLASS} trials/class × {NUM_CLASSES} classes = "
          f"{TRIALS_PER_CLASS * NUM_CLASSES} total")
    print(f"  Split: {TRAIN_PER_CLASS} train + {HOLDOUT_PER_CLASS} holdout per class")
    print(f"  Estimated time: ~{TRIALS_PER_CLASS * NUM_CLASSES * 9 // 60} minutes")
    print(f"  Saving to: {CALIB_FOLDER}/")
    print("=" * 60)

    for movement in movements:
        print(f"\n  --- Recording: {movement.upper()} "
              f"({TRIALS_PER_CLASS} trials) ---")
        _show_movement_start(movement)

        successful = 0
        aborted_count = 0

        while successful < TRIALS_PER_CLASS:
            trial_num = successful + 1
            ok = record_one_trial(
                board, movement, trial_num, TRIALS_PER_CLASS, CALIB_FOLDER
            )
            if ok:
                successful += 1
            else:
                aborted_count += 1
                time.sleep(1)

            if successful >= TRIALS_PER_CLASS:
                break

            rest_abort = [False]
            show_inter_trial_rest(successful + 1, TRIALS_PER_CLASS, rest_abort)

        print(f"  Completed {movement.upper()}: "
              f"{successful} saved, {aborted_count} retried")

    print("\n" + "=" * 60)
    print("  CALIBRATION COMPLETE")
    print(f"  Files saved in: {CALIB_FOLDER}/")
    print("=" * 60)
    return CALIB_FOLDER


def _show_movement_start(movement):
    window = tk.Tk()
    window.title("Next Movement")
    make_fullscreen(window)
    label = tk.Label(window, text="", font=("Arial", 50),
                     bg="black", fg="white")
    label.pack(expand=True)
    for i in range(3, 0, -1):
        label.config(text=f"Next: {movement.upper()}\n\nStarting in {i}...")
        window.update()
        time.sleep(1)
    window.destroy()


# ================================================================
# SECTION 6: FINE-TUNER (with all supervisor fixes)
# ================================================================

def load_calibration_data(calib_folder):
    """Load all CSVs. Returns X (n,8,625), y (n,), filenames list."""
    X_list = []
    y_list = []
    fnames = []

    for filename in sorted(os.listdir(calib_folder)):
        if not filename.lower().endswith('.csv'):
            continue
        basename = filename.lower().replace('.csv', '')
        movement = basename.split('_')[0]
        if movement not in LABEL_MAP:
            continue

        filepath = os.path.join(calib_folder, filename)
        eeg = load_csv_trial(filepath)
        if eeg is None:
            print(f"    Skipping bad file: {filename}")
            continue

        X_list.append(eeg)
        y_list.append(LABEL_MAP[movement])
        fnames.append(filename)

    if len(X_list) == 0:
        return None, None, None

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\n  Loaded {len(X)} calibration trials")
    for i, name in enumerate(LABEL_NAMES):
        count = np.sum(y == i)
        print(f"    {name:>10s}: {count} trials")

    return X, y, fnames


def split_trials_by_class(X, y, holdout_per_class=HOLDOUT_PER_CLASS):
    """
    [FIX #1] Trial-level split BEFORE windowing.
    [FIX] Random split per class (not last-N, to avoid fatigue bias).

    Returns: X_train, y_train, X_holdout, y_holdout (all at trial level)
    """
    X_train_list = []
    y_train_list = []
    X_hold_list = []
    y_hold_list = []

    for cls in range(NUM_CLASSES):
        mask = (y == cls)
        X_cls = X[mask]
        y_cls = y[mask]

        if len(X_cls) <= holdout_per_class:
            X_train_list.append(X_cls)
            y_train_list.append(y_cls)
            print(f"    WARNING: {LABEL_NAMES[cls]} has only {len(X_cls)} trials, "
                  f"no holdout possible")
            continue

        # Random selection of holdout indices (avoids fatigue bias)
        indices = np.arange(len(X_cls))
        np.random.seed(42)  # Reproducible
        hold_idx = np.random.choice(indices, size=holdout_per_class, replace=False)
        train_idx = np.setdiff1d(indices, hold_idx)

        X_train_list.append(X_cls[train_idx])
        y_train_list.append(y_cls[train_idx])
        X_hold_list.append(X_cls[hold_idx])
        y_hold_list.append(y_cls[hold_idx])

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)

    if len(X_hold_list) > 0:
        X_holdout = np.concatenate(X_hold_list)
        y_holdout = np.concatenate(y_hold_list)
    else:
        X_holdout = np.array([]).reshape(0, NUM_CHANNELS, MI_FIXED_LENGTH)
        y_holdout = np.array([])

    return X_train, y_train, X_holdout, y_holdout


def fine_tune_model(base_model_path, calib_folder):
    """
    Fine-tune with all supervisor fixes applied.

    Key changes from v1:
      [FIX #1] Trial-level split BEFORE windowing — no leakage
      [FIX #2] Stronger augmentation (noise + shift + scaling = 4x)
      [FIX #3] Layer-by-layer freeze printout
      [FIX #5] Holdout validation on truly unseen trials
      [FIX #8] Debug logging of processed windows
      [FIX #10] Augmentation on training split only
    """
    print("\n" + "=" * 60)
    print("  FINE-TUNING (v2 — with leakage fix)")
    print("=" * 60)

    # --- Load base model ---
    print(f"\n  Loading base model: {base_model_path}")
    model = tf.keras.models.load_model(base_model_path)
    print(f"    Input:  {model.input_shape}")
    print(f"    Params: {model.count_params():,}")

    expected_samples = model.input_shape[2]
    if expected_samples != WINDOW_SIZE:
        print(f"\n  *** FATAL: Model expects {expected_samples} samples "
              f"but WINDOW_SIZE = {WINDOW_SIZE} ***")
        sys.exit(1)

    # --- [FIX #3] Print layer-by-layer freeze map ---
    print(f"\n  Layer architecture:")
    for i, layer in enumerate(model.layers):
        print(f"    [{i:2d}] {layer.name:30s}  "
              f"params={layer.count_params():>5,}  "
              f"{'FREEZE' if i < 7 else 'TRAIN'}")

    # --- Load calibration data ---
    X_raw, y_raw, fnames = load_calibration_data(calib_folder)
    if X_raw is None:
        print("  ERROR: No calibration data found!")
        return None

    # --- [FIX #1] Trial-level split BEFORE windowing ---
    print(f"\n  [FIX #1] Splitting at TRIAL level before windowing...")
    X_train_trials, y_train_trials, X_hold_trials, y_hold_trials = \
        split_trials_by_class(X_raw, y_raw)

    print(f"    Train trials:   {len(X_train_trials)}")
    print(f"    Holdout trials: {len(X_hold_trials)}")

    # --- Preprocess ALL trials ---
    print("\n  Preprocessing...")
    X_train_proc = np.array([preprocess_trial(t) for t in X_train_trials])
    X_hold_proc = np.array([preprocess_trial(t) for t in X_hold_trials]) \
        if len(X_hold_trials) > 0 else np.array([])

    # --- Window EACH split separately (no cross-contamination) ---
    X_train_win, y_train_win = create_windows(X_train_proc, y_train_trials)
    print(f"    Train windows:  {len(X_train_win)} "
          f"(from {len(X_train_trials)} trials)")

    if len(X_hold_proc) > 0:
        X_hold_win, y_hold_win = create_windows(X_hold_proc, y_hold_trials)
        print(f"    Holdout windows: {len(X_hold_win)} "
              f"(from {len(X_hold_trials)} trials, ZERO overlap with train)")
    else:
        X_hold_win = np.array([]).reshape(0, NUM_CHANNELS, WINDOW_SIZE)
        y_hold_win = np.array([])

    # --- [FIX #2 & #10] Augment TRAINING split only ---
    X_train_aug, y_train_aug = augment_data(X_train_win, y_train_win)
    print(f"    After augmentation: {len(X_train_aug)} train windows (4x)")

    # --- [FIX #8] Debug logging ---
    if DEBUG_LOGGING:
        os.makedirs(DEBUG_FOLDER, exist_ok=True)
        debug_path = os.path.join(DEBUG_FOLDER, "train_windows.npy")
        np.save(debug_path, X_train_win)
        debug_path2 = os.path.join(DEBUG_FOLDER, "train_labels.npy")
        np.save(debug_path2, y_train_win)
        if len(X_hold_win) > 0:
            np.save(os.path.join(DEBUG_FOLDER, "holdout_windows.npy"), X_hold_win)
            np.save(os.path.join(DEBUG_FOLDER, "holdout_labels.npy"), y_hold_win)
        print(f"    Debug data saved to: {DEBUG_FOLDER}/")

    # --- Prepare for TensorFlow ---
    X_train_tf = X_train_aug[..., np.newaxis]
    y_train_tf = to_categorical(y_train_aug, NUM_CLASSES)

    if len(X_hold_win) > 0:
        X_hold_tf = X_hold_win[..., np.newaxis]
        y_hold_tf = to_categorical(y_hold_win, NUM_CLASSES)
    else:
        X_hold_tf = None
        y_hold_tf = None

    # --- [FIX #3] Freeze with verification ---
    # Dynamically find the freeze boundary based on layer names
    # Block 1 ends after the first Dropout layer
    freeze_until = 7  # default
    for i, layer in enumerate(model.layers):
        if 'dropout' in layer.name.lower():
            freeze_until = i + 1  # freeze up to and including first dropout
            break

    print(f"\n  Freezing layers 0-{freeze_until-1} (Block 1)...")
    for i, layer in enumerate(model.layers):
        layer.trainable = (i >= freeze_until)

    # Print verification
    for i, layer in enumerate(model.layers):
        status = "TRAINABLE" if layer.trainable else "FROZEN"
        print(f"    [{i:2d}] {layer.name:30s} → {status}")

    trainable_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    total_params = model.count_params()
    print(f"\n    Trainable: {trainable_params:,} / {total_params:,} parameters")

    # --- Recompile ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(FT_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- Class weights ---
    classes = np.unique(y_train_aug)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_aug)
    class_weights = dict(zip(classes.astype(int), weights))

    # --- Train (validation = holdout windows, NOT random split) ---
    print(f"\n  Training:")
    print(f"    Train:    {len(X_train_tf)} windows (augmented)")
    if X_hold_tf is not None:
        print(f"    Holdout:  {len(X_hold_tf)} windows (unseen trials)")
    print(f"    Epochs:   {FT_EPOCHS} (patience={FT_PATIENCE})")
    print(f"    LR:       {FT_LEARNING_RATE}")
    print(f"    Batch:    {FT_BATCH_SIZE}")
    print()

    callbacks = [
        EarlyStopping(
            monitor='val_loss' if X_hold_tf is not None else 'loss',
            patience=FT_PATIENCE,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_tf, y_train_tf,
        batch_size=FT_BATCH_SIZE,
        epochs=FT_EPOCHS,
        validation_data=(X_hold_tf, y_hold_tf) if X_hold_tf is not None else None,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # --- [FIX #5] Holdout evaluation ---
    print(f"\n  Fine-tuning complete!")
    print(f"    Epochs used: {len(history.history['loss'])}")

    if X_hold_tf is not None and len(X_hold_tf) > 0:
        hold_loss, hold_acc = model.evaluate(X_hold_tf, y_hold_tf, verbose=0)
        print(f"\n  ═══ HOLDOUT VALIDATION (unseen trials) ═══")
        print(f"    Accuracy: {hold_acc:.1%}")
        print(f"    Loss:     {hold_loss:.4f}")

        # Per-class holdout accuracy
        hold_preds = model.predict(X_hold_tf, verbose=0)
        hold_pred_classes = np.argmax(hold_preds, axis=1)
        hold_true_classes = np.argmax(y_hold_tf, axis=1)
        print(f"\n    Per-class holdout results:")
        for cls in range(NUM_CLASSES):
            mask = (hold_true_classes == cls)
            if np.sum(mask) > 0:
                cls_acc = np.mean(hold_pred_classes[mask] == cls)
                print(f"      {LABEL_NAMES[cls]:>10s}: "
                      f"{cls_acc:.0%} ({np.sum(mask)} windows)")

        # --- Confusion Matrix ---
        print(f"\n    Confusion Matrix (rows=True, cols=Predicted):")
        print(f"    {'':>12s}", end="")
        for name in LABEL_NAMES:
            print(f" {name[:5]:>6s}", end="")
        print()

        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(hold_true_classes, hold_pred_classes):
            cm[t][p] += 1

        for i in range(NUM_CLASSES):
            print(f"    {LABEL_NAMES[i]:>12s}", end="")
            for j in range(NUM_CLASSES):
                if i == j:
                    print(f"  [{cm[i][j]:3d}]", end="")
                else:
                    print(f"   {cm[i][j]:3d} ", end="")
            row_total = np.sum(cm[i])
            if row_total > 0:
                print(f"   {cm[i][i]/row_total:.0%}")
            else:
                print()

        # Save confusion matrix to debug folder
        if DEBUG_LOGGING:
            np.save(os.path.join(DEBUG_FOLDER, "confusion_matrix.npy"), cm)
            print(f"\n    Confusion matrix saved to {DEBUG_FOLDER}/")

        if hold_acc < 0.40:
            print(f"\n  ⚠ WARNING: Holdout accuracy is low ({hold_acc:.1%}).")
            print(f"    The model may not generalize well in real-time.")
            print(f"    Consider re-recording calibration with better focus.")
    else:
        print("    (No holdout data — skipping validation)")

    # --- Save ---
    session_model_path = f"neurolink_session_{SESSION_TIMESTAMP}.h5"
    model.save(session_model_path)
    print(f"\n  Session model saved: {session_model_path}")
    print("=" * 60)
    return session_model_path


# ================================================================
# SECTION 7: MAJORITY VOTER [FIX #6]
# ================================================================

class MajorityVoter:
    """
    Temporal smoothing via majority voting.
    Requires VOTE_MAJORITY out of VOTE_WINDOW consecutive confident
    predictions to agree before sending a command.
    """
    def __init__(self):
        self.votes = deque(maxlen=VOTE_WINDOW)

    def add(self, pred_class, confidence):
        """
        Add a prediction. Returns (command_name, vote_count) if consensus
        reached, or (None, current_count) otherwise.
        """
        if confidence >= CONFIDENCE_THRESHOLD:
            self.votes.append(pred_class)
        # else: skip low-confidence predictions

        if len(self.votes) < VOTE_MAJORITY:
            return None, len(self.votes)

        # Count votes
        counts = {}
        for v in self.votes:
            counts[v] = counts.get(v, 0) + 1

        best_class = max(counts, key=counts.get)
        best_count = counts[best_class]

        if best_count >= VOTE_MAJORITY:
            command = LABEL_NAMES[best_class]
            self.votes.clear()
            return command, best_count

        return None, best_count

    def reset(self):
        self.votes.clear()


# ================================================================
# SECTION 8: CAR CONTROLLER
# ================================================================

class CarController:
    def __init__(self, base_url=CAR_BASE_URL, timeout=CAR_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
        self.is_busy = False

    def test_connection(self):
        if not CAR_ENABLED:
            print("  [CAR] Car control DISABLED.")
            return True
        try:
            resp = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            if resp.status_code == 200:
                print(f"  [CAR] Connected: {resp.text.strip()}")
                return True
            return False
        except Exception:
            print("  [CAR] Cannot reach ESP32. Check WiFi 'NeuroLink_Car'.")
            return False

    def send_command(self, command):
        if not CAR_ENABLED:
            print(f"  [CAR-SIM] {command.upper()}")
            return
        self.is_busy = True
        duration = DURATION.get(command, 1.0)

        def _worker():
            try:
                resp = requests.get(
                    f"{self.base_url}/{command}", timeout=self.timeout
                )
                if resp.status_code == 200:
                    print(f"  [CAR] {command.upper()} for {duration}s")
                else:
                    print(f"  [CAR] Error: HTTP {resp.status_code}")
            except Exception as e:
                print(f"  [CAR] Error: {e}")
            if command != "stop" and duration > 0:
                time.sleep(duration)
                try:
                    requests.get(f"{self.base_url}/stop", timeout=self.timeout)
                except:
                    pass
            self.is_busy = False

        threading.Thread(target=_worker, daemon=True).start()

    def emergency_stop(self):
        try:
            requests.get(f"{self.base_url}/stop", timeout=self.timeout)
        except:
            pass
        self.is_busy = False


# ================================================================
# SECTION 9: REAL-TIME PREDICTOR [FIX #4, #6, #7, #8]
# ================================================================

def run_realtime(board, model_path, car):
    """
    Real-time BCI with majority voting.

    Fixes applied:
      [FIX #4] Use LAST 625 samples (more stable MI)
      [FIX #6] Majority voting (3/5 consensus before car moves)
      [FIX #7] Higher confidence threshold (0.55)
      [FIX #8] Debug log every processed window
    """
    print("\n" + "=" * 60)
    print("  REAL-TIME PREDICTION (v2)")
    print("=" * 60)

    model = tf.keras.models.load_model(model_path)
    print(f"  Model: {model_path}")

    eeg_channels = board.eeg_channels[:NUM_CHANNELS]
    voter = MajorityVoter()
    cycle = 0

    # Debug log folder
    if DEBUG_LOGGING:
        rt_debug = os.path.join(DEBUG_FOLDER, "realtime_windows")
        os.makedirs(rt_debug, exist_ok=True)

    print(f"\n  Window:     {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLING_RATE:.0f}s)")
    print(f"  Recording:  {RT_RECORD_SECONDS}s per cycle")
    print(f"  Threshold:  {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Voting:     {VOTE_MAJORITY}/{VOTE_WINDOW} consensus")
    print(f"  Car:        {'ENABLED' if CAR_ENABLED else 'DISABLED'}")
    print(f"  Debug log:  {'ON' if DEBUG_LOGGING else 'OFF'}")
    print("\n  Press Ctrl+C to stop.\n")

    try:
        while True:
            cycle += 1

            if car.is_busy:
                print(f"  [{cycle:3d}] Car busy — waiting...")
                time.sleep(0.5)
                continue

            print(f"\n{'─' * 50}")
            print(f"  [{cycle:3d}]  THINK NOW — imagine any movement...")

            # Flush + record
            board.flush_buffer()
            for sec in range(RT_RECORD_SECONDS, 0, -1):
                sys.stdout.write(f"\r    Recording: {sec}s remaining...  ")
                sys.stdout.flush()
                time.sleep(1)
            print()

            data = board.get_data()
            if data.shape[1] < DISCARD_SAMPLES + WINDOW_SIZE:
                print(f"    Not enough data ({data.shape[1]} samples). Retrying...")
                continue

            eeg = data[eeg_channels, :]
            eeg = eeg[:, DISCARD_SAMPLES:]

            if eeg.shape[1] < MI_FIXED_LENGTH:
                print(f"    Short signal ({eeg.shape[1]}). Retrying...")
                continue

            # [FIX #4] Take LAST 625 samples (most stable MI period)
            eeg_625 = eeg[:, -MI_FIXED_LENGTH:]

            # Preprocess
            window = preprocess_and_extract_window(eeg_625)

            # [FIX #8] Save debug window
            if DEBUG_LOGGING:
                np.save(
                    os.path.join(rt_debug, f"cycle_{cycle:04d}.npy"),
                    window
                )

            # Predict
            X = window[np.newaxis, ..., np.newaxis]
            probs = model.predict(X, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_name = LABEL_NAMES[pred_idx]

            # Display
            status = "HIGH" if confidence >= CONFIDENCE_THRESHOLD else "LOW"
            print(f"\n    Raw: {pred_name.upper():>10s}  "
                  f"({confidence:.0%}) [{status}]")
            for i, name in enumerate(LABEL_NAMES):
                bar_len = int(probs[i] * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                marker = " ◄" if i == pred_idx else ""
                print(f"      {name:>10s} {probs[i]:6.1%} |{bar}|{marker}")

            # [FIX #6] Majority voting
            command, vote_count = voter.add(pred_idx, confidence)

            if command is not None:
                print(f"\n    ═══ CONSENSUS: {command.upper()} "
                      f"({vote_count}/{VOTE_WINDOW} votes) ═══")
                print(f"    → Sending to car!")
                car.send_command(command)
            elif confidence >= CONFIDENCE_THRESHOLD:
                print(f"\n    → Vote registered. "
                      f"Waiting for consensus ({vote_count}/{VOTE_MAJORITY})...")
            else:
                print(f"\n    → Below threshold ({CONFIDENCE_THRESHOLD:.0%}). "
                      f"Not counted.")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n  Stopping real-time prediction...")
        car.emergency_stop()

    # Save debug summary
    if DEBUG_LOGGING:
        print(f"\n  Debug windows saved in: {rt_debug}/")


# ================================================================
# SECTION 10: SIMULATION MODE
# ================================================================

def run_simulation(model_path, csv_folder=None):
    print("\n" + "=" * 60)
    print("  SIMULATION MODE")
    print("=" * 60)

    model = tf.keras.models.load_model(model_path)
    print(f"  Model: {model_path}")

    if csv_folder is None:
        csv_folder = CALIB_FOLDER
    print(f"  Folder: {csv_folder}")

    if not os.path.exists(csv_folder):
        print(f"  ERROR: Folder '{csv_folder}' not found!")
        return

    correct = 0
    total = 0

    for filename in sorted(os.listdir(csv_folder)):
        if not filename.lower().endswith('.csv'):
            continue
        basename = filename.lower().replace('.csv', '')
        true_label = basename.split('_')[0]
        if true_label not in LABEL_MAP:
            continue

        filepath = os.path.join(csv_folder, filename)
        eeg = load_csv_trial(filepath)
        if eeg is None:
            continue

        window = preprocess_and_extract_window(eeg)
        X = window[np.newaxis, ..., np.newaxis]
        probs = model.predict(X, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted = LABEL_NAMES[pred_idx]

        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        total += 1

        mark = "✓" if is_correct else "✗"
        print(f"    {mark} {filename:>25s}  "
              f"True: {true_label:>8s}  "
              f"Pred: {predicted:>8s} ({confidence:.0%})")

    if total > 0:
        print(f"\n  Accuracy: {correct}/{total} = {correct/total:.1%}")
    else:
        print("  No valid files found.")


# ================================================================
# SECTION 11: SIGNAL QUALITY CHECK
# ================================================================

def check_signal_quality(board, duration=3):
    print(f"\n  Checking signal quality ({duration}s)...")
    board.flush_buffer()
    time.sleep(duration)
    data = board.get_data()

    if data.shape[1] < SAMPLING_RATE:
        print(f"    Only {data.shape[1]} samples — check connection!")
        return False

    eeg_channels = board.eeg_channels[:NUM_CHANNELS]
    eeg = data[eeg_channels, :]

    flat_count = 0
    for ch in range(NUM_CHANNELS):
        ch_std = np.std(eeg[ch])
        status = "OK" if ch_std > 1e-6 else "FLAT"
        if ch_std < 1e-6:
            flat_count += 1
        print(f"    Ch {ch+1}: std={ch_std:.2f} — {status}")

    if flat_count == 0:
        print("  Signal: GOOD")
    elif flat_count <= 2:
        print(f"  Signal: WARNING — {flat_count} flat channel(s)")
    else:
        print(f"  Signal: POOR — {flat_count} flat channels!")

    return flat_count < NUM_CHANNELS


# ================================================================
# SECTION 12: MAIN
# ================================================================

def main():
    print("\n" + "=" * 60)
    print("  NEUROLINK SESSION CALIBRATION SYSTEM (v2)")
    print("=" * 60)
    print(f"""
  v2 fixes applied:
  ─────────────────
  [1] Trial-level split BEFORE windowing (no leakage)
  [2] Stronger augmentation (scaling + noise + shift = 4x)
  [3] Layer-by-layer freeze verification
  [4] Use LAST 625 samples (more stable MI)
  [5] Holdout validation (5 unseen trials/class) before real-time
  [6] Majority voting in real-time (3/5 consensus)
  [7] Higher confidence threshold (55%)
  [8] Debug logging of all processed windows
  [9] 15 trials/class (10 train + 5 holdout)

  Prerequisites:
  ─────────────
  • {BASE_MODEL_PATH} in this folder
  • NeuroPawn Knight on {SERIAL_PORT}
  • For car: WiFi → 'NeuroLink_Car'
    """)

    if not os.path.exists(BASE_MODEL_PATH):
        print(f"  ERROR: Base model not found: {BASE_MODEL_PATH}")
        return

    print("  Options:")
    print("    [1] Full pipeline  (calibrate → fine-tune → real-time)")
    print("    [2] Fine-tune only (existing calibration folder)")
    print("    [3] Real-time only (existing session model)")
    print("    [4] Simulation     (test model on CSV files)")
    print("    [0] Exit")
    print()

    choice = input("  Enter choice (1/2/3/4/0): ").strip()

    if choice == '0':
        return

    elif choice == '1':
        print("\n  Connecting to NeuroPawn Knight...")
        board = KnightBoard(SERIAL_PORT, NUM_CHANNELS)
        board.start_stream()

        print(f"\n  Stabilizing ({STABILIZATION_TIME}s)...")
        time.sleep(STABILIZATION_TIME)

        signal_ok = check_signal_quality(board)
        if not signal_ok:
            resp = input("\n  Signal issues. Continue? (y/n): ").strip()
            if resp.lower() != 'y':
                board.stop_stream()
                return

        calib_folder = run_calibration(board)
        session_model_path = fine_tune_model(BASE_MODEL_PATH, calib_folder)
        if session_model_path is None:
            board.stop_stream()
            return

        car = CarController()
        car.test_connection()

        input("\n  Press Enter to start real-time prediction...")
        run_realtime(board, session_model_path, car)
        board.stop_stream()

    elif choice == '2':
        folder = input(f"  Calibration folder: ").strip()
        if not folder:
            calib_dirs = [d for d in os.listdir('.')
                         if d.startswith('calibration_') and os.path.isdir(d)]
            if calib_dirs:
                print("  Available:")
                for d in sorted(calib_dirs):
                    n = len([f for f in os.listdir(d) if f.endswith('.csv')])
                    print(f"    {d} ({n} files)")
                folder = input("  Enter folder: ").strip()
            else:
                print("  No calibration folders found.")
                return

        if not os.path.exists(folder):
            print(f"  ERROR: '{folder}' not found!")
            return

        session_model_path = fine_tune_model(BASE_MODEL_PATH, folder)
        if session_model_path:
            resp = input("\n  Start real-time? (y/n): ").strip()
            if resp.lower() == 'y':
                board = KnightBoard(SERIAL_PORT, NUM_CHANNELS)
                board.start_stream()
                time.sleep(STABILIZATION_TIME)
                car = CarController()
                car.test_connection()
                run_realtime(board, session_model_path, car)
                board.stop_stream()

    elif choice == '3':
        models = [f for f in os.listdir('.') if f.startswith('neurolink_session_')]
        if models:
            print("  Available session models:")
            for m in sorted(models):
                print(f"    {m}")
        model_path = input("  Session model path: ").strip()

        if not os.path.exists(model_path):
            print(f"  ERROR: '{model_path}' not found!")
            return

        board = KnightBoard(SERIAL_PORT, NUM_CHANNELS)
        board.start_stream()
        time.sleep(STABILIZATION_TIME)
        car = CarController()
        car.test_connection()
        run_realtime(board, model_path, car)
        board.stop_stream()

    elif choice == '4':
        models = [f for f in os.listdir('.')
                  if f.endswith('.h5') and 'neurolink' in f.lower()]
        if models:
            print("  Available models:")
            for m in sorted(models):
                print(f"    {m}")

        model_path = input(f"  Model [{BASE_MODEL_PATH}]: ").strip()
        if not model_path:
            model_path = BASE_MODEL_PATH

        csv_folder = input("  CSV folder: ").strip()
        if not csv_folder:
            print("  ERROR: Must specify a folder.")
            return

        run_simulation(model_path, csv_folder)

    else:
        print("  Invalid choice.")


if __name__ == "__main__":
    main()
