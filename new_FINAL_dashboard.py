# streamlit_dashboard.py
#
# Streamlit dashboard for interactive CycleGAN fault visualization
# Uses ModelTester and paths from load_model_and_test.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from scipy.ndimage import maximum_filter1d, uniform_filter1d
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

from new_load_model_and_test import ModelTester, DATA_DIR, OUT_DIR, PROJECT_ROOT 
import torch

import joblib

import plotly.graph_objects as go   

import io
import base64
import time
import json
import threading
from datetime import datetime


import gspread
from scipy.interpolate import make_interp_spline
from scipy.interpolate import UnivariateSpline


# --- Portable path setup ---
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"
OUT_DIR = PROJECT_ROOT / "resources" / "models" /"epoch_models"

OUT_DIR.mkdir(parents=True, exist_ok=True)
# --- end portable path setup ---

# Must be the first Streamlit command
st.set_page_config(page_title="CycleGAN Fault Dashboard", layout="wide")

#MODEL_PATH_base = OUT_DIR/'generators_all_faults'/'model_acc_100.00_epoch_756.pth'
MODEL_PATH_base = OUT_DIR/'generators_all_faults'/'model_acc_100.00_epoch_291.pth'
MODEL_PATH_exclude_fault = OUT_DIR/'generators_no_2_fault'/'model_acc_100.00_epoch_360.pth'




#DATA_PATH = DATA_DIR / 'dataset_fft_for_cyclegan_case1_512 (1).npz'

#data_with_noise = "fresh_dataset_fft_for_cyclegan_case1_512_test.npz"

#data_with_noise = "awgn_5_fresh_dataset_fft_for_cyclegan_case1_512_test.npz" 
#data_with_noise = "awgn_10_fresh_dataset_fft_for_cyclegan_case1_512_test.npz" 
#data_with_noise = "awgn_15_fresh_dataset_fft_for_cyclegan_case1_512_test.npz"
#data_with_noise = "awgn_20_fresh_dataset_fft_for_cyclegan_case1_512_test.npz" 
#data_with_noise = "awgn_0_fresh_dataset_fft_for_cyclegan_case1_512_test.npz" 

#data_with_noise = "lp_10_fresh_dataset_fft_for_cyclegan_case1_512_test.npz"
#data_with_noise = "lp_5_fresh_dataset_fft_for_cyclegan_case1_512_test.npz"
#data_with_noise = "lp_0_fresh_dataset_fft_for_cyclegan_case1_512_test.npz"

#data_with_noise = "pink_20_fresh_dataset_fft_for_cyclegan_case1_512_test.npz"
#data_with_noise = "pink_10_fresh_dataset_fft_for_cyclegan_case1_512_test.npz"
#data_with_noise = "pink_5_fresh_dataset_fft_for_cyclegan_case1_512_test.npz"

#DATA_PATH = DATA_DIR / data_with_noise

# data = np.load(DATA_DIR / data_with_noise)


DATA_PATH = DATA_DIR / "NEW_ALL_TEST_SETS_COMBINED_PLUS_1HP.npz"
REF_DATA_PATH = DATA_DIR / "dataset_fft_for_cyclegan_case1_512 (1).npz"

@st.cache_resource
def _load_data():
    return np.load(DATA_PATH)

data = _load_data()




# ── Authentication ────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("## Research Experiment Access")
        st.markdown("This application is restricted to study participants.")
        pwd = st.text_input("Password", type="password", key="_login_pwd")
        if st.button("Submit", use_container_width=True):
            try:
                correct = st.secrets["APP_PASSWORD"]
            except (KeyError, FileNotFoundError):
                st.error("APP_PASSWORD secret is not configured.")
                st.stop()
            if pwd == correct:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
    st.stop()
# ── End authentication ────────────────────────────────────────────────────────

# Block F5 / Ctrl+R page refresh at all times
st.iframe("""
<script>
(function() {
    var doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.key === 'F5' || ((e.ctrlKey || e.metaKey) && e.key === 'r')) {
            e.preventDefault();
            e.stopPropagation();
        }
    }, true);
})();
</script>
""", height=1)

# Remove top padding
st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            /* Remove empty white gaps */
            .main > div {
                padding-top: 0rem;
            }
        </style>
    """, unsafe_allow_html=True)


# Smaller subheader style
st.markdown("""
<style>
.small-subheader {
    font-size: 1.3rem;     /* default subheader ~1.6rem */
    font-weight: 600;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Title style
st.markdown("""
<style>
.small-title {
    font-size: 2rem;     /* default subheader ~1.6rem */
    font-weight: 600;
    margin-top: -1rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Experiment logging (Google Sheets)
# -----------------------------
LOG_SHEET = "trust_logs"

LOG_COLUMNS = [
    # core
    "datetime_local",
    "loaded_experiment",
    "svm_subdir",
    "scenario_mode",
    "noise_enabled",
    "noise_level",
    "svm_full_accuracy",

    # sample + model outputs
    "sample_index",
    "true_label",
    "predicted_label",

    # human decision + time
    "decision",                 # trust / dont_trust
    "elapsed_seconds",

    # similarity summary (top-3)
    "top1_label", "top1_score",
    "top2_label", "top2_score",
    "top3_label", "top3_score",

    # extra useful context
    "true_label_was_shown",
    "selected_generated_idx",
    
    #User Evaluation
    "penalty_score",
]

# -----------------------------
# Fault category helper
# -----------------------------
def _fault_category(label):
    """
    Maps fault labels to high-level categories.
    Adjust mapping here if your dataset encoding differs.
    """
    if label is None:
        return None

    # Example mapping (adapt ONLY if your encoding is different):
    # 0,1,2 = ball
    # 3,4,5 = inner race
    # 6,7,8 = outer race
    # 9 = normal
    BALL = {0, 1, 2}
    INNER_RACE = {3, 4, 5}
    OUTER_RACE = {6, 7, 8}
    NORMAL = {9}
    

    if label in INNER_RACE:
        return "inner_race"
    if label in OUTER_RACE:
        return "outer_race"
    if label in BALL:
        return "ball"
    if label in NORMAL:
        return "normal"

    return "other_error"

# -----------------------------
# Penalty / reward computation
# -----------------------------
def _compute_penalty_score(decision, true_label, predicted_label):
    """
    Returns the penalty/reward score according to the defined rules.
    Higher score = lower simulated maintenance cost.
    """
    if true_label is None or predicted_label is None:
        return None

    correct = (true_label == predicted_label)

    true_cat = _fault_category(true_label)
    pred_cat = _fault_category(predicted_label)

    same_category = (true_cat is not None and true_cat == pred_cat)

     # ----- TRUST cases -----
    if decision == "trust":
        if correct:
            return +5                       # trusted & correct
        if not correct and same_category:
            return -2                       # trusted & minor mistake
        if not correct and not same_category:
            return -5                       # trusted & major mistake

    # ----- DON'T TRUST cases -----
    if decision == "dont_trust":
        if correct:
            return -3                       # unnecessary override
        if not correct and same_category:
            return +2                       # avoided minor mistake
        if not correct and not same_category:
            return +5                       # avoided major mistake

    return 0



@st.cache_resource
def _gsheet_spreadsheet() -> gspread.Spreadsheet:
    info = dict(st.secrets["gcp_service_account"])
    client = gspread.service_account_from_dict(info)
    sheet_id = st.secrets["GSHEET_ID"].strip()
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
    return client.open_by_url(url)


def _get_or_create_worksheet(sheet_name: str, columns: list[str]) -> gspread.Worksheet:
    """Return the named worksheet, creating it with a header row if it doesn't exist."""
    ss = _gsheet_spreadsheet()
    try:
        ws = ss.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = ss.add_worksheet(title=sheet_name, rows=5000, cols=len(columns))
        ws.append_row(columns, value_input_option="RAW")
    return ws


@st.cache_resource
def _gsheets_lock() -> threading.Lock:
    """Single in-process lock — prevents concurrent API calls from the same worker."""
    return threading.Lock()


def _append_log_row(sheet_name: str, columns: list[str], row_dict: dict) -> None:
    row = [str(row_dict.get(col, "")) if row_dict.get(col) is not None else "" for col in columns]
    with _gsheets_lock():
        ws = _get_or_create_worksheet(sheet_name, columns)
        ws.append_row(row, value_input_option="RAW")

# -----------------------------
# Timer helpers
# -----------------------------
def _reset_and_start_timer() -> None:
    st.session_state.timer_elapsed_s = 0.0
    st.session_state.timer_running = True
    st.session_state.timer_start_ts = time.time()

def _stop_timer_and_get_elapsed() -> float:
    running = bool(st.session_state.get("timer_running", False))
    start_ts = st.session_state.get("timer_start_ts", None)
    base_elapsed = float(st.session_state.get("timer_elapsed_s", 0.0))

    if running and start_ts is not None:
        base_elapsed += (time.time() - float(start_ts))

    st.session_state.timer_elapsed_s = float(base_elapsed)
    st.session_state.timer_running = False
    st.session_state.timer_start_ts = None
    return float(base_elapsed)

# -----------------------------
# Unseen-sample helper
# -----------------------------
def _ensure_unseen_pool(n_total: int) -> None:
    if "unseen_test_indices" not in st.session_state or not isinstance(st.session_state.unseen_test_indices, list):
        st.session_state.unseen_test_indices = []
    if "seen_test_indices" not in st.session_state or not isinstance(st.session_state.seen_test_indices, set):
        st.session_state.seen_test_indices = set()

    # If pool empty, re-fill with all not-yet-seen (or reset fully if everything seen)
    if len(st.session_state.unseen_test_indices) == 0:
        remaining = [i for i in range(n_total) if i not in st.session_state.seen_test_indices]
        if len(remaining) == 0:
            # all seen -> start over
            st.session_state.seen_test_indices = set()
            remaining = list(range(n_total))

        # shuffle
        perm = np.random.permutation(len(remaining)).tolist()
        st.session_state.unseen_test_indices = [remaining[p] for p in perm]

def _pop_next_unseen_index(n_total: int) -> int:
    _ensure_unseen_pool(n_total)
    idx = int(st.session_state.unseen_test_indices.pop(0))
    st.session_state.seen_test_indices.add(idx)
    return idx



def _format_elapsed_seconds(total_seconds: float) -> str:
    """Format elapsed time as MM:SS (and HH:MM:SS when >= 1 hour)."""
    total_seconds = max(0, int(total_seconds))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


from typing import Optional

def _pick_random_index_for_class(y: np.ndarray, cls: int, exclude_index: Optional[int] = None) -> Optional[int]:
    y = y.flatten()
    idxs = np.where(y == cls)[0].tolist()
    if exclude_index is not None and exclude_index in idxs:
        idxs.remove(exclude_index)
    if len(idxs) == 0:
        return None
    return int(np.random.choice(idxs))


def run_full_svm_accuracy_test(svm_model, tester):
    scaler = load_svm_scaler(st.session_state.get("svm_subdir", "baseline"))
    if scaler is None:
        st.session_state.full_svm_accuracy = None
        st.write("NO SCALER")
        return

    X = tester.test_X.astype(np.float32)
    y = tester.test_Y.flatten().astype(int)


    X_scaled = scaler.transform(X)
    y_pred = svm_model.predict(X_scaled)

    accuracy = float((y_pred == y).mean())
    mis_idx = np.where(y_pred != y)[0]

    mis_by_class = {}
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        mis_by_class[int(cls)] = cls_idx[y_pred[cls_idx] != y[cls_idx]].tolist()


    mis_into_class = {}
    for cls in np.unique(y_pred):
        pred_idx = np.where(y_pred == cls)[0]
        wrong_idx = pred_idx[y[pred_idx] != y_pred[pred_idx]]
        mis_into_class[int(cls)] = wrong_idx.tolist()

    st.session_state.full_svm_accuracy = accuracy
    st.session_state.full_svm_mis_idx = mis_idx.tolist()
    st.session_state.full_svm_mis_by_class = mis_by_class
    st.session_state.full_svm_mis_into_class = mis_into_class

    # st.write(st.session_state.full_svm_accuracy)
    # st.write(st.session_state.full_svm_mis_idx)
    # st.write(st.session_state.full_svm_mis_by_class)








def _drop_first_point_for_scoring(x, y):
    # Convert locally; do NOT modify caller arrays
    x = np.asarray(x).reshape(-1).astype(np.float32)
    y = np.asarray(y).reshape(-1).astype(np.float32)

    # Exclude exactly the first sample point (index 0) from both signals
    if x.size > 1 and y.size > 1:
        x = x[1:]
        y = y[1:]
    else:
        # If too short, fall back to empty-safe behavior
        x = x[:0]
        y = y[:0]
    return x, y





def sim_nrmse_as_displayed(real_signal: np.ndarray, generated_signal: np.ndarray, eps: float = 1e-12) -> float:
    """
    Score using the exact preprocessing the user sees in the Plotly overlay:
      - float32 cast
      - clip to non-negative
      - RMS-match generated to real (same as plot_elif8_overlay_plotly)
      - drop first point (same as sim_nrmse)
      - NRMSE similarity mapping: 1/(1+nrmse)
    """

    # Match plot_elif8_overlay_plotly preprocessing :contentReference[oaicite:2]{index=2}
    real = np.asarray(real_signal, dtype=np.float32).reshape(-1)
    gen  = np.asarray(generated_signal, dtype=np.float32).reshape(-1)

    real = np.clip(real, 0.0, None)
    gen  = np.clip(gen, 0.0, None)

    real_rms = float(np.sqrt(np.mean(real ** 2)))
    gen_rms  = float(np.sqrt(np.mean(gen ** 2)))

    if gen_rms < eps:
        gen = np.full_like(gen, real_rms)
    else:
        gen = gen * (real_rms / (gen_rms + eps))

    # Match sim_nrmse drop-first behavior :contentReference[oaicite:3]{index=3}
    real, gen = _drop_first_point_for_scoring(real, gen)
    if real.size == 0 or gen.size == 0:
        return 0.0

    rmse = float(np.sqrt(np.mean((real - gen) ** 2)))

    # Choose ONE denominator:
    # (A) Keep your current std-normalized behavior (what your sim_nrmse currently does) :contentReference[oaicite:4]{index=4}
    denom = float(np.std(real) + eps)

    # (B) OR use RMS normalization (often more intuitive for FFT magnitudes):
    # denom = float(np.sqrt(np.mean(real ** 2)) + eps)

    nrmse = rmse / denom
    return float(1.0 / (1.0 + nrmse))


def sim_nrmse(x, y, eps=1e-12):
    x, y = _drop_first_point_for_scoring(x, y)
    if x.size == 0 or y.size == 0:
        return 0.0
    rmse = np.sqrt(np.mean((x - y) ** 2))
    #denom = (np.std(x) + eps)
    denom = np.sqrt(np.mean(x**2))
    nrmse = rmse / denom
    return float(1.0 / (1.0 + nrmse))  # (0,1], higher is better






def scale_generated_like_real(real_signal: np.ndarray,
                              generated_signal: np.ndarray):
    """
    Apply the same scaling / preprocessing to `generated_signal` as in the
    main visualization: match amplitude range, mean, and clamp negatives.
    Returns (scaled_generated, ymin, ymax).
    """
    real_signal = real_signal.astype(np.float32)
    generated_signal = generated_signal.astype(np.float32).copy()

    # --- STEP 1: enforce non-negative magnitudes (FFT) ---
    real_signal = np.clip(real_signal, 0.0, None)
    generated_signal = np.clip(generated_signal, 0.0, None)

    # --- STEP 2: RMS-match generated to real (multiplicative only) ---
    eps = 1e-12
    real_rms = float(np.sqrt(np.mean(real_signal ** 2)))
    gen_rms  = float(np.sqrt(np.mean(generated_signal ** 2)))

    if gen_rms < eps:
        # generated is (almost) flat/zero → show a flat line at real RMS level
        generated_signal = np.full_like(generated_signal, real_rms)
    else:
        generated_signal = generated_signal * (real_rms / (gen_rms + eps))




    ymin = float(min(real_signal.min(), generated_signal.min()))
    ymax = float(max(real_signal.max(), generated_signal.max()))
    return generated_signal, ymin, ymax



from typing import Optional


def plot_elif8_overlay_plotly(real_signal: np.ndarray,
                              generated_signal: np.ndarray,
                              title: str,
                              class_label=None):
    """
    Plotly version of plot_elif8_overlay:
    - Same scaling of generated signal to real signal
    - Same mean-matching and clamping
    - Same diff computation (using real_signal.std())
    - Same red/blue band extent in y as Matplotlib imshow
    - Same line colors and behavior
    """
    window_env = 25
    smooth_env = 12
    diff_clip = 0.8
    alpha = 0.45

    # Ensure float32
    real_signal = real_signal.astype(np.float32)
    generated_signal = generated_signal.astype(np.float32)

    # --- STEP 1: enforce non-negative magnitudes (FFT) ---
    real_signal = np.clip(real_signal, 0.0, None)
    generated_signal = np.clip(generated_signal, 0.0, None)

    # --- STEP 2: RMS-match generated to real (multiplicative only) ---
    eps = 1e-12
    real_rms = float(np.sqrt(np.mean(real_signal ** 2)))
    gen_rms  = float(np.sqrt(np.mean(generated_signal ** 2)))

    if gen_rms < eps:
        generated_signal = np.full_like(generated_signal, real_rms)
    else:
        generated_signal = generated_signal * (real_rms / (gen_rms + eps))




    # Envelope computation (same as original function, although not used in final diff)
    env = maximum_filter1d(real_signal, size=window_env)
    env = uniform_filter1d(env, size=smooth_env)
    env = env + 1e-8  # avoid divide-by-zero

    # Signed deviation (EXACTLY like plot_elif8_overlay)
    # diff = (generated_signal - real_signal) / env   # <- old idea, now commented out in Matplotlib
    scale = real_signal.std() + 1e-8
    diff = (generated_signal - real_signal) / scale
    diff = uniform_filter1d(diff, size=5)
    diff = np.clip(diff, -diff_clip, diff_clip)

    # Y-extent identical to Matplotlib imshow(extent=[..., min(...), max(...)])
    ymin = float(min(real_signal.min(), generated_signal.min()))
    ymax = float(max(real_signal.max(), generated_signal.max()))

    x_vals = np.arange(len(diff))
    # Two rows so the band fills the vertical range [ymin, ymax]
    y_vals = [ymin, ymax]
    z = np.vstack([diff, diff])

    fig = go.Figure()

    # Background heat band (equivalent to imshow with cmap=bwr, TwoSlopeNorm)
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x_vals,
            y=y_vals,
            # colorscale = [
            #     [0.0,  "rgb(0, 0, 255)"],      # strong blue
            #     [0.5,  "rgb(255, 255, 255)"],  # white
            #     [1.0,  "rgb(255, 0, 0)"]       # strong red
            # ], 
            colorscale = [
                [0.00, "rgb(0, 0, 225)"],       # dark blue (very extreme)
                [0.10, "rgb(100, 150, 255)"],   # light blue
                [0.25, "rgb(210, 225, 255)"],   # very light blue (near white)
                [0.40, "rgb(255, 255, 255)"],   # white
                [0.60, "rgb(255, 255, 255)"],   # white (flat neutral band)
                [0.75, "rgb(255, 225, 210)"],   # very light red (near white)
                [0.90, "rgb(255, 150, 150)"],   # light red
                [1.00, "rgb(225, 0, 0)"],       # dark red (very extreme)
            ],
            zmin=-diff_clip,
            zmax=diff_clip,
            showscale=False,
            hoverinfo="skip",
            opacity=alpha,
        )
    )

    # Foreground lines – same style as Matplotlib
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(real_signal)),
            y=real_signal,
            mode="lines",
            name="Test spectrum (measured)",
            line=dict(color="black", width=3),
        )
    )

    # fig.add_trace(
    #     go.Scatter(
    #         x=np.arange(len(generated_signal)),
    #         y=generated_signal,
    #         mode="lines",
    #         name="Generated fault spectrum (CycleGAN)",
    #         line=dict(color="green", width=2.5),
    #     )
    # )



    # fig.add_trace(
    #     go.Scatter(
    #         x=np.arange(len(generated_signal)),
    #         y=generated_signal,
    #         mode="lines",
    #         name="Generated fault spectrum (CycleGAN)",
    #         line=dict(color="green", width=2.5),
    #         line_shape="spline",   # <-- makes it curved
    #         # optional: smoothing=1.0
    #     )
    # )

    x = np.arange(len(generated_signal))
    x_s = np.linspace(x.min(), x.max(), len(x) * 4)
    spl = UnivariateSpline(x, generated_signal, k=3, s=4 * len(x))  # increase s => smoother
    y_s = spl(x_s)

    fig.add_trace(
        go.Scatter(
            x=x_s,
            y=y_s,
            mode="lines",
            name="Generated normal spectrum (CycleGAN)" if class_label == 9 else "Generated fault spectrum (CycleGAN)",
            line=dict(color="green", width=3),
        )
    )


    fig.update_layout(
        title=dict(text=title, font=dict(size=22)),
        height=400,
        template="plotly_white",
        xaxis_title="FFT bin index (0–511)",
        yaxis_title="FFT magnitude",
        xaxis=dict(title_font=dict(size=20)),
        yaxis=dict(title_font=dict(size=20)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=15),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Show grid similar to ax.grid(True, alpha=0.3)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, range=[ymin, ymax])

    st.plotly_chart(fig, use_container_width=True)





@st.cache_resource
def load_svm_classifier(svm_subdir: str = "baseline"):
    svm_path = PROJECT_ROOT / "resources" / "svm_models" / svm_subdir / "new_svm_fault_classifier.pkl"
    if not svm_path.exists():
        return None
    return joblib.load(svm_path)


@st.cache_resource
def load_svm_scaler(svm_subdir: str = "baseline"):
    scaler_path = (
        PROJECT_ROOT / "resources" / "svm_models" / svm_subdir / "new_svm_fault_classifier_scaler.pkl"
    )
    if not scaler_path.exists():
        return None
    return joblib.load(scaler_path)



# ================================
# Helper: load ModelTester (cached)
# ================================
@st.cache_resource
def get_tester(model_path: Path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = ModelTester(model_path, DATA_PATH, device=device, ref_data_path=REF_DATA_PATH)
    return tester



# ══════════════════════════════════════════════════════════════════════════════
# STUDY MODE
# ══════════════════════════════════════════════════════════════════════════════

STUDY_POOLS_PATH = PROJECT_ROOT / "study_pools_VERIFIED.json"

#STUDY_POOLS_PATH = PROJECT_ROOT / "study_pools_N3_FINAL.json"
STUDY_LOG_SHEET  = "study_events"
# ── experimental toggle ──────────────────────────────────────────────────────
_COMPARE_WITH_REAL = True   # set False (or delete guarded blocks) to disable

# One row per event — 3 event types per trial (+ N × thumbnail_select in S3):
#   trial_loaded     – fires once when trial loads; carries all fixed-at-load data
#   thumbnail_select – fires on every S3 hypothesis click (0–N times)
#   trial_end        – fires once on confidence submit; carries decision outcome
EVENT_LOG_COLUMNS = [
    # ── identity (all events) ─────────────────────────────────────────────────
    "participant_id", "participant_name",
    "participant_number",   # 1–N; determines counterbalancing group
    "participant_group",    # 1/2/3 rotating
    "pool_id", "scenario",
    # ── trial coordinates (all events) ───────────────────────────────────────
    "trial_id",             # "B{block}_T{position:02d}"
    "block_number",         # S1→1, S2→2, S3→3
    "trial_position",       # 1-indexed within scenario
    "noise_condition",      # "awgn_15" / "awgn_5" / "1HP_20_AWGN"
    "hp_trial",             # True for N3 robustness trials (S3 only)
    # ── sample / SVM (all events) ────────────────────────────────────────────
    "sample_index", "true_label", "predicted_label",
    "fault_category", "svm_correct",
    # ── event fields (all events) ────────────────────────────────────────────
    "event_type",           # trial_loaded | thumbnail_select | trial_end
    "event_value",          # class_id / confidence / None
    "timestamp",
    # ── per-interaction payload ───────────────────────────────────────────────
    #   trial_loaded:     class_id=predicted_label, nrmse_rank/score of predicted class
    #   thumbnail_select: class_id=clicked class, nrmse_rank/score of that hypothesis
    "class_id", "nrmse_rank", "nrmse_score",
    # ── decision outcome (trial_end only) ────────────────────────────────────
    "decision", "confidence_rating", "appropriate_decision",
    "elapsed_seconds",
    # ── fixed-at-load (trial_loaded only) ────────────────────────────────────
    "trial_start_ts",
    "predict_ts",
    "generate_ts",
    "svm_inference_s",
    "cyclegan_inference_s",
    # ── top-3 thumbnail NRMSE hypotheses (trial_loaded, S3 only) ─────────────
    "top1_label", "top1_score",
    "top2_label", "top2_score",
    "top3_label", "top3_score",
    # ── all per-class NRMSE scores (trial_loaded, S3 only) ───────────────────
    "s3_nrmse_class0", "s3_nrmse_class1", "s3_nrmse_class2",
    "s3_nrmse_class3", "s3_nrmse_class4", "s3_nrmse_class5",
    "s3_nrmse_class6", "s3_nrmse_class7", "s3_nrmse_class8",
    "s3_nrmse_variance",
    # ── reference comparison (trial_loaded only) ─────────────────────────────
    "s2_ref_nrmse",       # NRMSE(test, real training ref shown in S2 / shadow ref in S3)
    "true_class_nrmse",   # band-colour score of true class (S3 CycleGAN hypothesis)
    # ── misc ──────────────────────────────────────────────────────────────────
    "svm_variant",
]

_SCENARIO_BLOCK = {"S1": 1, "S2": 2, "S3": 3}


def _log_study_event(event_type: str, **kwargs) -> None:
    """Buffer one event row — flushed to disk on Submit & Continue.

    Fixed-at-load columns (timing, top3, S3 NRMSE per-class, reference NRMSE)
    are only populated on 'trial_loaded'; all other events leave them None.
    """
    ss       = st.session_state
    trial    = ss.get("study_current_trial") or {}
    pred     = ss.get("study_trial_prediction")
    true_lbl = trial.get("true_label")
    svm_correct = (pred == true_lbl) if pred is not None and true_lbl is not None else None

    scenario     = ss.get("active_scenario") or ""
    block_number = _SCENARIO_BLOCK.get(scenario, 0)
    trial_pos    = (ss.get("trial_index") or 0) + 1
    condition    = trial.get("condition", "")
    trial_id     = f"B{block_number}_T{trial_pos:02d}"

    pnum  = int(ss.get("participant_number", 1))
    group = (pnum - 1) % 3 + 1

    # ── identity + event fields (every row) ──────────────────────────────────
    row = {
        "participant_id":        ss.get("participant_id"),
        "participant_name":      ss.get("participant_name"),
        "participant_number":    pnum,
        "participant_group":     group,
        "pool_id":               ss.get("study_pool_id"),
        "scenario":              scenario,
        "trial_id":              trial_id,
        "block_number":          block_number,
        "trial_position":        trial_pos,
        "noise_condition":       condition,
        "hp_trial":              condition == "1HP_20_AWGN",
        "sample_index":          trial.get("index"),
        "true_label":            true_lbl,
        "predicted_label":       pred,
        "fault_category":        trial.get("fault_category"),
        "svm_correct":           svm_correct,
        "svm_variant":           ss.get("svm_subdir", "baseline"),
        "event_type":            event_type,
        "event_value":           kwargs.get("event_value"),
        "timestamp":             datetime.now().isoformat(timespec="milliseconds"),
        # per-interaction payload (trial_loaded + thumbnail_select)
        "class_id":              kwargs.get("class_id"),
        "nrmse_rank":            kwargs.get("nrmse_rank"),
        "nrmse_score":           kwargs.get("nrmse_score"),
        # decision outcome (trial_end)
        "decision":              kwargs.get("decision"),
        "confidence_rating":     kwargs.get("confidence_rating"),
        "appropriate_decision":  kwargs.get("appropriate_decision"),
        "elapsed_seconds":       kwargs.get("elapsed_seconds"),
        # fixed-at-load — default None, filled below for trial_loaded only
        "trial_start_ts":        None,
        "predict_ts":            None,
        "generate_ts":           None,
        "svm_inference_s":    None,
        "cyclegan_inference_s":    None,
        "top1_label": None, "top1_score": None,
        "top2_label": None, "top2_score": None,
        "top3_label": None, "top3_score": None,
        "s2_ref_nrmse":          None,
        "true_class_nrmse":      None,
    }
    for c in range(9):
        row[f"s3_nrmse_class{c}"] = None
    row["s3_nrmse_variance"] = None

    # ── fixed-at-load block (trial_loaded only) ───────────────────────────────
    if event_type == "trial_loaded":
        row["trial_start_ts"] = (
            datetime.fromtimestamp(ss["study_trial_start_ts_ms"] / 1000)
            if ss.get("study_trial_start_ts_ms") else None)
        row["predict_ts"] = (
            datetime.fromtimestamp(ss["study_predict_ts_ms"] / 1000)
            if ss.get("study_predict_ts_ms") else None)
        row["generate_ts"] = (
            datetime.fromtimestamp(ss["study_generate_ts_ms"] / 1000)
            if ss.get("study_generate_ts_ms") else None)
        row["svm_inference_s"] = (
            round((ss["study_predict_ts_ms"] - ss["study_trial_start_ts_ms"]) / 1000, 3)
            if ss.get("study_predict_ts_ms") and ss.get("study_trial_start_ts_ms") else None)
        row["cyclegan_inference_s"] = (
            round((ss["study_generate_ts_ms"] - ss["study_trial_start_ts_ms"]) / 1000, 3)
            if ss.get("study_generate_ts_ms") and ss.get("study_trial_start_ts_ms") else None)

        # top1/2/3 + per-class NRMSE — S3 only
        s3_scores = ss.get("study_trial_nrmse_scores") or []
        s3_labels = ss.get("study_trial_s3_labels") or []
        if s3_scores and s3_labels and len(s3_scores) == len(s3_labels):
            sorted_idx = sorted(range(len(s3_scores)),
                                key=lambda i: s3_scores[i], reverse=True)
            for rank, i in enumerate(sorted_idx[:3], 1):
                row[f"top{rank}_label"] = int(s3_labels[i])
                row[f"top{rank}_score"] = float(s3_scores[i])
            nrmse_map = {int(lbl): float(sc) for lbl, sc in zip(s3_labels, s3_scores)}
            for c in range(9):
                row[f"s3_nrmse_class{c}"] = nrmse_map.get(c)
            row["s3_nrmse_variance"] = (
                float(np.var(s3_scores)) if len(s3_scores) > 1 else None)
            if true_lbl is not None:
                row["true_class_nrmse"] = nrmse_map.get(int(true_lbl))

        # s2_ref_nrmse: S2 reference removed; S3 shadow ref kept for cross-scenario comparison
        if scenario == "S3":
            row["s2_ref_nrmse"] = ss.get("study_trial_s3_pred_ref_nrmse")

    ss.setdefault("study_log_buffer", []).append(row)


def _flush_log_buffer() -> None:
    """Write all buffered event rows to Google Sheets under the shared lock."""
    ss  = st.session_state
    buf = ss.get("study_log_buffer", [])
    if not buf:
        return
    with _gsheets_lock():
        ws = _get_or_create_worksheet(STUDY_LOG_SHEET, EVENT_LOG_COLUMNS)
        rows = [
            [str(row.get(col, "")) if row.get(col) is not None else "" for col in EVENT_LOG_COLUMNS]
            for row in buf
        ]
        ws.append_rows(rows, value_input_option="RAW")
    ss.study_log_buffer = []


# Counterbalanced pool assignment — groups of 10 participants (1–30)
#   Group 1 (p  1–10): S1=A  S2=B  S3=C
#   Group 2 (p 11–20): S1=B  S2=C  S3=A
#   Group 3 (p 21–30): S1=C  S2=A  S3=B
_COUNTERBALANCE: dict[int, dict[str, str]] = {
    1: {"S1": "A", "S2": "B", "S3": "C"},
    2: {"S1": "B", "S2": "C", "S3": "A"},
    3: {"S1": "C", "S2": "A", "S3": "B"},
}

def _get_scenario_pool(scenario: str) -> str:
    """Return the pool ID for *scenario* based on this participant's group (1–3, rotating)."""
    pnum  = int(st.session_state.get("participant_number", 1))
    group = (pnum - 1) % 3 + 1
    return _COUNTERBALANCE[group][scenario]

_CONDITION_LABEL = {
    "awgn_15":     "15 dB AWGN",
    "awgn_5":      "5 dB AWGN",
    "1HP_20_AWGN": "1HP + AWGN",
}
_SC_INFO = {
    "S1": {"title": "Scenario 1 (S1)", "pool": "A", "aid": "Plain spectrum — no overlay"},
    "S2": {"title": "Scenario 2 (S2)", "pool": "B", "aid": "Spectrum + Real Examples (Paper)"},
    "S3": {"title": "Scenario 3 (S3)", "pool": "C", "aid": "Spectrum + CycleGAN hypotheses"},
}
_STATUS_ICON = {"not_started": "⬜", "in_progress": "🔄", "finished": "✅"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_study_pools() -> Optional[dict]:
    if not STUDY_POOLS_PATH.exists():
        return None
    with open(STUDY_POOLS_PATH, "r") as fh:
        return json.load(fh)


def _generate_participant_id(name: str) -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    initials = "".join(w[0].upper() for w in name.strip().split() if w)[:4]
    return f"P-{date_str}-{initials}"


def _scenario_trial_count(scenario: str, pool_data: Optional[dict]) -> int:
    if pool_data is None:
        return 0
    pool = pool_data.get(_get_scenario_pool(scenario), {})
    n = len(pool.get("main_trials", []))
    n += len(pool.get("n3_trials", []))
    return n


def _init_study_state() -> None:
    ss = st.session_state
    ss.setdefault("study_phase",              "participant_entry")
    ss.setdefault("participant_name",          "")
    ss.setdefault("participant_id",            "")
    ss.setdefault("study_pool_id",             "A")
    ss.setdefault("study_pool_data",           None)
    ss.setdefault("study_scenario_status",     {"S1": "not_started", "S2": "not_started", "S3": "not_started"})
    ss.setdefault("active_scenario",           None)
    ss.setdefault("trial_queue",               [])
    ss.setdefault("trial_index",               0)
    ss.setdefault("study_trial_loaded",        False)
    ss.setdefault("study_trial_signal",        None)
    ss.setdefault("study_trial_prediction",    None)
    ss.setdefault("study_trial_generated",     None)
    ss.setdefault("study_trial_s3_labels",     None)
    ss.setdefault("study_trial_nrmse_scores",  None)
    ss.setdefault("study_trial_real_true_nrmse",   None)
    ss.setdefault("study_trial_s3_pred_ref_nrmse", None)
    ss.setdefault("study_trial_slot_rank",     None)
    ss.setdefault("study_trial_top3_idx",      set())
    ss.setdefault("study_trial_top3",          [])
    ss.setdefault("study_trial_selected_idx",  None)
    ss.setdefault("study_trial_start_ts",      None)
    ss.setdefault("study_s2_refs",             None)
    ss.setdefault("study_current_trial",       None)
    ss.setdefault("study_svm_accuracy",        None)
    # event-log timing
    ss.setdefault("study_trial_start_ts_ms",   None)
    ss.setdefault("study_predict_ts_ms",       None)
    ss.setdefault("study_generate_ts_ms",      None)
    # confidence-slider phase
    ss.setdefault("study_awaiting_confidence", False)
    ss.setdefault("study_last_decision",       None)
    ss.setdefault("study_log_buffer",          [])
    # sub-block transition banner
    ss.setdefault("study_show_subblock_transition", False)
    # practice
    ss.setdefault("practice_done",              False)
    ss.setdefault("practice_queue",             [])
    ss.setdefault("practice_index",             0)
    ss.setdefault("practice_trial_loaded",      False)
    ss.setdefault("practice_signal",            None)
    ss.setdefault("practice_pred",              None)
    ss.setdefault("practice_awaiting_feedback", False)
    ss.setdefault("practice_last_decision",     None)
    ss.setdefault("practice_show_result",       False)
    ss.setdefault("practice_conf_val",          5)
    ss.setdefault("participant_number",         1)


_SCENARIO_SEED_OFFSET = {"S1": 1, "S2": 2, "S3": 3}

def _start_scenario(scenario: str) -> None:
    import random
    ss      = st.session_state
    pnum    = ss.get("participant_number", 1)
    rng     = random.Random(pnum * 1000 + _SCENARIO_SEED_OFFSET[scenario])
    pool_id = _get_scenario_pool(scenario)
    pool    = ss.study_pool_data[pool_id]

    # Reproducibly shuffle main trials, then append N3 at the end
    trials = list(pool.get("main_trials", []))
    rng.shuffle(trials)

    n3 = list(pool.get("n3_trials", []))
    rng.shuffle(n3)
    trials = trials + n3

    ss = st.session_state
    ss.active_scenario    = scenario
    ss.study_pool_id      = pool_id               # track which pool is active
    ss.trial_queue        = trials
    ss.trial_index        = 0
    ss.study_trial_loaded = False
    ss.study_phase        = "trial"

    status = dict(ss.study_scenario_status)
    if status[scenario] == "not_started":
        status[scenario] = "in_progress"
    ss.study_scenario_status = status


def _start_practice() -> None:
    import random
    ss  = st.session_state
    rng = random.Random(ss.get("participant_number", 1) * 1000)
    trials = list((ss.study_pool_data or {}).get("practice_trials", []))
    rng.shuffle(trials)
    ss.practice_queue        = trials[:2]
    ss.practice_index        = 0
    ss.practice_trial_loaded = False
    ss.study_phase           = "practice"


def _render_practice(svm_model) -> None:
    ss      = st.session_state
    queue   = ss.practice_queue
    idx     = ss.practice_index
    n_total = len(queue)

    st.markdown(
        f"<div class='trial-banner'>"
        f"<div class='tb-left'>Practice</div>"
        f"<div class='tb-mid'>Trial {min(idx + 1, n_total)} / {n_total}"
        f" &nbsp;·&nbsp; Familiarisation — decisions are not recorded</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(idx, n_total) / n_total)

    if idx >= n_total:
        ss.practice_done = True
        st.success("Practice complete! You can now start the study scenarios.")
        if st.button("Go to scenarios →", type="primary"):
            ss.study_phase = "scenario_menu"
            st.rerun()
        return

    if not ss.practice_trial_loaded:
        trial     = queue[idx]
        condition = trial["condition"]
        X      = data[f"{condition}_X"].astype(np.float32)
        signal = X[int(trial["index"])].copy()
        scaler = load_svm_scaler("baseline")
        pred   = int(svm_model.predict(scaler.transform(signal.reshape(1, -1)))[0])
        ss.practice_signal       = signal
        ss.practice_pred         = pred
        ss.practice_trial_loaded = True
        st.rerun()
        return

    signal      = ss.practice_signal
    pred        = ss.practice_pred
    true_label  = queue[idx].get("true_label")
    cat         = _fault_category(pred)
    awaiting    = ss.practice_awaiting_feedback
    show_result = ss.get("practice_show_result", False)

    c_pred, c_trust, c_dont, c_back = st.columns([5, 2, 2, 2])
    with c_back:
        if st.button("← Scenarios", use_container_width=True, key=f"prac_back_inline_{idx}"):
            ss.study_phase = "scenario_menu"
            st.rerun()
    with c_pred:
        health       = "Healthy" if cat == "normal" else "Faulty"
        health_color = "#2e7d32" if cat == "normal" else "#c62828"
        health_bg    = "#f1f8f1" if cat == "normal" else "#fff5f5"
        st.markdown(
            f"<div class='pred-badge'>SVM → Class {pred}"
            f"<span class='pb-label'>({cat})</span>"
            f"<span class='pb-health' style='background:{health_bg};color:{health_color};'>{health}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    if not awaiting and not show_result:
        with c_trust:
            if st.button("✅ Trust", use_container_width=True, key=f"prac_trust_{idx}"):
                ss.practice_last_decision     = "trust"
                ss.practice_awaiting_feedback = True
                st.rerun()
        with c_dont:
            if st.button("❌ Don't Trust", use_container_width=True, key=f"prac_dont_{idx}"):
                ss.practice_last_decision     = "dont_trust"
                ss.practice_awaiting_feedback = True
                st.rerun()

    if not (awaiting and not show_result):
        _render_trial_s1(signal)

    # ── Confidence slider (after decision, before result) ─────────────────────
    if awaiting and not show_result:
        st.markdown("<br>", unsafe_allow_html=True)
        _sl, sl_col, _sr = st.columns([1, 4, 1])
        with sl_col:
            dec_icon = "✅" if ss.practice_last_decision == "trust" else "❌"
            dec_text = "Trust" if ss.practice_last_decision == "trust" else "Don't Trust"
            st.markdown(
                f"<div class='conf-card'>"
                f"<div class='cc-title'>How confident are you in your decision?</div>"
                f"<div class='cc-decision'>You chose: {dec_icon} <b>{dec_text}</b></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            conf_val = st.slider(
                "Confidence",
                min_value=0, max_value=10, value=5,
                key=f"prac_conf_slider_{idx}",
                label_visibility="collapsed",
            )
            st.markdown(
                "<div style='display:flex;justify-content:space-between;"
                "padding:0 2px;margin-top:-8px;font-size:1.0rem;color:#666;'>"
                + "".join(f"<span>{v}</span>" for v in range(11))
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='display:flex;justify-content:space-between;"
                "padding:0 2px;margin-top:6px;font-size:1.4rem;color:#444;'>"
                "<span>← Not at all confident</span>"
                "<span>Completely certain →</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            sub_btn_l, sub_btn_col, sub_btn_r = st.columns([1, 1, 1])
            with sub_btn_col:
                st.markdown("""
                <style>
                div[data-testid="stButton"] button[kind="primary"] {
                    background-color: #ADD8E6 !important;
                    border-color: #ADD8E6 !important;
                    color: #000000 !important;
                }
                div[data-testid="stButton"] button[kind="primary"]:hover {
                    background-color: #87CEEB !important;
                    border-color: #87CEEB !important;
                }
                </style>
                """, unsafe_allow_html=True)
                if st.button("Submit →", key=f"prac_conf_submit_{idx}",
                             use_container_width=True, type="primary"):
                    ss.practice_conf_val          = conf_val
                    ss.practice_show_result       = False
                    ss.practice_awaiting_feedback = False
                    # advance to next trial immediately (no feedback screen)
                    ss.practice_index            += 1
                    ss.practice_trial_loaded      = False
                    ss.practice_last_decision     = None
                    st.rerun()
        return

    # ── Minimal result + go back ───────────────────────────────────────────────
    if show_result:
        decision     = ss.practice_last_decision
        svm_correct  = (true_label == pred) if true_label is not None else None
        trust_correct = (decision == "trust" and svm_correct) or \
                        (decision == "dont_trust" and not svm_correct)
        result_icon = "✅" if trust_correct else "❌"
        result_text = "Correct" if trust_correct else "Incorrect"
        true_cat    = _fault_category(true_label) if true_label is not None else "—"
        st.markdown(
            f"<div style='padding:8px 14px;margin-top:10px;font-size:1.05rem;color:#444;'>"
            f"{result_icon} <b>{result_text}</b> &nbsp;·&nbsp; "
            f"True label: <b>Class {true_label} ({true_cat})</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        _, next_col, back_col, _ = st.columns([2, 2, 2, 2])

        def _reset_practice_trial():
            ss.practice_index            += 1
            ss.practice_trial_loaded      = False
            ss.practice_awaiting_feedback = False
            ss.practice_last_decision     = None
            ss.practice_show_result       = False

        with next_col:
            if st.button("Next trial →", use_container_width=True, key=f"prac_next_{idx}"):
                _reset_practice_trial()
                st.rerun()
        with back_col:
            if st.button("← Go back to scenarios", use_container_width=True, key=f"prac_back_{idx}"):
                _reset_practice_trial()
                ss.study_phase = "scenario_menu"
                st.rerun()


def _study_load_current_trial(tester, svm_model) -> None:
    """Load signal, run SVM, generate CycleGAN outputs (S3) or real refs (S2)."""
    ss        = st.session_state
    scenario  = ss.active_scenario
    trial     = ss.trial_queue[ss.trial_index]
    condition = trial["condition"]
    data_key  = condition  # keys match: "awgn_15", "awgn_5", "1HP_20_AWGN"

    # ── record trial-start timestamp before anything else ────────────────────
    now_ms = int(time.time() * 1000)
    ss.study_trial_start_ts_ms = now_ms
    ss.study_predict_ts_ms     = None
    ss.study_generate_ts_ms    = None
    # store in session state early so _log_study_event can read trial coords
    ss.study_current_trial = trial

    X          = data[f"{data_key}_X"].astype(np.float32)
    Y          = data[f"{data_key}_Y"].astype(np.int32)
    sample_idx = int(trial["index"])
    signal     = X[sample_idx].copy()

    # Update tester split so reference lookups stay consistent
    tester.test_X = X
    tester.test_Y = Y

    # SVM prediction
    scaler   = load_svm_scaler("baseline")
    x_scaled = scaler.transform(signal.reshape(1, -1))
    pred     = int(svm_model.predict(x_scaled)[0])
    ss.study_trial_prediction = pred          # needed by _log_study_event
    ss.study_predict_ts_ms    = int(time.time() * 1000)

    # SVM accuracy (cached per condition to avoid repeated full-dataset inference)
    acc_key = f"study_acc_{data_key}"
    if ss.get(acc_key) is None:
        X_sc = scaler.transform(X)
        y_pr = svm_model.predict(X_sc)
        ss[acc_key] = float((y_pr == Y.flatten()).mean())
    ss.study_svm_accuracy = ss[acc_key]

    # ── S3: CycleGAN generation ───────────────────────────────────────────────
    generated_faults = None
    s3_labels        = None
    nrmse_scores     = None
    slot_rank        = None
    top3_idx_set     = set()
    top3             = []
    s2_refs          = None

    if scenario == "S3":
        svm_variant = ss.get("svm_subdir", "baseline")
        s3_labels   = [0, 1, 3, 4, 5, 6, 7, 8] if svm_variant == "exclude_2" else list(range(9))

        if pred != 9:
            real_batch = signal[np.newaxis, :]
            norm_label = np.array([0], dtype=np.int32)
            gen_normal = tester.gan.generate_samples(real_batch, norm_label, generator="g_BA")[0]

            gen_list = []
            for f in s3_labels:
                lab   = np.array([int(f)], dtype=np.int32)
                gen_f = tester.gan.generate_samples(gen_normal[np.newaxis, :], lab, generator="g_AB")
                gen_list.append(gen_f[0])
            generated_faults = np.array(gen_list)

            nrmse_scores = [sim_nrmse_as_displayed(signal, generated_faults[i])
                            for i in range(len(s3_labels))]
            all_sorted   = list(np.argsort(nrmse_scores)[::-1])
            slot_rank    = {int(s): r + 1 for r, s in enumerate(all_sorted)}
            top3_idx_set = set(np.argsort(nrmse_scores)[-3:])
            top3         = [{"label": int(s3_labels[int(i)]), "score": float(nrmse_scores[int(i)])}
                            for i in list(np.argsort(nrmse_scores)[-3:][::-1])]
        else:
            # pred == 9: cycle-consistency path
            real_batch = signal[np.newaxis, :]
            lab_f      = np.array([0], dtype=np.int32)
            gen_fault  = tester.gan.generate_samples(real_batch, lab_f, generator="g_AB")[0]
            gen_normal_back = tester.gan.generate_samples(
                gen_fault[np.newaxis, :], np.array([0], dtype=np.int32), generator="g_BA"
            )[0]
            generated_faults = gen_normal_back[np.newaxis, :]
            s3_labels        = [9]
            score            = sim_nrmse_as_displayed(signal, gen_normal_back)
            nrmse_scores     = [score]
            slot_rank        = {0: 1}
            top3_idx_set     = {0}
            top3             = [{"label": 9, "score": float(score)}]

    # ── S2: reference sheet (deactivated — s2_refs stays None) ──────────────

    # ── record generate timestamp (S3 only) ───────────────────────────────────
    if scenario == "S3":
        ss.study_generate_ts_ms = int(time.time() * 1000)

    # ── S3 shadow reference: real training sample of predicted class ───────────
    # Mirrors exactly what S2 shows, computed for logging only (not displayed).
    # Stored in study_trial_s3_pred_ref_nrmse → logged as s2_ref_nrmse for S3.
    _s3_pred_ref_nrmse = None
    if scenario == "S3" and pred is not None:
        try:
            if int(pred) == 9:
                _arr = tester.domain_A_train_X
                _ref = _arr[np.random.randint(0, _arr.shape[0])].astype(np.float32)
            else:
                _arr = tester.domain_B_train_X_by_class.get(int(pred))
                _ref = _arr[np.random.randint(0, _arr.shape[0])].astype(np.float32) \
                       if _arr is not None else None
            if _ref is not None:
                _s3_pred_ref_nrmse = float(sim_nrmse_as_displayed(signal, _ref))
        except Exception:
            _s3_pred_ref_nrmse = None
    ss.study_trial_s3_pred_ref_nrmse = _s3_pred_ref_nrmse

    # ── Commit ────────────────────────────────────────────────────────────────
    ss.study_current_trial       = trial
    ss.study_trial_signal        = signal
    ss.study_trial_prediction    = pred
    ss.study_trial_generated     = generated_faults
    ss.study_trial_s3_labels     = s3_labels
    ss.study_trial_nrmse_scores  = nrmse_scores
    ss.study_trial_slot_rank     = slot_rank
    ss.study_trial_top3_idx      = top3_idx_set
    ss.study_trial_top3          = top3
    ss.study_trial_selected_idx  = None
    ss.study_s2_refs             = s2_refs
    ss.study_trial_loaded        = True
    ss.study_trial_start_ts      = time.time()
    ss.study_awaiting_confidence = False
    ss.study_last_decision       = None

    # ── [_COMPARE_WITH_REAL] real training sample of true class ─────────────────
    if _COMPARE_WITH_REAL:
        _true_lbl = trial.get("true_label")
        _real_nrmse = None
        if _true_lbl is not None:
            try:
                if int(_true_lbl) == 9:
                    _arr = tester.domain_A_train_X
                    _ref = _arr[np.random.randint(0, _arr.shape[0])].astype(np.float32)
                else:
                    _arr = tester.domain_B_train_X_by_class.get(int(_true_lbl))
                    _ref = _arr[np.random.randint(0, _arr.shape[0])].astype(np.float32) \
                           if _arr is not None else None
                if _ref is not None:
                    _real_nrmse = float(sim_nrmse_as_displayed(signal, _ref))
            except Exception:
                _real_nrmse = None
        ss.study_trial_real_true_nrmse = _real_nrmse

    # ── log trial_loaded (merges former trial_start + prediction_shown) ─────────
    _log_study_event(
        "trial_loaded",
        class_id=pred,
        nrmse_rank=trial.get("nrmse_rank_predicted"),
        nrmse_score=trial.get("nrmse_score_predicted"),
    )


def _study_on_trust_decision(decision: str) -> None:
    """Called when participant presses Trust / Don't Trust.
    Logs the trust_decision event, then waits for confidence rating."""
    ss = st.session_state
    ss.study_last_decision       = decision
    ss.study_awaiting_confidence = True


def _study_on_confidence(confidence: int) -> None:
    """Called after participant submits confidence slider.
    Logs confidence_rating + trial_end events, then advances to next trial."""
    ss      = st.session_state
    trial   = ss.study_current_trial or {}
    pred    = ss.study_trial_prediction
    true_lbl = trial.get("true_label")
    decision = ss.study_last_decision
    elapsed  = time.time() - float(ss.study_trial_start_ts or time.time())

    svm_correct = (pred == true_lbl) if pred is not None and true_lbl is not None else None
    appropriate = None
    if decision is not None and svm_correct is not None:
        appropriate = (decision == "trust" and svm_correct) or \
                      (decision == "dont_trust" and not svm_correct)

    _log_study_event(
        "trial_end",
        decision=decision,
        confidence_rating=confidence,
        appropriate_decision=appropriate,
        elapsed_seconds=round(elapsed, 3),
    )
    _flush_log_buffer()

    # Advance
    ss.trial_index               += 1
    ss.study_trial_loaded         = False
    ss.study_trial_prediction     = None
    ss.study_trial_generated      = None
    ss.study_trial_selected_idx   = None
    ss.study_trial_top3           = []
    ss.study_awaiting_confidence  = False
    ss.study_last_decision        = None
    ss.study_predict_ts_ms         = None
    ss.study_generate_ts_ms        = None
    ss.study_trial_start_ts_ms     = None
    ss.study_trial_real_true_nrmse   = None
    ss.study_trial_s3_pred_ref_nrmse = None


# ── render functions ──────────────────────────────────────────────────────────


def _render_participant_entry() -> None:
    #st.title("CycleGAN Trust Calibration Study")
    st.markdown("Please enter your name to begin. The study pool is assigned automatically.")
    st.markdown("---")

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.subheader("Participant Registration")
        name = st.text_input("Full name", value=st.session_state.get("participant_name", ""))

        pnum = st.number_input(
            "Participant number",
            min_value=1, step=1,
            value=st.session_state.get("participant_number", 1),
            help="Determines pool assignment (1→G1, 2→G2, 3→G3, 4→G1, …) and trial order.",
        )

        if st.button("Begin Study →", disabled=not name.strip(), use_container_width=True):
            pools_data = _load_study_pools()
            if pools_data is None:
                st.error("study_pools.json not found. Run generate_study_pools.py first.")
                return
            pid = _generate_participant_id(name.strip())
            ss = st.session_state
            ss.participant_name      = name.strip()
            ss.participant_id        = pid
            ss.participant_number    = int(pnum)
            ss.study_pool_id         = "A"
            ss.study_pool_data       = pools_data
            ss.study_scenario_status = {"S1": "not_started", "S2": "not_started", "S3": "not_started"}
            ss.study_phase           = "scenario_menu"
            st.rerun()

        if name.strip():
            st.caption(f"Your participant ID will be: `{_generate_participant_id(name.strip())}`")


def _render_scenario_menu() -> None:
    ss     = st.session_state
    status = ss.study_scenario_status

    #st.title("CycleGAN Trust Calibration Study")
    st.markdown(
        f"**Participant:** {ss.participant_name} &emsp; "
        f"**ID:** `{ss.participant_id}`"
    )
    st.markdown("---")

    _left, inner, _right = st.columns([1, 8, 1])
    with inner:
        st.subheader("Select a scenario to continue")
        cols = st.columns(4)

        # ── Practice card (leftmost) ──────────────────────────────────────────
        prac_done  = ss.get("practice_done", False)
        prac_stkey = "finished" if prac_done else "not_started"
        prac_icon  = "✅" if prac_done else "⬜"
        n_prac     = len((ss.study_pool_data or {}).get("practice_trials", []))
        with cols[0]:
            st.markdown(
                f"<div class='sc-card {prac_stkey}'>"
                f"<div class='sc-title'>{prac_icon} Practice</div>"
                f"<div class='sc-row'>🛠 Plain spectrum — familiarisation</div>"
                f"<div class='sc-row'>📋 {n_prac} trials</div>"
                f"<div class='sc-status {prac_stkey}'>{'Completed' if prac_done else 'Not Started'}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            label = "Restart Practice →" if prac_done else "Start Practice →"
            if st.button(label, key="prac_btn", use_container_width=True, type="primary"):
                _start_practice()
                st.rerun()

        # ── S1 / S2 / S3 cards ───────────────────────────────────────────────
        for i, sc in enumerate(["S1", "S2", "S3"]):
            info     = _SC_INFO[sc]
            icon     = _STATUS_ICON[status[sc]]
            n_total  = _scenario_trial_count(sc, ss.study_pool_data)
            st_key   = status[sc]
            st_label = st_key.replace("_", " ").title()
            with cols[i + 1]:
                st.markdown(
                    f"<div class='sc-card {st_key}'>"
                    f"<div class='sc-title'>{icon} {info['title']}</div>"
                    f"<div class='sc-row'>Pool &nbsp;<b>{_get_scenario_pool(sc)}</b></div>"
                    f"<div class='sc-row'>🛠 {info['aid']}</div>"
                    f"<div class='sc-row'>📋 {n_total} trials</div>"
                    f"<div class='sc-status {st_key}'>{st_label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                label = "Restart →" if status[sc] == "finished" else ("Resume →" if status[sc] == "in_progress" else "Start →")
                if st.button(label, key=f"sc_btn_{sc}", use_container_width=True,
                             type="primary"):
                    _start_scenario(sc)
                    st.rerun()

        done = sum(1 for s in ["S1", "S2", "S3"] if status[s] == "finished")
        st.progress(done / 3, text=f"Progress: {done} / 3 scenarios completed")
        if done == 3:
            st.success("🎉 You have completed all scenarios! Thank you for participating.")


def _render_trial_s1(signal: np.ndarray) -> None:
    """Plain spectrum only (used by S1 and as base for S2/S3 before selection)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(signal)), y=signal,
        mode="lines", line=dict(color="black", width=2), showlegend=False,
    ))
    fig.update_layout(
        height=420, template="plotly_white",
        xaxis_title="FFT bin index (0–511)", yaxis_title="FFT magnitude",
        xaxis=dict(title_font=dict(size=20)), yaxis=dict(title_font=dict(size=20)),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, use_container_width=True)


def _render_trial_s2(signal: np.ndarray) -> None:
    """Plain spectrum only — reference sheet is deactivated."""
    _render_trial_s1(signal)
    # _render_reference_sheet(st.session_state.get("_tester_ref"))


# def _render_reference_sheet(tester) -> None:
#     """Static reference sheet: one training example per fault class (0–9).
#     Paper-format 2×5 grid. Shown once per S2 scenario, not per trial."""
#     CLASSES = list(range(9)) + [9]
#     CAT_LABEL = {
#         0: "Ball (0)", 1: "Ball (1)", 2: "Ball (2)",
#         3: "Inner Race (3)", 4: "Inner Race (4)", 5: "Inner Race (5)",
#         6: "Outer Race (6)", 7: "Outer Race (7)", 8: "Outer Race (8)",
#         9: "Normal (9)",
#     }
#     samples = []
#     for cls in CLASSES:
#         if cls == 9:
#             arr = tester.domain_A_train_X
#             sample = arr[0].astype(np.float32)
#         else:
#             arr = tester.domain_B_train_X_by_class.get(cls)
#             sample = arr[0].astype(np.float32) if arr is not None else np.zeros(512, dtype=np.float32)
#         samples.append(sample)
#     fig, axes = plt.subplots(2, 5, figsize=(14, 4.5))
#     axes = axes.flatten()
#     for i, (cls, sig) in enumerate(zip(CLASSES, samples)):
#         ax = axes[i]
#         ax.plot(sig, color="black", linewidth=0.9)
#         ax.set_title(CAT_LABEL[cls], fontsize=9, fontweight="bold")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.grid(True, alpha=0.3, linewidth=0.5)
#         for spine in ax.spines.values():
#             spine.set_linewidth(0.8)
#     fig.suptitle("Reference Examples — All Fault Classes", fontsize=11, fontweight="bold", y=1.02)
#     fig.tight_layout()
#     st.pyplot(fig, use_container_width=True)
#     plt.close(fig)


@st.cache_data(max_entries=300, show_spinner=False)
def _thumbnail_b64(
    gen_sig_bytes: bytes,
    sig_shape: tuple,
    ymin: float,
    ymax: float,
    is_selected: bool,
    is_top3: bool,
    is_pred_tile: bool,
) -> str:
    """Render one S3 thumbnail to a base64 PNG string (cached per unique combo)."""
    gen_sig = np.frombuffer(gen_sig_bytes, dtype=np.float32).reshape(sig_shape)
    fig, ax = plt.subplots(figsize=(4.3, 1.8))
    ax.set_facecolor("#b8b8b8" if is_selected else "white")
    ax.plot(gen_sig, color="green", linewidth=0.8)

    border_w = 5.0 if is_top3 else 2.0
    for spine in ax.spines.values():
        spine.set_linewidth(border_w)
        if is_pred_tile:
            spine.set_edgecolor("#ffc107")

    if is_selected:
        ax.plot(0.90, 0.80, "D", color="#000000", markersize=20,
                transform=ax.transAxes, zorder=5, markeredgewidth=0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _render_trial_s3(signal: np.ndarray, pred: int) -> None:
    """CycleGAN grid + overlay when a class tile is selected."""
    ss               = st.session_state
    generated_faults = ss.study_trial_generated
    s3_labels        = ss.study_trial_s3_labels
    nrmse_scores     = ss.study_trial_nrmse_scores
    slot_rank        = ss.study_trial_slot_rank
    top3_idx_set     = ss.study_trial_top3_idx
    selected_idx     = ss.get("study_trial_selected_idx")

    # Main visualization
    if selected_idx is not None and generated_faults is not None:
        raw_ref     = generated_faults[selected_idx]
        class_label = int(s3_labels[selected_idx]) if s3_labels else selected_idx
        ref_sig, _, _ = scale_generated_like_real(signal, raw_ref)
        plot_elif8_overlay_plotly(
            signal, ref_sig,
            f"Test Sample vs Generated Class {class_label} ({_fault_category(class_label)})",
            class_label=class_label,
        )
    else:
        _render_trial_s1(signal)

    hyp_label = "Normal Hypothesis" if pred == 9 else "Fault Hypotheses"
    if generated_faults is None:
        st.caption(f"Click **Generate Alternative Versions** to produce CycleGAN {hyp_label.lower()}.")
        return

    st.subheader(f"Generated {hyp_label} (CycleGAN)")
    n_thumbs = len(s3_labels)

    # Inject CSS to highlight the predicted class column with an amber border
    pred_slot = next(
        (i for i, lbl in enumerate(s3_labels) if pred is not None and int(lbl) == int(pred)),
        None,
    )
    if pred_slot is not None and n_thumbs > 1:
        nth = pred_slot + 1
        st.markdown(
            f"""<style>
            div[data-testid="stHorizontalBlock"]:has(
                > div[data-testid="stColumn"]:nth-child({n_thumbs}):last-child
            ) > div[data-testid="stColumn"]:nth-child({nth}) > div[data-testid="stVerticalBlock"] {{
                border: 3px solid #ffc107;
                border-radius: 10px;
                padding: 6px 6px 10px 6px;
                background: #fffde7;
            }}
            div[data-testid="stHorizontalBlock"]:has(
                > div[data-testid="stColumn"]:nth-child({n_thumbs}):last-child
            ) div[data-testid="stButton"] button {{
                min-height: 3rem !important;
                padding: 0.4rem 0.8rem !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(
                > div[data-testid="stColumn"]:nth-child({n_thumbs}):last-child
            ) div[data-testid="stButton"] button p {{
                font-size: 1.35rem !important;
            }}
            </style>""",
            unsafe_allow_html=True,
        )
    elif n_thumbs >= 1:
        st.markdown(
            f"""<style>
            div[data-testid="stHorizontalBlock"]:has(
                > div[data-testid="stColumn"]:nth-child({n_thumbs}):last-child
            ) div[data-testid="stButton"] button {{
                min-height: 3rem !important;
                padding: 0.4rem 0.8rem !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(
                > div[data-testid="stColumn"]:nth-child({n_thumbs}):last-child
            ) div[data-testid="stButton"] button p {{
                font-size: 1.35rem !important;
            }}
            </style>""",
            unsafe_allow_html=True,
        )

    if n_thumbs == 1:
        # pred==9 path: don't let the single thumbnail stretch full-width
        thumb_cols = st.columns([1, 1, 4])
        cols = [thumb_cols[0]]
    else:
        cols = st.columns(n_thumbs)

    for slot, class_label in enumerate(s3_labels):
        with cols[slot]:
            raw_gen = generated_faults[slot]
            gen_sig, ymin, ymax = scale_generated_like_real(signal, raw_gen)

            is_selected  = (selected_idx == slot)
            is_top3      = slot in top3_idx_set
            is_pred_tile = (pred is not None and int(class_label) == int(pred))

            b64 = _thumbnail_b64(
                gen_sig.astype(np.float32).tobytes(),
                gen_sig.shape,
                float(ymin), float(ymax),
                bool(is_selected), bool(is_top3), bool(is_pred_tile),
            )
            st.markdown(
                f'<img src="data:image/png;base64,{b64}" style="width:100%;display:block;"/>',
                unsafe_allow_html=True,
            )

            if st.button(
                f"Select Class {class_label}",
                key=f"study_s3_{class_label}_{slot}_{ss.trial_index}",
            ):
                ss.study_trial_selected_idx = slot
                _log_study_event(
                    "thumbnail_select",
                    event_value=int(class_label),
                    class_id=int(class_label),
                    nrmse_rank=slot_rank.get(slot) if slot_rank else None,
                    nrmse_score=float(nrmse_scores[slot]) if nrmse_scores else None,
                )
                st.rerun()

            if nrmse_scores and slot_rank:
                rank_color = "#e65100" if slot_rank[slot] <= 3 else "inherit"
                st.markdown(
                    f"<div style='font-size:1.05rem; line-height:1.6;'>"
                    f"<b style='font-size:1.4rem; color:{rank_color};'>Rank #{slot_rank[slot]}</b><br>"
                    f"Score: <b>{nrmse_scores[slot]:.3f}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def _render_trial(tester, svm_model) -> None:
    ss       = st.session_state
    scenario = ss.active_scenario
    queue    = ss.trial_queue
    idx      = ss.trial_index
    n_total  = len(queue)

    if idx >= n_total:
        status = dict(ss.study_scenario_status)
        status[scenario] = "finished"
        ss.study_scenario_status = status
        ss.study_phase = "scenario_complete"
        st.rerun()
        return

    if not ss.get("study_trial_loaded", False):
        with st.spinner(f"Loading trial {idx + 1} of {n_total}…"):
            _study_load_current_trial(tester, svm_model)
        st.rerun()
        return

    trial   = ss.study_current_trial
    signal  = ss.study_trial_signal
    pred    = ss.study_trial_prediction
    info    = _SC_INFO[scenario]
    elapsed = time.time() - float(ss.study_trial_start_ts)

    # ── Header banner ─────────────────────────────────────────────────────────
    elapsed_str = _format_elapsed_seconds(elapsed)
    st.markdown(
        f"<div class='trial-banner'>"
        f"<div class='tb-left'>{info['title']}</div>"
        f"<div class='tb-mid'>Trial {idx + 1} / {n_total}</div>"
        f"<div class='tb-timer' style='display:none;'>⏱ {elapsed_str}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(idx / n_total)

    # ── Prediction badge + control row ────────────────────────────────────────
    awaiting_conf = ss.get("study_awaiting_confidence", False)
    cat = _fault_category(pred)

    c_pred, c_trust, c_dont = st.columns([5, 2, 2])

    with c_pred:
        health = "Healthy" if cat == "normal" else "Faulty"
        health_color = "#2e7d32" if cat == "normal" else "#c62828"
        health_bg = "#f1f8f1" if cat == "normal" else "#fff5f5"
        st.markdown(
            f"<div class='pred-badge'>SVM → Class {pred}"
            f"<span class='pb-label'>({cat})</span>"
            f"<span class='pb-health' style='background:{health_bg};color:{health_color};'>{health}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not awaiting_conf:
        st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(3):last-child)
        div[data-testid="stButton"] button {
            min-height: 3.6rem !important;
            padding-top: 0.7rem !important;
            padding-bottom: 0.7rem !important;
        }
        div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(3):last-child)
        div[data-testid="stButton"] button p {
            font-size: 1.4rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
        with c_trust:
            if st.button("✅ Trust", use_container_width=True, key=f"trust_{idx}", ):
                _study_on_trust_decision("trust")
                st.rerun()
        with c_dont:
            if st.button("❌ Don't Trust", use_container_width=True, key=f"dont_{idx}"):
                _study_on_trust_decision("dont_trust")
                st.rerun()

    # ── Confidence slider (appears after Trust/Don't Trust, before advancing) ─
    if awaiting_conf:
        st.markdown("<br>", unsafe_allow_html=True)
        _sl, sl_col, _sr = st.columns([1, 4, 1])
        with sl_col:
            dec_icon = "✅" if ss.study_last_decision == "trust" else "❌"
            dec_text = "Trust" if ss.study_last_decision == "trust" else "Don't Trust"
            st.markdown(
                f"<div class='conf-card'>"
                f"<div class='cc-title'>How confident are you in your decision?</div>"
                f"<div class='cc-decision'>You chose: {dec_icon} <b>{dec_text}</b></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            conf_val = st.slider(
                "Confidence",
                min_value=0, max_value=10, value=5,
                key=f"conf_slider_{idx}",
                label_visibility="collapsed",
            )
            # tick numbers
            st.markdown(
                "<div style='display:flex;justify-content:space-between;"
                "padding:0 2px;margin-top:-8px;font-size:1.0rem;color:#666;'>"
                + "".join(f"<span>{v}</span>" for v in range(11))
                + "</div>",
                unsafe_allow_html=True,
            )
            # side labels
            st.markdown(
                "<div style='display:flex;justify-content:space-between;"
                "padding:0 2px;margin-top:6px;font-size:1.4rem;color:#444;'>"
                "<span>← Not at all confident</span>"
                "<span>Completely certain →</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            sub_btn_l, sub_btn_col, _ = st.columns([1, 1, 1])
            with sub_btn_l:
                if st.button("← Change decision", key=f"conf_back_{idx}",
                             use_container_width=True):
                    ss.study_awaiting_confidence = False
                    ss.study_last_decision       = None
                    st.rerun()
            with sub_btn_col:
                st.markdown("""
                <style>
                div[data-testid="stButton"] button[kind="primary"] {
                    background-color: #ADD8E6 !important;
                    border-color: #ADD8E6 !important;
                    color: #000000 !important;
                }
                div[data-testid="stButton"] button[kind="primary"]:hover {
                    background-color: #87CEEB !important;
                    border-color: #87CEEB !important;
                }
                </style>
                """, unsafe_allow_html=True)
                if st.button("Submit & continue →", key=f"conf_submit_{idx}",
                            use_container_width=True, type="primary"):
                    _study_on_confidence(conf_val)
                    st.rerun()
        return                           # don't render visualization while awaiting

    st.markdown("---")

    # ── Visualization ─────────────────────────────────────────────────────────
    if scenario == "S3":
        _render_trial_s3(signal, pred)
    elif scenario == "S2":
        _render_trial_s2(signal)
    else:
        _render_trial_s1(signal)

    # ── Escape hatch disabled ─────────────────────────────────────────────────
    # if st.button("↩ Abandon scenario and return to menu", type="secondary"):
    #     ss.study_phase    = "scenario_menu"
    #     ss.active_scenario = None
    #     st.rerun()


def _render_scenario_complete() -> None:
    scenario = st.session_state.active_scenario
    info     = _SC_INFO.get(scenario or "S1", {})
    st.title(f"{info.get('title', scenario)} — Complete!")
    st.success("✅ All trials for this scenario are done.")
    st.markdown("Take a short break if needed, then proceed to the next scenario.")
    if st.button("→ Return to Scenario Menu", use_container_width=True):
        st.session_state.study_phase    = "scenario_menu"
        st.session_state.active_scenario = None
        st.rerun()


_STUDY_CSS = """
<style>
/* ── Scenario cards ────────────────────────────────────────────── */
.sc-card {
    border-radius: 12px;
    padding: 18px 20px 14px 20px;
    margin-bottom: 6px;
    border: 1.5px solid #ccc;
    background: #fff;
    min-height: 200px;
}
.sc-card.finished    { border-color: #aaa; }
.sc-card.in_progress { border-color: #888; }
.sc-card .sc-title   { font-size: 1.15rem; font-weight: 700; margin-bottom: 10px; }
.sc-card .sc-row     { font-size: 1rem; color: #555; margin-bottom: 3px; }
.sc-card .sc-status  { display:inline-block; font-size:0.8rem; font-weight:600;
                       border-radius:20px; padding:2px 10px; margin-top:6px;
                       background:#e9ecef; color:#495057; }

/* ── Trial header banner ───────────────────────────────────────── */
.trial-banner {
    background: linear-gradient(90deg, #1a1a2e 0%, #2d3a6b 100%);
    color: #fff;
    border-radius: 10px;
    padding: 16px 26px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.trial-banner .tb-left { font-size: 1.9rem; font-weight: 700; }
.trial-banner .tb-mid  { font-size: 1.7rem;  color: #dce9fc; font-weight: 500; }
.trial-banner .tb-timer {
    background: rgba(255,255,255,0.15);
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 1.05rem;
    font-family: monospace;
    letter-spacing: 0.04em;
}

/* ── Prediction badge ──────────────────────────────────────────── */
.pred-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #fff8e1;
    border: 1.5px solid #ffc107;
    border-radius: 24px;
    padding: 7px 18px;
    font-size: 1.35rem;
    font-weight: 700;
    color: #333;
    margin: 4px 0 8px 0;
}
.pred-badge .pb-label { font-size: 1.0rem; font-weight: 400; color: #666; margin-left:4px; }
.pred-badge .pb-health { font-size: 1.0rem; font-weight: 600; padding: 4px 12px; border-radius: 10px; margin-left: 10px; }

/* ── Trust (col 2) and Don't Trust (col 3) buttons ─────────────── */
/* Scoped to the row containing .pred-badge so other buttons        */
/* (Submit, Abandon, Start →) are not affected                      */
div[data-testid="stHorizontalBlock"]:has(.pred-badge)
    > div:nth-child(2) button {
    background-color: #f1f8f1 !important;
    border-color:     #81c784 !important;
    color:            #2e7d32 !important;
}
div[data-testid="stHorizontalBlock"]:has(.pred-badge)
    > div:nth-child(2) button p {
    font-size: 1.15rem !important;
    color:     #2e7d32 !important;
}
div[data-testid="stHorizontalBlock"]:has(.pred-badge)
    > div:nth-child(2) button:hover {
    background-color: #dcedc8 !important;
}
div[data-testid="stHorizontalBlock"]:has(.pred-badge)
    > div:nth-child(3) button {
    background-color: #fff5f5 !important;
    border-color:     #e57373 !important;
    color:            #c62828 !important;
}
div[data-testid="stHorizontalBlock"]:has(.pred-badge)
    > div:nth-child(3) button p {
    font-size: 1.15rem !important;
    color:     #c62828 !important;
}
div[data-testid="stHorizontalBlock"]:has(.pred-badge)
    > div:nth-child(3) button:hover {
    background-color: #ffebee !important;
}

/* ── Sub-block transition ──────────────────────────────────────── */
.subblock-banner {
    background: #f5f5f5;
    border-left: 4px solid #888;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 6px 0 12px 0;
    font-size: 0.95rem;
    color: #333;
}

/* ── Confidence card ───────────────────────────────────────────── */
.conf-card {
    background: #fafafa;
    border: 1.5px solid #ccc;
    border-radius: 14px;
    padding: 26px 30px 20px 30px;
    margin: 4px 0;
}
.conf-card .cc-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    color: #1a1a1a;
}
.conf-card .cc-decision {
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 12px;
}
</style>
"""


def render_study_mode(tester, svm_model) -> None:
    """Top-level dispatcher for study mode. Replaces the main dashboard UI."""
    _init_study_state()
    st.markdown(_STUDY_CSS, unsafe_allow_html=True)

    # Sidebar: minimal, read-only info
    ss    = st.session_state
    phase = ss.get("study_phase", "participant_entry")
    if phase not in ("participant_entry",):
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Participant:** {ss.get('participant_name', '—')}")
        st.sidebar.markdown(f"**ID:** `{ss.get('participant_id', '—')}`")
        st.sidebar.markdown(f"**Pool:** {ss.get('study_pool_id', '—')}")
        if phase == "trial" and ss.get("study_current_trial"):
            st.sidebar.markdown(f"**Sample ID:** `{ss.study_current_trial.get('index', '—')}`")
        prac_icon = "✅" if ss.get("practice_done") else "⬜"
        st.sidebar.markdown(f"{prac_icon} Practice")
        for sc in ["S1", "S2", "S3"]:
            icon = _STATUS_ICON.get(ss.study_scenario_status.get(sc, "not_started"), "⬜")
            st.sidebar.markdown(f"{icon} {sc}")

        # Researcher access link (passcode protected)
        st.sidebar.markdown("---")
        dl_pwd = st.sidebar.text_input("Researcher passcode", type="password", key="_dl_pwd")
        try:
            correct_dl = st.secrets["RESEARCHER_PASSWORD"]
        except (KeyError, FileNotFoundError):
            correct_dl = None
        if correct_dl and dl_pwd == correct_dl:
            sheet_id = st.secrets.get("GSHEET_ID", "")
            sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
            st.sidebar.markdown(f"[Open study log in Google Sheets]({sheet_url})")

    if phase == "participant_entry":
        _render_participant_entry()
    elif phase == "scenario_menu":
        _render_scenario_menu()
    elif phase == "practice":
        _render_practice(svm_model)
    elif phase == "trial":
        _render_trial(tester, svm_model)
    elif phase == "scenario_complete":
        _render_scenario_complete()


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════
def main():

    #st.title("CycleGAN Fault Identification Dashboard")
    st.markdown(
                        f"<div class='small-title'>SVM Trust-Assessment Dashboard with CycleGAN Support v1</div>",
                        unsafe_allow_html=True,
                    )

    # --- session-state defaults (MUST be before any reads) ---
    st.session_state.setdefault("loaded_experiment", "All faults (baseline)")
    st.session_state.setdefault("tester_model_path", MODEL_PATH_base)
    st.session_state.setdefault("svm_subdir", "baseline")
    st.session_state.setdefault("full_svm_accuracy", None)
    st.session_state.setdefault("full_svm_mis_idx", [])
    st.session_state.setdefault("full_svm_mis_by_class", {})
    st.session_state.setdefault("full_svm_mis_into_class", {})

    st.session_state.study_mode_active = True
    _study_tester = get_tester(MODEL_PATH_base)
    _study_svm = load_svm_classifier("baseline")
    render_study_mode(_study_tester, _study_svm)
    return

    # Sidebar: class & sample selection
    st.sidebar.header("Test Sample Selection")
    
    dataset_options = [
        "clean",
        # "awgn_0", 
        "awgn_5", 
        #"awgn_10", 
        "awgn_15", 
        # "awgn_20",
        #"lp_0", "lp_5", "lp_10","lp_15","lp_20",
        #"pink_0","pink_5", "pink_10","pink_15", "pink_20",
        "1HP_20_AWGN",
    ]

    selected_dataset = st.sidebar.selectbox(
        "Test dataset",
        dataset_options,
        index=0
    )





    # -----------------
    # Experiment selection + explicit load trigger
    # -----------------
    st.sidebar.subheader("Experiment")
    st.sidebar.caption(f"Loaded: {st.session_state['loaded_experiment']}")

    experiment = st.sidebar.selectbox(
        "Model / SVM variant",
        ["All faults (baseline)", "Exclude fault 2"],
        index=0,
    )

    # default selection = baseline
    selected_subdir = "baseline" if experiment.startswith("All faults") else "exclude_2"

    # persist currently loaded experiment (default baseline)
    if "loaded_experiment" not in st.session_state:
        st.session_state["loaded_experiment"] = "All faults (baseline)"

    # button to switch (or first load if you want)
    switch_clicked = st.sidebar.button("Load / Switch model")

    # Resolve model path based on selection
    target_model_path = MODEL_PATH_base if experiment.startswith("All faults") else MODEL_PATH_exclude_fault
    target_subdir = "baseline" if experiment.startswith("All faults") else "exclude_2"


    # If user didn't click switch, keep whatever is already loaded (baseline by default)
    if not switch_clicked:
        # force baseline load on first run
        if "tester_model_path" not in st.session_state:
            st.session_state["tester_model_path"] = MODEL_PATH_base
            st.session_state["loaded_experiment"] = "All faults (baseline)"
    else:
        # user explicitly requested change
        st.session_state["tester_model_path"] = target_model_path
        st.session_state["loaded_experiment"] = experiment
        st.session_state["svm_subdir"] = target_subdir



    model_path = st.session_state["tester_model_path"]

    with st.spinner(f"Loading {st.session_state['loaded_experiment']} model/data..."):
        try:
            tester = get_tester(st.session_state["tester_model_path"])
        except Exception as e:
            st.session_state.messages.append(("error", f"Error loading ModelTester / model / data:\n{e}"))
            return


    # Ensure session state variables exist
    if "test_fault_class" not in st.session_state:
        st.session_state.test_fault_class = 0
    if "test_sample_index" not in st.session_state:
        st.session_state.test_sample_index = None
    if "test_sample_signal" not in st.session_state:
        st.session_state.test_sample_signal = None
    if "original_test_feature" not in st.session_state:
        st.session_state.original_test_feature = None
    if "prediction_ready_feature" not in st.session_state:
        st.session_state.prediction_ready_feature = None
    if "generated_faults" not in st.session_state:
        st.session_state.generated_faults = None
    if "selected_generated_idx" not in st.session_state:
        st.session_state.selected_generated_idx = None
    if "generation_done" not in st.session_state:
        st.session_state.generation_done = False
    if "true_label_shown" not in st.session_state:
        st.session_state.true_label_shown = False    
    if "noise_enabled" not in st.session_state:
        st.session_state.noise_enabled = False
    if "noise_level" not in st.session_state:
        # default: moderate noise
        st.session_state.noise_level = "Medium-High (SNR ≈ 15 dB)"
    if "full_svm_mis_idx" not in st.session_state:
        st.session_state.full_svm_mis_idx = []
    # --- scenario + random real reference storage ---
    if "scenario_mode" not in st.session_state:
        st.session_state.scenario_mode = "S3: Overlay + generated (CycleGAN) references"
    if "random_real_refs" not in st.session_state:
        st.session_state.random_real_refs = None  # shape (9, L)
    if "random_real_ref_indices" not in st.session_state:
        st.session_state.random_real_ref_indices = None  # list of 9 indices
    # --- operator decision buttons ---
    if "operator_trust" not in st.session_state:
        st.session_state.operator_trust = None  # "trust" / "dont_trust" / None
    if "s1_display_order" not in st.session_state:
        st.session_state.s1_display_order = None  # list[int] mapping grid slot -> underlying index
    if "s1_ref_kind" not in st.session_state:
        st.session_state.s1_ref_kind = None  # "fault" or "normal"
    if "last_scenario_mode" not in st.session_state:
        st.session_state.last_scenario_mode = st.session_state.scenario_mode
    if "random_real_pred_label" not in st.session_state:
        st.session_state.random_real_pred_label = None  # int predicted label used to sample refs
    if "random_real_ref_source" not in st.session_state:
        st.session_state.random_real_ref_source = None  # "fault_train" or "normal_train"
    if "refs_locked" not in st.session_state:
        st.session_state.refs_locked = False

    if "s1_lock_key" not in st.session_state:
        st.session_state.s1_lock_key = None  # tuple to detect when a new lock is needed

    if "noise_type" not in st.session_state:
        st.session_state.noise_type = "White Gaussian (AWGN)"
    
    #st.session_state.setdefault("noise_type", "White Gaussian (AWGN)")

    # --- timer state ---
    st.session_state.setdefault("timer_running", False)
    st.session_state.setdefault("timer_start_ts", None)
    st.session_state.setdefault("timer_elapsed_s", 0.0)

    # --- logging / top3 cache ---
    st.session_state.setdefault("top3_similar", [])  # list of dicts: [{"label":..., "score":...}, ...]
    st.session_state.setdefault("last_sample_load_dt", None)

    # --- unseen-pool state ---
    st.session_state.setdefault("unseen_test_indices", [])
    st.session_state.setdefault("seen_test_indices", set())


    # Ensure session keys exist
    if "loaded_experiment" not in st.session_state:
        st.session_state["loaded_experiment"] = "All faults (baseline)"

    if "tester_model_path" not in st.session_state:
        st.session_state["tester_model_path"] = MODEL_PATH_base



    # Load the EXACT dataset used during SVM training
    #data = np.load(DATA_DIR / 'dataset_fft_for_cyclegan_case1_512 (1).npz')
    
    #data = np.load(DATA_DIR / "fresh_dataset_fft_for_cyclegan_case1_512_test.npz")
    #data = np.load(DATA_DIR / "awgn_10_fresh_dataset_fft_for_cyclegan_case1_512_test.npz")
    #data = np.load(DATA_DIR / "awgn_5_fresh_dataset_fft_for_cyclegan_case1_512_test.npz")
    

    # streamlit_test_X = data["test_X"]
    # streamlit_test_Y = data["test_Y"]


    # streamlit_test_X = data[f"{selected_dataset}_X"]
    # streamlit_test_Y = data[f"{selected_dataset}_Y"]
    streamlit_test_X = data[f"{selected_dataset}_X"].astype(np.float32)
    streamlit_test_Y = data[f"{selected_dataset}_Y"].astype(np.int32)

    # update tester test set
    tester.test_X = streamlit_test_X
    tester.test_Y = streamlit_test_Y

    
    






    svm_model = load_svm_classifier(st.session_state.get("svm_subdir", "baseline"))



    # -----------------
    # Noise configuration
    # -----------------
    # st.sidebar.subheader("Noise Injection")

    # noise_enabled = st.sidebar.checkbox(
    #     "Add sensor noise to test sample",
    #     value=st.session_state.noise_enabled,
    # )
    # st.session_state.noise_enabled = noise_enabled

    # noise_type_options = [
    #     "White Gaussian (AWGN)",
    #     "Laplacian",
    #     "Pink (1/f) approx",
    #     "Brown (1/f^2) approx",
    # ]

    # try:
    #     default_noise_type_index = noise_type_options.index(st.session_state.noise_type)
    # except Exception:
    #     default_noise_type_index = 0

    # noise_type = st.sidebar.selectbox(
    #     "Noise type",
    #     options=noise_type_options,
    #     index=default_noise_type_index,
    #     disabled=not noise_enabled,
    # )

    # if noise_enabled:
    #     st.session_state.noise_type = noise_type

    # noise_options = [
    #     "Low (SNR ≈ 30 dB)",
    #     "Medium (SNR ≈ 20 dB)",
    #     "Medium-High (SNR ≈ 15 dB)",
    #     "High (SNR ≈ 10 dB)",
    #     "Very High (SNR ≈ 5 dB)"
        
    # ]

    # try:
    #     default_noise_index = noise_options.index(st.session_state.noise_level)
    # except Exception:
    #     default_noise_index = 2  # default = Medium-High

    # noise_level = st.sidebar.selectbox(
    #     "Noise level (approx. SNR)",
    #     options=noise_options,
    #     index=default_noise_index,
    #     disabled=not noise_enabled,
    # )

    # if noise_enabled:
    #     st.session_state.noise_level = noise_level

    # st.sidebar.markdown("---")


    # --- auto-run full SVM accuracy when configuration changes ---
    sig = (
        st.session_state.get("svm_subdir"),
        selected_dataset,
        st.session_state.get("noise_enabled"),
        st.session_state.get("noise_level"),
        st.session_state.get("noise_type"),
    )

    if st.session_state.get("full_svm_acc_signature") != sig:
        run_full_svm_accuracy_test(svm_model, tester)
        st.session_state.full_svm_acc_signature = sig
    
    st.session_state.setdefault("full_svm_acc_signature", None)



    # -----------------
    # Selection mode
    # -----------------
    selection_mode = st.sidebar.radio(
        "Selection mode",
        options=[
            "By fault class (random index)",
            "By test index",
            "Random misclassified sample (after full SVM eval)",
            "Random sample from all test samples",
        ],
        index=3,
    )




    # -----------------------
    # Scenario mode (S1/S2/S3)
    # -----------------------
    scenario_mode = st.sidebar.radio(
        "Scenario",
        options=[
            "S1: No overlay + random real references",
            "S2: Overlay + random real reference",
            "S3: Overlay + generated (CycleGAN) references",
        ],
        index=2,  # default to your current behavior (S3)
    )

    st.session_state.scenario_mode = scenario_mode

    scenario = st.session_state.scenario_mode

    # If user switches scenario, keep the selected test sample, but reset selection state
    if st.session_state.last_scenario_mode != st.session_state.scenario_mode:
        st.session_state.selected_generated_idx = None  # prevents stale selection crash
        st.session_state.operator_trust = None          # optional: clear prior decision
        st.session_state.last_scenario_mode = st.session_state.scenario_mode




    unique_classes = sorted(list(set(streamlit_test_Y.flatten().tolist())))

    # ---------- Mode 1: random sample from a chosen fault class ----------
    if selection_mode == "By fault class (random index)":
        fault_class = st.sidebar.selectbox(
            "Choose fault class (from test set labels)",
            options=unique_classes,
            index=0,
        )
        st.session_state.test_fault_class = int(fault_class)

        if st.sidebar.button("Select Random Test Sample"):
            st.session_state.last_prediction = None
            st.session_state.true_label_shown = False

            indices = np.where(streamlit_test_Y.flatten() == fault_class)[0]
            if len(indices) == 0:
                st.warning(f"No test samples found for class {fault_class}.")
            else:
                idx = int(np.random.choice(indices))
                st.session_state.test_sample_index = idx

                # Clean feature (FFT)
                clean_feature = streamlit_test_X[idx].astype(np.float32).copy()
                st.session_state.original_test_feature = clean_feature.copy()

                

                # These are used downstream
                st.session_state.prediction_ready_feature = clean_feature.copy()
                st.session_state.test_sample_signal = clean_feature.copy()
                st.session_state.sample_loaded = True

                # Reset generator state
                st.session_state.generated_faults = None
                st.session_state.selected_generated_idx = None
                st.session_state.generation_done = False

                # Reset S1 reference lock when loading a new test sample
                st.session_state.refs_locked = False
                st.session_state.s1_lock_key = None


                # Reset S1/S2 reference preview state
                st.session_state.random_real_refs = None
                st.session_state.random_real_ref_indices = None
                st.session_state.random_real_pred_label = None
                st.session_state.random_real_ref_source = None


                msg_noise = (
                    f" with noise ({st.session_state.noise_level})"
                    if st.session_state.noise_enabled
                    else " (no added noise)"
                )
                st.session_state.messages.append(
                    ("success", f"Selected test sample index {idx}{msg_noise}.")
                )

                # Y-scaling from the loaded (possibly noisy) sample
                real = st.session_state.test_sample_signal
                st.session_state.y_min = float(real.min())
                st.session_state.y_max = float(real.max())

    # ---------- Mode 2: directly specify the test index ----------
    elif selection_mode == "By test index":
        max_idx = streamlit_test_X.shape[0] - 1
        default_idx = (
            st.session_state.test_sample_index
            if st.session_state.test_sample_index is not None
            else 0
        )

        idx_input = st.sidebar.number_input(
            f"Test sample index (0..{max_idx})",
            min_value=0,
            max_value=int(max_idx),
            value=int(default_idx),
            step=1,
        )

        if st.sidebar.button("Load Test Sample by Index"):
            idx = int(idx_input)
            st.session_state.test_sample_index = idx

            clean_feature = streamlit_test_X[idx].astype(np.float32).copy()
            st.session_state.original_test_feature = clean_feature.copy()



            st.session_state.prediction_ready_feature = clean_feature.copy()
            st.session_state.test_sample_signal = clean_feature.copy()
            st.session_state.sample_loaded = True

            st.session_state.last_prediction = None
            st.session_state.true_label_shown = False
            st.session_state.generated_faults = None
            st.session_state.selected_generated_idx = None
            st.session_state.generation_done = False

            # Reset S1 reference lock when loading a new test sample
            st.session_state.refs_locked = False
            st.session_state.s1_lock_key = None


            # Reset S1/S2 reference preview state
            st.session_state.random_real_refs = None
            st.session_state.random_real_ref_indices = None
            st.session_state.random_real_pred_label = None
            st.session_state.random_real_ref_source = None




            true_cls = int(streamlit_test_Y.flatten()[idx])
            st.session_state.test_fault_class = true_cls

            msg_noise = (
                f" with noise ({st.session_state.noise_level})"
                if st.session_state.noise_enabled
                else " (no added noise)"
            )
            st.session_state.messages.append(
                ("success", f"Loaded test sample index {idx} (class {true_cls}){msg_noise}.")
            )

            real = st.session_state.test_sample_signal
            st.session_state.y_min = float(real.min())
            st.session_state.y_max = float(real.max())

    # ---------- Mode 3: random sample from misclassified indices ----------
    elif selection_mode == "Random misclassified sample (after full SVM eval)":

        if st.sidebar.button("Select Random Misclassified Sample"):
            mis_list = st.session_state.get("full_svm_mis_idx", [])

            if not mis_list:
                st.warning(
                    "No misclassified indices available. "
                    "Run 'Run Full SVM Accuracy Test' first."
                )
            else:
                idx = int(np.random.choice(mis_list))
                st.session_state.test_sample_index = idx

                clean_feature = streamlit_test_X[idx].astype(np.float32).copy()
                st.session_state.original_test_feature = clean_feature.copy()



                st.session_state.prediction_ready_feature = clean_feature.copy()
                st.session_state.test_sample_signal = clean_feature.copy()
                st.session_state.sample_loaded = True

                st.session_state.last_prediction = None
                st.session_state.true_label_shown = False
                st.session_state.generated_faults = None
                st.session_state.selected_generated_idx = None
                st.session_state.generation_done = False

                # Reset S1 reference lock when loading a new test sample
                st.session_state.refs_locked = False
                st.session_state.s1_lock_key = None

                # Reset S1/S2 reference preview state
                st.session_state.random_real_refs = None
                st.session_state.random_real_ref_indices = None
                st.session_state.random_real_pred_label = None
                st.session_state.random_real_ref_source = None





                true_cls = int(streamlit_test_Y.flatten()[idx])
                st.session_state.test_fault_class = true_cls

                msg_noise = (
                    f" with noise ({st.session_state.noise_level})"
                    if st.session_state.noise_enabled
                    else " (no added noise)"
                )
                st.session_state.messages.append(
                    #("success",
                    # f"Loaded MISCLASSIFIED test sample index {idx} (class {true_cls}){msg_noise}.")
                    ("success",
                     f"Loaded MISCLASSIFIED test sample index {idx} (class unknown){msg_noise}.")
                )

                real = st.session_state.test_sample_signal
                st.session_state.y_min = float(real.min())
                st.session_state.y_max = float(real.max())

    # ---------- Mode 4: random sample from entire test set ----------
    elif selection_mode == "Random sample from all test samples":

        if st.sidebar.button("Select Random Test Sample (All)"):
            #max_idx = streamlit_test_X.shape[0] - 1
            idx = _pop_next_unseen_index(streamlit_test_X.shape[0])
            st.session_state.test_sample_index = idx

            clean_feature = streamlit_test_X[idx].astype(np.float32).copy()
            st.session_state.original_test_feature = clean_feature.copy()



            st.session_state.prediction_ready_feature = clean_feature.copy()
            st.session_state.test_sample_signal = clean_feature.copy()
            st.session_state.sample_loaded = True

            # ---- experiment timing: reset to 0 and start immediately on sample load ----
            st.session_state.last_sample_load_dt = datetime.now().isoformat(timespec="seconds")
            _reset_and_start_timer()


            st.session_state.last_prediction = None
            st.session_state.true_label_shown = False
            st.session_state.generated_faults = None
            st.session_state.selected_generated_idx = None
            st.session_state.generation_done = False

            # Reset S1 reference lock when loading a new test sample
            st.session_state.refs_locked = False
            st.session_state.s1_lock_key = None


            # Reset S1/S2 reference preview state
            st.session_state.random_real_refs = None
            st.session_state.random_real_ref_indices = None
            st.session_state.random_real_pred_label = None
            st.session_state.random_real_ref_source = None



            true_cls = int(streamlit_test_Y.flatten()[idx])
            st.session_state.test_fault_class = true_cls

            msg_noise = (
                f" with noise ({st.session_state.noise_level})"
                if st.session_state.noise_enabled
                else " (no added noise)"
            )
            st.session_state.messages.append(
                #("success",
                # f"Loaded random test sample index {idx} (class {true_cls}){msg_noise}.")
                ("success",
                 f"Loaded random test sample index {idx} (class unknown){msg_noise}.")
            )

            real = st.session_state.test_sample_signal
            st.session_state.y_min = float(real.min())
            st.session_state.y_max = float(real.max())


        

    # --------------------------------------------------------
    # SVM EVALUATION ON FULL TEST SET WITH CURRENT NOISE LEVEL
    # --------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Evaluate SVM on Entire Test Set")



    if st.sidebar.button("Run Full SVM Accuracy Test"):
        run_full_svm_accuracy_test(svm_model, tester)

    with st.sidebar:
        st.write(st.session_state.full_svm_accuracy)
        st.write(st.session_state.full_svm_mis_idx)
        st.write(st.session_state.full_svm_mis_by_class)
        st.write(st.session_state.full_svm_mis_into_class)
    
    

    # -----------------------
    # TOP ROW: SVM prediction / true label
    # -----------------------
    if st.session_state.get("sample_loaded"):

        pred_col, label_col = st.columns([5,4])

        # --- SVM prediction panel ---
        with pred_col:
            pred = st.session_state.get("last_prediction")

            text_col, btn_col, trust_col, dont_col, empt_col_r = st.columns([3, 2, 2, 2, 5])

            with text_col:
                if pred is not None:
                    st.markdown(
                        f"<div class='small-subheader'>SVM Prediction: {pred}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        #"<div class='small-subheader'>SVM Prediction: –</div>",
                        f"<div class='small-subheader'>SVM Prediction: {pred}</div>",
                        unsafe_allow_html=True,
                    )

            with btn_col:
                if st.button(
                    "Predict",
                    key="predict_fault_label",
                    use_container_width=True,
                    disabled=pred is not None,
                ):
                    x = st.session_state.prediction_ready_feature.reshape(1, -1)
                    scaler = load_svm_scaler(st.session_state.get("svm_subdir", "baseline"))
                    x_scaled = scaler.transform(x)
                    pred = svm_model.predict(x_scaled)[0]

                    st.session_state.last_prediction = int(pred)

                    # ------------------------------------------------------------
                    # S2: immediately sample TWO real TRAIN reference for predicted label
                    # (so you no longer need the old "Sample ..." button)
                    # ------------------------------------------------------------
                    scenario_now = st.session_state.get("scenario_mode", "")
                    if str(scenario_now).startswith("S2"):
                        pred_now = int(st.session_state.last_prediction)

                        K = 2
                        refs = []
                        ref_indices = []

                        if pred_now == 9:
                            refs = []
                            for _ in range(2):
                                idx = np.random.randint(0, tester.domain_A_train_X.shape[0])
                                refs.append(tester.domain_A_train_X[idx].astype(np.float32))
                        else:
                            # predicted FAULT → sample from CycleGAN training samples of predicted class
                            refs = tester.get_random_fault_references_by_label(pred_now, 2)

                        st.session_state.random_real_refs = np.stack(refs, axis=0)

                        st.session_state.random_real_ref_indices = ref_indices
                        st.session_state.s1_display_order = None
                        st.session_state.s1_ref_kind = "normal" if pred_now == 9 else "fault"

                    st.rerun()
            
            acc = st.session_state.get("full_svm_accuracy")
            if acc is None:
                st.caption("SVM accuracy (full test set): –")
            else:
                st.caption(f"SVM accuracy (full test set): {acc*100:.2f}%")
            # -----------------------
            # Operator trust decision (keeps your layout)
            # -----------------------
            pred_now = st.session_state.get("last_prediction")


            # -----------------------------
            # Operator decision: Trust / Don't Trust
            # - stops timer
            # - logs one Excel row
            # - loads next unseen random test sample
            # -----------------------------

            def _log_and_advance(decision: str):
                # stop timer and compute elapsed
                elapsed_s = _stop_timer_and_get_elapsed()

                idx = st.session_state.get("test_sample_index")
                true_lbl = None
                if idx is not None:
                    try:
                        true_lbl = int(streamlit_test_Y.flatten()[int(idx)])
                    except Exception:
                        true_lbl = None

                pred_lbl = st.session_state.get("last_prediction")

                top3 = st.session_state.get("top3_similar", []) or []
                def _top(i, field):
                    if i < len(top3):
                        return top3[i].get(field, None)
                    return None

                row = {
                    "datetime_local": datetime.now().isoformat(timespec="milliseconds"),
                    "loaded_experiment": st.session_state.get("loaded_experiment"),
                    "svm_subdir": st.session_state.get("svm_subdir"),
                    "scenario_mode": st.session_state.get("scenario_mode"),
                    "noise_enabled": bool(st.session_state.get("noise_enabled", False)),
                    "noise_level": st.session_state.get("noise_level"),
                    "svm_full_accuracy": st.session_state.get("full_svm_accuracy"),

                    "sample_index": idx,
                    "true_label": true_lbl,
                    "predicted_label": pred_lbl,

                    "decision": decision,
                    "elapsed_seconds": float(elapsed_s),

                    "top1_label": _top(0, "label"),
                    "top1_score": _top(0, "score"),
                    "top2_label": _top(1, "label"),
                    "top2_score": _top(1, "score"),
                    "top3_label": _top(2, "label"),
                    "top3_score": _top(2, "score"),

                    "true_label_was_shown": bool(st.session_state.get("true_label_shown", False)),
                    "selected_generated_idx": st.session_state.get("selected_generated_idx"),

                    "penalty_score": _compute_penalty_score(decision, true_lbl,pred_lbl,),

                }

                # append to Excel
                _append_log_row(LOG_SHEET, LOG_COLUMNS, row)

                # load next unseen random test sample (same behavior as "random sample from all test samples")
                n_total = int(streamlit_test_X.shape[0])
                next_idx = _pop_next_unseen_index(n_total)
                st.session_state.test_sample_index = int(next_idx)

                clean_feature = streamlit_test_X[next_idx].astype(np.float32).copy()
                st.session_state.original_test_feature = clean_feature.copy()


                st.session_state.prediction_ready_feature = clean_feature.copy()
                st.session_state.test_sample_signal = clean_feature.copy()
                st.session_state.sample_loaded = True

                # reset decision + dependent UI state
                st.session_state.operator_trust = None
                st.session_state.last_prediction = None
                st.session_state.true_label_shown = False
                st.session_state.generated_faults = None
                st.session_state.selected_generated_idx = None
                st.session_state.generation_done = False

                # Reset S1 lock
                st.session_state.refs_locked = False
                st.session_state.s1_lock_key = None

                # Reset S1/S2 preview state
                st.session_state.random_real_refs = None
                st.session_state.random_real_ref_indices = None
                st.session_state.random_real_pred_label = None
                st.session_state.random_real_ref_source = None


                # update stored "test_fault_class" (you sometimes hide it in messages, but state is fine)
                try:
                    st.session_state.test_fault_class = int(streamlit_test_Y.flatten()[next_idx])
                except Exception:
                    pass

                # timing: reset to 0 and auto-start for the newly loaded sample
                st.session_state.last_sample_load_dt = datetime.now().isoformat(timespec="seconds")
                _reset_and_start_timer()

                st.rerun()

            with trust_col:
                if st.button("Trust", use_container_width=False, disabled=(st.session_state.get("test_sample_index") is None),icon="✅"):
                    _log_and_advance("trust")

            with dont_col:
                if st.button("Don't trust", use_container_width=False, disabled=(st.session_state.get("test_sample_index") is None),icon="❌"):
                    _log_and_advance("dont_trust")

           





       # --- True label panel ---
        with label_col:
            idx = st.session_state.get("test_sample_index")
            true_label = int(streamlit_test_Y.flatten()[idx]) if idx is not None else None
            label_shown = st.session_state.get("true_label_shown", False)

            # Layout: True Label | Show | Start | Time | Stop
            text_col, btn_col, start_col, time_col, stop_col = st.columns([2, 1, 1, 2, 1])

            with text_col:
                if label_shown and true_label is not None:
                    st.markdown(
                        f"<div class='small-subheader'>True Label: {true_label}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='small-subheader'>True Label: –</div>",
                        unsafe_allow_html=True,
                    )

            with btn_col:
                if st.button(
                    "Show",
                    key="show_true_label",
                    use_container_width=True,
                    disabled=label_shown or true_label is None,
                ):
                    st.session_state.true_label_shown = True
                    st.rerun()

            running = bool(st.session_state.get("timer_running", False))

            # Trigger periodic reruns while running so the timer display updates
            if running:
                try:
                    _autorefresh = getattr(st, "autorefresh", None) or getattr(st, "st_autorefresh", None)
                    if _autorefresh is not None:
                        _autorefresh(interval=500, key="timer_autorefresh")
                except Exception:
                    pass

            with start_col:
                st.markdown("<div style='display:none;'>", unsafe_allow_html=True)
                start_clicked = st.button(
                    "Start",
                    key="start_timer",
                    use_container_width=True,
                    disabled=running,
                )
                st.markdown("</div>", unsafe_allow_html=True)
                if start_clicked:
                    st.session_state.timer_running = True
                    st.session_state.timer_start_ts = time.time()
                    st.rerun()

            with time_col:
                pass  # timer display hidden

            with stop_col:
                st.markdown("<div style='display:none;'>", unsafe_allow_html=True)
                stop_clicked = st.button(
                    "Stop",
                    key="stop_timer",
                    use_container_width=True,
                    disabled=not running,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                if stop_clicked and running:
                    if st.session_state.timer_start_ts is not None:
                        st.session_state.timer_elapsed_s = float(st.session_state.timer_elapsed_s) + (
                            time.time() - float(st.session_state.timer_start_ts)
                        )
                    st.session_state.timer_running = False
                    st.session_state.timer_start_ts = None
                    st.rerun()


    # ----------------------------------------------
    # MAIN VISUALIZATION (Interactive Plotly version)
    # ----------------------------------------------
    st.subheader("Main Visualization")
    st.markdown('<div class="main-plot-container">', unsafe_allow_html=True)

    if st.session_state.test_sample_signal is None:
        # Same behavior as original Matplotlib version: push message to bottom area
        st.session_state.messages.append(
            ("info", "Use the sidebar to select a fault class and load a random test sample.")
        )
    else:
        real_sig = st.session_state.test_sample_signal

        # If no generated fault selected yet → plain view (single line, no band)
        if st.session_state.selected_generated_idx is None:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(real_sig)),
                    y=real_sig,
                    mode="lines",
                    name="Test spectrum",
                    line=dict(color="black", width=2),
                )
            )

            fig.update_layout(
                title=(
                    f"Test Sample – Index {st.session_state.test_sample_index} "
                    # f"(Class {st.session_state.test_fault_class})"
                ),
                height=400,
                template="plotly_white",
                xaxis_title="FFT bin index (0–511)",
                yaxis_title="FFT magnitude",
                margin=dict(l=40, r=20, t=60, b=40),
                showlegend=False,
            )
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)

            st.plotly_chart(fig, use_container_width=True)

        # Generated fault selected → overlay with red/blue band (exact Matplotlib behavior)
        else:
            # ---------------------------
            # S1/S2: MAIN = plain test only
            # ---------------------------
            if scenario.startswith("S1") or scenario.startswith("S2"):
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(real_sig)),
                        y=real_sig,
                        mode="lines",
                        name="Test spectrum",
                        line=dict(color="black", width=2),
                    )
                )
                fig.update_layout(
                    title=f"Test Sample – Index {st.session_state.test_sample_index}",
                    height=400,
                    template="plotly_white",
                    xaxis_title="FFT bin index (0–511)",
                    yaxis_title="FFT magnitude",
                    margin=dict(l=40, r=20, t=60, b=40),
                    showlegend=False,
                )
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)
                st.plotly_chart(fig, use_container_width=True)

            # ---------------------------
            # S3: overlay visualization (unchanged behavior)
            # ---------------------------
            else:
                idx_sel = st.session_state.selected_generated_idx

                # existing S3 guard
                if st.session_state.generated_faults is None:
                    st.session_state.selected_generated_idx = None
                    st.session_state.messages.append(("info", "S3 selected: click 'Generate Alternative Fault Versions' first."))
                    st.rerun()

                raw_ref = st.session_state.generated_faults[idx_sel]
                s3_labels = st.session_state.get("s3_class_labels", list(range(9)))
                class_label = int(s3_labels[idx_sel]) if idx_sel < len(s3_labels) else int(idx_sel)

                title = f"Test Sample {st.session_state.test_sample_index} vs Generated Class {class_label}"

                ref_sig, _, _ = scale_generated_like_real(real_sig, raw_ref)
                plot_elif8_overlay_plotly(real_sig, ref_sig, title)

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------
    # S2: Secondary plain main visualization (REFERENCE)
    # ----------------------------------------------
    if st.session_state.scenario_mode.startswith("S2"):

        refs_container = st.session_state.get("random_real_refs")
        if refs_container is None:
            pass  # nothing to show yet
        else:
            # Normalize to list of 1D spectra
            if isinstance(refs_container, np.ndarray) and refs_container.ndim == 2:
                refs_list = [refs_container[i, :] for i in range(refs_container.shape[0])]
            elif isinstance(refs_container, np.ndarray) and refs_container.ndim == 1:
                refs_list = [refs_container]
            else:
                refs_list = list(refs_container)

            st.subheader(f"Predicted (Class {pred}), SVM TRAIN Sample References")
            # st.markdown(
            #             f"<div class='small-subheader'>Predicted (Class {pred}), SVM TRAIN Sample References </div>",
            #             unsafe_allow_html=True,
            #         )

            def make_ref_fig(sig, j):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=np.arange(len(sig)),
                    y=sig,
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                ))
                fig.update_layout(
                    title=f"Real TRAIN Reference – Sample {j + 1}",
                    height=200,
                    template="plotly_white",
                    xaxis_title="FFT bin index (0–511)",
                    yaxis_title="FFT magnitude",
                    margin=dict(l=40, r=20, t=60, b=40),
                    showlegend=False,
                )
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)
                return fig

            col1, col2 = st.columns([2, 2])
            cols = [col1, col2]

            for j in range(min(2, len(refs_list))):
                with cols[j]:
                    st.plotly_chart(make_ref_fig(refs_list[j], j), use_container_width=True)


    # -----------------------------
    # PREVIEW PANEL (UPDATED WITH BORDER + 3 COLUMNS)
    # -----------------------------
    
    scenario = st.session_state.scenario_mode
    pred_now = st.session_state.get("last_prediction")

    if scenario.startswith("S3"):
        st.subheader("Generated fault hypotheses (CycleGAN, classes 0–8) conditioned on the selected test sample")
    elif scenario.startswith("S2"):
        pass
    #     st.subheader(f"Real TRAIN References for Predicted Label = {pred_now} (No overlay)")

    
    scenario = st.session_state.scenario_mode

    if scenario.startswith("S3"):
        gen_btn_label = "Generate CycleGAN References (1 per class)"
    elif scenario.startswith("S2"):
        gen_btn_label = "Sample Real TRAIN References (predicted label)"
    # else:
    #     gen_btn_label = "Lock Real TRAIN References (predicted label)"  # S1 baseline


    # --- S3 class label list (used for both generation + preview grid) ---
    exclude_fault_2_active = (st.session_state.get("loaded_experiment") == "Exclude fault 2")
    S3_CLASS_LABELS = [0, 1, 3, 4, 5, 6, 7, 8] if exclude_fault_2_active else list(range(9))

    # store so main-plot + preview can both use the same mapping
    st.session_state["s3_class_labels"] = S3_CLASS_LABELS

    # Generate button remains unchanged
    if st.session_state.test_sample_signal is not None:
         
        if scenario.startswith("S3"):

            pred_now = st.session_state.get("last_prediction")
            can_generate = (pred_now is not None)

            if st.button(gen_btn_label, disabled=not can_generate):
                # -----------------------
                # S1/S2: random real references (NOT the test slice)
                # -----------------------
                if scenario.startswith("S1") or scenario.startswith("S2"):

                    pred_now = st.session_state.get("last_prediction")
                    
                    # -----------------------
                    # S1 lock: prevent re-sampling references for the same test+prediction
                    # -----------------------
                    if scenario.startswith("S1") or scenario.startswith("S2"):
                        test_idx = st.session_state.get("test_sample_index")
                        lock_key = (test_idx, pred_now)

                        if st.session_state.get("refs_locked", False) and st.session_state.get("s1_lock_key") == lock_key:
                            st.session_state.messages.append(
                                ("info", "References are locked for this test sample and prediction. Select a new test sample to refresh.")
                            )
                            st.rerun()

                        # Allow sampling once for this (test sample, prediction) and then lock it
                        st.session_state.s1_lock_key = lock_key

                    # Require a prediction so S1/S2 can sample "predicted-class-only" references
                    if pred_now is None:
                        st.session_state.messages.append(("info", "Run 'Predict' first so S1/S2 can sample references from the predicted class."))
                        st.rerun()

                    # Always sample 4 references to keep your 9-column layout unchanged
                    K = 2 if scenario.startswith("S2") else 1

                    refs = []
                    ref_indices = []

                    if int(pred_now) == 9:
                        # sample normal references from training set
                        for _ in range(K):
                            ridx = int(np.random.randint(0, tester.domain_A_train_X.shape[0]))
                            refs.append(tester.domain_A_train_X[ridx].astype(np.float32))
                            ref_indices.append(ridx)

                        st.session_state.random_real_ref_source = "normal_train"

                    else:
                        # sample from CycleGAN training samples of predicted fault class
                        refs = tester.get_random_fault_references_by_label(int(pred_now), K)
                        ref_indices = list(range(len(refs)))

                        st.session_state.random_real_ref_source = "fault_train"

                    st.session_state.random_real_pred_label = int(pred_now)
                    st.session_state.random_real_refs = np.stack(refs, axis=0)
                    st.session_state.random_real_ref_indices = ref_indices
                    st.session_state.s1_display_order = None
                    st.session_state.s1_ref_kind = "normal" if int(pred_now) == 9 else "fault"
                    st.session_state.selected_generated_idx = None
                # -----------------------
                # S3: your existing CycleGAN generation (unchanged)
                # -----------------------
                else:
                    pred_now = st.session_state.get("last_prediction")

                    # ------------------------------------------------------------
                    # If predicted NORMAL (9): cycle consistency test
                    # normal -> (generate fault via g_AB) -> back to normal via g_BA
                    # ------------------------------------------------------------
                    if pred_now == 9:
                        real_normal = st.session_state.test_sample_signal.astype(np.float32)
                        real_normal_batch = real_normal[np.newaxis, :]

                        target_norm_label = np.array([0], dtype=np.int32)

                        # pick ONE fault label to run the cycle through
                        f = 0
                        lab_f = np.array([f], dtype=np.int32)

                        gen_fault = tester.gan.generate_samples(
                            real_normal_batch,
                            lab_f,
                            generator="g_AB",
                        )[0].astype(np.float32)

                        gen_back_normal = tester.gan.generate_samples(
                            gen_fault[np.newaxis, :],
                            target_norm_label,
                            generator="g_BA",
                        )[0].astype(np.float32)

                        # Store as shape (1, 512) so the rest of your code can treat it like a grid
                        st.session_state.generated_faults = gen_back_normal[np.newaxis, :]
                        st.session_state.selected_generated_idx = None
                        st.session_state.messages.append(("success", "S3 (pred=9): Generated ONE normal reference (Normal→Fault→Normal)."))

                        # keep your y-limits expansion as-is
                        real_sig = st.session_state.test_sample_signal.astype(np.float32)
                        for f in st.session_state.generated_faults:
                            f_scaled, ymin, ymax = scale_generated_like_real(real_sig, f)
                            st.session_state.y_min = min(st.session_state.y_min, float(ymin))
                            st.session_state.y_max = max(st.session_state.y_max, float(ymax))

                    # ------------------------------------------------------------
                    # Otherwise (pred != 9): keep your existing S3 fault pipeline unchanged
                    # ------------------------------------------------------------
                    else:
                        real_fault = st.session_state.test_sample_signal.astype(np.float32)
                        real_fault_batch = real_fault[np.newaxis, :]

                        target_norm_label = np.array([0], dtype=np.int32)
                        generated_normal = tester.gan.generate_samples(
                            real_fault_batch,
                            target_norm_label,
                            generator='g_BA'
                        )[0]

                        generated_faults = []
                        s3_labels = st.session_state.get("s3_class_labels", list(range(9)))

                        for f in s3_labels:
                            lab = np.array([int(f)], dtype=np.int32)
                            gen_f = tester.gan.generate_samples(
                                generated_normal[np.newaxis, :],
                                lab,
                                generator="g_AB",
                            )
                            generated_faults.append(gen_f[0])

                        st.session_state.generated_faults = np.array(generated_faults)
                        st.session_state.selected_generated_idx = None
                        st.session_state.messages.append(("success", "Generated alternative fault versions (0–8)."))

                        real_sig = st.session_state.test_sample_signal.astype(np.float32)
                        for f in st.session_state.generated_faults:
                            f_scaled, ymin, ymax = scale_generated_like_real(real_sig, f)
                            st.session_state.y_min = min(st.session_state.y_min, float(ymin))
                            st.session_state.y_max = max(st.session_state.y_max, float(ymax))
            if not can_generate:
                st.caption("Run 'Predict' first to enable generation / sampling.")


            else:
                scenario = st.session_state.scenario_mode
                
                # if scenario.startswith("S3"):
                #     st.info("Select a test sample first, then click 'Generate CycleGAN References (1 per class)'.")
                # elif scenario.startswith("S2"):
                #     st.info("Select a test sample first. Then click 'Predict' and sample real TRAIN references.")
                # else:
                #     # S1: no reference area / no generate button
                #     st.info("Select a test sample first.")

    # PREVIEW GRID
    scenario = st.session_state.scenario_mode
    # grid_ready = (
    #     (scenario.startswith("S3") and st.session_state.generated_faults is not None) or
    #     ((scenario.startswith("S1") or scenario.startswith("S2")) and st.session_state.random_real_refs is not None)
    # )
    grid_ready = (
        scenario.startswith("S3") and st.session_state.generated_faults is not None
    )

    if grid_ready:
        real_sig = st.session_state.test_sample_signal  # current main signal

        # -----------------------
        # S3: N columns (9 normally, 8 when excluding fault 2)
        # -----------------------
        if scenario.startswith("S3"):
            s3_labels = st.session_state.get("s3_class_labels", list(range(9)))
            cols = st.columns(len(s3_labels))

            n_slots = len(s3_labels)
            n_gen = int(st.session_state.generated_faults.shape[0])
            pred_now = st.session_state.get("last_prediction")

            # If pred==9, you stored only ONE generated sample; keep layout and center it
            pred9_single = (pred_now == 9 and n_gen == 1)
            mid_slot = n_slots // 2

            # ---------- FIRST PASS: compute NRMSE scores ONLY ----------
            # Keep list length == n_slots so later indexing stays consistent.
            s3_nrmse_scores = [-1e18] * n_slots
            real_sig = st.session_state.test_sample_signal

            if pred9_single:
                raw_gen_sig = st.session_state.generated_faults[0]
                gen_sig, _, _ = scale_generated_like_real(real_sig, raw_gen_sig)
                s3_nrmse_scores[mid_slot] = sim_nrmse(real_sig, gen_sig)
            else:
                for i, class_label in enumerate(s3_labels):
                    raw_gen_sig = st.session_state.generated_faults[i]
                    gen_sig, _, _ = scale_generated_like_real(real_sig, raw_gen_sig)
                    #########################################################################################################s3_nrmse_scores[i] = sim_nrmse(real_sig, gen_sig)
                    s3_nrmse_scores[i] = sim_nrmse_as_displayed(real_sig, raw_gen_sig)
            # Top-3 BEST scores (higher = better in your dashboard)
            _top3_idx = set(np.argsort(s3_nrmse_scores)[-3:])

            # Rank every slot: rank 1 = best score
            _all_sorted_desc = list(np.argsort(s3_nrmse_scores)[::-1])
            slot_rank = {int(slot_i): rank + 1 for rank, slot_i in enumerate(_all_sorted_desc)}


            # Cache Top-3 (label, score) for logging
            _top3_sorted = list(np.argsort(s3_nrmse_scores)[-3:][::-1])  # best -> 2nd -> 3rd

            top3 = []
            for slot_i in _top3_sorted:
                score_i = float(s3_nrmse_scores[int(slot_i)])
                #########################################################################score_i = float(s3_nrmse_scores[int(slot_i)])
                if pred9_single:
                    lbl_i = 9  # centered tile represents "9" in your UI
                else:
                    lbl_i = int(s3_labels[int(slot_i)])
                top3.append({"label": lbl_i, "score": score_i})

            st.session_state.top3_similar = top3

            # ---------- SECOND PASS: render previews ----------
            for slot, class_label in enumerate(s3_labels):
                with cols[slot]:

                    # Empty slots when pred==9 (keep tile spacing)
                    if pred9_single and slot != mid_slot:
                        st.markdown("<div style='height:140px;'></div>", unsafe_allow_html=True)
                        continue

                    # Choose which generated signal to use
                    gen_idx = 0 if pred9_single else slot
                    raw_gen_sig = st.session_state.generated_faults[gen_idx]
                    gen_sig, ymin, ymax = scale_generated_like_real(real_sig, raw_gen_sig)

                    # Display label: show "9" for the centered pred==9 tile
                    display_label = 9 if pred9_single else int(class_label)

                    is_selected = (st.session_state.get("selected_generated_idx") == gen_idx)
                    is_top3 = (slot in _top3_idx)

                    # predicted-tile indicator (Option C) — make the centered tile count as predicted when pred==9
                    is_pred_tile = (pred_now is not None and ((pred9_single and slot == mid_slot) or (int(class_label) == int(pred_now))))

                    fig, ax = plt.subplots(figsize=(4.3, 1.8))

                    # ---- background highlight for selected tile ----
                    if is_selected:
                        ax.set_facecolor("#b8b8b8")
                    else:
                        ax.set_facecolor("white")

                    # ---- Option C: border color for predicted tile ----
                    if is_pred_tile:
                        for spine in ax.spines.values():
                            spine.set_edgecolor("red")

                    ax.plot(gen_sig, color="green", linewidth=0.8)

                    # ---- frame thickness on the GRAPH ITSELF ----
                    border_width = 2.0
                    if is_top3:
                        border_width = 5.0

                    for spine in ax.spines.values():
                        spine.set_linewidth(border_width)

                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_ylim(ymin, ymax)
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)
                    plt.close(fig)

                    if st.button(
                        f"Select Class {display_label}",
                        key=f"select_preview_S3_{display_label}_{slot}",
                    ):
                        st.session_state.selected_generated_idx = gen_idx
                        st.session_state.selected_generated_class_label = int(display_label)
                        st.rerun()

                    # Display SAME score used for ranking (slot-based)
                    s_nrmse = s3_nrmse_scores[slot]
                    s_rank = slot_rank[slot]

                    st.markdown(
                        f"<div style='font-size:0.75rem; line-height:1.1;'>"
                        f"<b>Rank #{s_rank}</b> &nbsp;|&nbsp; "
                        f"Similarity-Score(NRMSE): <b>{s_nrmse:.3f}</b>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        # -----------------------
        # S1/S2: keep 9-slot layout with spacers (your existing behavior)
        # -----------------------
        else:
            cols = st.columns(9)
            slot_positions = [4]
            slot_to_ref_idx = {
                slot_positions[i]: i
                for i in range(min(4, st.session_state.random_real_refs.shape[0]))
            }

            for slot in range(9):
                with cols[slot]:
                    if slot not in slot_to_ref_idx:
                        st.markdown("<div style='height:140px;'></div>", unsafe_allow_html=True)
                        continue

                    f_idx = slot_to_ref_idx[slot]
                    raw_ref_sig = st.session_state.random_real_refs[f_idx]
                    ref_sig, ymin, ymax = scale_generated_like_real(real_sig, raw_ref_sig)


                    
                    is_selected = (st.session_state.get("selected_generated_idx") == f_idx)
                    css_class = "preview-box selected" if is_selected else "preview-box"
                    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)

                    fig, ax = plt.subplots(figsize=(4.3, 1.8))

                    if is_selected:
                        ax.set_facecolor("#b8b8b8") # very light grey
                    else:
                        ax.set_facecolor("white")

                    ax.plot(ref_sig, color="green", linewidth=0.8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_ylim(ymin, ymax)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

                    


                    if not scenario.startswith("S1"):
                        if st.button(f"Select Ref #{f_idx + 1}", key=f"select_preview_{scenario}_{slot}_{f_idx}"):
                            st.session_state.selected_generated_idx = f_idx
                            st.rerun()
                    else:
                        st.markdown("<div style='height:2.2rem;'></div>", unsafe_allow_html=True)

                    # Similarity scores for THIS tile (higher is better)
                    # (Only show for S2; S1 is intentionally "no overlay / no guidance")
                    if scenario.startswith("S2"):
                        s_nrmse = sim_nrmse(real_sig, ref_sig)
                        #s_cos   = sim_cosine_zscore(real_sig, ref_sig)
                        #s_spec  = sim_weighted_spectrum(real_sig, ref_sig)

                        st.markdown(
                            f"<div style='font-size:0.75rem; line-height:1.1;'>"
                            f"NRMSE: <b>{s_nrmse:.3f}</b><br>"
                            #f"Cos(z): <b>{s_cos:.3f}</b><br>"
                            #f"Spec(w): <b>{s_spec:.3f}</b>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    


                    


        # ------------------------------------------------------------
        # Show which label each metric selects as "best match"
        # ------------------------------------------------------------
        # if not scenario.startswith("S1"):
        #     if (
        #         'best_nrmse_label' in st.session_state and
        #         'best_cos_label' in st.session_state and
        #         'best_spec_label' in st.session_state
        #     ):
        #         st.markdown(
        #             "<div style='font-size:0.95rem; line-height:1.25;'>"
        #             "<b>Best label by each similarity metric (higher = better):</b><br>"
        #             f"NRMSE → <b>{st.session_state.best_nrmse_label}</b> "
        #             f"(score {st.session_state.best_nrmse_score:.3f})<br>"
        #             f"Cos(z) → <b>{st.session_state.best_cos_label}</b> "
        #             f"(score {st.session_state.best_cos_score:.3f})<br>"
        #             f"Spec(w) → <b>{st.session_state.best_spec_label}</b> "
        #             f"(score {st.session_state.best_spec_score:.3f})"
        #             "</div>",
        #             unsafe_allow_html=True
        #         )
            
        if scenario.startswith("S1"):
            st.caption("Reference gallery (unlabeled). Use it only for context; decide whether to trust the prediction.")
        else:
            st.caption("Click a preview to see the detailed overlay on the left.")


    
    # ============================================================
    # BOTTOM: SHOW MESSAGES (auto-clear after rendering)
    # ============================================================
    message_area = st.container()
    with message_area:
        for mtype, text in st.session_state.messages:
            if mtype == "success":
                st.success(text)
            elif mtype == "info":
                st.info(text)
            elif mtype == "warning":
                st.warning(text)
            elif mtype == "error":
                st.error(text)

    # Clear messages so they only appear once
    st.session_state.messages = []

if __name__ == "__main__":
    main()
