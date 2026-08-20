"""
Feature Extraction — Morlet Scalograms + PLV Connectivity
---------------------------------------------------------
Step 4: After preprocessing (step 3) + channel selection.

Features extracted (both NON-LINEAR):
  1. Morlet Scalograms  — time-frequency power per channel per trial
  2. PLV Matrices       — phase locking value between channel pairs per band

Input:  S01_clean.npz through S15_clean.npz  (795 × 64 × 300)
Output per subject:  {subj_id}_features.npz containing:
  - scalograms : (n_trials, 20, n_freqs, n_times)  float32, log10 power
  - plv        : (n_trials, 20, 20, 5)              float32, PLV per band
  - labels     : (n_trials,)                         int, 0-4
  - channels   : (20,)                               channel names used
  - freqs      : (n_freqs,)                          CWT center frequencies
  - bands      : (5,)                                PLV band names

Run:  py -3.11 step4_feature_extraction.py
"""

import os
import sys
import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
PREPROCESSED_FOLDER = r"C:\Users\kkl24\Downloads\CMRG_PROJECT_EEG_BASED_IMAGINED_SPEECH\preprocessed_data"
OUTPUT_FOLDER = r"C:\Users\kkl24\Downloads\CMRG_PROJECT_EEG_BASED_IMAGINED_SPEECH\features"
PLOT_FOLDER = r"C:\Users\kkl24\Downloads\CMRG_PROJECT_EEG_BASED_IMAGINED_SPEECH\feature_validation_plots"

# ---------------------------------------------------------------
# PASTE YOUR VALIDATED 20 CHANNELS FROM CHANNEL SELECTION OUTPUT
# (ordered by consistency rank, artifact channels excluded)
# ---------------------------------------------------------------
TOP_20_CHANNELS = [
    "C6",  "C4",  "CP4", "C3",  "FC4",
    "CP3", "FC3", "C5",  "Cz",  "FC1",
    "FC2", "C1",  "C2",  "CP5", "F7",
    "F5",  "FC5", "T7",  "TP7", "Fz",
]

FS = 256
BASELINE_SAMPLES = int(0.5 * FS)  # 128 samples = 500ms pre-stimulus

# --- Scalogram parameters ---
N_FREQS = 30                       # log-spaced frequency bins
FREQ_MIN, FREQ_MAX = 1.0, 50.0    # Hz (full bandpass range)
OMEGA0 = 6                        # Morlet wavelet parameter
TIME_DS = 2                        # downsample factor for time axis

# --- PLV parameters ---
BANDS = {
    "theta":     (4, 8),
    "alpha":     (8, 13),
    "low_beta":  (13, 20),
    "high_beta": (20, 30),
    "low_gamma": (30, 50),
}

# Precompute CWT frequencies and widths
FREQS = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQS)
WIDTHS = OMEGA0 * FS / (2 * np.pi * FREQS)

CLASS_NAMES = ["Hello", "Help me", "Stop", "Thank you", "Yes"]


# ============================================================
# FEATURE 1: MORLET SCALOGRAMS (NON-LINEAR)
# ============================================================
def extract_scalograms(x, ch_indices):
    """
    Continuous Wavelet Transform with Morlet wavelet.
    Input:  x (795, 64, 300)
    Output: (300, 20, 30, n_times_ds) — log10 power, float32
    """
    x_speech = x[BASELINE_SAMPLES:]          # (667, 64, 300)
    n_time = x_speech.shape[0]
    n_trials = x_speech.shape[2]
    n_ch = len(ch_indices)
    n_times_ds = len(range(0, n_time, TIME_DS))

    scalograms = np.zeros((n_trials, n_ch, N_FREQS, n_times_ds), dtype=np.float32)

    for t in range(n_trials):
        for ci, ch_idx in enumerate(ch_indices):
            sig = x_speech[:, ch_idx, t]
            coefs, _ = pywt.cwt(sig, WIDTHS, wavelet='cmor1.5-1.0', sampling_period=1.0/FS)
            power = np.abs(coefs) ** 2
            scalograms[t, ci] = np.log10(power[:, ::TIME_DS] + 1e-12)

        if (t + 1) % 50 == 0 or t == 0:
            print(f"    scalograms: {t+1}/{n_trials} trials done")

    return scalograms


# ============================================================
# FEATURE 2: PLV CONNECTIVITY MATRICES (NON-LINEAR)
# ============================================================
def extract_plv(x, ch_indices):
    """
    Phase Locking Value between all channel pairs per frequency band.
    Input:  x (795, 64, 300)
    Output: (300, 20, 20, 5) — PLV per band, float32
    """
    x_speech = x[BASELINE_SAMPLES:]          # (667, 64, 300)
    n_trials = x_speech.shape[2]
    n_ch = len(ch_indices)
    n_bands = len(BANDS)

    # Select 20 channels: (667, 20, 300)
    x_sel = x_speech[:, ch_indices, :]
    plv_all = np.zeros((n_trials, n_ch, n_ch, n_bands), dtype=np.float32)

    BATCH = 50  # trials per batch (memory management)

    for bi, (band_name, (lo, hi)) in enumerate(BANDS.items()):
        # Bandpass filter all channels/trials at once
        sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
        filtered = sosfiltfilt(sos, x_sel, axis=0)     # (667, 20, 300)

        # Instantaneous phase via Hilbert transform
        analytic = hilbert(filtered, axis=0)            # (667, 20, 300)
        phases = np.angle(analytic)                     # (667, 20, 300)
        phases_t = phases.transpose(2, 1, 0)            # (300, 20, 667)

        # Vectorized PLV computation in batches
        for t0 in range(0, n_trials, BATCH):
            t1 = min(t0 + BATCH, n_trials)
            batch = phases_t[t0:t1]                     # (batch, 20, 667)

            # Phase difference matrix: (batch, 20, 20, 667)
            pd = batch[:, :, np.newaxis, :] - batch[:, np.newaxis, :, :]
            plv_all[t0:t1, :, :, bi] = np.abs(
                np.mean(np.exp(1j * pd), axis=3)
            ).astype(np.float32)

        print(f"    PLV: {band_name} ({lo}-{hi} Hz) done")

    return plv_all


# ============================================================
# VALIDATION PLOTS
# ============================================================
def plot_sample_scalogram(scalograms, labels, channels, subj_id):
    """Plot scalogram for 1 trial per class, channel 0 (top-ranked)."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    n_times = scalograms.shape[3]
    t_axis = np.linspace(0, n_times * TIME_DS / FS, n_times)

    for cls in range(5):
        trial_idx = np.where(labels == cls)[0][0]
        ax = axes[cls]
        im = ax.pcolormesh(t_axis, FREQS, scalograms[trial_idx, 0],
                           cmap="jet", shading="auto")
        ax.set_yscale("log")
        ax.set_ylabel("Freq (Hz)" if cls == 0 else "")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{CLASS_NAMES[cls]}", fontsize=10)
        ax.set_yticks([1, 4, 8, 13, 20, 30, 50])
        ax.set_yticklabels(["1", "4", "8", "13", "20", "30", "50"])

    plt.suptitle(f"{subj_id} — Morlet Scalogram — {channels[0]} (top channel)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, f"{subj_id}_scalogram_sample.png"), dpi=150)
    plt.close()


def plot_sample_plv(plv, labels, channels, subj_id):
    """Plot mean PLV matrix per class for one band (theta)."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    band_idx = 0  # theta

    for cls in range(5):
        cls_mask = labels == cls
        mean_plv = np.mean(plv[cls_mask, :, :, band_idx], axis=0)
        ax = axes[cls]
        im = ax.imshow(mean_plv, cmap="hot", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{CLASS_NAMES[cls]}", fontsize=10)
        if cls == 0:
            ax.set_xticks(range(0, 20, 4))
            ax.set_yticks(range(0, 20, 4))
            ax.set_xticklabels([channels[i] for i in range(0, 20, 4)],
                               fontsize=6, rotation=45)
            ax.set_yticklabels([channels[i] for i in range(0, 20, 4)],
                               fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.colorbar(im, ax=axes, fraction=0.02, label="PLV")
    plt.suptitle(f"{subj_id} — Mean PLV Matrix — Theta Band (4-8 Hz)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, f"{subj_id}_plv_sample.png"), dpi=150)
    plt.close()


def plot_plv_class_difference(plv, labels, channels, subj_id):
    """Plot PLV difference between two classes to show discriminability."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    band_names = list(BANDS.keys())

    # Compare class 0 (Hello) vs class 2 (Stop) across all bands
    mask_0 = labels == 0
    mask_2 = labels == 2

    for bi in range(5):
        mean_0 = np.mean(plv[mask_0, :, :, bi], axis=0)
        mean_2 = np.mean(plv[mask_2, :, :, bi], axis=0)
        diff = mean_0 - mean_2

        ax = axes[bi]
        vmax = max(abs(diff.min()), abs(diff.max()), 0.05)
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(f"{band_names[bi]}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.colorbar(im, ax=axes, fraction=0.02, label="PLV difference")
    plt.suptitle(f"{subj_id} — PLV Difference: Hello vs Stop (red = Hello higher)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, f"{subj_id}_plv_difference.png"), dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    npz_files = sorted([f for f in os.listdir(PREPROCESSED_FOLDER)
                        if f.endswith("_clean.npz")])
    if not npz_files:
        print(f"ERROR: No _clean.npz files in {PREPROCESSED_FOLDER}")
        return

    n_subj = len(npz_files)
    print(f"Subjects: {n_subj}")
    print(f"Channels: {len(TOP_20_CHANNELS)} — {TOP_20_CHANNELS[:5]}...")
    print(f"Scalogram: {N_FREQS} freqs, {FREQ_MIN}-{FREQ_MAX} Hz, "
          f"time downsample ×{TIME_DS}")
    print(f"PLV bands: {list(BANDS.keys())}")
    print(f"Feature tags: SCALOGRAMS=non-linear, PLV=non-linear\n")

    for i, fname in enumerate(npz_files):
        filepath = os.path.join(PREPROCESSED_FOLDER, fname)
        subj_id = fname.replace("_clean.npz", "")

        print(f"{'='*50}")
        print(f"{subj_id} ({i+1}/{n_subj})")
        print(f"{'='*50}")

        # Load preprocessed data
        d = np.load(filepath, allow_pickle=True)
        x = d["x"]                          # (795, 64, 300)
        y = d["y"]                          # (5, 300)
        clab = list(d["clab"])
        labels = np.argmax(y, axis=0)       # (300,) values 0-4

        # Map channel names → indices
        try:
            ch_indices = [clab.index(ch) for ch in TOP_20_CHANNELS]
        except ValueError as e:
            print(f"  ERROR: Channel not found — {e}")
            print(f"  Available: {clab}")
            sys.exit(1)

        t_start = time.time()

        # --- Feature 1: Scalograms ---
        print("  [NON-LINEAR] Extracting Morlet scalograms...")
        scalograms = extract_scalograms(x, ch_indices)
        print(f"    Output shape: {scalograms.shape}")

        # --- Feature 2: PLV ---
        print("  [NON-LINEAR] Extracting PLV connectivity...")
        plv = extract_plv(x, ch_indices)
        print(f"    Output shape: {plv.shape}")

        elapsed = time.time() - t_start
        print(f"  Extraction time: {elapsed:.1f}s")

        # --- Save ---
        save_path = os.path.join(OUTPUT_FOLDER, f"{subj_id}_features.npz")
        np.savez_compressed(
            save_path,
            scalograms=scalograms,
            plv=plv,
            labels=labels,
            channels=np.array(TOP_20_CHANNELS),
            freqs=FREQS,
            bands=np.array(list(BANDS.keys())),
        )
        fsize = os.path.getsize(save_path) / (1024**2)
        print(f"  Saved: {save_path} ({fsize:.1f} MB)")

        # --- Validation plots (Subject 1 only) ---
        if i == 0:
            print("  Generating validation plots...")
            plot_sample_scalogram(scalograms, labels, TOP_20_CHANNELS, subj_id)
            print(f"    Saved: {subj_id}_scalogram_sample.png")
            plot_sample_plv(plv, labels, TOP_20_CHANNELS, subj_id)
            print(f"    Saved: {subj_id}_plv_sample.png")
            plot_plv_class_difference(plv, labels, TOP_20_CHANNELS, subj_id)
            print(f"    Saved: {subj_id}_plv_difference.png")

        print()

    # --- Summary ---
    print("=" * 50)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"  Output folder : {OUTPUT_FOLDER}")
    print(f"  Plots folder  : {PLOT_FOLDER}")
    print(f"  Per subject   :")
    print(f"    scalograms  : {scalograms.shape}  (trial × ch × freq × time)")
    print(f"    plv         : {plv.shape}  (trial × ch × ch × band)")
    print(f"    labels      : {labels.shape}  (0=Hello, 1=Help me, "
          f"2=Stop, 3=Thank you, 4=Yes)")
    print(f"\n  Feature summary:")
    print(f"    Scalograms  — NON-LINEAR — Morlet CWT log-power")
    print(f"    PLV         — NON-LINEAR — phase locking value")
    print("=" * 50)


if __name__ == "__main__":
    main()
