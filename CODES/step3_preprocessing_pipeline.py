"""
Standard Preprocessing Pipeline — Imagined Speech EEG
------------------------------------------------------
Steps (in order):
  1. Bandpass filter 0.5-50 Hz
  2. Notch filter at 50 Hz (power line noise)
  3. ICA artifact removal (eye blinks)
  4. Common Average Reference (CAR)
  5. Baseline correction (-500ms to 0ms)

Generates validation plots for Subject 1 (one per step).
Saves cleaned data for all 15 subjects as .npz files.

Install:  pip install numpy scipy matplotlib mne
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch as scipy_welch
import mne
from mne.preprocessing import ICA

mne.set_log_level("WARNING")

# ============================================================
# CONFIG
# ============================================================
TRAIN_FOLDER = r"C:\Users\kkl24\Downloads\pq7vb-osfstorage-Track#3 Imagined speech classification-archive\Training set"
OUTPUT_FOLDER = "preprocessed_data"
PLOT_FOLDER = "validation_plots"
# ============================================================


# ============================================================
# LOADING
# ============================================================
def load_subject(filepath):
    mat = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    epo_key = [k for k in mat.keys() if k.startswith("epo")][0]
    epo = mat[epo_key]
    x = np.array(epo.x, dtype=np.float64)   # (795, 64, 300)
    y = np.array(epo.y)                      # (5, 300)
    fs = int(epo.fs)                         # 256
    clab = [str(c) for c in epo.clab]        # 64 channel names
    return x, y, fs, clab


# ============================================================
# STEP 1: BANDPASS FILTER
# ============================================================
def bandpass_filter(x, fs, low=0.5, high=50.0, order=4):
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def validate_bandpass(x_raw, x_bp, fs, clab):
    ch = clab.index("C3")
    trial = 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PSD comparison
    f1, p1 = scipy_welch(x_raw[:, ch, trial], fs=fs, nperseg=256)
    f2, p2 = scipy_welch(x_bp[:, ch, trial], fs=fs, nperseg=256)
    axes[0].semilogy(f1, p1, label="Before", alpha=0.8, color="#e74c3c")
    axes[0].semilogy(f2, p2, label="After bandpass", alpha=0.8, color="#2ecc71")
    axes[0].axvline(0.5, color="grey", linestyle="--", alpha=0.5, label="0.5 Hz cutoff")
    axes[0].axvline(50, color="grey", linestyle="--", alpha=0.5, label="50 Hz cutoff")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (µV²/Hz)")
    axes[0].set_title("Step 1 Validation: PSD — C3, Trial 1")
    axes[0].legend(fontsize=8)

    # Time domain
    axes[1].plot(x_raw[:, ch, trial], label="Before", alpha=0.6, color="#e74c3c")
    axes[1].plot(x_bp[:, ch, trial], label="After bandpass", alpha=0.8, color="#2ecc71")
    axes[1].set_xlabel("Samples")
    axes[1].set_ylabel("Amplitude (µV)")
    axes[1].set_title("Step 1 Validation: Time Domain — C3, Trial 1")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step1_bandpass.png"), dpi=150)
    print("  Saved: step1_bandpass.png")
    plt.close()


# ============================================================
# STEP 2: NOTCH FILTER
# ============================================================
def notch_filter(x, fs, freq=50.0, Q=30):
    b, a = iirnotch(freq, Q, fs=fs)
    return filtfilt(b, a, x, axis=0)


def validate_notch(x_bp, x_notch, fs, clab):
    ch = clab.index("C3")
    trial = 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PSD zoomed around 50 Hz
    f1, p1 = scipy_welch(x_bp[:, ch, trial], fs=fs, nperseg=256)
    f2, p2 = scipy_welch(x_notch[:, ch, trial], fs=fs, nperseg=256)
    mask = (f1 >= 40) & (f1 <= 60)
    axes[0].plot(f1[mask], p1[mask], "o-", label="After bandpass (before notch)",
                 alpha=0.8, color="#e74c3c")
    axes[0].plot(f2[mask], p2[mask], "o-", label="After notch",
                 alpha=0.8, color="#2ecc71")
    axes[0].axvline(50, color="grey", linestyle="--", alpha=0.7, label="50 Hz line noise")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (µV²/Hz)")
    axes[0].set_title("Step 2 Validation: PSD Zoomed 40-60 Hz — C3, Trial 1")
    axes[0].legend(fontsize=8)

    # Full PSD for context
    axes[1].semilogy(f1, p1, label="Before notch", alpha=0.6, color="#e74c3c")
    axes[1].semilogy(f2, p2, label="After notch", alpha=0.8, color="#2ecc71")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD (µV²/Hz)")
    axes[1].set_title("Step 2 Validation: Full PSD — C3, Trial 1")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step2_notch.png"), dpi=150)
    print("  Saved: step2_notch.png")
    plt.close()


# ============================================================
# STEP 3: ICA ARTIFACT REMOVAL
# ============================================================
def apply_ica(x, fs, clab):
    # Convert to MNE: (n_epochs, n_channels, n_times) in Volts
    data = x.transpose(2, 1, 0) * 1e-6     # (300, 64, 795)
    info = mne.create_info(ch_names=clab, sfreq=fs, ch_types="eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)

    # Create a 1 Hz highpass copy for ICA fitting (MNE recommendation)
    # ICA needs 1 Hz+ highpass for stable component separation
    # But we keep the original 0.5 Hz data and apply ICA to that
    epochs_for_ica = epochs.copy().filter(l_freq=1.0, h_freq=None,
                                          method="iir", verbose=False)

    # Fit ICA on the 1 Hz filtered copy
    ica = ICA(n_components=20, random_state=42, max_iter="auto")
    ica.fit(epochs_for_ica, verbose=False)

    # Auto-detect eye components using the 1 Hz filtered data
    sources = ica.get_sources(epochs_for_ica).get_data()  # (300, 20, 795)
    ep_data = epochs_for_ica.get_data()                   # (300, 64, 795)
    fp1_i = clab.index("Fp1")
    fp2_i = clab.index("Fp2")

    eog_comps = []
    eog_corrs = []
    for c in range(sources.shape[1]):
        cf = sources[:, c, :].ravel()
        r1 = abs(np.corrcoef(cf, ep_data[:, fp1_i, :].ravel())[0, 1])
        r2 = abs(np.corrcoef(cf, ep_data[:, fp2_i, :].ravel())[0, 1])
        max_r = max(r1, r2)
        if max_r > 0.3:
            eog_comps.append(c)
            eog_corrs.append(max_r)

    # Remove eye components
    ica.exclude = eog_comps
    clean = ica.apply(epochs.copy(), verbose=False)

    # Back to original format
    x_clean = clean.get_data().transpose(2, 1, 0) / 1e-6   # (795, 64, 300) µV
    return x_clean, eog_comps, eog_corrs


def validate_ica(x_before, x_after, fs, clab, eog_comps, eog_corrs):
    fp1 = clab.index("Fp1")
    c3 = clab.index("C3")
    trial = 0
    t = np.arange(x_before.shape[0]) / fs - 0.5   # time in seconds

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Fp1 before/after (eye channel — should show biggest change)
    axes[0, 0].plot(t, x_before[:, fp1, trial], alpha=0.7, color="#e74c3c", label="Before ICA")
    axes[0, 0].plot(t, x_after[:, fp1, trial], alpha=0.8, color="#2ecc71", label="After ICA")
    axes[0, 0].set_title("Step 3 Validation: Fp1 (eye channel) — Trial 1")
    axes[0, 0].set_ylabel("Amplitude (µV)")
    axes[0, 0].legend(fontsize=8)

    # Fp2 before/after
    fp2 = clab.index("Fp2")
    axes[0, 1].plot(t, x_before[:, fp2, trial], alpha=0.7, color="#e74c3c", label="Before ICA")
    axes[0, 1].plot(t, x_after[:, fp2, trial], alpha=0.8, color="#2ecc71", label="After ICA")
    axes[0, 1].set_title("Step 3 Validation: Fp2 (eye channel) — Trial 1")
    axes[0, 1].set_ylabel("Amplitude (µV)")
    axes[0, 1].legend(fontsize=8)

    # C3 before/after (should change minimally)
    axes[1, 0].plot(t, x_before[:, c3, trial], alpha=0.7, color="#e74c3c", label="Before ICA")
    axes[1, 0].plot(t, x_after[:, c3, trial], alpha=0.8, color="#2ecc71", label="After ICA")
    axes[1, 0].set_title("Step 3 Validation: C3 (speech channel) — minimal change expected")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude (µV)")
    axes[1, 0].legend(fontsize=8)

    # Summary text
    axes[1, 1].axis("off")
    summary = "ICA SUMMARY\n\n"
    summary += f"Components removed: {len(eog_comps)}\n\n"
    for c, r in zip(eog_comps, eog_corrs):
        summary += f"  IC{c}: correlation = {r:.2f}\n"
    summary += f"\nExpected behavior:\n"
    summary += f"  Fp1/Fp2: large amplitude changes (blinks removed)\n"
    summary += f"  C3: minimal change (speech signal preserved)"
    axes[1, 1].text(0.1, 0.5, summary, fontsize=12, fontfamily="monospace",
                    verticalalignment="center", transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step3_ica.png"), dpi=150)
    print("  Saved: step3_ica.png")
    plt.close()


# ============================================================
# STEP 4: COMMON AVERAGE REFERENCE (CAR)
# ============================================================
def apply_car(x):
    avg = np.mean(x, axis=1, keepdims=True)   # mean across 64 channels
    return x - avg


def validate_car(x_before, x_after, fs, clab):
    c3 = clab.index("C3")
    trial = 0
    t = np.arange(x_before.shape[0]) / fs - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Global Field Power (GFP) before/after
    gfp_before = np.std(x_before[:, :, trial], axis=1)
    gfp_after = np.std(x_after[:, :, trial], axis=1)
    axes[0].plot(t, gfp_before, label="Before CAR", alpha=0.7, color="#e74c3c")
    axes[0].plot(t, gfp_after, label="After CAR", alpha=0.8, color="#2ecc71")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("GFP (µV)")
    axes[0].set_title("Step 4 Validation: Global Field Power — Trial 1")
    axes[0].legend(fontsize=8)

    # C3 before/after
    axes[1].plot(t, x_before[:, c3, trial], alpha=0.7, color="#e74c3c", label="Before CAR")
    axes[1].plot(t, x_after[:, c3, trial], alpha=0.8, color="#2ecc71", label="After CAR")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (µV)")
    axes[1].set_title("Step 4 Validation: C3 — Trial 1")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step4_car.png"), dpi=150)
    print("  Saved: step4_car.png")
    plt.close()


# ============================================================
# STEP 5: BASELINE CORRECTION
# ============================================================
def baseline_correct(x, fs):
    n_bl = int(0.5 * fs)   # 128 samples = -500ms to 0ms
    bl_mean = np.mean(x[:n_bl, :, :], axis=0, keepdims=True)
    return x - bl_mean


def validate_baseline(x_before, x_after, fs, clab):
    c3 = clab.index("C3")
    n_bl = int(0.5 * fs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Baseline mean per trial (C3) — before vs after
    bl_means_before = np.mean(x_before[:n_bl, c3, :], axis=0)   # (300,)
    bl_means_after = np.mean(x_after[:n_bl, c3, :], axis=0)     # (300,)

    axes[0].hist(bl_means_before, bins=30, alpha=0.7, color="#e74c3c", label="Before correction")
    axes[0].hist(bl_means_after, bins=30, alpha=0.7, color="#2ecc71", label="After correction")
    axes[0].axvline(0, color="black", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Mean baseline amplitude (µV)")
    axes[0].set_ylabel("Number of trials")
    axes[0].set_title("Step 5 Validation: Baseline Means — C3, All 300 Trials")
    axes[0].legend(fontsize=8)

    # Single trial time course
    trial = 0
    t = np.arange(x_before.shape[0]) / fs - 0.5
    axes[1].plot(t, x_before[:, c3, trial], alpha=0.7, color="#e74c3c", label="Before correction")
    axes[1].plot(t, x_after[:, c3, trial], alpha=0.8, color="#2ecc71", label="After correction")
    axes[1].axvline(0, color="black", linestyle="--", alpha=0.5, label="Stimulus onset")
    axes[1].axhline(0, color="grey", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (µV)")
    axes[1].set_title("Step 5 Validation: Time Course — C3, Trial 1")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step5_baseline.png"), dpi=150)
    print("  Saved: step5_baseline.png")
    plt.close()


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    files = sorted([f for f in os.listdir(TRAIN_FOLDER)
                    if f.endswith(".mat") and f.startswith("Data_Sample")])
    if not files:
        print("ERROR: No .mat files found! Check TRAIN_FOLDER path.")
        return

    print(f"Found {len(files)} subjects.\n")
    print("Pipeline: Bandpass → Notch → ICA → CAR → Baseline\n")

    for i, fname in enumerate(files):
        filepath = os.path.join(TRAIN_FOLDER, fname)
        subj_id = f"S{i+1:02d}"
        print(f"{'='*55}")
        print(f"Subject {i+1}/{len(files)}: {fname}")
        print(f"{'='*55}")

        # --- Load ---
        x_raw, y, fs, clab = load_subject(filepath)
        print(f"  Loaded: {x_raw.shape}")

        # --- Step 1: Bandpass ---
        print("  Step 1: Bandpass 0.5-50 Hz ...", end=" ", flush=True)
        x_bp = bandpass_filter(x_raw, fs)
        print("done")
        if i == 0:
            validate_bandpass(x_raw, x_bp, fs, clab)

        # --- Step 2: Notch ---
        print("  Step 2: Notch 50 Hz ...", end=" ", flush=True)
        x_notch = notch_filter(x_bp, fs)
        print("done")
        if i == 0:
            validate_notch(x_bp, x_notch, fs, clab)

        # --- Step 3: ICA ---
        print("  Step 3: ICA ...", end=" ", flush=True)
        x_ica, eog_comps, eog_corrs = apply_ica(x_notch, fs, clab)
        comp_str = ", ".join([f"IC{c}({r:.2f})" for c, r in zip(eog_comps, eog_corrs)])
        print(f"done — removed {len(eog_comps)} eye components: {comp_str}")
        if i == 0:
            validate_ica(x_notch, x_ica, fs, clab, eog_comps, eog_corrs)

        # --- Step 4: CAR ---
        print("  Step 4: CAR ...", end=" ", flush=True)
        x_car = apply_car(x_ica)
        print("done")
        if i == 0:
            validate_car(x_ica, x_car, fs, clab)

        # --- Step 5: Baseline correction ---
        print("  Step 5: Baseline correction ...", end=" ", flush=True)
        x_final = baseline_correct(x_car, fs)
        print("done")
        if i == 0:
            validate_baseline(x_car, x_final, fs, clab)

        # --- Save ---
        save_path = os.path.join(OUTPUT_FOLDER, f"{subj_id}_clean.npz")
        np.savez_compressed(save_path,
                            x=x_final, y=y, fs=fs,
                            clab=np.array(clab))
        print(f"  Saved: {save_path}\n")

    print("=" * 55)
    print("ALL SUBJECTS PREPROCESSED")
    print(f"Cleaned data saved to: {OUTPUT_FOLDER}/")
    print(f"Validation plots saved to: {PLOT_FOLDER}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
