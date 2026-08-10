"""
Final Validation Plots — All 5 steps, cleaned layout
-----------------------------------------------------
Changes from previous versions:
  Step 1: PSD only (removed time domain plot)
  Step 4: 3 panels (removed variance scatter)
  Step 5: 2 panels (histograms only, removed bottom row)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch as scipy_welch
import mne
from mne.preprocessing import ICA

mne.set_log_level("WARNING")

TRAIN_FOLDER = r"C:\Users\kkl24\Downloads\pq7vb-osfstorage-Track#3 Imagined speech classification-archive\Training set"
PLOT_FOLDER = "validation_plots"


def load_subject(filepath):
    mat = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    epo_key = [k for k in mat.keys() if k.startswith("epo")][0]
    epo = mat[epo_key]
    return (np.array(epo.x, dtype=np.float64), np.array(epo.y),
            int(epo.fs), [str(c) for c in epo.clab])


def bandpass_filter(x, fs):
    sos = butter(4, [0.5, 50.0], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def notch_filter(x, fs):
    b, a = iirnotch(50.0, 30, fs=fs)
    return filtfilt(b, a, x, axis=0)


def apply_ica(x, fs, clab):
    data = x.transpose(2, 1, 0) * 1e-6
    info = mne.create_info(ch_names=clab, sfreq=fs, ch_types="eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)
    epochs_ica = epochs.copy().filter(l_freq=1.0, h_freq=None, method="iir", verbose=False)
    ica = ICA(n_components=20, random_state=42, max_iter="auto")
    ica.fit(epochs_ica, verbose=False)
    sources = ica.get_sources(epochs_ica).get_data()
    ep_data = epochs_ica.get_data()
    fp1_i, fp2_i = clab.index("Fp1"), clab.index("Fp2")
    eog_comps, eog_corrs = [], []
    for c in range(sources.shape[1]):
        cf = sources[:, c, :].ravel()
        r1 = abs(np.corrcoef(cf, ep_data[:, fp1_i, :].ravel())[0, 1])
        r2 = abs(np.corrcoef(cf, ep_data[:, fp2_i, :].ravel())[0, 1])
        if max(r1, r2) > 0.3:
            eog_comps.append(c)
            eog_corrs.append(max(r1, r2))
    ica.exclude = eog_comps
    clean = ica.apply(epochs.copy(), verbose=False)
    return clean.get_data().transpose(2, 1, 0) / 1e-6, eog_comps, eog_corrs


def apply_car(x):
    return x - np.mean(x, axis=1, keepdims=True)


def baseline_correct(x, fs):
    n_bl = int(0.5 * fs)
    return x - np.mean(x[:n_bl, :, :], axis=0, keepdims=True)


# ============================================================
# STEP 1 — PSD only
# ============================================================
def plot_step1(x_raw, x_bp, fs, clab):
    ch = clab.index("C3")
    f1, p1 = scipy_welch(x_raw[:, ch, 0], fs=fs, nperseg=256)
    f2, p2 = scipy_welch(x_bp[:, ch, 0], fs=fs, nperseg=256)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(f1, p1, label="Before", alpha=0.8, color="#e74c3c")
    ax.semilogy(f2, p2, label="After bandpass", alpha=0.8, color="#2ecc71")
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5, label="0.5 Hz cutoff")
    ax.axvline(50, color="grey", linestyle="--", alpha=0.5, label="50 Hz cutoff")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title("Step 1 Validation: Bandpass 0.5–50 Hz — C3, Trial 1")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step1_bandpass.png"), dpi=150)
    print("  Saved: step1_bandpass.png")
    plt.close()


# ============================================================
# STEP 2 — same as before (2 panels)
# ============================================================
def plot_step2(x_bp, x_notch, fs, clab):
    ch = clab.index("C3")
    f1, p1 = scipy_welch(x_bp[:, ch, 0], fs=fs, nperseg=256)
    f2, p2 = scipy_welch(x_notch[:, ch, 0], fs=fs, nperseg=256)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mask = (f1 >= 40) & (f1 <= 60)
    axes[0].plot(f1[mask], p1[mask], "o-", label="Before notch", alpha=0.8, color="#e74c3c")
    axes[0].plot(f2[mask], p2[mask], "o-", label="After notch", alpha=0.8, color="#2ecc71")
    axes[0].axvline(50, color="grey", linestyle="--", alpha=0.7, label="50 Hz")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (µV²/Hz)")
    axes[0].set_title("Step 2 Validation: PSD Zoomed 40–60 Hz — C3")
    axes[0].legend(fontsize=9)

    axes[1].semilogy(f1, p1, label="Before notch", alpha=0.6, color="#e74c3c")
    axes[1].semilogy(f2, p2, label="After notch", alpha=0.8, color="#2ecc71")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD (µV²/Hz)")
    axes[1].set_title("Step 2 Validation: Full PSD — C3")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step2_notch.png"), dpi=150)
    print("  Saved: step2_notch.png")
    plt.close()


# ============================================================
# STEP 3 — same as before (4 panels)
# ============================================================
def plot_step3(x_before, x_after, fs, clab, eog_comps, eog_corrs):
    fp1, fp2, c3 = clab.index("Fp1"), clab.index("Fp2"), clab.index("C3")
    t = np.arange(x_before.shape[0]) / fs - 0.5

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(t, x_before[:, fp1, 0], alpha=0.7, color="#e74c3c", label="Before ICA")
    axes[0, 0].plot(t, x_after[:, fp1, 0], alpha=0.8, color="#2ecc71", label="After ICA")
    axes[0, 0].set_title("Step 3: Fp1 (eye channel) — Trial 1")
    axes[0, 0].set_ylabel("Amplitude (µV)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t, x_before[:, fp2, 0], alpha=0.7, color="#e74c3c", label="Before ICA")
    axes[0, 1].plot(t, x_after[:, fp2, 0], alpha=0.8, color="#2ecc71", label="After ICA")
    axes[0, 1].set_title("Step 3: Fp2 (eye channel) — Trial 1")
    axes[0, 1].set_ylabel("Amplitude (µV)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(t, x_before[:, c3, 0], alpha=0.7, color="#e74c3c", label="Before ICA")
    axes[1, 0].plot(t, x_after[:, c3, 0], alpha=0.8, color="#2ecc71", label="After ICA")
    axes[1, 0].set_title("Step 3: C3 (speech channel) — minimal change expected")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude (µV)")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axis("off")
    summary = "ICA SUMMARY\n\n"
    summary += f"Components removed: {len(eog_comps)}\n\n"
    for c, r in zip(eog_comps, eog_corrs):
        summary += f"  IC{c}: correlation = {r:.2f}\n"
    summary += f"\nExpected behavior:\n"
    summary += f"  Fp1/Fp2: large change (blinks removed)\n"
    summary += f"  C3: minimal change (speech preserved)"
    axes[1, 1].text(0.1, 0.5, summary, fontsize=12, fontfamily="monospace",
                    verticalalignment="center", transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step3_ica.png"), dpi=150)
    print("  Saved: step3_ica.png")
    plt.close()


# ============================================================
# STEP 4 — 3 panels (no variance scatter)
# ============================================================
def plot_step4(x_before, x_after, fs, clab):
    c3 = clab.index("C3")
    t = np.arange(x_before.shape[0]) / fs - 0.5

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: common average signal
    common_avg = np.mean(x_before[:, :, 0], axis=1)
    axes[0].plot(t, common_avg, color="#e74c3c", linewidth=1.5)
    axes[0].set_title("What CAR removes:\nCommon Average Signal — Trial 1")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].axhline(0, color="grey", linestyle=":", alpha=0.5)

    # Panel 2: C3 before vs after
    before_range = x_before[:, c3, 0].max() - x_before[:, c3, 0].min()
    after_range = x_after[:, c3, 0].max() - x_after[:, c3, 0].min()
    axes[1].plot(t, x_before[:, c3, 0], alpha=0.7, color="#e74c3c",
                 label=f"Before CAR (range: {before_range:.1f} µV)")
    axes[1].plot(t, x_after[:, c3, 0], alpha=0.8, color="#2ecc71",
                 label=f"After CAR (range: {after_range:.1f} µV)")
    axes[1].set_title("C3 Before vs After CAR — Trial 1")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (µV)")
    axes[1].legend(fontsize=9)

    # Panel 3: amplitude range all channels
    range_before = x_before[:, :, 0].max(axis=0) - x_before[:, :, 0].min(axis=0)
    range_after = x_after[:, :, 0].max(axis=0) - x_after[:, :, 0].min(axis=0)
    ch_idx = np.arange(64)
    axes[2].bar(ch_idx - 0.2, range_before, width=0.4, color="#e74c3c", alpha=0.7,
                label=f"Before CAR (mean: {range_before.mean():.1f} µV)")
    axes[2].bar(ch_idx + 0.2, range_after, width=0.4, color="#2ecc71", alpha=0.7,
                label=f"After CAR (mean: {range_after.mean():.1f} µV)")
    axes[2].set_title("Amplitude Range — All 64 Channels")
    axes[2].set_xlabel("Channel index")
    axes[2].set_ylabel("Peak-to-peak (µV)")
    axes[2].legend(fontsize=9)

    plt.suptitle("STEP 4 — CAR VALIDATION (Subject 1)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step4_car.png"), dpi=150)
    print("  Saved: step4_car.png")
    plt.close()


# ============================================================
# STEP 5 — 2 panels (histograms only)
# ============================================================
def plot_step5(x_before, x_after, fs, clab):
    c3 = clab.index("C3")
    n_bl = int(0.5 * fs)

    bl_before = np.mean(x_before[:n_bl, c3, :], axis=0)
    bl_after = np.mean(x_after[:n_bl, c3, :], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(bl_before, bins=30, color="#e74c3c", alpha=0.8, edgecolor="black")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=2)
    axes[0].set_title(f"BEFORE Correction — C3 Baseline Means\n"
                      f"Mean={bl_before.mean():.2f} µV, Std={bl_before.std():.2f} µV")
    axes[0].set_xlabel("Mean baseline amplitude (µV)")
    axes[0].set_ylabel("Number of trials")
    axes[0].set_xlim(-18, 15)

    axes[1].hist(bl_after, bins=30, color="#2ecc71", alpha=0.8, edgecolor="black")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=2)
    axes[1].set_title(f"AFTER Correction — C3 Baseline Means\n"
                      f"Mean={bl_after.mean():.2f} µV, Std={bl_after.std():.2f} µV")
    axes[1].set_xlabel("Mean baseline amplitude (µV)")
    axes[1].set_ylabel("Number of trials")
    axes[1].set_xlim(-18, 15)

    plt.suptitle("STEP 5 — BASELINE CORRECTION VALIDATION (Subject 1)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step5_baseline.png"), dpi=150)
    print("  Saved: step5_baseline.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    filepath = os.path.join(TRAIN_FOLDER, "Data_Sample01.mat")
    print("Loading Subject 1...")
    x_raw, y, fs, clab = load_subject(filepath)

    print("Step 1: Bandpass...")
    x_bp = bandpass_filter(x_raw, fs)
    plot_step1(x_raw, x_bp, fs, clab)

    print("Step 2: Notch...")
    x_notch = notch_filter(x_bp, fs)
    plot_step2(x_bp, x_notch, fs, clab)

    print("Step 3: ICA...")
    x_ica, eog_comps, eog_corrs = apply_ica(x_notch, fs, clab)
    plot_step3(x_notch, x_ica, fs, clab, eog_comps, eog_corrs)

    print("Step 4: CAR...")
    x_car = apply_car(x_ica)
    plot_step4(x_ica, x_car, fs, clab)

    print("Step 5: Baseline...")
    x_final = baseline_correct(x_car, fs)
    plot_step5(x_car, x_final, fs, clab)

    print("\nDone! All plots saved to validation_plots/")


if __name__ == "__main__":
    main()
