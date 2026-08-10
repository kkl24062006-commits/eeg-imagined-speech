"""
Multi-Trial Validation Plots — Preprocessing Pipeline
------------------------------------------------------
Same 5-step validation as before, but instead of showing 1 trial,
shows 10 random trials per step. This proves the preprocessing works
consistently across trials, not just one lucky example.

Output: 5 PNG files, one per step, each with 10 subplots (one per trial).
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
PLOT_FOLDER = r"C:\Users\kkl24\Downloads\BCI_project\validation_plots_10trials"
N_TRIALS_TO_SHOW = 10
RANDOM_SEED = 42  # for reproducible trial selection
# ============================================================


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
    eog_comps = []
    for c in range(sources.shape[1]):
        cf = sources[:, c, :].ravel()
        r1 = abs(np.corrcoef(cf, ep_data[:, fp1_i, :].ravel())[0, 1])
        r2 = abs(np.corrcoef(cf, ep_data[:, fp2_i, :].ravel())[0, 1])
        if max(r1, r2) > 0.3:
            eog_comps.append(c)
    ica.exclude = eog_comps
    clean = ica.apply(epochs.copy(), verbose=False)
    return clean.get_data().transpose(2, 1, 0) / 1e-6


def apply_car(x):
    return x - np.mean(x, axis=1, keepdims=True)


def baseline_correct(x, fs):
    n_bl = int(0.5 * fs)
    return x - np.mean(x[:n_bl, :, :], axis=0, keepdims=True)


# ============================================================
# STEP 1 — Bandpass: PSD comparison across 10 trials
# ============================================================
def plot_step1_multi(x_raw, x_bp, fs, clab, trial_idx):
    ch = clab.index("C3")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for i, tr in enumerate(trial_idx):
        ax = axes[i // 5, i % 5]
        f1, p1 = scipy_welch(x_raw[:, ch, tr], fs=fs, nperseg=256)
        f2, p2 = scipy_welch(x_bp[:, ch, tr], fs=fs, nperseg=256)
        ax.semilogy(f1, p1, alpha=0.7, color="#e74c3c", label="Before")
        ax.semilogy(f2, p2, alpha=0.8, color="#2ecc71", label="After")
        ax.axvline(0.5, color="grey", linestyle="--", alpha=0.4)
        ax.axvline(50, color="grey", linestyle="--", alpha=0.4)
        ax.set_title(f"Trial {tr+1}", fontsize=10)
        ax.set_xlabel("Freq (Hz)", fontsize=8)
        ax.set_ylabel("PSD", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f"STEP 1 — BANDPASS 0.5-50 Hz | Channel C3 | 10 Random Trials",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step1_bandpass_10trials.png"), dpi=120)
    print("  Saved: step1_bandpass_10trials.png")
    plt.close()


# ============================================================
# STEP 2 — Notch: PSD zoomed 40-60 Hz across 10 trials
# ============================================================
def plot_step2_multi(x_bp, x_notch, fs, clab, trial_idx):
    ch = clab.index("C3")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for i, tr in enumerate(trial_idx):
        ax = axes[i // 5, i % 5]
        f1, p1 = scipy_welch(x_bp[:, ch, tr], fs=fs, nperseg=256)
        f2, p2 = scipy_welch(x_notch[:, ch, tr], fs=fs, nperseg=256)
        mask = (f1 >= 40) & (f1 <= 60)
        ax.plot(f1[mask], p1[mask], "o-", alpha=0.7, color="#e74c3c",
                label="Before", markersize=3)
        ax.plot(f2[mask], p2[mask], "o-", alpha=0.8, color="#2ecc71",
                label="After", markersize=3)
        ax.axvline(50, color="grey", linestyle="--", alpha=0.5)
        ax.set_title(f"Trial {tr+1}", fontsize=10)
        ax.set_xlabel("Freq (Hz)", fontsize=8)
        ax.set_ylabel("PSD", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f"STEP 2 — NOTCH 50 Hz | Channel C3 | 10 Random Trials (zoomed 40-60 Hz)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step2_notch_10trials.png"), dpi=120)
    print("  Saved: step2_notch_10trials.png")
    plt.close()


# ============================================================
# STEP 3 — ICA: Fp1 (eye) before/after across 10 trials
# ============================================================
def plot_step3_multi(x_before, x_after, fs, clab, trial_idx):
    fp1 = clab.index("Fp1")
    t = np.arange(x_before.shape[0]) / fs - 0.5

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, tr in enumerate(trial_idx):
        ax = axes[i // 5, i % 5]
        ax.plot(t, x_before[:, fp1, tr], alpha=0.7, color="#e74c3c", label="Before")
        ax.plot(t, x_after[:, fp1, tr], alpha=0.8, color="#2ecc71", label="After")
        ax.set_title(f"Trial {tr+1}", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("µV", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f"STEP 3 — ICA | Channel Fp1 (eye) | 10 Random Trials\n"
                 f"(Blinks should disappear after ICA)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step3_ica_fp1_10trials.png"), dpi=120)
    print("  Saved: step3_ica_fp1_10trials.png")
    plt.close()

    # Also for C3 (should NOT change much)
    c3 = clab.index("C3")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, tr in enumerate(trial_idx):
        ax = axes[i // 5, i % 5]
        ax.plot(t, x_before[:, c3, tr], alpha=0.7, color="#e74c3c", label="Before")
        ax.plot(t, x_after[:, c3, tr], alpha=0.8, color="#2ecc71", label="After")
        ax.set_title(f"Trial {tr+1}", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("µV", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f"STEP 3 — ICA | Channel C3 (speech) | 10 Random Trials\n"
                 f"(Signal should be preserved after ICA)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step3_ica_c3_10trials.png"), dpi=120)
    print("  Saved: step3_ica_c3_10trials.png")
    plt.close()


# ============================================================
# STEP 4 — CAR: C3 before/after across 10 trials
# ============================================================
def plot_step4_multi(x_before, x_after, fs, clab, trial_idx):
    c3 = clab.index("C3")
    t = np.arange(x_before.shape[0]) / fs - 0.5

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, tr in enumerate(trial_idx):
        ax = axes[i // 5, i % 5]
        r_before = x_before[:, c3, tr].max() - x_before[:, c3, tr].min()
        r_after = x_after[:, c3, tr].max() - x_after[:, c3, tr].min()
        ax.plot(t, x_before[:, c3, tr], alpha=0.7, color="#e74c3c",
                label=f"Before ({r_before:.0f} µV)")
        ax.plot(t, x_after[:, c3, tr], alpha=0.8, color="#2ecc71",
                label=f"After ({r_after:.0f} µV)")
        ax.set_title(f"Trial {tr+1}", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("µV", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    plt.suptitle(f"STEP 4 — CAR | Channel C3 | 10 Random Trials\n"
                 f"(Amplitude range should decrease after CAR)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step4_car_10trials.png"), dpi=120)
    print("  Saved: step4_car_10trials.png")
    plt.close()


# ============================================================
# STEP 5 — Baseline: C3 before/after across 10 trials
# ============================================================
def plot_step5_multi(x_before, x_after, fs, clab, trial_idx):
    c3 = clab.index("C3")
    n_bl = int(0.5 * fs)
    t = np.arange(x_before.shape[0]) / fs - 0.5

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, tr in enumerate(trial_idx):
        ax = axes[i // 5, i % 5]
        bl_before = np.mean(x_before[:n_bl, c3, tr])
        bl_after = np.mean(x_after[:n_bl, c3, tr])
        ax.plot(t, x_before[:, c3, tr], alpha=0.7, color="#e74c3c",
                label=f"Before (bl mean {bl_before:+.1f})")
        ax.plot(t, x_after[:, c3, tr], alpha=0.8, color="#2ecc71",
                label=f"After (bl mean {bl_after:+.2f})")
        ax.axvline(0, color="black", linestyle="--", alpha=0.4)
        ax.axhline(0, color="grey", linestyle=":", alpha=0.4)
        ax.set_title(f"Trial {tr+1}", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("µV", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    plt.suptitle(f"STEP 5 — BASELINE CORRECTION | Channel C3 | 10 Random Trials\n"
                 f"(Baseline mean should be ~0 after correction)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "step5_baseline_10trials.png"), dpi=120)
    print("  Saved: step5_baseline_10trials.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    filepath = os.path.join(TRAIN_FOLDER, "Data_Sample01.mat")
    print("Loading Subject 1...")
    x_raw, y, fs, clab = load_subject(filepath)

    # Pick 10 random trials (reproducible with seed)
    rng = np.random.default_rng(RANDOM_SEED)
    trial_idx = sorted(rng.choice(x_raw.shape[2], size=N_TRIALS_TO_SHOW, replace=False))
    print(f"Selected trials: {[t+1 for t in trial_idx]}\n")

    print("Step 1: Bandpass...")
    x_bp = bandpass_filter(x_raw, fs)
    plot_step1_multi(x_raw, x_bp, fs, clab, trial_idx)

    print("Step 2: Notch...")
    x_notch = notch_filter(x_bp, fs)
    plot_step2_multi(x_bp, x_notch, fs, clab, trial_idx)

    print("Step 3: ICA...")
    x_ica = apply_ica(x_notch, fs, clab)
    plot_step3_multi(x_notch, x_ica, fs, clab, trial_idx)

    print("Step 4: CAR...")
    x_car = apply_car(x_ica)
    plot_step4_multi(x_ica, x_car, fs, clab, trial_idx)

    print("Step 5: Baseline...")
    x_final = baseline_correct(x_car, fs)
    plot_step5_multi(x_car, x_final, fs, clab, trial_idx)

    print(f"\n{'='*55}")
    print(f"All plots saved to: {PLOT_FOLDER}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
