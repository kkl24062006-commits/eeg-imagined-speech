"""
Validation — Cross-Subject Channel Consistency (Welch version)
--------------------------------------------------------------
Uses the SAME Welch PSD method as Step 2, but vectorized across
trials so it runs in ~2-3 minutes instead of hanging forever.

Key optimization: instead of calling welch() once per trial (76,800
calls per subject), we call it once per channel with all 300 trials
passed together (64 calls per subject).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch as scipy_welch
from scipy.stats import f_oneway

# ============================================================
# CONFIG — point this to your Training Set FOLDER
# ============================================================
TRAIN_FOLDER = r"C:\Users\kkl24\Downloads\pq7vb-osfstorage-Track#3 Imagined speech classification-archive\Training set"
# ============================================================

TOP_N = 20

BANDS = {
    "theta":    (4, 8),
    "alpha":    (8, 13),
    "low_beta": (13, 20),
    "low_gamma":(30, 50),
}

BROCA    = {"F7", "F5", "FC5", "FT7", "FT9", "FC3"}
WERNICKE = {"T7", "TP7", "TP9", "CP5", "P7"}
SMA      = {"Fz", "FC1", "FC2", "Cz", "C1", "C2"}
MOTOR    = {"C3", "C4", "C5", "C6", "CP3", "CP4"}
ARTIFACT = {"Fp1", "Fp2", "AF7", "AF8"}
ALL_SPEECH = BROCA | WERNICKE | SMA | MOTOR


def get_region(ch):
    if ch in BROCA:    return "Broca"
    if ch in WERNICKE: return "Wernicke"
    if ch in SMA:      return "SMA"
    if ch in MOTOR:    return "Motor"
    if ch in ARTIFACT: return "Eye-artifact"
    return ""


def compute_bandpowers_welch(x_speech, fs):
    """
    Compute band power using Welch's method, vectorized across trials.
    For each channel, passes ALL trials into welch() in one call.
    64 calls total instead of 76,800.
    """
    n_time, n_ch, n_trials = x_speech.shape
    nperseg = min(n_time, 256)  # 1-second window, same as Step 2

    band_names = list(BANDS.keys())
    n_bands = len(band_names)
    features = np.zeros((n_ch, n_trials, n_bands))

    for ch in range(n_ch):
        # x_speech[:, ch, :] is shape (n_time, n_trials)
        # welch with axis=0 computes PSD for each trial column at once
        # Result: freqs shape (n_freqs,), psd shape (n_freqs, n_trials)
        freqs, psd = scipy_welch(x_speech[:, ch, :], fs=fs,
                                 nperseg=nperseg, axis=0)

        for b, bname in enumerate(band_names):
            low, high = BANDS[bname]
            mask = (freqs >= low) & (freqs <= high)
            # Mean power in band for each trial
            features[ch, :, b] = np.mean(psd[mask, :], axis=0)

    return features


def rank_channels_for_subject(filepath):
    """Load one subject, compute ANOVA F-scores for all 64 channels."""
    mat = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    epo_key = [k for k in mat.keys() if k.startswith("epo")][0]
    epo = mat[epo_key]

    x = np.array(epo.x)            # (795, 64, 300)
    y = np.array(epo.y)            # (5, 300) one-hot
    fs = int(epo.fs)               # 256
    clab = [str(c) for c in epo.clab]

    n_time, n_ch, n_trials = x.shape
    labels = np.argmax(y, axis=0)  # flat class labels

    # Skip baseline (-500ms = 128 samples at 256 Hz)
    baseline_samples = int(0.5 * fs)
    x_speech = x[baseline_samples:, :, :]

    # Welch-based band power extraction (vectorized)
    features = compute_bandpowers_welch(x_speech, fs)

    # ANOVA per channel across 5 classes
    n_bands = len(BANDS)
    f_scores = np.zeros(n_ch)

    for ch in range(n_ch):
        band_f_values = []
        for b in range(n_bands):
            groups = [features[ch, labels == c, b] for c in range(5)]
            f_val, _ = f_oneway(*groups)
            band_f_values.append(f_val)
        f_scores[ch] = np.mean(band_f_values)

    return f_scores, clab


def main():
    # --- Find all subject files ---
    files = sorted([f for f in os.listdir(TRAIN_FOLDER)
                    if f.endswith(".mat") and f.startswith("Data_Sample")])
    if len(files) == 0:
        print("ERROR: No .mat files found! Check TRAIN_FOLDER path.")
        print("       Make sure Data_Sample files are directly in this folder.")
        return
    print(f"Found {len(files)} subject files.\n")

    # --- Run ranking for each subject ---
    all_f_scores = []
    clab = None

    for i, fname in enumerate(files):
        filepath = os.path.join(TRAIN_FOLDER, fname)
        print(f"Processing Subject {i+1:2d}/{len(files)}: {fname} ...",
              end=" ", flush=True)
        scores, clab = rank_channels_for_subject(filepath)
        all_f_scores.append(scores)
        print(f"done (top: {clab[np.argmax(scores)]})")

    n_ch = len(clab)
    n_subj = len(files)
    score_matrix = np.array(all_f_scores)  # (n_subj, 64)

    # --- Compute rankings per subject ---
    rank_matrix = np.zeros_like(score_matrix, dtype=int)
    for s in range(n_subj):
        rank_matrix[s] = np.argsort(np.argsort(-score_matrix[s]))  # 0 = best

    # --- Consistency: how often does each channel appear in top N? ---
    in_top_n = np.sum(rank_matrix < TOP_N, axis=0)  # (64,)
    consistency_order = np.argsort(-in_top_n)

    print("\n" + "=" * 70)
    print(f"CROSS-SUBJECT CONSISTENCY (in top {TOP_N} out of {n_subj} subjects)")
    print("=" * 70)
    print(f"{'Rank':>4}  {'Channel':>7}  {'In top 20':>9}  {'Avg rank':>9}  "
          f"{'Region':>12}  {'Verdict'}")
    print("-" * 70)

    for rank, idx in enumerate(consistency_order, 1):
        count = in_top_n[idx]
        avg_rank = np.mean(rank_matrix[:, idx]) + 1  # fixed: [:, idx]
        region = get_region(clab[idx])

        if count >= int(n_subj * 0.75):
            verdict = "STRONG"
        elif count >= int(n_subj * 0.5):
            verdict = "MODERATE"
        elif count >= int(n_subj * 0.25):
            verdict = "WEAK"
        else:
            verdict = "unreliable"

        marker = ""
        if region in ("Broca", "Wernicke", "SMA", "Motor"):
            marker = " <<<"
        elif region == "Eye-artifact":
            marker = " [!]"

        print(f"{rank:4d}  {clab[idx]:>7s}  {count:5d}/{n_subj:<3d}  "
              f"{avg_rank:9.1f}  {region:>12s}  {verdict}{marker}")

    # --- Final recommended subsets ---
    print("\n" + "=" * 60)
    print("VALIDATED CHANNEL SUBSETS")
    print("=" * 60)

    validated_20 = []
    for idx in consistency_order:
        if clab[idx] not in ARTIFACT and len(validated_20) < 20:
            validated_20.append(clab[idx])

    validated_8 = validated_20[:8]

    speech_count = len([c for c in validated_20 if c in ALL_SPEECH])
    print(f"\n20-channel subset ({speech_count} are speech-region channels):")
    for i, ch in enumerate(validated_20, 1):
        reg = get_region(ch)
        tag = f" [{reg}]" if reg else ""
        print(f"  {i:2d}. {ch}{tag}")

    print(f"\n8-channel compact subset:")
    for i, ch in enumerate(validated_8, 1):
        reg = get_region(ch)
        tag = f" [{reg}]" if reg else ""
        print(f"  {i}. {ch}{tag}")

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(18, 7))
    top30_idx = consistency_order[:30]
    heatmap_data = rank_matrix[:, top30_idx].T  # (30, n_subj)
    heatmap_labels = [clab[i] for i in top30_idx]

    im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=63)
    ax.set_yticks(range(len(heatmap_labels)))
    ax.set_yticklabels(heatmap_labels, fontsize=8)
    ax.set_xticks(range(n_subj))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_subj)], fontsize=8)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Channel (sorted by consistency)")
    ax.set_title("Channel Rank per Subject — Welch Method "
                 "(green = top-ranked, red = bottom)")
    plt.colorbar(im, ax=ax, label="Rank (0 = best, 63 = worst)")
    plt.tight_layout()
    plt.savefig("channel_consistency_heatmap_welch.png", dpi=150)
    print("\nSaved: channel_consistency_heatmap_welch.png")
    plt.show()

    # --- Consistency bar chart ---
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = []
    for idx in consistency_order:
        ch = clab[idx]
        if ch in BROCA:      colors.append("#e74c3c")
        elif ch in WERNICKE: colors.append("#2ecc71")
        elif ch in SMA:      colors.append("#3498db")
        elif ch in MOTOR:    colors.append("#f39c12")
        elif ch in ARTIFACT: colors.append("#95a5a6")
        else:                colors.append("#bdc3c7")

    ax.bar(range(n_ch), in_top_n[consistency_order], color=colors)
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels([clab[i] for i in consistency_order],
                       rotation=90, fontsize=7)
    ax.set_ylabel(f"Number of subjects in top {TOP_N}")
    ax.set_title("Channel Consistency Across All 15 Subjects — Welch Method")
    ax.axhline(y=int(n_subj * 0.75), color="red", linestyle="--",
               alpha=0.7, label="75% threshold (STRONG)")
    ax.axhline(y=int(n_subj * 0.5), color="orange", linestyle="--",
               alpha=0.7, label="50% threshold (MODERATE)")

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#e74c3c", label="Broca's"),
        Patch(facecolor="#2ecc71", label="Wernicke's"),
        Patch(facecolor="#3498db", label="SMA"),
        Patch(facecolor="#f39c12", label="Motor"),
        Patch(facecolor="#95a5a6", label="Eye-artifact"),
        Patch(facecolor="#bdc3c7", label="Other"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("channel_consistency_bar_welch.png", dpi=150)
    print("Saved: channel_consistency_bar_welch.png")
    plt.show()


if __name__ == "__main__":
    main()
