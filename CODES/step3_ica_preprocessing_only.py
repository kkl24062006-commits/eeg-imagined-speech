"""
Step 3 — Preprocessing (ICA) + Re-validation on Cleaned Data
-------------------------------------------------------------
For each of the 15 subjects:
  1. Load raw epoched data from .mat
  2. Bandpass filter 1-50 Hz
  3. Fit ICA → auto-detect eye-blink components by correlating with Fp1/Fp2
  4. Remove eye components → cleaned data
  5. Run Welch-based channel ranking on cleaned data

Then aggregates cross-subject consistency and shows final validated channels.

Install first:  pip install mne
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch as scipy_welch
from scipy.stats import f_oneway
import mne
from mne.preprocessing import ICA

# Suppress MNE's verbose logging
mne.set_log_level("WARNING")

# ============================================================
# CONFIG — same Training Set folder as before
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


def preprocess_subject(filepath):
    """
    Load one subject's .mat → filter → ICA → return cleaned data.
    """
    # --- Load .mat ---
    mat = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    epo_key = [k for k in mat.keys() if k.startswith("epo")][0]
    epo = mat[epo_key]

    x = np.array(epo.x, dtype=np.float64)   # (795, 64, 300)
    y = np.array(epo.y)                      # (5, 300)
    fs = int(epo.fs)                         # 256
    clab = [str(c) for c in epo.clab]        # 64 channel names

    n_time, n_ch, n_trials = x.shape

    # --- Convert to MNE format ---
    # MNE expects (n_epochs, n_channels, n_times) in Volts
    data = x.transpose(2, 1, 0)    # (300, 64, 795)
    data = data * 1e-6             # µV → Volts

    # Create MNE info object (all channels = EEG)
    info = mne.create_info(ch_names=clab, sfreq=fs, ch_types="eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)

    # --- Bandpass filter: 1-50 Hz ---
    # 1 Hz high-pass recommended for stable ICA fitting
    epochs.filter(l_freq=1.0, h_freq=50.0, verbose=False)

    # --- Fit ICA ---
    ica = ICA(n_components=20, random_state=42, max_iter="auto")
    ica.fit(epochs, verbose=False)

    # --- Auto-detect eye-blink components ---
    # Correlate each ICA component with Fp1 and Fp2 (eye channels)
    sources = ica.get_sources(epochs).get_data()  # (300, 20, 795)
    epoch_data = epochs.get_data()                # (300, 64, 795)

    fp1_idx = clab.index("Fp1")
    fp2_idx = clab.index("Fp2")

    eog_comps = []
    eog_details = []
    for comp in range(sources.shape[1]):
        comp_flat = sources[:, comp, :].flatten()
        fp1_flat = epoch_data[:, fp1_idx, :].flatten()
        fp2_flat = epoch_data[:, fp2_idx, :].flatten()

        corr_fp1 = abs(np.corrcoef(comp_flat, fp1_flat)[0, 1])
        corr_fp2 = abs(np.corrcoef(comp_flat, fp2_flat)[0, 1])
        max_corr = max(corr_fp1, corr_fp2)

        if max_corr > 0.3:  # threshold for eye-artifact component
            eog_comps.append(comp)
            eog_details.append(f"IC{comp} (corr={max_corr:.2f})")

    # --- Remove eye components ---
    ica.exclude = eog_comps
    epochs_clean = ica.apply(epochs.copy(), verbose=False)

    # --- Extract cleaned data back to original format ---
    x_clean = epochs_clean.get_data()       # (300, 64, 795)
    x_clean = x_clean.transpose(2, 1, 0)   # (795, 64, 300)
    x_clean = x_clean / 1e-6               # Volts → µV (back to original scale)

    return x_clean, y, fs, clab, eog_details


def rank_channels_welch(x, y, fs, clab):
    """Compute per-channel ANOVA F-scores using Welch PSD (vectorized)."""
    n_time, n_ch, n_trials = x.shape
    labels = np.argmax(y, axis=0)

    # Skip baseline
    baseline_samples = int(0.5 * fs)
    x_speech = x[baseline_samples:, :, :]

    nperseg = min(x_speech.shape[0], 256)
    band_names = list(BANDS.keys())
    n_bands = len(band_names)
    features = np.zeros((n_ch, n_trials, n_bands))

    for ch in range(n_ch):
        freqs, psd = scipy_welch(x_speech[:, ch, :], fs=fs,
                                 nperseg=nperseg, axis=0)
        for b, bname in enumerate(band_names):
            low, high = BANDS[bname]
            mask = (freqs >= low) & (freqs <= high)
            features[ch, :, b] = np.mean(psd[mask, :], axis=0)

    f_scores = np.zeros(n_ch)
    for ch in range(n_ch):
        band_f_values = []
        for b in range(n_bands):
            groups = [features[ch, labels == c, b] for c in range(5)]
            f_val, _ = f_oneway(*groups)
            band_f_values.append(f_val)
        f_scores[ch] = np.mean(band_f_values)

    return f_scores


def main():
    # --- Find files ---
    files = sorted([f for f in os.listdir(TRAIN_FOLDER)
                    if f.endswith(".mat") and f.startswith("Data_Sample")])
    if not files:
        print("ERROR: No .mat files found! Check TRAIN_FOLDER path.")
        return
    print(f"Found {len(files)} subject files.\n")
    print("=" * 65)
    print("PHASE 1: Preprocessing (filter + ICA) + Channel Ranking")
    print("=" * 65)

    all_f_scores = []
    clab = None

    for i, fname in enumerate(files):
        filepath = os.path.join(TRAIN_FOLDER, fname)
        print(f"\nSubject {i+1:2d}/{len(files)}: {fname}")
        print(f"  Filtering (1-50 Hz) + fitting ICA...", end=" ", flush=True)

        x_clean, y, fs, clab, eog_details = preprocess_subject(filepath)

        if eog_details:
            print(f"done")
            print(f"  Eye components removed: {len(eog_details)} → {', '.join(eog_details)}")
        else:
            print(f"done")
            print(f"  Eye components removed: 0 (none detected)")

        print(f"  Ranking channels on cleaned data...", end=" ", flush=True)
        scores = rank_channels_welch(x_clean, y, fs, clab)
        all_f_scores.append(scores)
        print(f"done (top: {clab[np.argmax(scores)]})")

    n_ch = len(clab)
    n_subj = len(files)
    score_matrix = np.array(all_f_scores)

    # --- Compute rankings per subject ---
    rank_matrix = np.zeros_like(score_matrix, dtype=int)
    for s in range(n_subj):
        rank_matrix[s] = np.argsort(np.argsort(-score_matrix[s]))

    # --- Consistency ---
    in_top_n = np.sum(rank_matrix < TOP_N, axis=0)
    consistency_order = np.argsort(-in_top_n)

    print("\n\n" + "=" * 70)
    print(f"PHASE 2: CROSS-SUBJECT CONSISTENCY — ICA-CLEANED DATA")
    print(f"(in top {TOP_N} out of {n_subj} subjects)")
    print("=" * 70)
    print(f"{'Rank':>4}  {'Channel':>7}  {'In top 20':>9}  {'Avg rank':>9}  "
          f"{'Region':>12}  {'Verdict'}")
    print("-" * 70)

    for rank, idx in enumerate(consistency_order, 1):
        count = in_top_n[idx]
        avg_rank = np.mean(rank_matrix[:, idx]) + 1
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

    # --- Validated subsets ---
    print("\n" + "=" * 60)
    print("VALIDATED CHANNEL SUBSETS (ICA-CLEANED)")
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

    # --- Check: did eye channels drop after ICA? ---
    print("\n" + "=" * 60)
    print("ARTIFACT CHANNEL CHECK (did ICA fix the problem?)")
    print("=" * 60)
    for ch_name in ["AF8", "Fp2", "Fp1", "AF7"]:
        idx = clab.index(ch_name)
        count = in_top_n[idx]
        rank_pos = np.where(consistency_order == idx)[0][0] + 1
        print(f"  {ch_name}: {count}/{n_subj} subjects in top 20 "
              f"(rank #{rank_pos}/64)")

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(18, 7))
    top30_idx = consistency_order[:30]
    heatmap_data = rank_matrix[:, top30_idx].T
    heatmap_labels = [clab[i] for i in top30_idx]

    im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=63)
    ax.set_yticks(range(len(heatmap_labels)))
    ax.set_yticklabels(heatmap_labels, fontsize=8)
    ax.set_xticks(range(n_subj))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_subj)], fontsize=8)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Channel (sorted by consistency)")
    ax.set_title("Channel Rank per Subject — After ICA Cleaning "
                 "(green = top, red = bottom)")
    plt.colorbar(im, ax=ax, label="Rank (0 = best, 63 = worst)")
    plt.tight_layout()
    plt.savefig("channel_consistency_heatmap_ICA.png", dpi=150)
    print("\nSaved: channel_consistency_heatmap_ICA.png")
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
    ax.set_title("Channel Consistency — After ICA Cleaning")
    ax.axhline(y=int(n_subj * 0.75), color="red", linestyle="--",
               alpha=0.7, label="75% (STRONG)")
    ax.axhline(y=int(n_subj * 0.5), color="orange", linestyle="--",
               alpha=0.7, label="50% (MODERATE)")

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
    plt.savefig("channel_consistency_bar_ICA.png", dpi=150)
    print("Saved: channel_consistency_bar_ICA.png")
    plt.show()


if __name__ == "__main__":
    main()
