"""
Step 2 — Channel Selection for Imagined Speech BCI
---------------------------------------------------
For each of the 64 channels, extracts band power in 4 frequency bands
(theta, alpha, low-beta, low-gamma), runs one-way ANOVA across the 5
classes, and ranks channels by discriminability.

Then cross-references the top-ranked channels against known speech-
relevant brain regions (Broca's, Wernicke's, SMA, motor cortex).

Outputs:
  1. Printed ranking table
  2. Bar chart of all 64 channels' F-scores
  3. Topographic-style summary of selected channels
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import f_oneway

# ============================================================
# CONFIG — same path as Step 1
# ============================================================
DATA_PATH = r"C:\Users\kkl24\Downloads\pq7vb-osfstorage-Track#3 Imagined speech classification-archive\Training set\Data_sample01.mat"
# ============================================================

# --- Frequency bands of interest (from your project design) ---
BANDS = {
    "theta":      (4, 8),
    "alpha":      (8, 13),
    "low_beta":   (13, 20),
    "low_gamma":  (30, 50),
}

# --- Speech-relevant channels (literature-informed) ---
# Broca's area (left inferior frontal — speech production planning)
BROCA = {"F7", "F5", "FC5", "FT7", "FT9", "FC3"}
# Wernicke's area (left posterior temporal — speech comprehension)
WERNICKE = {"T7", "TP7", "TP9", "CP5", "P7"}
# SMA / Premotor (midline — speech initiation & planning)
SMA = {"Fz", "FC1", "FC2", "Cz", "C1", "C2"}
# Primary motor cortex (articulatory — tongue, lips, jaw area)
MOTOR = {"C3", "C4", "C5", "C6", "CP3", "CP4"}
# Artifact-prone (eye movement) — expect these to rank poorly
ARTIFACT_PRONE = {"Fp1", "Fp2", "AF7", "AF8"}

ALL_SPEECH = BROCA | WERNICKE | SMA | MOTOR


def get_region(ch_name):
    """Return which speech region a channel belongs to, if any."""
    if ch_name in BROCA:
        return "Broca"
    elif ch_name in WERNICKE:
        return "Wernicke"
    elif ch_name in SMA:
        return "SMA"
    elif ch_name in MOTOR:
        return "Motor"
    elif ch_name in ARTIFACT_PRONE:
        return "Eye-artifact"
    else:
        return ""


def extract_bandpower(signal, fs, band):
    """Compute average power in a frequency band using Welch's method."""
    low, high = band
    nperseg = min(len(signal), 256)  # 1-second window or less
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    return np.mean(psd[mask]) if mask.any() else 0.0


def main():
    # --- Load data ---
    print("Loading data...\n")
    mat = loadmat(DATA_PATH, struct_as_record=False, squeeze_me=True)
    epo_key = [k for k in mat.keys() if k.startswith("epo")][0]
    epo = mat[epo_key]

    x = np.array(epo.x)            # (795, 64, 300)
    y = np.array(epo.y)            # (5, 300) one-hot
    fs = int(epo.fs)               # 256
    clab = [str(c) for c in epo.clab]
    class_names = list(epo.className)

    n_time, n_ch, n_trials = x.shape
    labels = np.argmax(y, axis=0)  # convert one-hot to flat labels

    # --- Use only the imagined-speech period (skip baseline) ---
    # Baseline = -500ms to 0ms = first 128 samples at 256 Hz
    baseline_samples = int(0.5 * fs)  # 128
    x_speech = x[baseline_samples:, :, :]  # keep only post-stimulus
    print(f"Using speech period only: samples {baseline_samples}–{n_time} "
          f"({x_speech.shape[0]} samples = {x_speech.shape[0]/fs:.2f}s)\n")

    # --- Extract band power for each channel × trial × band ---
    print("Extracting band power features (this may take ~30 seconds)...\n")
    n_bands = len(BANDS)
    band_names = list(BANDS.keys())
    features = np.zeros((n_ch, n_trials, n_bands))

    for ch in range(n_ch):
        for tr in range(n_trials):
            signal = x_speech[:, ch, tr]
            for b, band_name in enumerate(band_names):
                features[ch, tr, b] = extract_bandpower(signal, fs, BANDS[band_name])

    # --- ANOVA: for each channel, test if band powers differ across classes ---
    print("Running ANOVA per channel...\n")
    f_scores = np.zeros(n_ch)           # combined F-score across all bands
    f_per_band = np.zeros((n_ch, n_bands))

    for ch in range(n_ch):
        band_f_values = []
        for b in range(n_bands):
            groups = [features[ch, labels == c, b] for c in range(5)]
            f_val, p_val = f_oneway(*groups)
            f_per_band[ch, b] = f_val
            band_f_values.append(f_val)
        # Combined score = mean F across bands
        f_scores[ch] = np.mean(band_f_values)

    # --- Rank channels ---
    ranking = np.argsort(f_scores)[::-1]  # highest F first

    print("=" * 72)
    print(f"{'Rank':>4}  {'Ch#':>3}  {'Channel':>7}  {'F-score':>8}  "
          f"{'Region':>12}  {'theta':>7} {'alpha':>7} {'l-beta':>7} {'l-gamma':>7}")
    print("=" * 72)
    for rank, idx in enumerate(ranking, 1):
        region = get_region(clab[idx])
        marker = " <<<" if region in ("Broca", "Wernicke", "SMA", "Motor") else ""
        if region == "Eye-artifact":
            marker = " [!]"
        print(f"{rank:4d}  {idx+1:3d}  {clab[idx]:>7s}  {f_scores[idx]:8.2f}  "
              f"{region:>12s}  {f_per_band[idx,0]:7.2f} {f_per_band[idx,1]:7.2f} "
              f"{f_per_band[idx,2]:7.2f} {f_per_band[idx,3]:7.2f}{marker}")
    print()

    # --- Summary: how many speech-region channels are in the top 20? ---
    top_20 = [clab[i] for i in ranking[:20]]
    speech_in_top20 = [ch for ch in top_20 if ch in ALL_SPEECH]
    print(f"Speech-relevant channels in top 20: {len(speech_in_top20)}/20")
    print(f"  → {speech_in_top20}\n")

    # --- Suggested channel subset ---
    # Take top 20 channels, but exclude artifact-prone ones
    suggested = []
    for idx in ranking:
        if clab[idx] not in ARTIFACT_PRONE and len(suggested) < 20:
            suggested.append((clab[idx], get_region(clab[idx]), f_scores[idx]))
    
    print("=" * 50)
    print("SUGGESTED 20-CHANNEL SUBSET (data + anatomy)")
    print("=" * 50)
    for i, (ch, reg, score) in enumerate(suggested, 1):
        tag = f"  [{reg}]" if reg else ""
        print(f"  {i:2d}. {ch:>5s}  (F = {score:.2f}){tag}")
    print()

    # Also show a compact 8-channel subset (for low-cost headset viability)
    compact = suggested[:8]
    print("COMPACT 8-CHANNEL SUBSET (for low-cost deployment)")
    print("-" * 50)
    for i, (ch, reg, score) in enumerate(compact, 1):
        tag = f"  [{reg}]" if reg else ""
        print(f"  {i}. {ch:>5s}  (F = {score:.2f}){tag}")
    print()

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = []
    for idx in ranking:
        ch = clab[idx]
        if ch in BROCA:
            colors.append("#e74c3c")        # red
        elif ch in WERNICKE:
            colors.append("#2ecc71")        # green
        elif ch in SMA:
            colors.append("#3498db")        # blue
        elif ch in MOTOR:
            colors.append("#f39c12")        # orange
        elif ch in ARTIFACT_PRONE:
            colors.append("#95a5a6")        # grey
        else:
            colors.append("#bdc3c7")        # light grey

    ax.bar(range(n_ch), f_scores[ranking], color=colors)
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels([clab[i] for i in ranking], rotation=90, fontsize=7)
    ax.set_ylabel("Mean ANOVA F-score (across 4 bands)")
    ax.set_title("Channel Discriminability Ranking — Subject 01")

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#e74c3c", label="Broca's area"),
        Patch(facecolor="#2ecc71", label="Wernicke's area"),
        Patch(facecolor="#3498db", label="SMA / Premotor"),
        Patch(facecolor="#f39c12", label="Motor cortex"),
        Patch(facecolor="#95a5a6", label="Eye-artifact prone"),
        Patch(facecolor="#bdc3c7", label="Other"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("channel_ranking.png", dpi=150)
    print("Saved: channel_ranking.png")
    plt.show()


if __name__ == "__main__":
    main()
