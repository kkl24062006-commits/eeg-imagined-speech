"""
Hemisphere Verification — Left vs Right Speech Channels
--------------------------------------------------------
Investigates whether the right-hemisphere dominance in channel selection
is a real finding or a preprocessing artifact.

Three checks:
  1. Aggregate F-score comparison (left vs right speech channels)
  2. Per-subject hemisphere preference (how many of 15 favor left vs right)
  3. Cross-preprocessing comparison (raw vs ICA-cleaned vs full preprocessing)

Uses preprocessed .npz files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch as scipy_welch
from scipy.stats import f_oneway, ttest_rel, wilcoxon

# ============================================================
# CONFIG
# ============================================================
PREPROCESSED_FOLDER = r"C:\Users\kkl24\Downloads\BCI_project\preprocessed_data"
RAW_FOLDER = r"C:\Users\kkl24\Downloads\pq7vb-osfstorage-Track#3 Imagined speech classification-archive\Training set"
OUTPUT_FOLDER = "hemisphere_verification"
# ============================================================

BANDS = {
    "theta":    (4, 8),
    "alpha":    (8, 13),
    "low_beta": (13, 20),
    "low_gamma":(30, 50),
}

# --- LEFT hemisphere speech-related channels ---
LEFT_SPEECH = {
    "F5":  "Broca (left frontal)",
    "F7":  "Broca (left frontal)",
    "FC5": "Broca (left premotor)",
    "FT7": "Broca (left)",
    "FC3": "Broca/motor (left)",
    "T7":  "Wernicke (left temporal)",
    "TP7": "Wernicke (left temporo-parietal)",
    "CP5": "Wernicke (left parietal)",
    "C3":  "Motor (left)",
    "C5":  "Motor (left)",
    "CP3": "Motor (left)",
}

# --- RIGHT hemisphere homologues ---
RIGHT_SPEECH = {
    "F6":  "Right frontal (homologue of F5)",
    "F8":  "Right frontal (homologue of F7)",
    "FC6": "Right premotor (homologue of FC5)",
    "FT8": "Right (homologue of FT7)",
    "FC4": "Right motor (homologue of FC3)",
    "T8":  "Right temporal (homologue of T7)",
    "TP8": "Right temporo-parietal (homologue of TP7)",
    "CP6": "Right parietal (homologue of CP5)",
    "C4":  "Right motor (homologue of C3)",
    "C6":  "Right motor (homologue of C5)",
    "CP4": "Right motor (homologue of CP3)",
}


def rank_channels_welch(x, y, fs):
    """Same Welch + ANOVA method used across all channel selection scripts."""
    n_time, n_ch, n_trials = x.shape
    labels = np.argmax(y, axis=0)
    baseline_samples = int(0.5 * fs)
    x_speech = x[baseline_samples:, :, :]
    nperseg = min(x_speech.shape[0], 256)

    band_names = list(BANDS.keys())
    features = np.zeros((n_ch, n_trials, len(band_names)))

    for ch in range(n_ch):
        freqs, psd = scipy_welch(x_speech[:, ch, :], fs=fs, nperseg=nperseg, axis=0)
        for b, bname in enumerate(band_names):
            low, high = BANDS[bname]
            mask = (freqs >= low) & (freqs <= high)
            features[ch, :, b] = np.mean(psd[mask, :], axis=0)

    f_scores = np.zeros(n_ch)
    for ch in range(n_ch):
        band_f = []
        for b in range(len(band_names)):
            groups = [features[ch, labels == c, b] for c in range(5)]
            f_val, _ = f_oneway(*groups)
            band_f.append(f_val)
        f_scores[ch] = np.mean(band_f)
    return f_scores


def load_raw(filepath):
    mat = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    epo_key = [k for k in mat.keys() if k.startswith("epo")][0]
    epo = mat[epo_key]
    return np.array(epo.x), np.array(epo.y), int(epo.fs), [str(c) for c in epo.clab]


def load_preprocessed(filepath):
    d = np.load(filepath, allow_pickle=True)
    return d["x"], d["y"], int(d["fs"]), list(d["clab"])


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ==========================================================
    # RUN ON BOTH RAW AND PREPROCESSED DATA
    # ==========================================================
    raw_files = sorted([f for f in os.listdir(RAW_FOLDER)
                        if f.endswith(".mat") and f.startswith("Data_Sample")])
    prep_files = sorted([f for f in os.listdir(PREPROCESSED_FOLDER)
                         if f.endswith("_clean.npz")])

    n_subj = len(prep_files)
    print(f"Processing {n_subj} subjects...\n")

    all_raw_scores = []
    all_prep_scores = []
    clab = None

    for i in range(n_subj):
        print(f"Subject {i+1}/{n_subj}...", end=" ", flush=True)

        # Raw
        x_raw, y, fs, clab = load_raw(os.path.join(RAW_FOLDER, raw_files[i]))
        raw_scores = rank_channels_welch(x_raw, y, fs)
        all_raw_scores.append(raw_scores)

        # Preprocessed
        x_prep, y, fs, clab_p = load_preprocessed(os.path.join(PREPROCESSED_FOLDER, prep_files[i]))
        prep_scores = rank_channels_welch(x_prep, y, fs)
        all_prep_scores.append(prep_scores)

        print("done")

    raw_scores = np.array(all_raw_scores)     # (15, 64)
    prep_scores = np.array(all_prep_scores)   # (15, 64)

    # Get channel indices for left/right speech
    left_idx = [clab.index(ch) for ch in LEFT_SPEECH if ch in clab]
    right_idx = [clab.index(ch) for ch in RIGHT_SPEECH if ch in clab]

    left_labels = [ch for ch in LEFT_SPEECH if ch in clab]
    right_labels = [ch for ch in RIGHT_SPEECH if ch in clab]

    # ==========================================================
    # CHECK 1: Aggregate F-score comparison
    # ==========================================================
    print("\n" + "=" * 70)
    print("CHECK 1: AGGREGATE F-SCORE (mean across all subjects)")
    print("=" * 70)

    for name, scores in [("RAW DATA", raw_scores), ("PREPROCESSED", prep_scores)]:
        left_mean = scores[:, left_idx].mean()
        right_mean = scores[:, right_idx].mean()
        diff = right_mean - left_mean
        pct_diff = (diff / left_mean) * 100

        print(f"\n{name}:")
        print(f"  Left  hemisphere mean F-score: {left_mean:.3f}")
        print(f"  Right hemisphere mean F-score: {right_mean:.3f}")
        print(f"  Difference: {diff:+.3f} ({pct_diff:+.1f}%)")
        if diff > 0:
            print(f"  → RIGHT dominant by {pct_diff:.1f}%")
        else:
            print(f"  → LEFT dominant by {-pct_diff:.1f}%")

    # ==========================================================
    # CHECK 2: Per-subject hemisphere preference
    # ==========================================================
    print("\n" + "=" * 70)
    print("CHECK 2: PER-SUBJECT PREFERENCE (raw + preprocessed)")
    print("=" * 70)
    print(f"{'Subj':>5} | {'RAW Left mean':>14} {'RAW Right mean':>14} {'Winner':>8}"
          f" | {'PREP Left mean':>15} {'PREP Right mean':>15} {'Winner':>8}")
    print("-" * 100)

    raw_left_wins, raw_right_wins = 0, 0
    prep_left_wins, prep_right_wins = 0, 0

    for i in range(n_subj):
        rl = raw_scores[i, left_idx].mean()
        rr = raw_scores[i, right_idx].mean()
        raw_winner = "LEFT" if rl > rr else "RIGHT"
        if rl > rr: raw_left_wins += 1
        else: raw_right_wins += 1

        pl = prep_scores[i, left_idx].mean()
        pr = prep_scores[i, right_idx].mean()
        prep_winner = "LEFT" if pl > pr else "RIGHT"
        if pl > pr: prep_left_wins += 1
        else: prep_right_wins += 1

        print(f"S{i+1:2d}   | {rl:14.3f} {rr:14.3f} {raw_winner:>8}"
              f" | {pl:15.3f} {pr:15.3f} {prep_winner:>8}")

    print("-" * 100)
    print(f"RAW    total:  LEFT wins {raw_left_wins}/{n_subj}, RIGHT wins {raw_right_wins}/{n_subj}")
    print(f"PREP   total:  LEFT wins {prep_left_wins}/{n_subj}, RIGHT wins {prep_right_wins}/{n_subj}")

    # ==========================================================
    # CHECK 3: Statistical significance (paired test)
    # ==========================================================
    print("\n" + "=" * 70)
    print("CHECK 3: PAIRED STATISTICAL TEST (Wilcoxon signed-rank)")
    print("Tests if left vs right difference is significant across subjects")
    print("=" * 70)

    raw_left_per_subj = raw_scores[:, left_idx].mean(axis=1)
    raw_right_per_subj = raw_scores[:, right_idx].mean(axis=1)
    prep_left_per_subj = prep_scores[:, left_idx].mean(axis=1)
    prep_right_per_subj = prep_scores[:, right_idx].mean(axis=1)

    stat_raw, p_raw = wilcoxon(raw_left_per_subj, raw_right_per_subj)
    stat_prep, p_prep = wilcoxon(prep_left_per_subj, prep_right_per_subj)

    print(f"\nRAW DATA:          W={stat_raw:.2f}, p={p_raw:.4f}")
    if p_raw < 0.05:
        winner = "RIGHT" if raw_right_per_subj.mean() > raw_left_per_subj.mean() else "LEFT"
        print(f"  → SIGNIFICANT difference. {winner} hemisphere dominant.")
    else:
        print(f"  → NO significant difference (both hemispheres similar).")

    print(f"\nPREPROCESSED DATA: W={stat_prep:.2f}, p={p_prep:.4f}")
    if p_prep < 0.05:
        winner = "RIGHT" if prep_right_per_subj.mean() > prep_left_per_subj.mean() else "LEFT"
        print(f"  → SIGNIFICANT difference. {winner} hemisphere dominant.")
    else:
        print(f"  → NO significant difference (both hemispheres similar).")

    # ==========================================================
    # CHECK 4: Channel-by-channel left vs right pairs
    # ==========================================================
    print("\n" + "=" * 70)
    print("CHECK 4: LEFT vs RIGHT PAIRED CHANNELS (preprocessed data)")
    print("=" * 70)

    pairs = [
        ("F5", "F6"), ("F7", "F8"), ("FC5", "FC6"), ("FT7", "FT8"),
        ("FC3", "FC4"), ("T7", "T8"), ("TP7", "TP8"), ("CP5", "CP6"),
        ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"),
    ]

    print(f"{'Pair':>10} | {'Left mean':>10} {'Right mean':>11} {'Winner':>8} {'Diff %':>8}")
    print("-" * 60)

    left_wins_pair, right_wins_pair = 0, 0
    for left_ch, right_ch in pairs:
        if left_ch not in clab or right_ch not in clab:
            continue
        l_score = prep_scores[:, clab.index(left_ch)].mean()
        r_score = prep_scores[:, clab.index(right_ch)].mean()
        winner = "LEFT" if l_score > r_score else "RIGHT"
        if l_score > r_score: left_wins_pair += 1
        else: right_wins_pair += 1
        pct = ((r_score - l_score) / l_score) * 100
        print(f"{left_ch:>4}/{right_ch:>4} | {l_score:10.3f} {r_score:11.3f} {winner:>8} {pct:+7.1f}%")

    print(f"\nPairwise winner count: LEFT {left_wins_pair}, RIGHT {right_wins_pair} out of {len(pairs)} pairs")

    # ==========================================================
    # PLOT
    # ==========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot: per-subject preference
    x_pos = np.arange(n_subj)
    width = 0.35
    axes[0].bar(x_pos - width/2, raw_left_per_subj, width, label="Raw - Left",
                color="#3498db", alpha=0.7)
    axes[0].bar(x_pos + width/2, raw_right_per_subj, width, label="Raw - Right",
                color="#e74c3c", alpha=0.7)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"S{i+1}" for i in range(n_subj)], fontsize=8)
    axes[0].set_ylabel("Mean F-score")
    axes[0].set_title(f"Raw Data: Left vs Right\n(Left wins {raw_left_wins}/{n_subj})")
    axes[0].legend()

    axes[1].bar(x_pos - width/2, prep_left_per_subj, width, label="Prep - Left",
                color="#3498db", alpha=0.7)
    axes[1].bar(x_pos + width/2, prep_right_per_subj, width, label="Prep - Right",
                color="#e74c3c", alpha=0.7)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"S{i+1}" for i in range(n_subj)], fontsize=8)
    axes[1].set_ylabel("Mean F-score")
    axes[1].set_title(f"Preprocessed Data: Left vs Right\n(Left wins {prep_left_wins}/{n_subj})")
    axes[1].legend()

    plt.suptitle("HEMISPHERE VERIFICATION", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "hemisphere_comparison.png"), dpi=150)
    print(f"\nSaved: hemisphere_comparison.png")
    plt.close()

    # ==========================================================
    # VERDICT
    # ==========================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if p_prep < 0.05 and prep_right_per_subj.mean() > prep_left_per_subj.mean():
        if raw_right_per_subj.mean() > raw_left_per_subj.mean():
            print("Right-hemisphere dominance is a ROBUST finding:")
            print("  ✓ Present in BOTH raw and preprocessed data")
            print("  ✓ Statistically significant (Wilcoxon p < 0.05)")
            print(f"  ✓ {prep_right_wins}/{n_subj} subjects favor right hemisphere")
            print("\n  This is likely a REAL neurophysiological finding for this dataset,")
            print("  not a preprocessing artifact. Discuss in report as: whole-word")
            print("  imagined speech engaging bilateral or right-lateralized motor")
            print("  networks, especially the right motor articulatory strip.")
        else:
            print("Right dominance emerged AFTER preprocessing:")
            print("  ⚠ Raw data was more balanced or left-dominant")
            print("  ⚠ Preprocessing (likely ICA) may have removed left components")
            print("  → Recommend investigating ICA components before finalizing")
    else:
        print("NO clear hemisphere preference:")
        print("  ✓ Difference not statistically significant")
        print("  → Include BOTH hemispheres in final channel selection")
        print("  → Bilateral motor cortex is the safest interpretation")


if __name__ == "__main__":
    main()
