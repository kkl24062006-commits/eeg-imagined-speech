"""
Channel Selection on Preprocessed Data — Welch + ANOVA + Topo Maps
------------------------------------------------------------------
Same pipeline as Steps 2-3 (Welch PSD, ANOVA F-score, cross-subject
consistency), now running on ICA-cleaned preprocessed .npz data,
with topographic head maps added.

Outputs:
  1. Full 64-channel ranking table
  2. Per-subject topo maps
  3. Grand average topo map
  4. Consistency bar chart (color-coded by brain region)
  5. Consistency heatmap (channel × subject)
  6. Validated channel subsets (20ch, 8ch)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch as scipy_welch
from scipy.stats import f_oneway
from scipy.interpolate import griddata
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")

# ============================================================
# CONFIG
# ============================================================
PREPROCESSED_FOLDER = r"C:\Users\kkl24\Downloads\BCI_project\preprocessed_data"
ORIGINAL_MAT = r"C:\Users\kkl24\Downloads\pq7vb-osfstorage-Track#3 Imagined speech classification-archive\Training set\Data_Sample01.mat"
OUTPUT_FOLDER = r"C:\Users\kkl24\Downloads\BCI_project\channel_selection_results"
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


def region_color(ch):
    if ch in BROCA:    return "#e74c3c"
    if ch in WERNICKE: return "#2ecc71"
    if ch in SMA:      return "#3498db"
    if ch in MOTOR:    return "#f39c12"
    if ch in ARTIFACT: return "#95a5a6"
    return "#bdc3c7"


# ============================================================
# LOADING
# ============================================================
def load_montage(mat_path):
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    mnt = mat["mnt"]
    return np.array(mnt.x), np.array(mnt.y), [str(c) for c in mnt.clab]


def load_preprocessed(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d["x"], d["y"], int(d["fs"]), list(d["clab"])


# ============================================================
# CHANNEL RANKING (same Welch + ANOVA as before)
# ============================================================
def rank_channels(x, y, fs):
    n_time, n_ch, n_trials = x.shape
    labels = np.argmax(y, axis=0)

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
    f_per_band = np.zeros((n_ch, n_bands))
    for ch in range(n_ch):
        for b in range(n_bands):
            groups = [features[ch, labels == c, b] for c in range(5)]
            f_val, _ = f_oneway(*groups)
            f_per_band[ch, b] = f_val
        f_scores[ch] = np.mean(f_per_band[ch])

    return f_scores, f_per_band


# ============================================================
# TOPOGRAPHIC MAP
# ============================================================
def plot_topomap(scores, mnt_x, mnt_y, clab, title, ax=None,
                 highlight_speech=True):
    grid_x, grid_y = np.mgrid[
        min(mnt_x):max(mnt_x):200j,
        min(mnt_y):max(mnt_y):200j
    ]
    grid_z = griddata((mnt_x, mnt_y), scores, (grid_x, grid_y), method="cubic")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.contourf(grid_x, grid_y, grid_z, levels=100, cmap="RdYlBu_r")
    ax.scatter(mnt_x, mnt_y, c="k", s=8)

    if highlight_speech:
        speech = BROCA | WERNICKE | MOTOR | SMA
        for i, ch in enumerate(clab):
            if ch in speech:
                ax.scatter(mnt_x[i], mnt_y[i], c="lime", s=40,
                           edgecolors="k", linewidths=1, zorder=5)
                ax.annotate(ch, (mnt_x[i], mnt_y[i]), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")

    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- Load montage ---
    print("Loading montage...")
    mnt_x, mnt_y, mnt_clab = load_montage(ORIGINAL_MAT)

    # --- Find preprocessed files ---
    npz_files = sorted([f for f in os.listdir(PREPROCESSED_FOLDER)
                        if f.endswith("_clean.npz")])
    if not npz_files:
        print(f"ERROR: No _clean.npz files found in {PREPROCESSED_FOLDER}")
        return

    n_subj = len(npz_files)
    print(f"Found {n_subj} preprocessed subjects.\n")

    # --- Process each subject ---
    all_f_scores = []
    clab = None

    for i, fname in enumerate(npz_files):
        filepath = os.path.join(PREPROCESSED_FOLDER, fname)
        subj_name = fname.replace("_clean.npz", "")
        print(f"Processing {subj_name} ({i+1}/{n_subj}) ...", end=" ", flush=True)

        x, y, fs, clab = load_preprocessed(filepath)
        scores, _ = rank_channels(x, y, fs)
        all_f_scores.append(scores)

        # Normalize for topo map
        s_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        # Per-subject topo map
        fig, ax = plt.subplots(figsize=(5, 5))
        im = plot_topomap(s_norm, mnt_x, mnt_y, clab,
                          title=f"{subj_name} — Channel Importance", ax=ax)
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"{subj_name}_topomap.png"), dpi=150)
        plt.close()
        print(f"done (top: {clab[np.argmax(scores)]})")

    n_ch = len(clab)
    score_matrix = np.array(all_f_scores)  # (n_subj, 64)

    # ============================================================
    # CROSS-SUBJECT CONSISTENCY
    # ============================================================
    rank_matrix = np.zeros_like(score_matrix, dtype=int)
    for s in range(n_subj):
        rank_matrix[s] = np.argsort(np.argsort(-score_matrix[s]))

    in_top_n = np.sum(rank_matrix < TOP_N, axis=0)
    consistency_order = np.argsort(-in_top_n)

    # --- Print full ranking ---
    print("\n" + "=" * 75)
    print(f"CROSS-SUBJECT CONSISTENCY (in top {TOP_N} out of {n_subj} subjects)")
    print("Preprocessed data (bandpass + notch + ICA + CAR + baseline)")
    print("=" * 75)
    print(f"{'Rank':>4}  {'Channel':>7}  {'In top 20':>9}  {'Avg rank':>9}  "
          f"{'Region':>12}  {'Verdict'}")
    print("-" * 75)

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
    validated_20 = []
    for idx in consistency_order:
        if clab[idx] not in ARTIFACT and len(validated_20) < 20:
            validated_20.append(clab[idx])
    validated_8 = validated_20[:8]

    speech_count = len([c for c in validated_20 if c in ALL_SPEECH])
    print(f"\n{'='*60}")
    print(f"VALIDATED 20-CHANNEL SUBSET ({speech_count} speech-region channels)")
    print(f"{'='*60}")
    for i, ch in enumerate(validated_20, 1):
        reg = get_region(ch)
        tag = f" [{reg}]" if reg else ""
        print(f"  {i:2d}. {ch}{tag}")

    print(f"\n8-CHANNEL COMPACT SUBSET:")
    for i, ch in enumerate(validated_8, 1):
        reg = get_region(ch)
        tag = f" [{reg}]" if reg else ""
        print(f"  {i}. {ch}{tag}")

    # ============================================================
    # PLOT 1: Grand average topo map
    # ============================================================
    mean_scores = score_matrix.mean(axis=0)
    mean_norm = (mean_scores - mean_scores.min()) / (mean_scores.max() - mean_scores.min() + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = plot_topomap(mean_norm, mnt_x, mnt_y, clab,
                        title="Grand Average Channel Importance\n(all 15 subjects, preprocessed)",
                        ax=axes[0])
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Consistency-based topo (count in top 20)
    count_norm = in_top_n / n_subj
    im2 = plot_topomap(count_norm, mnt_x, mnt_y, clab,
                        title=f"Consistency Score\n(fraction of subjects in top {TOP_N})",
                        ax=axes[1])
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.suptitle("CHANNEL SELECTION — Preprocessed Data", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "grand_average_topomap.png"), dpi=150)
    print(f"\nSaved: grand_average_topomap.png")
    plt.show()

    # ============================================================
    # PLOT 2: Consistency bar chart
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = [region_color(clab[idx]) for idx in consistency_order]

    ax.bar(range(n_ch), in_top_n[consistency_order], color=colors)
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels([clab[i] for i in consistency_order], rotation=90, fontsize=7)
    ax.set_ylabel(f"Number of subjects in top {TOP_N}")
    ax.set_title("Channel Consistency — Preprocessed Data (Welch + ANOVA)")
    ax.axhline(y=int(n_subj * 0.75), color="red", linestyle="--", alpha=0.7,
               label="75% (STRONG)")
    ax.axhline(y=int(n_subj * 0.5), color="orange", linestyle="--", alpha=0.7,
               label="50% (MODERATE)")

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
    plt.savefig(os.path.join(OUTPUT_FOLDER, "consistency_bar.png"), dpi=150)
    print("Saved: consistency_bar.png")
    plt.show()

    # ============================================================
    # PLOT 3: Consistency heatmap (channel × subject)
    # ============================================================
    fig, ax = plt.subplots(figsize=(18, 7))
    top30_idx = consistency_order[:30]
    heatmap_data = rank_matrix[:, top30_idx].T
    heatmap_labels = [clab[i] for i in top30_idx]

    im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=63)
    ax.set_yticks(range(len(heatmap_labels)))
    ax.set_yticklabels(heatmap_labels, fontsize=8)
    ax.set_xticks(range(n_subj))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_subj)], fontsize=8)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Channel (sorted by consistency)")
    ax.set_title("Channel Rank per Subject — Preprocessed Data "
                 "(green = top, red = bottom)")
    plt.colorbar(im, ax=ax, label="Rank (0 = best, 63 = worst)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "consistency_heatmap.png"), dpi=150)
    print("Saved: consistency_heatmap.png")
    plt.show()

    # ============================================================
    # PLOT 4: Selected channels highlighted on topo map
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 6))
    im = plot_topomap(mean_norm, mnt_x, mnt_y, clab,
                       title="Selected 20 Channels (highlighted)",
                       ax=ax, highlight_speech=False)

    # Highlight selected 20
    for i, ch in enumerate(clab):
        if ch in validated_20:
            rank_pos = validated_20.index(ch) + 1
            ax.scatter(mnt_x[i], mnt_y[i], c="lime", s=60,
                       edgecolors="black", linewidths=1.5, zorder=5)
            ax.annotate(f"{rank_pos}.{ch}", (mnt_x[i], mnt_y[i]),
                        fontsize=6, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points")

    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "selected_channels_topomap.png"), dpi=150)
    print("Saved: selected_channels_topomap.png")
    plt.show()


if __name__ == "__main__":
    main()
