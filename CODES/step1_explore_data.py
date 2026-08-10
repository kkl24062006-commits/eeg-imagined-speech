"""
Step 1 — Explore the Track#3 Imagined Speech dataset
----------------------------------------------------
Loads ONE subject's .mat file, prints its structure, and plots a few trials.
Run this first to confirm the data is loading correctly before doing
channel selection.

Dataset structs (from the description file):
    epo.x         -> raw data (time x channels x trials)
    epo.y         -> class labels
    epo.fs        -> sampling frequency
    epo.t         -> time points (-500 ms ~ 2600 ms)
    epo.className -> class names (Hello, Help me, Stop, Thank you, Yes)
    epo.clab      -> channel labels (64 channels)
    mnt           -> channel montage / positions
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG  --  change this ONE line to point at your file
# ============================================================
# Windows example (note the 'r' before the quotes — it handles backslashes):
DATA_PATH = r"C:\Users\kkl24\Downloads\pq7vb-osfstorage-Track#3 Imagined speech classification-archive\Training set\Data_sample01.mat"
# ============================================================


def load_mat(path):
    """Load a .mat file. Handles both old (<v7.3) and new (v7.3/HDF5) formats."""
    try:
        from scipy.io import loadmat
        mat = loadmat(path, struct_as_record=False, squeeze_me=True)
        return mat, "scipy"
    except NotImplementedError:
        # v7.3 files are HDF5 — need mat73 instead
        import mat73
        mat = mat73.loadmat(path)
        return mat, "mat73"


def get_field(epo, name, loader):
    """Grab a field from the epo struct regardless of loader used."""
    if loader == "scipy":
        return getattr(epo, name)
    else:  # mat73 returns dicts
        return epo[name]


def main():
    print(f"Loading: {DATA_PATH}\n")
    mat, loader = load_mat(DATA_PATH)
    print(f"Loaded with: {loader}")
    print(f"Top-level keys: {[k for k in mat.keys() if not k.startswith('__')]}\n")

    # Auto-detect the key name (epo_train / epo_validation / epo_test / epo)
    epo_key = None
    for k in mat.keys():
        if k.startswith("epo"):
            epo_key = k
            break
    if epo_key is None:
        raise KeyError("No epo-related key found in the .mat file!")
    print(f"Using key: '{epo_key}'\n")
    epo = mat[epo_key]

    # --- Pull the main fields ---
    x = np.array(get_field(epo, "x", loader))          # time x channels x trials
    y = np.array(get_field(epo, "y", loader))          # labels
    fs = get_field(epo, "fs", loader)                  # sampling freq
    clab = get_field(epo, "clab", loader)              # channel labels
    class_names = get_field(epo, "className", loader)  # class names

    # --- Print structure ---
    print("=" * 55)
    print("DATA STRUCTURE")
    print("=" * 55)
    print(f"epo.x shape (time x channels x trials): {x.shape}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Number of channels: {len(clab)}")
    print(f"Class names: {list(class_names)}")
    print(f"epo.y shape: {np.array(y).shape}")
    print()

    # --- Channel labels ---
    print("=" * 55)
    print("CHANNEL LABELS (all 64)")
    print("=" * 55)
    clab_list = [str(c) for c in clab]
    for i, ch in enumerate(clab_list, 1):
        print(f"{i:2d}: {ch:5s}", end="   ")
        if i % 5 == 0:
            print()
    print("\n")

    # --- Class distribution ---
    # epo.y is often one-hot (classes x trials). Convert to a flat label list.
    y_arr = np.array(y)
    if y_arr.ndim == 2:                     # one-hot encoded
        labels = np.argmax(y_arr, axis=0)
    else:
        labels = y_arr.astype(int).ravel()
    unique, counts = np.unique(labels, return_counts=True)
    print("=" * 55)
    print("TRIALS PER CLASS")
    print("=" * 55)
    for u, c in zip(unique, counts):
        print(f"  class {u}: {c} trials")
    print()

    # --- Visualize: 5 channels from the first trial ---
    trial_idx = 0
    n_channels_to_plot = 5
    plt.figure(figsize=(11, 6))
    for ch in range(n_channels_to_plot):
        plt.plot(x[:, ch, trial_idx] + ch * 40,   # offset each channel vertically
                 label=clab_list[ch])
    plt.title(f"First trial — first {n_channels_to_plot} channels (stacked)")
    plt.xlabel("Time samples")
    plt.ylabel("Amplitude (offset per channel)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("trial_preview.png", dpi=120)
    print("Saved plot to: trial_preview.png")
    plt.show()


if __name__ == "__main__":
    main()
