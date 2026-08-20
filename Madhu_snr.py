# =============================================================================
# SNR VALIDATION: Raw vs Preprocessed EEG
# Track#3 Multi-class Imagined Speech Dataset
# =============================================================================
# Paste each "# %% [CELL n]" block into a separate Google Colab cell.
#
# Definitions used (based on your stated preprocessing):
#   - Bandpass applied: 0.5-50 Hz  -> this is the "signal" range
#   - Notch applied: 50 Hz (line noise) -> a narrow band around 50 Hz is
#     treated as NOISE even though it sits inside the 0.5-50 Hz range
#   - Anything outside 0.5-50 Hz is treated as NOISE
# =============================================================================

# %% [CELL 1] --- Install / import dependencies ------------------------------
# If needed, install dependencies in a shell before running this script:
# pip install scipy h5py

import numpy as np
import scipy.io as sio
import h5py
import os
import glob
from scipy.signal import welch


# EDIT THESE two paths to your actual folders
RAW_DATA_DIR = r"C:\Users\kkl24\Downloads\pq7vb-osfstorage-Track#3 Imagined speech classification-archive\Training set"       # raw data
PREPROCESSED_DATA_DIR = r"C:\Users\kkl24\Downloads\BCI_project\preprocessed_data"  # your cleaned data

# %% [CELL 3] --- Loader: reads epo (signals) + mnt (electrode positions) ----
# Same loader as before. If your PREPROCESSED files are saved with a
# different struct/variable name than the raw files, add that name to
# POSSIBLE_EPO_KEYS below.

def load_epo_mnt(filepath):
    POSSIBLE_EPO_KEYS = ['epo_train', 'epo_validation', 'epo_test', 'epo',
                         'epo_preprocessed', 'epo_clean']  # add your own name here if needed

    try:
        mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=False)

        epo_key = next((k for k in POSSIBLE_EPO_KEYS if k in mat), None)
        if epo_key is None:
            real_keys = [k for k in mat.keys() if not k.startswith('__')]
            raise KeyError(
                f"None of {POSSIBLE_EPO_KEYS} found in {filepath}. "
                f"Actual variables in this file: {real_keys}"
            )

        epo_raw = mat[epo_key]
        epo = {
            'x': np.array(epo_raw.x),
            'y': np.array(epo_raw.y),
            'fs': float(epo_raw.fs),
            'clab': [str(c) for c in np.array(epo_raw.clab)],
        }
        return epo

    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            epo_key = next((k for k in POSSIBLE_EPO_KEYS if k in f), None)
            if epo_key is None:
                real_keys = [k for k in f.keys() if not k.startswith('__')]
                raise KeyError(
                    f"None of {POSSIBLE_EPO_KEYS} found in {filepath}. "
                    f"Actual variables in this file: {real_keys}"
                )
            epo_grp = f[epo_key]
            def deref_str(ref_array):
                return [''.join(chr(c[0]) for c in f[ref][:]) for ref in ref_array]

            epo = {
                'x': np.array(epo_grp['x']).T,
                'y': np.array(epo_grp['y']).T,
                'fs': float(np.array(epo_grp['fs']).squeeze()),
                'clab': deref_str(epo_grp['clab'][:].squeeze()),
            }
        return epo


# %% [CELL 4] --- SANITY CHECK: confirm BOTH raw and preprocessed load -------

raw_test = load_epo_mnt(os.path.join(RAW_DATA_DIR, "Data_Sample01.mat"))
print("RAW   -- x shape:", raw_test['x'].shape, " fs:", raw_test['fs'])

# EDIT the filename below if your preprocessed files are named differently
preproc_test = load_epo_mnt(os.path.join(PREPROCESSED_DATA_DIR, "Data_Sample01.mat"))
print("CLEAN -- x shape:", preproc_test['x'].shape, " fs:", preproc_test['fs'])

# If both print real shapes, you're clear to continue.

# %% [CELL 5] --- SNR calculation (signal band vs noise band) ----------------

SIGNAL_LOW = 0.5
SIGNAL_HIGH = 50.0
NOTCH_FREQ = 50.0
NOTCH_WIDTH = 2.0   # excludes 49-51 Hz from the "signal" band, treats it as noise

def signal_noise_power(sig, fs):
    """
    sig: 1D array, one channel, one trial
    Returns: (signal_power, noise_power)
      signal_power = power within [SIGNAL_LOW, SIGNAL_HIGH] Hz, EXCLUDING
                     the narrow notch band around NOTCH_FREQ
      noise_power  = power everywhere else (out-of-band + the notch band)
    """
    freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))

    notch_lo = NOTCH_FREQ - NOTCH_WIDTH / 2
    notch_hi = NOTCH_FREQ + NOTCH_WIDTH / 2

    in_signal_range = (freqs >= SIGNAL_LOW) & (freqs <= SIGNAL_HIGH)
    in_notch_band = (freqs >= notch_lo) & (freqs <= notch_hi)

    signal_mask = in_signal_range & ~in_notch_band
    noise_mask = ~signal_mask   # everything else, including the notch band

    signal_power = psd[signal_mask].sum()
    noise_power = psd[noise_mask].sum()
    return signal_power, noise_power


def compute_snr_db(x, fs):
    """
    x: array (time, channels, trials)
    Computes SNR (in dB) for EVERY channel/trial combination, then returns
    the average SNR across all of them -- one overall SNR number for
    this subject's file.
    """
    n_time, n_ch, n_trials = x.shape
    snr_values = []

    for ch in range(n_ch):
        for tr in range(n_trials):
            sig = x[:, ch, tr]
            sig_power, noise_power = signal_noise_power(sig, fs)
            if noise_power > 0 and sig_power > 0:
                snr_db = 10 * np.log10(sig_power / noise_power)
                snr_values.append(snr_db)

    return np.mean(snr_values)


# %% [CELL 6] --- Run for ONE subject first, to confirm it works -----------

raw_snr = compute_snr_db(raw_test['x'], raw_test['fs'])
clean_snr = compute_snr_db(preproc_test['x'], preproc_test['fs'])

print(f"Subject 1 -- Raw SNR:          {raw_snr:.2f} dB")
print(f"Subject 1 -- Preprocessed SNR: {clean_snr:.2f} dB")
print(f"Subject 1 -- Improvement:      {clean_snr - raw_snr:+.2f} dB")

# %% [CELL 7] --- Run across all 15 subjects, compare raw vs preprocessed ---

raw_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "Data_Sample*.mat")))
preproc_files = sorted(glob.glob(os.path.join(PREPROCESSED_DATA_DIR, "Data_Sample*.mat")))

print(f"Found {len(raw_files)} raw files, {len(preproc_files)} preprocessed files\n")

results = []
print(f"{'Subject':14s} {'Raw SNR (dB)':>14s} {'Clean SNR (dB)':>16s} {'Improvement':>14s}")

for raw_fp, clean_fp in zip(raw_files, preproc_files):
    subj_name = os.path.basename(raw_fp).replace('.mat', '')

    raw_epo = load_epo_mnt(raw_fp)
    clean_epo = load_epo_mnt(clean_fp)

    raw_snr = compute_snr_db(raw_epo['x'], raw_epo['fs'])
    clean_snr = compute_snr_db(clean_epo['x'], clean_epo['fs'])
    improvement = clean_snr - raw_snr

    results.append((subj_name, raw_snr, clean_snr, improvement))
    print(f"{subj_name:14s} {raw_snr:14.2f} {clean_snr:16.2f} {improvement:+14.2f}")

# %% [CELL 8] --- Summary across all subjects --------------------------------

raw_snrs = np.array([r[1] for r in results])
clean_snrs = np.array([r[2] for r in results])
improvements = np.array([r[3] for r in results])

print("\n=== Summary across all subjects ===")
print(f"Mean Raw SNR:          {raw_snrs.mean():.2f} dB  (std {raw_snrs.std():.2f})")
print(f"Mean Preprocessed SNR: {clean_snrs.mean():.2f} dB  (std {clean_snrs.std():.2f})")
print(f"Mean Improvement:      {improvements.mean():+.2f} dB  (std {improvements.std():.2f})")
print(f"Subjects improved:     {(improvements > 0).sum()} / {len(improvements)}")