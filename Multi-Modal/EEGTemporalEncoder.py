import numpy as np
import mne # Library for EEG Data
from scipy import signal

class EEGTemporalEncoder:
    def __init__(self):
        self.eeg_data = None
        self.fs = None # Sampling rate
        
    def fetch_eeg(self):
        print("[EEG] Downloading MNE Sample Dataset ( Biological Signals)...")
        data_path = mne.datasets.sample.data_path()
        raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
        
        # Load data and pick only EEG channels
        raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
        raw.pick_types(meg=False, eeg=True, eog=False)
        
        # Get data as numpy array
        self.eeg_data = raw.get_data()
        self.fs = raw.info['sfreq']
        print(f"   [EEG] Loaded {self.eeg_data.shape[0]} channels, {self.eeg_data.shape[1]} timepoints.")

    def compute_spectral_features(self):
        if self.eeg_data is None: self.fetch_eeg()

        # Average across all channels (Global Field Power equivalent)
        avg_signal = np.mean(self.eeg_data, axis=0)
        
        # Compute Power Spectral Density (PSD)
        freqs, psd = signal.welch(avg_signal, self.fs, nperseg=2048)
        
        # Define Bands
        # Theta: 4-8 Hz (Increases in Dementia)
        # Alpha: 8-12 Hz (Decreases in Dementia)
        theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
        alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)
        
        theta_power = np.mean(psd[theta_idx])
        alpha_power = np.mean(psd[alpha_idx])
        
        # Biomarker: Theta/Alpha Ratio (TAR)
        tar = theta_power / alpha_power if alpha_power > 0 else 0
        print(f"[EEG] Measured Theta/Alpha Ratio: {tar:.4f}")
        
        # Normalize TAR to Risk (0-1)
        # In this healthy sample, TAR should be low.
        risk = (tar - 0.5) 
        return max(0, min(risk, 1.0))