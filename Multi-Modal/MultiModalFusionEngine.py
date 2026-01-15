from MRISpatialEncoder import MRISpatialEncoder
from EEGTemporalEncoder import EEGTemporalEncoder
import matplotlib.pyplot as plt
from nilearn import plotting
from scipy import signal
import numpy as np

class MultiModalFusionEngine:
    def __init__(self):
        self.mri_encoder = MRISpatialEncoder()
        self.eeg_encoder = EEGTemporalEncoder()
        self.results = {}

    def run_diagnosis(self, mri_func, mri_conf):
        # 1. MRI Stream
        spatial_risk = self.mri_encoder.process_volume(mri_func, mri_conf)
        
        # 2. EEG Stream
        self.eeg_encoder.fetch_eeg()
        temporal_risk = self.eeg_encoder.compute_spectral_features()
        
        # 3. Fusion Logic
        # MRI detects structure, EEG detects function.
        fused_risk = (0.6 * spatial_risk) + (0.4 * temporal_risk)
        
        self.results = {
            "MRI_Risk": spatial_risk,
            "EEG_Risk": temporal_risk,
            "Fused_Risk": fused_risk
        }
        return self.results

    def generate_multimodal_dashboard(self):
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 2)
        
        # A. MRI Connectome
        ax1 = fig.add_subplot(gs[0, 0])
        plotting.plot_connectome(
            self.mri_encoder.adj_matrix, 
            self.mri_encoder.atlas.region_coords,
            edge_threshold='90%',
            node_color='blue',
            display_mode='z',
            axes=ax1,
            title="Spatial Atrophy Map (fMRI)"
        )
        
        # B. EEG Spectral Plot
        ax2 = fig.add_subplot(gs[1, 0])
        # Re-compute for plotting
        avg_signal = np.mean(self.eeg_encoder.eeg_data, axis=0)
        freqs, psd = signal.welch(avg_signal, self.eeg_encoder.fs, nperseg=2048)
        
        ax2.plot(freqs, psd, color='purple', lw=2)
        ax2.set_xlim(0, 30)
        ax2.fill_between(freqs, psd, where=((freqs>=4)&(freqs<=8)), color='red', alpha=0.3, label='Theta')
        ax2.fill_between(freqs, psd, where=((freqs>=8)&(freqs<=12)), color='green', alpha=0.3, label='Alpha')
        ax2.set_title("Spectral Analysis (EEG)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.legend()
        
        # C. Fusion Risk Gauge
        ax3 = fig.add_subplot(gs[:, 1])
        ax3.axis('off')
        
        risk_pct = self.results['Fused_Risk'] * 100
        
        ax3.text(0.5, 0.8, "MULTI-MODAL DIAGNOSIS", ha='center', fontsize=18, weight='bold')
        ax3.text(0.5, 0.6, f"{risk_pct:.1f}%", ha='center', fontsize=60, weight='bold', color='green')
        ax3.text(0.5, 0.5, "Risk Score", ha='center', fontsize=12)
        ax3.text(0.5, 0.4, "(Low score expected for healthy controls)", ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig('Data_Analysis.png', dpi=300)
        print("[Output] Generated 'Data_Analysis.png'")