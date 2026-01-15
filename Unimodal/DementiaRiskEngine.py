# AI-Driven Dementia Risk Assessment Platform
# Author: Abdul Wasay Sardar

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from nilearn import datasets, input_data, plotting
from nilearn.connectome import ConnectivityMeasure
from math import pi

class DementiaRiskEngine:
    def __init__(self):
        # Load a high-resolution probabilistic atlas (MSDL)
        self.atlas = datasets.fetch_atlas_msdl()
        
        # FIX 1: Updated standardization strategy to silence FutureWarnings
        self.masker = input_data.NiftiMapsMasker(
            maps_img=self.atlas.maps,
            standardize='zscore_sample',  # Updated from True to 'zscore_sample'
            memory='nilearn_cache',
            verbose=0
        )
        self.labels = self.atlas.labels
        self.coords = self.atlas.region_coords
        self.time_series = None
        self.adj_matrix = None
        self.G = None
        self.risk_report = {}

    def ingest_mri_data(self, func_file, confounds_file):
        # Extracts BOLD time-series signals from the raw 3D MRI.
        print(f"   [Ingest] Processing volumetric data...")
        self.time_series = self.masker.fit_transform(func_file, confounds=confounds_file)
        return self.time_series.shape

    def build_functional_network(self, threshold=0.5):
        # Constructs the functional connectome.High threshold isolates the 'backbone' of the brain.
        # FIX 2: Explicitly set standardization to silence warnings in ConnectivityMeasure
        correlation_measure = ConnectivityMeasure(
            kind='correlation', 
            standardize='zscore_sample'
        )
        corr_matrix = correlation_measure.fit_transform([self.time_series])[0]
        np.fill_diagonal(corr_matrix, 0)
        
        # Thresholding to create a binary Graph for topological analysis
        self.adj_matrix = np.abs(corr_matrix)
        binary_matrix = (self.adj_matrix > threshold).astype(int)
        
        # Convert to NetworkX Graph
        self.G = nx.from_numpy_array(binary_matrix)
        return self.adj_matrix

    def analyze_network_biomarkers(self):
       # Calculates specific biomarkers known to degrade in Dementia.
        if self.G is None:
            raise ValueError("Graph not built.")
            
        # 1. Global Efficiency: Speed of information transfer (Drops in Dementia)
        glob_eff = nx.global_efficiency(self.G)
        
        # 2. Average Clustering: Local interconnectedness (Drops in Dementia)
        avg_clust = nx.average_clustering(self.G)
        
        # 3. Network Density
        density = nx.density(self.G)
        
        # 4. Hub Vulnerability: Check connections of the Posterior Cingulate Cortex (PCC)
        # FIX 3: Removed 'b' prefix. Labels are Strings, not Bytes.
        pcc_indices = [
            i for i, label in enumerate(self.labels) 
            if 'Cing' in str(label) or 'PCC' in str(label)
        ]
        
        pcc_strength = 0
        if pcc_indices:
            # We take the average degree of all Cingulate/PCC regions found
            degrees = [self.G.degree[n] for n in pcc_indices]
            pcc_strength = sum(degrees) / len(degrees)

        self.risk_report = {
            "Global Efficiency": glob_eff,
            "Local Clustering": avg_clust,
            "Network Density": density,
            "PCC Hub Strength": pcc_strength
        }
        return self.risk_report

    def predict_risk_score(self):
        # AI INFERENCE LAYER: Simulates a pre-trained classifier output based on extracted biomarkers.
        # Clinical Baselines (Simulated thresholds for 'Healthy' brains)
        baseline_eff = 0.45
        baseline_clust = 0.60
        
        metrics = self.risk_report
        risk_score = 0
        
        # Logic-based Risk Calculation
        if metrics['Global Efficiency'] < baseline_eff:
            risk_score += 35  # High penalty for efficiency loss
        
        if metrics['Local Clustering'] < baseline_clust:
            risk_score += 25
            
        if metrics['PCC Hub Strength'] < 3: # If the hub is disconnected
            risk_score += 30
            
        # Normalize to 0-100%
        final_risk = min(max(risk_score + np.random.randint(0, 10), 0), 99)
        self.risk_report['Predicted_Risk_Probability'] = final_risk
        return final_risk

    def generate_clinical_dashboard(self):
        fig = plt.figure(figsize=(18, 8), constrained_layout=True)
        gs = fig.add_gridspec(1, 3)

        # PLOT 1: The Disconnected Brain (3D View)
        ax1 = fig.add_subplot(gs[0, 0])
        display = plotting.plot_connectome(
            self.adj_matrix, self.coords,
            edge_threshold='95%', # Only show strongest links
            node_color='red',     # Nodes are 'hot spots'
            title='Functional Connectivity Backbone',
            display_mode='z',     # Top-down view
            axes=ax1,
            colorbar=True
        )

        # PLOT 2: Radar Chart (Biomarker Profile)
        ax2 = fig.add_subplot(gs[0, 1], polar=True)
        categories = ['Efficiency', 'Clustering', 'Hub Strength', 'Density']
        
        # Prevent division by zero if metrics are 0
        values = [
            self.risk_report['Global Efficiency'] / 0.6,
            self.risk_report['Local Clustering'] / 1.0,
            self.risk_report['PCC Hub Strength'] / 10.0 if self.risk_report['PCC Hub Strength'] > 0 else 0,
            self.risk_report['Network Density'] / 0.5
        ]
        values += values[:1] # Close the loop
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        ax2.plot(angles, values, linewidth=2, linestyle='solid', color='red')
        ax2.fill(angles, values, 'red', alpha=0.1)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_title("Network Integrity Profile", weight='bold', size=12, pad=20)

        # PLOT 3: AI Risk Prediction Output
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        risk = self.risk_report['Predicted_Risk_Probability']
        color = 'green' if risk < 40 else ('orange' if risk < 70 else 'red')
        
        ax3.text(0.5, 0.7, "DEMENTIA RISK ASSESSMENT", ha='center', fontsize=14, weight='bold')
        ax3.text(0.5, 0.5, f"{risk}%", ha='center', fontsize=50, weight='bold', color=color)
        ax3.text(0.5, 0.3, f"Status: {'High Risk' if risk > 70 else 'Monitor'}", ha='center', fontsize=16)

        plt.savefig('Dementia_Risk_Dashboard.png', dpi=300)
        print("   [Output] Clinical Dashboard saved as 'Dementia_Risk_Dashboard.png'")

if __name__ == "__main__":
    print("--- Starting AI Neuro-Diagnostic Pipeline ---")
    
    # 1. Load Data
    data = datasets.fetch_adhd(n_subjects=1)
    
    # 2. Initialize Engine
    engine = DementiaRiskEngine()
    
    # 3. Run Pipeline
    engine.ingest_mri_data(data.func[0], data.confounds[0])
    engine.build_functional_network(threshold=0.5)
    
    # 4. Extract Metrics & Predict
    biomarkers = engine.analyze_network_biomarkers()
    risk = engine.predict_risk_score()
    
    # 5. Visualize
    engine.generate_clinical_dashboard()
    
    print("\n--- BIOMARKER REPORT ---")
    for k, v in biomarkers.items():
        print(f"   {k}: {v}")