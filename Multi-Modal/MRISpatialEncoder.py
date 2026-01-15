import networkx as nx
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, input_data

class MRISpatialEncoder:
    def __init__(self):
        self.atlas = datasets.fetch_atlas_msdl()
        self.masker = input_data.NiftiMapsMasker(
            maps_img=self.atlas.maps,
            standardize='zscore_sample',
            memory='nilearn_cache',
            verbose=0
        )
        self.adj_matrix = None
        self.G = None

    def process_volume(self, func_file, confounds_file):
        print(f"[MRI] Extracting signals from {func_file[-20:]}...")
        time_series = self.masker.fit_transform(func_file, confounds=confounds_file)
        
        # Correlation -> Graph
        measure = ConnectivityMeasure(kind='correlation', standardize='zscore_sample')
        corr_matrix = measure.fit_transform([time_series])[0]
        np.fill_diagonal(corr_matrix, 0)
        
        # Threshold for strong connections
        self.adj_matrix = np.abs(corr_matrix)
        binary_matrix = (self.adj_matrix > 0.5).astype(int)
        self.G = nx.from_numpy_array(binary_matrix)
        
        return self.calculate_spatial_risk()

    def calculate_spatial_risk(self):
        if self.G is None: return 0
        
        # Metric: Global Efficiency
        eff = nx.global_efficiency(self.G)
        print(f"[MRI] Measured Global Efficiency: {eff:.4f}")
        
        # Transform to risk (Baseline healthy efficiency approx 0.5)
        risk_score = max(0, (0.5 - eff) * 2) 
        return min(risk_score, 1.0)