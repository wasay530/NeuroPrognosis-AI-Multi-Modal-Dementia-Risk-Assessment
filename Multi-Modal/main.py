from nilearn import datasets
from MultiModalFusionEngine import MultiModalFusionEngine

print("--- Initializing Data Dual-Stream Engine ---")
    
# 1. Fetch MRI (ADHD-200)
data = datasets.fetch_adhd(n_subjects=1)
    
# 2. Run Engine (Fetches EEG internally)
engine = MultiModalFusionEngine()
engine.run_diagnosis(data.func[0], data.confounds[0])
    
# 3. Generate Report
engine.generate_multimodal_dashboard()