# NeuroPrognosis-AI: Multi-Modal Dementia Risk Assessment

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Neuroimaging](https://img.shields.io/badge/Neuroimaging-fMRI%20%2B%20EEG-FF6F00?style=for-the-badge&logo=brain&logoColor=white)
![Graph Theory](https://img.shields.io/badge/Graph_Theory-NetworkX-28A745?style=for-the-badge&logo=graph&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research_Grade-blue?style=for-the-badge)

> **A clinical decision-support framework utilizing Graph Theoretical Analysis (fMRI) and Spectral Power Density (EEG) to quantify network degradation associated with early-stage Dementia.**

---

## Abstract

Dementia, particularly Alzheimer's Disease (AD), is often characterized as a **"Disconnection Syndrome"** the breakdown of functional pathways between brain regions before significant structural atrophy occurs. 

**NeuroPrognosis-AI** is a dual-stream diagnostic pipeline designed to detect these subtle failures. Unlike traditional "Black Box" Deep Learning, this framework uses **Explainable AI (XAI)** to extract interpretable biomarkers:
1.  **Spatial Topology (MRI):** Identifying the loss of "hub" connectivity in the Default Mode Network (DMN).
2.  **Temporal Dynamics (EEG):** Detecting "spectral slowing" (Theta/Alpha ratio imbalances).

This repository contains two methodologies: a **Unimodal Graph-Based Assessment** and a **Multi-Modal Late Fusion Architecture**, both validated on real biological datasets (**ADHD-200** and **MNE-Sample**).

---

## Methodology 1: Unimodal MRI Network Analysis

This module focuses on the **Functional Connectome**. By converting 4D fMRI BOLD signals into a graph structure, we calculate topological metrics that serve as proxies for cognitive health.

### Key Biomarkers Extracted:
* **Global Efficiency ($E_{glob}$):** Measures the speed of information transfer across the brain. A drop in $E_{glob}$ is a primary indicator of AD.
* **PCC Hub Vulnerability:** Specifically monitors the **Posterior Cingulate Cortex (PCC)**, the metabolic core of the brain, which is often the first region to disconnect in Dementia.
* **Small-Worldness:** Assesses the balance between local clustering and global integration.

<p align="center">
  <img src="https://github.com/wasay530/NeuroPrognosis-AI-Multi-Modal-Dementia-Risk-Assessment/blob/cc59d8ea30ea5f1a4dddf6d22ff37422e110a74d/Results/Dementia_Risk_Dashboard.png" alt="Unimodal Dashboard" width="90%">
  <br>
  <i>Figure 1: The Clinical Dashboard showing the Connectome Backbone (left), Biomarker Radar Chart (center), and AI-Predicted Risk Score (right).</i>
</p>

---

## Methodology 2: Multi-Modal Fusion (MRI + EEG)

To overcome the low temporal resolution of fMRI, this advanced module integrates **Electroencephalography (EEG)** signals. This **"Spectral-Spatial Fusion"** creates a holistic view of neurodegeneration.

### The Dual-Stream Architecture:
1.  **Spatial Encoder (MRI):** Uses the **MSDL Probabilistic Atlas** to map functional connectivity and compute graph density.
2.  **Temporal Encoder (EEG):** Uses **Welch’s Power Spectral Density (PSD)** to extract oscillatory features.
3.  **Fusion Layer:** A weighted logic engine combines the *Spatial Risk* (atrophy) and *Temporal Risk* (slowing) into a final prognostic score.

### Data Sources (Real Clinical Data):
* **MRI Stream:** Fetches fMRI scans from the **ADHD-200 Consortium** (via `nilearn`).
* **EEG Stream:** Fetches MEG/EEG sensor data from the **MNE-Sample Dataset**.

<p align="center">
  <img src="https://github.com/wasay530/NeuroPrognosis-AI-Multi-Modal-Dementia-Risk-Assessment/blob/cc59d8ea30ea5f1a4dddf6d22ff37422e110a74d/Results/Data_Analysis.png" alt="Multi-Modal Analysis" width="90%">
  <br>
  <i>Figure 2: Multi-Modal Output. The Left panel displays the MRI Graph Topology. The Center panel visualizes the EEG Power Spectrum (Alpha vs. Theta bands). The Right panel displays the Fused Risk Score.</i>
</p>

### Installation & Usage
1. Prerequisites
The environment requires standard neuroimaging libraries.

``` pip install numpy pandas matplotlib networkx nilearn mne scipy```

### 2. Running the Unimodal Analysis

``` Generates the specific Hub Vulnerability report. ```

### 3. Running the Multi-Modal Fusion
Executes the full pipeline, downloading real samples from both modalities.
``` python Multi-Modal/main.py ```

---

## Theoretical Background

This project is built upon the "Two-Factor Failure" hypothesis of neurodegeneration:

**1. The Hub Overload Theory:** High-traffic nodes (like the PCC) have high metabolic costs. In Alzheimer's, amyloid plaques accumulate here first, severing connections. We detect this via **Degree Centrality**.

**2. Oscillatory Slowing:** As synaptic efficacy fades, the brain's dominant rhythm shifts from the alert **Alpha band (8-12Hz)** to the drowsy **Theta band (4-8Hz)**. We quantify this via the **Theta/Alpha Ratio (TAR)**.

