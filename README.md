# Neuro-AI Playground (｡´ ‿｀♡)

A hands-on learning lab for exploring **EEG**, **MEG**, and **fMRI** data using **Python**, **MNE**, **Nilearn**, and **Machine Learning**.

This repository is designed for learners with a **Computer Science / AI** background who want to understand **neuroimaging data analysis** — from basic visualization to deep learning and multimodal integration.

## What You'll Learn

- How to **load and visualize** brain data (EEG, MEG, fMRI)
- How to **preprocess** and clean neuroimaging signals
- How to **analyze** brain activity (ERPs, GLMs, connectivity)
- How to **apply machine learning & deep learning** to brain data
- How to build **reproducible and open-source pipelines**

Each tutorial is standalone and focused on a single concept — perfect for self-paced learning or building a neuroimaging portfolio.

## Repository Structure

```
neuro-ai-playground/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── neurolib/             # Shared utilities library
│   ├── eeg_meg/          # EEG/MEG preprocessing, analysis, viz
│   ├── fmri/             # fMRI preprocessing, analysis, viz
│   └── ml/               # Machine learning utilities
│
├── tutorials/            # Learning notebooks organized by skill
│   ├── 01_foundations/
│   ├── 02_preprocessing/
│   ├── 03_time_domain/
│   ├── 04_frequency_domain/
│   ├── 05_spatial_analysis/
│   ├── 06_connectivity/
│   ├── 07_machine_learning/
│   ├── 08_deep_learning/
│   └── 09_multimodal/
│
└── datasets/             # Data download scripts
```

## Learning Roadmap

### Level 1: Foundations
Learn to load and visualize neuroimaging data

| Tutorial         | Focus                                                | Modality | Key Skills                                              |
| ---------------- | ---------------------------------------------------- | -------- | ------------------------------------------------------- |
| 01.1 EEG Basics  | Load and plot EEG signals, compute PSD               | EEG      | File I/O, time series plotting, PSD analysis            |
| 01.2 MEG Basics  | Visualize raw MEG and evoked responses, topographies | MEG      | Sensor layouts, evoked vs raw, topographic maps         |
| 01.3 fMRI Basics | Display mean images, glass brain, deviation maps     | fMRI     | NIfTI handling, anatomical overlay, temporal deviations |
|

**Status**: ദ്ദി(ᵔᗜᵔ) Complete

### Level 2: Preprocessing
Clean and prepare data for analysis

| Tutorial | Focus | Modality | Key Skills |
|----------|-------|----------|------------|
| 02.1 EEG/MEG Cleaning | Filtering, ICA, artifact removal | EEG/MEG | Signal processing, ICA decomposition |
| 02.2 fMRI Pipeline | Motion correction, slice timing | fMRI | BIDS, preprocessing workflows |
|

**Status**: ⏳ Planned

### Level 3: Time Domain Analysis
Analyze brain activity over time

| Tutorial | Focus | Modality | Key Skills |
|----------|-------|----------|------------|
| 03.1 ERP Analysis | Event-related potentials | EEG/MEG | Epoching, averaging, statistics |
| 03.2 GLM Activation | Task-based fMRI analysis | fMRI | General linear model, design matrices |
|

**Status**: ⏳ Planned


### Level 4: Frequency Domain
Understand brain oscillations

| Tutorial | Focus | Modality | Key Skills |
|----------|-------|----------|------------|
| 04.1 Time-Frequency | Spectrograms, wavelets | EEG/MEG | Wavelet transforms, multitaper methods |
| 04.2 Oscillatory Power | Band-specific analysis | EEG/MEG | Alpha, beta, gamma rhythms |
|

**Status**: ⏳ Planned


### Level 5: Spatial Analysis
Localize brain activity

| Tutorial | Focus | Modality | Key Skills |
|----------|-------|----------|------------|
| 05.1 Source Localization | Inverse solutions | MEG/EEG | Forward/inverse modeling, dipoles |
| 05.2 ROI Analysis | Region-of-interest extraction | fMRI | Atlases, parcellation, timecourses |
|

**Status**: ⏳ Planned


### Level 6: Connectivity
Map brain networks

| Tutorial | Focus | Modality | Key Skills |
|----------|-------|----------|------------|
| 06.1 EEG/MEG Connectivity | Phase locking, coherence | EEG/MEG | Functional connectivity metrics |
| 06.2 fMRI Networks | Resting-state networks | fMRI | Correlation matrices, graph theory |
|

**Status**: ⏳ Planned


### Level 7: Machine Learning
Decode brain states

| Tutorial | Focus | Modality | Key Skills |
|----------|-------|----------|------------|
| 07.1 EEG Decoding | Classification basics | EEG | Scikit-learn, cross-validation |
| 07.2 fMRI MVPA | Multi-voxel pattern analysis | fMRI | Searchlight, classification |
| 07.3 Cross-Validation | Proper validation strategies | All | Nested CV, permutation testing |
|

**Status**: ⏳ Planned


### Level 8: Deep Learning
End-to-end neural networks

| Tutorial | Focus | Modality | Key Skills |
|----------|-------|----------|------------|
| 08.1 EEG CNN | Convolutional neural nets | EEG | PyTorch, CNN architectures |
| 08.2 EEG RNN | Recurrent neural nets | EEG | LSTM, temporal modeling |
| 08.3 fMRI 3D CNN | Volumetric deep learning | fMRI | 3D convolutions, attention |
|

**Status**: ⏳ Planned



### Level 9: Multimodal Integration
Combine different brain imaging modalities

| Tutorial | Focus | Modality | Key Skills |
|----------|-------|----------|------------|
| 09.1 EEG-fMRI Fusion | Joint analysis | EEG + fMRI | Feature fusion, co-registration |
| 09.2 MEG-fMRI Fusion | Temporal + spatial integration | MEG + fMRI | Source space fusion |
|

**Status**: ⏳ Planned


## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/neuro-ai-playground.git
cd neuro-ai-playground
```

### 2. Create the conda environment
```bash
conda env create -f environment.yml
conda activate neuro-ai
```

### 3. Install the neurolib package
```bash
pip install -e .
```

### 4. Launch Jupyter
```bash
jupyter notebook
```

Then navigate to `tutorials/` and open any notebook to get started!


## Core Tools & Libraries

| Purpose | Libraries |
|---------|-----------|
| EEG/MEG Analysis | [MNE-Python](https://mne.tools/stable/index.html) |
| fMRI Analysis | [Nilearn](https://nilearn.github.io/stable/index.html) |
| Machine Learning | [scikit-learn](https://scikit-learn.org/) |
| Deep Learning | [PyTorch](https://pytorch.org/) |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Data Formats | `nibabel`, `pandas`, `numpy` |



## Example Datasets

All datasets can be downloaded using scripts in `datasets/download_scripts/`

| Dataset | Modality | Description | Size |
|---------|----------|-------------|------|
| MNE Sample | MEG/EEG | Auditory/Visual task | ~1.5 GB |
| OpenNeuro ds000117 | fMRI | Famous faces task | ~50 GB |
| OpenNeuro ds003775 | EEG | Motor imagery | ~5 GB |
| HCP Young Adult | fMRI/MEG | Resting state + tasks | Request access |
|

**To download the MNE sample dataset:**
```bash
python datasets/download_scripts/get_mne_sample.py
```

## Quick Start Guide

### 1. Start with EEG Basics
```bash
jupyter notebook tutorials/01_foundations/01_eeg_basics.ipynb
```

### 2. Try the shared library
```python
from neurolib.eeg_meg import preprocess_raw, plot_psd
import mne

# Load data
raw = mne.io.read_raw_fif('sample_data.fif', preload=True)

# Preprocess using neurolib
raw_clean = preprocess_raw(raw, filter_params={'l_freq': 1, 'h_freq': 40})

# Visualize
plot_psd(raw_clean)
```

### 3. Follow the learning path
Progress through tutorials from Level 1 → 9 based on your goals


## Contributing

Contributions are welcome! You can:

- Add new tutorials
- Improve existing notebooks
- Extend the `neurolib` package
- Add documentation or examples
- Report bugs or suggest features

Please open a Pull Request or Issue on GitHub.


## Resources & References

### Official Documentation
- [MNE-Python Tutorials](https://mne.tools/stable/auto_tutorials/index.html)
- [Nilearn Example Gallery](https://nilearn.github.io/stable/auto_examples/index.html)


## Citation

If you use this repository in your research or education, please cite:

```bibtex
@software{neuro_ai_playground,
  author = {Wang Chen, Yibei},
  title = {Neuro-AI Playground: A Learning Lab for Neuroimaging Analysis},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/neuro-ai-playground}
}
```


## License

This repository is licensed under the **MIT License**.  
You are free to use, modify, and distribute for educational and research purposes.


**Created with ♡ by Yibei Wang Chen**