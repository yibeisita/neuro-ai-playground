# Neuro-AI Playground (｡´ ‿｀♡)
A hands-on learning lab for exploring **EEG**, **MEG**, and **fMRI** data using **Python**, **MNE**, **Nilearn**, and **Machine Learning**.

This repository is designed for learners with a **Computer Science / AI** background who want to understand **neuroimaging data analysis** — from basic visualization to deep learning and multimodal integration.


## What You’ll Learn

- How to **load and visualize** brain data (EEG, MEG, fMRI)
- How to **preprocess** and clean neuroimaging signals
- How to **analyze** brain activity (ERPs, GLMs, connectivity)
- How to **apply machine learning & deep learning** to brain data
- How to build **reproducible and open-source pipelines**

Each mini-project is standalone and focused on a single concept — perfect for self-paced learning or building a neuroimaging portfolio.

## Folder Structure

````
neuro-ai-playground/
│
├── README.md
├── environment.yml
├── LICENSE
├── docs/
│   ├── learning_path.md
│   ├── references.md
│   └── glossary.md
│
└── projects/
    ├── 01_eeg_meg_analysis/
    ├── 02_fmri_visualize/
    ├── 03_eeg_preprocessing/
    ├── 04_fmri_preprocessing/
    ├── 05_eeg_erp_analysis/
    ├── 06_fmri_glm_analysis/
    ├── 07_signal_decoding_ml/
    ├── 08_time_frequency_analysis/
    ├── 09_source_localization_meg/
    ├── 10_connectivity_fmri/
    ├── 11_deep_learning_eeg/
    └── 12_multimodal_integration/
````


## Project Roadmap

| # | Project | Focus | Skills Learned | Tools | Status |
|---|----------|--------|----------------|-------|--------|
| 01 | EEG Read & Visualize | Load + visualize EEG signals | File I/O, PSD, plotting, preprocessing | `MNE` | ദ്ദി(ᵔᗜᵔ) DONE |
| 02 | fMRI Visualization | Display brain volumes | NIfTI handling, slicing | `nilearn`, `nibabel` | ⏳ Planned |
| 03 | EEG Preprocessing | Filtering, ICA | Signal cleaning | `MNE` | ⏳ Planned |
| 04 | fMRI Preprocessing | Motion correction, slice timing | BIDS, `fmriprep` | `nilearn`, `nipype` | ⏳ Planned |
| 05 | EEG ERP Analysis | Event-related potentials | Epoching, averaging | `MNE` | ⏳ Planned |
| 06 | fMRI GLM Analysis | Activation maps | GLM, design matrices | `nilearn.glm` | ⏳ Planned |
| 07 | EEG Decoding | Classification | ML, cross-validation | `scikit-learn` | ⏳ Planned |
| 08 | Time-Frequency Analysis | Brain oscillations | Wavelets, multitaper | `MNE` | ⏳ Planned |
| 09 | Source Localization | MEG inverse solutions | Forward/inverse modeling | `MNE` | ⏳ Planned |
| 10 | Functional Connectivity | Brain networks | Correlation, graph metrics | `nilearn`, `networkx` | ⏳ Planned |
| 11 | Deep Learning on EEG | End-to-end modeling | CNN/RNN, PyTorch | `PyTorch`, `MNE` | ⏳ Planned |
| 12 | Multimodal Integration | EEG + fMRI | Feature fusion | `MNE`, `nilearn` | ⏳ Planned |


## Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/neuro-ai-playground.git
cd neuro-ai-playground
````

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate neuro-ai
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

Then open any project notebook under `projects/<project_name>/notebooks/`.


## Tools & Libraries

| Purpose          | Libraries                                              |
| ---------------- | ------------------------------------------------------ |
| EEG/MEG Analysis | [MNE-Python](https://mne.tools/stable/index.html)      |
| fMRI Analysis    | [Nilearn](https://nilearn.github.io/stable/index.html) |
| Machine Learning | [scikit-learn](https://scikit-learn.org/)              |
| Deep Learning    | [PyTorch](https://pytorch.org/)                        |
| Visualization    | `matplotlib`, `seaborn`                                |
| Data Formats     | `nibabel`, `pandas`, `numpy`                           |


## Example Datasets

| Dataset            | Modality | Description                 | Source                                                                                         |
| ------------------ | -------- | --------------------------- | ---------------------------------------------------------------------------------------------- |
| MNE Sample         | MEG/EEG  | Auditory/Visual task sample | [`mne.datasets.sample`](https://mne.tools/stable/generated/mne.datasets.sample.data_path.html) |
| OpenNeuro Datasets | EEG/fMRI | Open-access BIDS datasets   | [openneuro.org](https://openneuro.org)                                                         |
| HCP                | fMRI/MEG | Human Connectome Project    | [humanconnectome.org](https://www.humanconnectome.org/)                                        |


## Learning Path

**Level 1 – Foundations**

> 01. EEG Read & Plot
> 02. fMRI Visualization
> 03. EEG / fMRI Preprocessing

**Level 2 – Analysis**

> 05. ERP / GLM
> 07. Decoding & Time-Frequency

**Level 3 – Advanced**

> 09. Source Localization
> 10. Connectivity
> 11. Deep Learning
> 12. Multimodal Integration


## How to Contribute

Contributions are welcome!
You can:

* Add new projects
* Improve notebooks or visualizations
* Add documentation under `/docs`

Please open a Pull Request or Issue on GitHub with your suggestions.

## References

* [MNE-Python Tutorials](https://mne.tools/stable/auto_tutorials/index.html)
* [Nilearn Example Gallery](https://nilearn.github.io/stable/auto_examples/index.html)
* [BIDS Standard](https://bids.neuroimaging.io/)
* [Human Connectome Project](https://www.humanconnectome.org/)

## License

This repository is licensed under the **MIT License**.
You are free to use, modify, and distribute for educational and research purposes.





Created by Yibei Wang Chen
