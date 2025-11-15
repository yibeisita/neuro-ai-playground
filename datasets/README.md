# Datasets Folder

This folder contains scripts to download and manage sample neuroimaging datasets for the **Neuro AI Playground** tutorials. **Raw datasets are not included in the repository** due to their size. This README explains how to obtain and use the data.

---

## Folder Structure

```

datasets/
├── download_scripts/       
│   ├── get_eeg_sample.py
│   ├── get_meg_sample.py
│   └── get_fmri_sample.py
├── eeg_sample/             # EEG BCI dataset (after download)
├── meg_sample/             # MNE MEG sample dataset (after download)          
└── fmri_sample/            # Sample fMRI NIfTI dataset (after download)

````

## Available Datasets

### 1. EEG BCI Sample
- Source: [Physionet EEG Motor Movement/Imagery Dataset](https://www.physionet.org/content/eegmmidb/1.0.0/)
- Download:
    ```bash
    python datasets/download_scripts/get_eeg_sample.py
    ````

### 2. MNE MEG Sample

* Source: [MNE sample dataset](https://mne.tools/1.6/documentation/datasets.html#sample)
* Download:

    ```bash
    python datasets/download_scripts/get_meg_sample.py
    ```

### 3. fMRI Sample

* Source: [Publicly available NIfTI 4D fMRI file](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_development_fmri.html).
* Download:

    ```bash
    python datasets/download_scripts/get_fmri_sample.py
    ```


## Usage Notes

* After running the download scripts, the data will be stored in the corresponding subfolders (`eeg_sample/`, `meg_sample/`, `fmri_sample/`).
* All tutorials use the **relative paths from the repo root**:

    ```python
    datasets/eeg_sample/
    datasets/meg_sample/
    datasets/fmri_sample/
    ```
