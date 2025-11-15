"""
Download a small EEG sample dataset from MNE.
Saves the file into:  neuro-ai-playground/datasets/eeg_sample/

MNE-Python’s EEGBCI (EEG Brain-Computer Interface) dataset, part of the PhysioNet EEG Motor Movement/Imagery dataset.
- Designed for brain-computer interface (BCI) research.
- Includes EEG recordings of motor movements and motor imagery tasks.
- seful for testing EEG preprocessing, feature extraction, frequency analysis, and machine learning decoding.

Tasks/Runs: 
------------
Each “run” in the dataset corresponds to a specific task:
- Run 1–2: Baseline (eyes open / eyes closed)
- Run 3–6: Actual movement or motor imagery:
  - Opening and closing the left fist
  - Opening and closing the right fist
  - Both fists
  - Both feet

Data format: 
------------
EDF files (European Data Format). Each file contains:
- Continuous EEG signals from 64 electrodes (standard 10-20 layout is optional)
- Sampling frequency: usually 160 Hz
- Channel types: EEG, EOG (eye), EMG (if recorded)
- Metadata: recording time, channel names, measurement info

[https://physionet.org/content/eegmmidb/1.0.0/]

"""

import os
from pathlib import Path
from mne.datasets import eegbci


def download_eeg_sample(subject=1, runs=[1]):
    base_dir = Path(__file__).resolve().parents[1]
    target_dir = base_dir / "eeg_sample"         

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading EEGBCI EEG sample data to:\n{target_dir}")

    files = eegbci.load_data(subject, runs, path=str(target_dir))

    print("\nDownloaded files:")
    for f in files:
        print(" -", f)


if __name__ == "__main__":
    download_eeg_sample(1, [1]) # Downloading Data from Subject 1, Run 1
