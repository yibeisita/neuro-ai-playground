"""
Download the MNE MEG sample dataset into the project datasets folder.


The MNE sample dataset is a multi-modal, full MEG/EEG recording from a single participant performing a simple audio-visual task. It’s used extensively in tutorials and testing.
- Source: MNE Python provides this dataset as a small, publicly available MEG/EEG dataset for tutorial and testing purposes.
- Acquisition system: Elekta Neuromag MEG system.
- Subject: One healthy adult participant.
- Stimuli: Auditory and visual events. 
  - Task: Passive perception (participant fixates, does not respond).
  - Purpose: Evoked response analysis (auditory ERFs, visual ERFs), topographic mapping, source localization, connectivity, etc.
"""

import os
from pathlib import Path
from mne.datasets import sample

def download_meg_sample():
    base_dir = Path(__file__).resolve().parents[1]
    target_dir = base_dir / "meg_sample"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MNE MEG sample dataset to:\n{target_dir}")
    sample_path = sample.data_path(path=str(target_dir), download=True)
    print(f"MEG sample data available at:\n{sample_path}")

    return sample_path

if __name__ == "__main__":
    download_meg_sample()
