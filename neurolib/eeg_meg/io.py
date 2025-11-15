from pathlib import Path
import mne


def load_eeg_raw(subject=1, run=1, preload=True):
    """
    Load EEGBCI sample dataset from datasets/eeg_sample.
    """
    repo_root = Path(__file__).resolve().parents[2]
    eeg_folder = repo_root / "datasets" / "eeg_sample" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"

    subject_str = f"S{subject:03d}" 
    run_str = f"R{run:02d}"

    raw_files = list(eeg_folder.glob(f"{subject_str}/{subject_str}{run_str}.edf"))

    if not raw_files:
        raise FileNotFoundError(
            f"EEG data for subject {subject}, run {run} not found in {eeg_folder}.\n"
            "Run get_mne_eeg_sample.py to download it."
        )

    raw = mne.io.read_raw(raw_files[0], preload=preload)
    return raw



def load_meg_sample():
    """
    Load MEG sample dataset from datasets/meg_sample folder.
    """
    repo_root = Path(__file__).resolve().parents[2]
    meg_folder = repo_root / "datasets" / "meg_sample" / "MNE-sample-data" / "MEG" / "sample"

    raw_file = meg_folder / "sample_audvis_raw.fif"
    if not raw_file.exists():
        raise FileNotFoundError(f"{raw_file} not found. Run get_meg_sample.py to download the dataset.")

    raw = mne.io.read_raw_fif(raw_file, preload=True)
    return raw
