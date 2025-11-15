from pathlib import Path
import nibabel as nib

def load_fmri_sample():
    """
    Load the sample fMRI NIfTI dataset from datasets/fmri_sample/.
    """
    repo_root = Path(__file__).resolve().parents[2]
    fmri_path = repo_root / "datasets" / "fmri_sample" / "sub-01_bold.nii.gz"

    if not fmri_path.exists():
        raise FileNotFoundError(
            f"fMRI sample not found: {fmri_path}. Run download script in datasets/download_scripts/"
        )

    img = nib.load(fmri_path)
    return img
