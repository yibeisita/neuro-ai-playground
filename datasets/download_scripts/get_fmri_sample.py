"""
Download a small sample fMRI dataset and save it in datasets/fmri_sample/.

This dataset is provided by Nilearn as part of their development datasets. 
It contains preprocessed functional MRI (fMRI) data from a single subject. 

Key features:
- 4D NIfTI image: 3 spatial dimensions (X, Y, Z) + time.
- Each voxel represents BOLD (blood-oxygen-level-dependent) signal intensity.
- The data can be used to learn basic fMRI concepts, including:
    - Loading and visualizing 4D fMRI data
    - Computing mean functional images
    - Exploring statistical maps
- Small and lightweight (~6 MB), suitable for tutorials and testing.
- Compatible with MNI152 space.
"""

from pathlib import Path
from nilearn import datasets
import shutil

def download_fmri_sample():
    repo_root = Path(__file__).resolve().parents[2]
    target_dir = repo_root / "datasets" / "fmri_sample"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading fMRI sample dataset to: {target_dir}")
    fmri_data = datasets.fetch_development_fmri(n_subjects=1, data_dir=target_dir, verbose=1)
    
    # The dataset returns paths to functional and confound files
    # We copy the functional NIfTI file to target_dir
    func_file = fmri_data.func[0]
    dest_file = target_dir / "sub-01_bold.nii.gz"
    
    shutil.copy(func_file, dest_file)
    
    print(f"fMRI sample saved to: {dest_file}")
    print("You can now load it in neurolib.fmri.io.load_fmri_sample()")

if __name__ == "__main__":
    download_fmri_sample()
