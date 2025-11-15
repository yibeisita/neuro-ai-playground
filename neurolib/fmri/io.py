from pathlib import Path
import nibabel as nib
import json
import os

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



def load_fmri_metadata():
    """
    Load metadata associated with a 3D or 4D fMRI NIfTI file.
    
    This function looks for:
        - sub-01_bold.nii.gz  (the fMRI image)
        - sub-01_bold.json    (optional metadata sidecar)
    """
    repo_root = Path(__file__).resolve().parents[2]
    fmri_path = repo_root / "datasets" / "fmri_sample" / "sub-01_bold.nii.gz"

    if not fmri_path.exists():
        raise FileNotFoundError(
            f"fMRI sample not found: {fmri_path}. "
            "Run download script in datasets/download_scripts/"
        )

    # Load NIfTI header metadata
    img = nib.load(str(fmri_path))
    header = img.header

    metadata = {
        "dim": header.get_data_shape(),
        "voxel_size": header.get_zooms(),
        "datatype": str(header.get_data_dtype()),
    }

    # Extract TR if 4D
    if len(header.get_zooms()) > 3:
        metadata["TR"] = header.get_zooms()[3]
    else:
        metadata["TR"] = None

    # Look for JSON sidecar next to the NIfTI
    json_path = fmri_path.with_suffix("").with_suffix(".json")

    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                json_meta = json.load(f)
            metadata.update(json_meta)
        except Exception as e:
            print(f"Warning: Failed to read JSON metadata: {e}")

    return metadata
