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

def load_motor_task_data(download_if_missing=True, verbose=1):
    """
    Load motor task fMRI dataset from Nilearn for GLM analysis.
    
    This dataset contains:
    - Functional MRI data from a motor task (button presses)
    - Events file with timing of left/right/both hand movements
    
    Parameters
    ----------
    download_if_missing : bool, default=True
        Download dataset if not cached locally
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    fmri_img : Nifti1Image
        4D functional MRI image
    events : pd.DataFrame
        Event dataframe with columns ['onset', 'duration', 'trial_type']
    tr : float
        Repetition time in seconds
    """
    # Download motor task dataset from Nilearn
    if verbose:
        print("Loading motor task dataset from Nilearn...")
    
    motor_data = datasets.fetch_localizer_button_task()
    
    # Load functional image
    fmri_img = nib.load(motor_data.epi)
    
    # Load events
    events = pd.read_csv(motor_data.events, sep='\t')
    
    # Extract TR from NIfTI header
    tr = fmri_img.header.get_zooms()[3]
    
    if verbose:
        print(f"✓ Dataset loaded successfully")
        print(f"  Image shape: {fmri_img.shape}")
        print(f"  TR: {tr}s")
        print(f"  Number of events: {len(events)}")
        print(f"  Event types: {events['trial_type'].unique()}")
    
    return fmri_img, events, tr


def load_localizer_dataset(task='language', download_if_missing=True):
    """
    Load various task-based fMRI datasets from Nilearn's localizer collection.
    
    Parameters
    ----------
    task : str, default='language'
        Task type: 'language', 'audio', 'visual', 'motor', 'spatial'
    download_if_missing : bool, default=True
        Download if not cached
        
    Returns
    -------
    fmri_img : Nifti1Image
        4D functional MRI image
    """
    if task == 'motor':
        data = datasets.fetch_localizer_button_task()
        return nib.load(data.epi)
    else:
        # Fetch general localizer dataset (includes various contrasts)
        data = datasets.fetch_localizer_contrasts(
            ["language"] if task == 'language' else [task],
            n_subjects=1,
            get_anats=False
        )
        return nib.load(data['cmaps'][0])


def load_fmri_with_confounds(subject_id=1, task='rest', download_if_missing=True):
    """
    Load fMRI data along with motion confounds for preprocessing.
    
    Parameters
    ----------
    subject_id : int, default=1
        Subject identifier
    task : str, default='rest'
        Task type ('rest' for resting-state)
    download_if_missing : bool, default=True
        Download if not cached
        
    Returns
    -------
    fmri_img : Nifti1Image
        4D functional MRI image
    confounds : pd.DataFrame
        Confound regressors (motion parameters, etc.)
    """
    # Fetch development dataset with confounds
    data = datasets.fetch_development_fmri(
        n_subjects=subject_id,
        reduce_confounds=True
    )
    
    fmri_img = nib.load(data.func[0])
    confounds = pd.read_csv(data.confounds[0], sep='\t')
    
    return fmri_img, confounds


def save_nifti(img, output_path, description=None):
    """
    Save a NIfTI image to disk with optional description.
    
    Parameters
    ----------
    img : Nifti1Image
        Image to save
    output_path : str or Path
        Output file path
    description : str, optional
        Description to add to NIfTI header
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if description:
        img.header['descrip'] = description.encode()[:79]  # Max 80 chars
    
    nib.save(img, output_path)
    print(f"✓ Saved: {output_path}")


def load_events_file(events_path, format='tsv'):
    """
    Load an events file (BIDS format or custom).
    
    Parameters
    ----------
    events_path : str or Path
        Path to events file
    format : str, default='tsv'
        File format ('tsv', 'csv', 'txt')
        
    Returns
    -------
    events : pd.DataFrame
        Events dataframe with columns ['onset', 'duration', 'trial_type']
        
    Examples
    --------
    >>> events = load_events_file('data/sub-01_task-motor_events.tsv')
    """
    events_path = Path(events_path)
    
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")
    
    # Determine separator
    sep = '\t' if format == 'tsv' else ','
    
    events = pd.read_csv(events_path, sep=sep)
    
    # Validate required columns
    required_cols = ['onset', 'duration', 'trial_type']
    missing_cols = [col for col in required_cols if col not in events.columns]
    
    if missing_cols:
        raise ValueError(
            f"Events file missing required columns: {missing_cols}\n"
            f"Found columns: {events.columns.tolist()}"
        )
    
    return events
