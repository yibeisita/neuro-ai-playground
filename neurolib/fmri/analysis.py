"""
GLM (General Linear Model) analysis functions for task-based fMRI.

This module provides core analysis functions for:
- Creating design matrices
- Fitting GLM models
- Computing contrasts
- Extracting activation clusters

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.first_level.hemodynamic_models import glover_hrf
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table


def create_simple_paradigm(
    n_scans: int,
    tr: float,
    conditions: List[str],
    n_events_per_condition: int = 10,
    duration: float = 1.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create a simple block-design experimental paradigm.
    
    Parameters
    ----------
    n_scans : int
        Total number of fMRI volumes (time points)
    tr : float
        Repetition time in seconds
    conditions : list of str
        Names of experimental conditions (e.g., ['left', 'right'])
    n_events_per_condition : int, default=10
        Number of events per condition
    duration : float, default=1.0
        Duration of each event in seconds
    random_seed : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    events : pd.DataFrame
        Event dataframe with columns ['onset', 'duration', 'trial_type']
    """
    np.random.seed(random_seed)
    
    total_time = n_scans * tr
    events_list = []
    
    for condition in conditions:
        # Generate random onsets, avoiding the first and last 10 seconds
        onsets = np.random.uniform(10, total_time - 10, n_events_per_condition)
        onsets = np.sort(onsets)
        
        for onset in onsets:
            events_list.append({
                'onset': onset,
                'duration': duration,
                'trial_type': condition
            })
    
    events = pd.DataFrame(events_list).sort_values('onset').reset_index(drop=True)
    
    return events


def create_design_matrix(
    frame_times: np.ndarray,
    events: pd.DataFrame,
    hrf_model: str = 'glover',
    drift_model: str = 'cosine',
    high_pass: float = 1./128,
    add_regs: Optional[pd.DataFrame] = None,
    add_reg_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a design matrix for GLM analysis.
    
    Parameters
    ----------
    frame_times : np.ndarray
        Time of each scan (in seconds)
    events : pd.DataFrame
        Event dataframe with ['onset', 'duration', 'trial_type']
    hrf_model : str, default='glover'
        Hemodynamic response function model
    drift_model : str, default='cosine'
        Drift model ('cosine', 'polynomial', None)
    high_pass : float, default=1./128
        High-pass filter cutoff in Hz
    add_regs : pd.DataFrame, optional
        Additional regressors (e.g., motion parameters)
    add_reg_names : list of str, optional
        Names for additional regressors
        
    Returns
    -------
    design_matrix : pd.DataFrame
        Design matrix with task and confound regressors
    """
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
        add_regs=add_regs,
        add_reg_names=add_reg_names
    )
    
    return design_matrix


def fit_glm(
    fmri_img,
    events: pd.DataFrame,
    tr: Optional[float] = None,
    smoothing_fwhm: float = 5.0,
    noise_model: str = 'ar1',
    high_pass: float = 1./128,
    mask_img=None,
    verbose: int = 1
) -> FirstLevelModel:
    """
    Fit a first-level GLM to fMRI data.
    
    Parameters
    ----------
    fmri_img : Niimg-like object
        4D fMRI data
    events : pd.DataFrame
        Event dataframe with ['onset', 'duration', 'trial_type']
    tr : float, optional
        Repetition time (extracted from image if None)
    smoothing_fwhm : float, default=5.0
        Spatial smoothing kernel FWHM in mm
    noise_model : str, default='ar1'
        Noise model ('ar1', 'ols', 'arN')
    high_pass : float, default=1./128
        High-pass filter cutoff in Hz
    mask_img : Niimg-like, optional
        Brain mask (auto-computed if None)
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    glm : FirstLevelModel
        Fitted GLM model
    """
    # Extract TR from image if not provided
    if tr is None:
        tr = fmri_img.header.get_zooms()[3]
    
    # Initialize GLM
    glm = FirstLevelModel(
        t_r=tr,
        noise_model=noise_model,
        standardize=False,
        hrf_model='glover',
        drift_model='cosine',
        high_pass=high_pass,
        mask_img=mask_img,
        smoothing_fwhm=smoothing_fwhm,
        verbose=verbose
    )
    
    # Fit GLM
    if verbose:
        print("Fitting GLM...")
    glm = glm.fit(fmri_img, events=events)
    
    if verbose:
        print("✓ GLM fitting complete!")
    
    return glm


def compute_contrast(
    glm: FirstLevelModel,
    contrast_spec: str,
    output_type: str = 'z_score'
):
    """
    Compute a contrast from a fitted GLM.
    
    Parameters
    ----------
    glm : FirstLevelModel
        Fitted GLM model
    contrast_spec : str
        Contrast specification (e.g., 'left', 'left - right')
    output_type : str, default='z_score'
        Output type: 'z_score', 't', 'p_value', 'effect_size'
        
    Returns
    -------
    contrast_map : Nifti1Image
        Statistical map
    """
    return glm.compute_contrast(contrast_spec, output_type=output_type)


def compute_multiple_contrasts(
    glm: FirstLevelModel,
    contrasts: Dict[str, str],
    output_type: str = 'z_score',
    verbose: int = 1
) -> Dict:
    """
    Compute multiple contrasts from a fitted GLM.
    
    Parameters
    ----------
    glm : FirstLevelModel
        Fitted GLM model
    contrasts : dict
        Dictionary mapping contrast names to specifications
    output_type : str, default='z_score'
        Output type for all contrasts
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    contrast_maps : dict
        Dictionary of contrast statistical maps
    """
    contrast_maps = {}
    
    for name, spec in contrasts.items():
        if verbose:
            print(f"Computing contrast: {name}")
        contrast_maps[name] = glm.compute_contrast(spec, output_type=output_type)
    
    if verbose:
        print(f"✓ Computed {len(contrast_maps)} contrasts")
    
    return contrast_maps


def fit_glm_and_compute_contrasts(
    fmri_img,
    events: pd.DataFrame,
    contrasts: Dict[str, str],
    tr: Optional[float] = None,
    smoothing_fwhm: float = 5.0,
    noise_model: str = 'ar1',
    high_pass: float = 1./128,
    verbose: int = 1
) -> Tuple[FirstLevelModel, Dict]:
    """
    Complete GLM analysis: fit model and compute contrasts.
    
    This is a convenience function that combines fit_glm and 
    compute_multiple_contrasts.
    
    Parameters
    ----------
    fmri_img : Niimg-like object
        4D fMRI data
    events : pd.DataFrame
        Event dataframe
    contrasts : dict
        Dictionary of contrast specifications
    tr : float, optional
        Repetition time
    smoothing_fwhm : float, default=5.0
        Spatial smoothing FWHM
    noise_model : str, default='ar1'
        Noise model
    high_pass : float, default=1./128
        High-pass filter cutoff
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    glm : FirstLevelModel
        Fitted GLM model
    contrast_maps : dict
        Dictionary of contrast statistical maps
    """
    # Fit GLM
    glm = fit_glm(
        fmri_img, events, tr=tr,
        smoothing_fwhm=smoothing_fwhm,
        noise_model=noise_model,
        high_pass=high_pass,
        verbose=verbose
    )
    
    contrast_maps = compute_multiple_contrasts(glm, contrasts, verbose=verbose)
    
    return glm, contrast_maps


def extract_cluster_table(
    stat_map,
    stat_threshold: float = 3.0,
    cluster_threshold: int = 10,
    two_sided: bool = True
) -> pd.DataFrame:
    """
    Extract cluster table from statistical map.
    
    Parameters
    ----------
    stat_map : Niimg-like object
        Statistical map (z-scores or t-values)
    stat_threshold : float, default=3.0
        Cluster-forming threshold
    cluster_threshold : int, default=10
        Minimum cluster size in voxels
    two_sided : bool, default=True
        Whether to consider both positive and negative activations
        
    Returns
    -------
    table : pd.DataFrame
        Cluster table with peak coordinates and statistics
    """
    try:
        table = get_clusters_table(
            stat_map,
            stat_threshold=stat_threshold,
            cluster_threshold=cluster_threshold,
            two_sided=two_sided
        )
        return table
    except Exception as e:
        print(f"Warning: Could not extract cluster table: {e}")
        return pd.DataFrame()


def threshold_map(
    stat_map,
    alpha: float = 0.001,
    height_control: str = 'fpr',
    cluster_threshold: int = 10
) -> Tuple:
    """
    Apply statistical thresholding to activation map.
    
    Parameters
    ----------
    stat_map : Niimg-like object
        Statistical map (z-scores)
    alpha : float, default=0.001
        Significance level
    height_control : str, default='fpr'
        Height control method ('fpr', 'fdr', 'bonferroni', 'fwe')
    cluster_threshold : int, default=10
        Minimum cluster size in voxels
        
    Returns
    -------
    thresholded_map : Nifti1Image
        Thresholded statistical map
    threshold_value : float
        Applied threshold value
    """
    thresholded_map, threshold_value = threshold_stats_img(
        stat_map,
        alpha=alpha,
        height_control=height_control,
        cluster_threshold=cluster_threshold
    )
    
    return thresholded_map, threshold_value


def compute_effect_size_map(
    glm: FirstLevelModel,
    contrast: str
):
    """
    Compute effect size (percent signal change) for a contrast.
    
    Parameters
    ----------
    glm : FirstLevelModel
        Fitted GLM model
    contrast : str
        Contrast specification
        
    Returns
    -------
    effect_size_map : Nifti1Image
        Effect size map
        
    """
    effect_map = glm.compute_contrast(contrast, output_type='effect_size')
    return effect_map


def quick_glm_analysis(
    fmri_img,
    events: pd.DataFrame,
    contrasts: Dict[str, str],
    tr: Optional[float] = None,
    threshold: float = 3.0,
    smoothing_fwhm: float = 5.0,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run complete GLM analysis pipeline.
    
    This convenience function performs:
    1. GLM fitting
    2. Contrast computation
    3. Cluster extraction
    4. Optional output saving
    
    Parameters
    ----------
    fmri_img : Niimg-like object
        4D fMRI data
    events : pd.DataFrame
        Event dataframe
    contrasts : dict
        Dictionary of contrasts
    tr : float, optional
        Repetition time
    threshold : float, default=3.0
        Statistical threshold for cluster extraction
    smoothing_fwhm : float, default=5.0
        Spatial smoothing FWHM
    output_dir : Path, optional
        Directory to save outputs
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'glm': Fitted GLM model
        - 'contrast_maps': Dictionary of statistical maps
        - 'cluster_tables': Dictionary of cluster tables
        - 'design_matrix': Design matrix
    """
    results = {}
    
    # Fit GLM and compute contrasts
    glm, contrast_maps = fit_glm_and_compute_contrasts(
        fmri_img, events, contrasts,
        tr=tr, smoothing_fwhm=smoothing_fwhm
    )
    
    results['glm'] = glm
    results['contrast_maps'] = contrast_maps
    results['design_matrix'] = glm.design_matrices_[0]
    
    # Extract cluster tables
    results['cluster_tables'] = {}
    for name, stat_map in contrast_maps.items():
        table = extract_cluster_table(stat_map, stat_threshold=threshold)
        results['cluster_tables'][name] = table
    
    # Save outputs if directory specified
    if output_dir is not None:
        from neurolib.fmri.io import save_nifti
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save contrast maps
        for name, stat_map in contrast_maps.items():
            save_nifti(
                stat_map,
                output_dir / f'contrast_{name}.nii.gz',
                description=f'Contrast: {name}'
            )
        
        # Save cluster tables
        for name, table in results['cluster_tables'].items():
            if len(table) > 0:
                table.to_csv(output_dir / f'clusters_{name}.csv', index=False)
        
        # Save design matrix
        results['design_matrix'].to_csv(
            output_dir / 'design_matrix.csv',
            index=False
        )
        
        print(f"✓ Results saved to {output_dir}")
    
    return results