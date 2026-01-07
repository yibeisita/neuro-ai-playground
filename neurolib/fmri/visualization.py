"""
Visualization functions for fMRI data analysis.

This module provides plotting functions for:
- Basic fMRI visualization (EPI, mean images, glass brains)
- GLM-specific visualization (HRF, activation maps, design matrices)
- Statistical maps and anatomical overlays
"""

import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn import plotting, datasets
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.glm.first_level.hemodynamic_models import glover_hrf
import numpy as np
from typing import Dict, Optional, Tuple, Union

MNI_TEMPLATE = datasets.load_mni152_template()


def plot_fmri_epi(img, title="fMRI Image", display_mode="ortho", cut_coords=None, cmap="gray"):
    """
    Wrapper for nilearn.plotting.plot_epi to visualize 3D/4D fMRI images.
    """
    plotting.plot_epi(
        img,
        title=title,
        display_mode=display_mode,
        cut_coords=cut_coords,
        cmap=cmap
    )

def plot_fmri_diffmap(diff_img, title=None):
    """Plot a difference map with strong contrast."""
    vmax = np.percentile(np.abs(diff_img.get_fdata()), 99)
    plotting.plot_epi(diff_img, vmin=-vmax, vmax=vmax, cmap="cold_hot", title=title)


def plot_mean_image(fmri_img, title=None):
    """
    Plot mean fMRI image across time over MNI152 template.
    """
    mean_data = np.mean(fmri_img.get_fdata(), axis=-1)
    mean_img = nib.Nifti1Image(mean_data, affine=fmri_img.affine)

    plotting.plot_stat_map(
        mean_img,
        bg_img=MNI_TEMPLATE,
        threshold=None,
        display_mode='ortho',
        title=title,
        colorbar=True,
        draw_cross=True
    )


def plot_stat_map_3d(stat_img, title=None, threshold=None):
    """
    Plot a 3D statistical map over MNI152 template.
    """
    plotting.plot_stat_map(
        stat_img,
        bg_img=MNI_TEMPLATE,
        threshold=threshold,
        display_mode='ortho',
        title=title,
        colorbar=True,
        draw_cross=True
    )


def plot_glass_brain(fmri_img, title=None):
    """
    Plot a 3D glass brain overview of fMRI data.
    """
    plotting.plot_glass_brain(
        fmri_img,
        display_mode='lyrz',
        colorbar=True,
        title=title
    )


# GLM-Specific Visualization

def plot_hrf(
    tr: float = 2.0,
    duration: float = 30.0,
    oversampling: int = 50
) -> plt.Figure:
    """
    Plot the canonical hemodynamic response function (HRF).
    
    Parameters
    ----------
    tr : float, default=2.0
        Repetition time in seconds
    duration : float, default=30.0
        Total duration to plot in seconds
    oversampling : int, default=50
        Temporal oversampling factor
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object

    """
    # Generate time points
    dt = tr / oversampling
    time_points = np.arange(0, duration, dt)
    
    # Compute HRF
    hrf = glover_hrf(tr, oversampling=oversampling, time_length=duration)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Plot HRF
    ax1.plot(time_points[:len(hrf)], hrf, linewidth=2, color='darkblue')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axvline(6, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Peak (~6s)')
    ax1.fill_between(time_points[:len(hrf)], 0, hrf, alpha=0.3, color='skyblue')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('HRF Amplitude', fontsize=12)
    ax1.set_title('Canonical Hemodynamic Response Function', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot convolution example
    stimulus = np.zeros_like(time_points[:len(hrf)])
    stimulus[(time_points[:len(hrf)] >= 5) & (time_points[:len(hrf)] < 6)] = 1
    
    # Convolve stimulus with HRF
    convolved = np.convolve(stimulus, hrf[:len(stimulus)], mode='same') * dt
    
    ax2.plot(time_points[:len(hrf)], stimulus, linewidth=2, label='Brief Stimulus', color='green')
    ax2.plot(time_points[:len(hrf)], convolved, linewidth=2, label='Predicted BOLD Response', color='darkred')
    ax2.fill_between(time_points[:len(hrf)], 0, convolved, alpha=0.3, color='salmon')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Signal Amplitude', fontsize=12)
    ax2.set_title('Stimulus → BOLD Response (Convolution)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_activation_map(
    stat_map,
    threshold: float = 3.0,
    display_mode: str = 'ortho',
    cut_coords: Optional[Union[Tuple, int]] = None,
    title: str = 'Activation Map',
    colorbar: bool = True,
    cmap: str = 'cold_hot',
    bg_img=None
) -> plotting.displays.OrthoSlicer:
    """
    Plot a single activation map.
    
    Parameters
    ----------
    stat_map : Niimg-like object
        Statistical map (z-scores or t-values)
    threshold : float, default=3.0
        Statistical threshold
    display_mode : str, default='ortho'
        Display mode ('ortho', 'x', 'y', 'z', 'tiled', 'mosaic')
    cut_coords : tuple or int, optional
        Coordinates for slice positions
    title : str, default='Activation Map'
        Plot title
    colorbar : bool, default=True
        Whether to display colorbar
    cmap : str, default='cold_hot'
        Colormap name
    bg_img : Niimg-like, optional
        Background image (uses MNI152 if None)
        
    Returns
    -------
    display : plotting display object
        Display object for further customization

    """
    if bg_img is None:
        bg_img = MNI_TEMPLATE
    
    display = plotting.plot_stat_map(
        stat_map,
        bg_img=bg_img,
        threshold=threshold,
        display_mode=display_mode,
        cut_coords=cut_coords,
        colorbar=colorbar,
        cmap=cmap,
        title=title
    )
    
    return display


def plot_activation_maps_grid(
    contrast_maps: Dict,
    threshold: float = 3.0,
    display_mode: str = 'ortho',
    cut_coords: Optional[Union[Tuple, int]] = None,
    colorbar: bool = True,
    cmap: str = 'cold_hot',
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot multiple activation maps in a grid.
    
    Parameters
    ----------
    contrast_maps : dict
        Dictionary mapping contrast names to statistical maps
    threshold : float, default=3.0
        Statistical threshold (z-score)
    display_mode : str, default='ortho'
        Display mode ('ortho', 'x', 'y', 'z', 'tiled')
    cut_coords : tuple or int, optional
        Coordinates for slice positions
    colorbar : bool, default=True
        Whether to display colorbar
    cmap : str, default='cold_hot'
        Colormap name
    figsize : tuple, default=(14, 10)
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
        
    """
    n_contrasts = len(contrast_maps)
    n_cols = min(2, n_contrasts)
    n_rows = int(np.ceil(n_contrasts / n_cols))
    
    fig = plt.figure(figsize=figsize)
    
    for idx, (name, stat_map) in enumerate(contrast_maps.items()):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        
        plotting.plot_stat_map(
            stat_map,
            bg_img=MNI_TEMPLATE,
            threshold=threshold,
            display_mode=display_mode,
            cut_coords=cut_coords,
            colorbar=colorbar,
            cmap=cmap,
            title=name.replace('_', ' ').title(),
            axes=ax
        )
    
    plt.tight_layout()
    return fig


def plot_glass_brains_grid(
    contrast_maps: Dict,
    threshold: float = 3.0,
    display_mode: str = 'lyrz',
    colorbar: bool = True,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """
    Plot multiple contrasts as glass brains in a grid.
    
    Parameters
    ----------
    contrast_maps : dict
        Dictionary of contrast statistical maps
    threshold : float, default=3.0
        Statistical threshold
    display_mode : str, default='lyrz'
        Display mode for glass brain
    colorbar : bool, default=True
        Whether to display colorbar
    figsize : tuple, default=(14, 12)
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
        
    """
    n_contrasts = len(contrast_maps)
    n_cols = 2
    n_rows = int(np.ceil(n_contrasts / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_contrasts == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, stat_map) in enumerate(contrast_maps.items()):
        plotting.plot_glass_brain(
            stat_map,
            threshold=threshold,
            display_mode=display_mode,
            colorbar=colorbar,
            title=name.replace('_', ' ').title(),
            axes=axes[idx]
        )
    
    # Hide unused subplots
    for idx in range(n_contrasts, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_design_matrix_summary(
    design_matrix,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot design matrix with correlation heatmap.
    
    Parameters
    ----------
    design_matrix : pd.DataFrame
        Design matrix from GLM
    figsize : tuple, default=(12, 8)
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot design matrix
    plot_design_matrix(design_matrix, ax=ax1)
    ax1.set_title('Design Matrix', fontsize=14, fontweight='bold')
    
    # Plot correlation matrix
    corr = design_matrix.corr()
    sns.heatmap(
        corr,
        annot=False,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Correlation'},
        ax=ax2
    )
    ax2.set_title('Predictor Correlations', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    return fig


def plot_event_timing(
    events,
    figsize: Tuple[int, int] = (14, 4),
    colors: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """
    Plot experimental event timing diagram.
    
    Parameters
    ----------
    events : pd.DataFrame
        Event dataframe with ['onset', 'duration', 'trial_type']
    figsize : tuple, default=(14, 4)
        Figure size
    colors : dict, optional
        Color mapping for trial types
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object

    """
    fig, ax = plt.subplots(figsize=figsize)
    
    trial_types = events['trial_type'].unique()
    
    # Default colors if not provided
    if colors is None:
        cmap = plt.cm.get_cmap('tab10')
        colors = {trial_type: cmap(i) for i, trial_type in enumerate(trial_types)}
    
    for trial_type in trial_types:
        trial_events = events[events['trial_type'] == trial_type]
        y_pos = list(trial_types).index(trial_type)
        ax.barh(
            y_pos,
            trial_events['duration'].values,
            left=trial_events['onset'].values,
            height=0.8,
            color=colors[trial_type],
            alpha=0.7,
            label=trial_type
        )
    
    ax.set_yticks(range(len(trial_types)))
    ax.set_yticklabels(trial_types)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Experimental Paradigm: Event Timing', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_contrasts_slices(
    glm,
    contrast_specs: Dict[str, str],
    display_mode: str = 'z',
    threshold: float = 3.0,
    cut_coords: int = 8
) -> plt.Figure:
    """
    Compare multiple contrasts side-by-side with slice views.
    
    Parameters
    ----------
    glm : FirstLevelModel
        Fitted GLM model
    contrast_specs : dict
        Dictionary of contrast specifications
    display_mode : str, default='z'
        Display mode ('x', 'y', 'z')
    threshold : float, default=3.0
        Statistical threshold
    cut_coords : int, default=8
        Number of slices to display
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
 
    """
    n_contrasts = len(contrast_specs)
    fig, axes = plt.subplots(n_contrasts, 1, figsize=(14, 4 * n_contrasts))
    
    if n_contrasts == 1:
        axes = [axes]
    
    for idx, (name, spec) in enumerate(contrast_specs.items()):
        z_map = glm.compute_contrast(spec, output_type='z_score')
        
        plotting.plot_stat_map(
            z_map,
            bg_img=MNI_TEMPLATE,
            threshold=threshold,
            display_mode=display_mode,
            cut_coords=cut_coords,
            colorbar=True,
            title=f"{name.replace('_', ' ').title()} (Z > {threshold})",
            axes=axes[idx],
            cmap='cold_hot'
        )
    
    plt.tight_layout()
    return fig


def plot_roi_overlay(
    stat_map,
    atlas_img,
    threshold: float = 3.0,
    title: str = 'Activation on Atlas',
    cut_coords: Optional[Tuple] = None,
    display_mode: str = 'ortho'
) -> plotting.displays.OrthoSlicer:
    """
    Plot activation map overlaid on anatomical atlas.
    
    Parameters
    ----------
    stat_map : Niimg-like object
        Statistical activation map
    atlas_img : Niimg-like object
        Atlas image (e.g., Harvard-Oxford)
    threshold : float, default=3.0
        Statistical threshold
    title : str, default='Activation on Atlas'
        Plot title
    cut_coords : tuple, optional
        Cut coordinates
    display_mode : str, default='ortho'
        Display mode
        
    Returns
    -------
    display : plotting display object
        Display object
        
    """
    display = plotting.plot_roi(
        atlas_img,
        title=title,
        cut_coords=cut_coords,
        display_mode=display_mode,
        cmap='Paired',
        alpha=0.3
    )
    
    display.add_overlay(
        stat_map,
        threshold=threshold,
        cmap='hot',
        alpha=0.8
    )
    
    return display


def plot_surface_activation(
    stat_map,
    threshold: float = 3.0,
    title: str = 'Surface Activation',
    cmap: str = 'cold_hot',
    colorbar: bool = True,
    hemispheres: str = 'both',
    views: list = ['lateral', 'medial']
) -> plt.Figure:
    """
    Plot activation on cortical surface.
    
    Parameters
    ----------
    stat_map : Niimg-like object
        Statistical map
    threshold : float, default=3.0
        Statistical threshold
    title : str, default='Surface Activation'
        Plot title
    cmap : str, default='cold_hot'
        Colormap
    colorbar : bool, default=True
        Show colorbar
    hemispheres : str, default='both'
        Hemispheres to plot ('left', 'right', 'both')
    views : list, default=['lateral', 'medial']
        Surface views to display
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
        
    """
    from nilearn import surface
    
    # Fetch surface mesh
    fsaverage = datasets.fetch_surf_fsaverage()
    
    # Project volume to surface
    if hemispheres in ['both', 'left']:
        texture_left = surface.vol_to_surf(stat_map, fsaverage.pial_left)
    if hemispheres in ['both', 'right']:
        texture_right = surface.vol_to_surf(stat_map, fsaverage.pial_right)
    
    # Determine subplot layout
    n_views = len(views)
    if hemispheres == 'both':
        n_plots = n_views * 2
        n_cols = 2
    else:
        n_plots = n_views
        n_cols = 1
    n_rows = n_views
    
    fig = plt.figure(figsize=(7 * n_cols, 5 * n_rows))
    
    plot_idx = 1
    for view in views:
        if hemispheres in ['both', 'left']:
            ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
            plotting.plot_surf_stat_map(
                fsaverage.pial_left,
                texture_left,
                hemi='left',
                view=view,
                threshold=threshold,
                cmap=cmap,
                colorbar=colorbar,
                title=f'{title} - Left {view.title()}',
                axes=ax
            )
            plot_idx += 1
        
        if hemispheres in ['both', 'right']:
            ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
            plotting.plot_surf_stat_map(
                fsaverage.pial_right,
                texture_right,
                hemi='right',
                view=view,
                threshold=threshold,
                cmap=cmap,
                colorbar=colorbar,
                title=f'{title} - Right {view.title()}',
                axes=ax
            )
            plot_idx += 1
    
    plt.tight_layout()
    return fig