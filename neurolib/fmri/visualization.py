import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
import numpy as np
from nilearn import datasets

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
