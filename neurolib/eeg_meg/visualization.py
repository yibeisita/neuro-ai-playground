import mne
import matplotlib.pyplot as plt
import numpy as np


def plot_raw(raw, n_channels=30, duration=10.0, title=None):
    """Plot MEG/EEG raw signals."""
    mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib', set_env=True)
    fig = raw.plot(n_channels=n_channels, duration=duration, scalings='auto', title=title, show=False)
    if fig is not None and title:
        ax = fig.axes[0] 
        ax.set_title(title)
    
    plt.show()

def plot_psd(raw, fmin=1, fmax=40):
    """Compute and plot PSD using the modern MNE API."""
    psd = raw.compute_psd(fmin=fmin, fmax=fmax)
    psd.plot(dB=True, picks="data", exclude="bads", amplitude=False)


def plot_meg_topomap(evoked, times=[0.1, 0.2, 0.3], title=None, ch_type='mag', figsize=(8, 4)):
    """
    Plot topomap of an Evoked MEG object at specified time points.
    """
    fig = evoked.plot_topomap(
        times=times,
        ch_type=ch_type,
        show=False,
        time_unit='s',
        size=3.0,
        colorbar=True
    )
    
    if title is not None:
        plt.suptitle(title, fontsize=10)

    plt.gcf().set_size_inches(figsize)