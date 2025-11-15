"""
Analysis utilities for EEG and MEG data
Includes ERP analysis, time-frequency, and statistical testing
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Union
from scipy import stats


def create_epochs(
    raw: mne.io.Raw,
    events: Optional[np.ndarray] = None,
    event_id: Optional[Dict[str, int]] = None,
    tmin: float = -0.2,
    tmax: float = 0.5,
    baseline: Optional[Tuple[float, float]] = (-0.2, 0),
    preload: bool = True,
    reject: Optional[Dict[str, float]] = None,
    picks: Optional[List[str]] = None
) -> mne.Epochs:
    """
    Create epochs from raw data around events.
    
    Epochs are time-locked segments of data centered around specific events
    (e.g., stimulus presentation, button press). This is the foundation of
    event-related potential (ERP) analysis.
    """
    
    if events is None:
        events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
    
    # Set default rejection criteria if not provided
    if reject is None:
        # Conservative defaults
        reject = dict(
            eeg=100e-6,  # 100 µV
            eog=200e-6   # 200 µV
        )
    
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=preload,
        reject=reject,
        picks=picks,
        verbose=False
    )
    
    return epochs


def compute_erp(
    epochs: mne.Epochs,
    condition: Optional[str] = None
) -> mne.Evoked:
    """
    Compute Event-Related Potential (ERP) by averaging epochs.
    
    ERPs represent the brain's consistent response to a specific type of event.
    Averaging removes random noise while preserving the time-locked signal.
    """
    if condition is not None:
        epochs = epochs[condition]
    
    evoked = epochs.average()
    
    return evoked


def compare_erps(
    epochs: mne.Epochs,
    conditions: List[str],
    channel: Optional[str] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None
) -> Tuple[Dict[str, mne.Evoked], plt.Figure]:
    """
    Compare ERPs across multiple conditions.
    """
    evokeds = {}
    
    for condition in conditions:
        evokeds[condition] = epochs[condition].average()
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for condition, evoked in evokeds.items():
        if channel is not None:
            ch_idx = evoked.ch_names.index(channel)
            data = evoked.data[ch_idx, :]
            times = evoked.times
        else:
            # Average across all channels
            data = evoked.data.mean(axis=0)
            times = evoked.times
        
        ax.plot(times, data * 1e6, label=condition, linewidth=2)
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude (µV)', fontsize=12)
    ax.set_title('ERP Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if tmin is not None and tmax is not None:
        ax.set_xlim(tmin, tmax)
    
    plt.tight_layout()
    
    return evokeds, fig


def find_peak_latency(
    evoked: mne.Evoked,
    tmin: float,
    tmax: float,
    channel: Optional[str] = None,
    mode: str = 'pos'
) -> Tuple[float, float]:
    """
    Find peak amplitude and latency in an ERP.
    
    Identifies the time point with maximum (or minimum) amplitude
    within a specified time window. This is useful for measuring
    components like P100, N170, P300, etc.
    """
    # Get time mask
    time_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
    times = evoked.times[time_mask]
    
    # Get data
    if channel is not None:
        ch_idx = evoked.ch_names.index(channel)
        data = evoked.data[ch_idx, time_mask]
    else:
        data = evoked.data[:, time_mask].mean(axis=0)
    
    # Find peak
    if mode == 'pos':
        peak_idx = np.argmax(data)
    elif mode == 'neg':
        peak_idx = np.argmin(data)
    elif mode == 'abs':
        peak_idx = np.argmax(np.abs(data))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    latency = times[peak_idx]
    amplitude = data[peak_idx]
    
    return latency, amplitude


def compute_contrast(
    evoked_a: mne.Evoked,
    evoked_b: mne.Evoked,
    operation: str = 'subtract'
) -> mne.Evoked:
    """
    Compute contrast between two ERPs.
    
    Contrasts reveal differences between conditions by subtracting
    or dividing ERPs. This highlights unique neural responses to
    different stimuli or tasks.
    """
    if operation == 'subtract':
        contrast = mne.combine_evoked([evoked_a, evoked_b], weights=[1, -1])
    elif operation == 'divide':
        # Create a copy and divide data
        contrast = evoked_a.copy()
        contrast.data = evoked_a.data / (evoked_b.data + 1e-10)  # Avoid division by zero
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return contrast


def permutation_cluster_test(
    epochs_a: mne.Epochs,
    epochs_b: mne.Epochs,
    n_permutations: int = 1000,
    tail: int = 0,
    threshold: Optional[float] = None,
    n_jobs: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform cluster-based permutation test between two conditions.
    
    This non-parametric statistical test identifies spatiotemporal clusters
    where two conditions significantly differ, while controlling for
    multiple comparisons across channels and time points.
    """
    from mne.stats import permutation_cluster_test
    
    # Get data arrays
    X = [epochs_a.get_data(), epochs_b.get_data()]
    
    # Run cluster test
    T_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
        X,
        n_permutations=n_permutations,
        tail=tail,
        threshold=threshold,
        n_jobs=n_jobs,
        seed=42
    )
    
    return T_obs, clusters, cluster_pv, H0


def plot_erp_components(
    evoked: mne.Evoked,
    components: Dict[str, Dict[str, Union[float, str]]],
    channel: Optional[str] = None
) -> plt.Figure:
    """
    Plot ERP with marked components (P100, N170, P300, etc.).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get data
    if channel is not None:
        ch_idx = evoked.ch_names.index(channel)
        data = evoked.data[ch_idx, :] * 1e6  # Convert to µV
        title = f'ERP Components - Channel {channel}'
    else:
        data = evoked.data.mean(axis=0) * 1e6
        title = 'ERP Components - Grand Average'
    
    times = evoked.times
    
    # Plot ERP
    ax.plot(times, data, 'k-', linewidth=2, label='ERP')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Mark components
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    
    for (comp_name, comp_params), color in zip(components.items(), colors):
        latency, amplitude = find_peak_latency(
            evoked,
            tmin=comp_params['tmin'],
            tmax=comp_params['tmax'],
            channel=channel,
            mode=comp_params['mode']
        )
        
        # Mark time window
        ax.axvspan(comp_params['tmin'], comp_params['tmax'],
                  alpha=0.2, color=color, label=f'{comp_name} window')
        
        # Mark peak
        ax.plot(latency, amplitude * 1e6, 'o', markersize=10,
               color=color, markeredgecolor='black', markeredgewidth=2)
        
        # Add annotation
        ax.annotate(
            f"{comp_name}\n{latency*1000:.0f}ms\n{amplitude*1e6:.1f}µV",
            xy=(latency, amplitude * 1e6),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude (µV)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compute_grand_average(
    evokeds_list: List[mne.Evoked]
) -> mne.Evoked:
    """
    Compute grand average across multiple subjects/sessions.
    """
    grand_avg = mne.grand_average(evokeds_list)
    return grand_avg


def extract_erp_features(
    evoked: mne.Evoked,
    time_windows: Dict[str, Tuple[float, float]],
    channels: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Extract features from ERP for statistical analysis.
    """
    features = {}
    
    if channels is None:
        channels = evoked.ch_names
    
    for window_name, (tmin, tmax) in time_windows.items():
        time_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
        
        for ch in channels:
            ch_idx = evoked.ch_names.index(ch)
            data = evoked.data[ch_idx, time_mask]
            
            key = f"{window_name}_{ch}"
            features[key] = {
                'mean_amplitude': np.mean(data),
                'peak_amplitude': np.max(np.abs(data)),
                'peak_latency': evoked.times[time_mask][np.argmax(np.abs(data))]
            }
    
    return features


def extract_cluster_time_window(cluster_mask, times):
    """
    Safely extract the time window of a cluster mask.
    """

    # Convert scalar or 1-D masks safely to 2-D
    mask = np.atleast_2d(cluster_mask)

    # If mask has wrong dimension or is empty → skip
    if mask.ndim != 2 or mask.size == 0:
        return None, None

    # Identify time indices where the cluster is active
    time_inds = np.where(mask.any(axis=0))[0]

    if len(time_inds) == 0:
        return None, None

    # Clip indices to the valid time array range
    max_valid = len(times) - 1
    start_idx = int(np.clip(time_inds[0], 0, max_valid))
    end_idx   = int(np.clip(time_inds[-1], 0, max_valid))

    tmin = times[start_idx]
    tmax = times[end_idx]
    return tmin, tmax
