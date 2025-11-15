import mne

def apply_bandpass_filter(raw, l_freq, h_freq):
    return raw.filter(l_freq=l_freq, h_freq=h_freq)

def apply_notch_filter(raw, freqs):
    return raw.notch_filter(freqs)

def compute_ica(raw, n_components=20, random_state=42):
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state)
    ica.fit(raw)
    return ica

def apply_ica_to_raw(raw, ica):
    return ica.apply(raw.copy())

def preprocess_raw(raw, 
                   filter_params=None, 
                   notch_freqs=None, 
                   resample_freq=None, 
                   apply_ica=False, 
                   ica_n_components=20, 
                   random_state=42, 
                   verbose=False):
    """
    Preprocess raw EEG/MEG data.
    """
    raw_clean = raw.copy()

    # 1. Bandpass filter
    if filter_params is not None:
        l_freq = filter_params.get('l_freq', None)
        h_freq = filter_params.get('h_freq', None)
        raw_clean = apply_bandpass_filter(raw_clean, l_freq, h_freq)
        if verbose:
            print(f"Applied band-pass filter: {l_freq}-{h_freq} Hz")

    # 2. Notch filter
    if notch_freqs is not None:
        raw_clean = apply_notch_filter(raw_clean, notch_freqs)
        if verbose:
            print(f"Applied notch filter at: {notch_freqs} Hz")

    # 3. Resample
    if resample_freq is not None:
        raw_clean.resample(resample_freq)
        if verbose:
            print(f"Resampled data to {resample_freq} Hz")

    # 4. ICA
    if apply_ica:
        ica = compute_ica(raw_clean, n_components=ica_n_components, random_state=random_state)
        raw_clean = apply_ica_to_raw(raw_clean, ica)
        if verbose:
            print(f"Applied ICA with {ica_n_components} components")

    return raw_clean