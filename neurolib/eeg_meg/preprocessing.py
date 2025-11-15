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
