import numpy as np
from biosppy.signals.tools import filter_signal
from scipy.signal import resample

def apply_filter(signal, filter_bandwidth, fs=100):
    ''' Bandpass filtering to remove noise, artifacts etc '''
    # Calculate filter order
    order = int(0.3 * fs)
    # Filter signal
    signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                order=order, frequency=filter_bandwidth, 
                                sampling_rate=fs)
    return signal

def scaling(seq, smooth=1e-8):
    return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1

def ecg_preprocessing(ecg_signal, band_pass=[0.05, 47]):

    assert ecg_signal.shape[0] == 12, "ecg_signal should have (12, signal_length) shape for pre-processing"

    # ecg_signal = resample(ecg_signal, int(ecg_signal.shape[-1] * (500/original_frequency)), axis=1) 
    # 500 hz is the highest and most common sampling rate found in literature and respects Shannon theorem, as max spectral component is said to be 150 hz

    ecg_signal = apply_filter(ecg_signal, band_pass) # this band focuses on dominant component of ecg waves

    return scaling(ecg_signal)