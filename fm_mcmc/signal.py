#!/usr/bin/env python3

import numpy as np

# Constants for the FM signal
fs = 10e6      # Sampling rate (Hz)
fc = 100e6     # Carrier frequency (Hz)
fm = 1e3       # Modulating frequency (Hz)
delta_f = 75e3 # Frequency deviation (Hz)
duration = 1   # Duration of the signal in seconds

def generate_fm_signal(d, e, f, t):
    """Generate an FM signal based on scaling factor d, modulating frequency e, and phase f."""
    return d * np.cos(2 * np.pi * fc * t + delta_f * np.sin(2 * np.pi * e * t + f))

def generate_audio_signal(t):
    """Generate the modulating sine wave signal."""
    return np.sin(2 * np.pi * fm * t)
