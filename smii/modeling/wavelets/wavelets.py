import numpy as np

def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency."""
    t = np.arange(length, dtype=np.float32) * dt - peak_time
    y = (1 - 2 * np.pi**2 * freq**2 * t**2) \
            * np.exp(-np.pi**2 * freq**2 * t**2)
    return y
