"""Test the propagators."""
import numpy as np
import scipy.special
import xarray as xr
from smii.modeling.propagators.propagators import (Scalar2D)
from smii.modeling.wavelets.wavelets import ricker
from smii.modeling.forward_model import forward_model


def green(x0, x1, dx, dt, v, f):
    """Use the 2D Green's function to determine the wavefield at a given
    location due to the given source.
    """
    nt = len(f)
    r = np.sqrt(np.sum((x1 - x0)**2))
    w = np.fft.rfftfreq(nt, dt)
    fw = np.fft.rfft(f)
    G = 1/4*1j*scipy.special.hankel1(0, -2*np.pi*w*r/v)
    G[0] = 0
    s = G * fw * dx**2
    return np.fft.irfft(s, nt)


def test_direct_2d(v=1500, freq=25, dx=5, dt=0.0001, nx=10,
                   nz=80):
    """Create a constant model, and the expected waveform at point,
       and compare with forward propagated wave.
    """
    model = xr.DataArray(np.ones([1, nz, nx], dtype=np.float32) * v,
                         dims=['property', 'z', 'x'],
                         coords={'property': ['wavespeed'],
                                 'z': np.arange(0, nz*dx, dx),
                                 'x': np.arange(0, nx*dx, dx)})

    nt = int(2*nz*dx/v/dt)

    sources = xr.DataArray(ricker(freq, nt, dt, 0.05).values[np.newaxis, :],
                           dims=['source_index', 'time'],
                           coords={'time': np.arange(0, nt * dt, dt),
                                   'sources_z': ('source_index', [1]),
                                   'sources_x': ('source_index', [nx//2])})
    receivers = xr.DataArray(np.zeros(nt, np.float32)[np.newaxis, :],
                           dims=['receiver_index', 'time'],
                           coords={'time': np.arange(0, nt * dt, dt),
                                   'receivers_z': ('receiver_index', [nz-5]),
                                   'receivers_x': ('receiver_index', [nx//2])})

    dataset = xr.Dataset({'sources': sources, 'receivers': receivers})
    propagator = Scalar2D

    expected = green(np.array([sources['sources_z'].values[0]*dx,
                               sources['sources_x'].values[0]*dx]),
                     np.array([receivers['receivers_z'].values[0]*dx,
                               receivers['receivers_x'].values[0]*dx]),
                     dx, dt, v, sources.values.ravel())

    prop = forward_model(model, dataset, propagator,
                              record_receivers=True,
                              store_wavefield=False)

    actual, _ = forward_model(model, dataset, propagator,
                              record_receivers=True,
                              store_wavefield=False)

    return expected, actual
    #assert np.allclose(expected.ravel(), actual.ravel(), atol=0.04)
