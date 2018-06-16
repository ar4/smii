"""Create constant and point scatterer models."""
import numpy as np
from timeit import repeat
from smii.modeling.propagators.propagators import (Scalar2D)
from smii.modeling.wavelets.wavelets import ricker
from smii.modeling.forward_model import forward_model
from smii.test.models_scalar import direct_2d_approx

def _versions():
    return [{'name': '1', 'propagator': Scalar2D}]

def _setup_propagator(propagator, c, freq, dx, dt, nt, nx):
    nx = np.array(nx)
    model = np.ones(nx, dtype=np.float32) * c

    x_s_idx = np.array([[1, 1]])
    f = ricker(freq, nt, dt, 0.05)

    source = {}
    source['amplitude'] = f[np.newaxis, :]
    source['locations'] = x_s_idx
    prop = propagator(model, dx, dt, source, pml_width=30)
    return prop

def verify_version(propagator):
    x_s_idx = propagator.source.locations.ravel()
    x_r_idx = x_s_idx+1
    receiver_locations = x_r_idx[np.newaxis, :]
    dx = propagator.geometry.dx
    dt = propagator.timestep.dt
    c = propagator.model.property['vp'][0, 0]
    f = propagator.source.amplitude.ravel()
    expected = direct_2d_approx(x_r_idx*dx, x_s_idx*dx, dx, dt, c, f)
    actual, _ = forward_model(propagator, receiver_locations)
    #np.allclose(expected, actual, atol=1)
    return expected, actual

def time_version(propagator):
    x_s_idx = propagator.source.locations.ravel()
    receiver_locations = x_s_idx[np.newaxis, :]+1
    nt = propagator.timestep.num_steps
    def closure():
        """Closure over variables so they can be used in repeat below."""
    #    forward_model(propagator, receiver_locations)
        propagator.step(int(nt/3))

    return np.min(repeat(closure, number=1))
    #return np.min(repeat(propagator.step, number=int(nt/3)))
        

def run_timing(c=1500, freq=25, dx=5, dt=0.0001, nt=1000, nx=[50, 50],
               versions=None):
    if versions is None:
        versions = _versions()

    timing = []
    for version in versions:
        propagator = _setup_propagator(version['propagator'],
                                       c, freq, dx, dt, nt, nx)
        runtime = time_version(propagator)
        timing.append({'name': version['name'], 'time': runtime})
    print(timing)
    return timing

def run_verify(c=1500, freq=25, dx=5, dt=0.0001, nt=1000, nx=[50, 50],
               versions=None):
    if versions is None:
        versions = _versions()

    o = []
    for version in versions:
        propagator = _setup_propagator(version['propagator'],
                                       c, freq, dx, dt, nt, nx)
        o.append(verify_version(propagator))
    return o

if __name__ == '__main__':
    run_timing()
