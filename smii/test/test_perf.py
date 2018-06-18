"""Create constant and point scatterer models."""
from timeit import repeat
import numpy as np
import pytest
from smii.modeling.propagators.propagators import Scalar2D as Scalar2Dcpu
from smii.modeling.propagators.scalar2d_gpu import Scalar2D as Scalar2Dgpu
from smii.modeling.wavelets.wavelets import ricker
from smii.modeling.forward_model import forward_model
from smii.test.models_scalar import direct_2d_approx

@pytest.fixture
def _versions(cpuprop_kwargs, gpuprop_kwargs):
    return [{'name': 'Scalar2Dcpu', 'propagator': Scalar2Dcpu,
             'kwargs': cpuprop_kwargs},
             {'name': 'Scalar2Dgpu', 'propagator': Scalar2Dgpu,
             'kwargs': gpuprop_kwargs}]

def _setup_propagator(c, freq, dx, dt, nt, nx, num_shots,
                      num_sources_per_shot, num_receivers_per_shot,
                      prop_kwargs):
    model = np.ones(nx, np.float32) * c

    x_s_idx = np.ones([num_shots, num_sources_per_shot, 2], np.int)
    x_s_idx[:, :, 1] = np.tile(np.linspace(1, nx[1]-1, num_sources_per_shot),
                               [num_shots, 1])
    f = ricker(freq, nt, dt, 0.05)

    source = {}
    source['amplitude'] = np.tile(f, [num_shots, num_sources_per_shot, 1])
    source['locations'] = x_s_idx

    x_r_idx = np.ones([num_shots, num_receivers_per_shot, 2], np.int)
    x_r_idx[:, :, 1] = np.tile(np.linspace(1, nx[1]-1, num_receivers_per_shot),
                               [num_shots, 1])
    x_r_idx[:, :, 0] = nx[0] - 1

    prop_args = (model, dx, dt, source)
    if len(prop_kwargs) == 0:
        prop_kwargs = {'pml_width': 30}

    return prop_args, prop_kwargs, x_r_idx


def verify_version(propagator, x_r_idx):
    dx = propagator.geometry.dx
    dt = propagator.timestep.dt * propagator.timestep.step_ratio
    c = propagator.model.property['vp'][0, 0]
    f = propagator.sources.amplitude[0, 0].ravel()
    expected = []
    for shotidx in range(propagator.sources.locations.shape[0]):
        expected.append([])
        for sourceidx in range(propagator.sources.locations.shape[1]):
            expected[-1].append([])
            x_s_idx = propagator.sources.locations[shotidx, sourceidx]
            for receiveridx in range(x_r_idx.shape[1]):
                x_r_idx0 = x_r_idx[shotidx, receiveridx]
                expected[-1][-1].append(direct_2d_approx(x_r_idx0*dx,
                                                         x_s_idx*dx,
                                                         dx, dt, c, f))
    expected = np.sum(np.array(expected), axis=1)
    actual, _ = forward_model(propagator, x_r_idx)
    #np.allclose(expected, actual, atol=1)
    return expected, actual


def time_version(propagator, prop_args, prop_kwargs, x_r_idx):
    def closure():
        """Closure over variables so they can be used in repeat below."""
        prop = propagator(*prop_args, **prop_kwargs)
        forward_model(prop, x_r_idx)

    return np.min(repeat(closure, number=1))


def run_timing(versions, c=1500, freq=25, dx=5, dt=0.005, nt=3000, nx=[64, 64],
               num_shots=3, num_sources_per_shot=4, num_receivers_per_shot=5):

    timing = []
    for version in versions:
        prop_args, prop_kwargs, x_r_idx = \
                _setup_propagator(c, freq, dx, dt, nt, nx,
                                  num_shots,
                                  num_sources_per_shot,
                                  num_receivers_per_shot,
                                  version['kwargs'])
        runtime = time_version(version['propagator'], prop_args, prop_kwargs,
                               x_r_idx)
        timing.append({'name': version['name'], 'time': runtime})
    print(timing)
    return timing


def run_verify(versions, c=1500, freq=25, dx=5, dt=0.005, nt=3000, nx=[64, 64],
               num_shots=3, num_sources_per_shot=4, num_receivers_per_shot=5):

    o = []
    for version in versions:
        prop_args, prop_kwargs, x_r_idx = \
                _setup_propagator(c, freq, dx, dt, nt, nx,
                                  num_shots,
                                  num_sources_per_shot,
                                  num_receivers_per_shot,
                                  version['kwargs'])
        propagator = version['propagator'](*prop_args, **prop_kwargs)
        o.append(verify_version(propagator, x_r_idx))
    return o


def test_verify(_versions):
    o = run_verify(versions=_versions)
    assert np.linalg.norm((o[0][0] - o[0][1].receivers).ravel()) < 2


if __name__ == '__main__':

    run_timing()
