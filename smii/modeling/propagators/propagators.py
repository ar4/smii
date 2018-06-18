"""Wave equation propagator.
"""
import numpy as np
from smii.modeling.propagators import (libscalar1d, libscalar2d)

class ScalarPropagator(object):
    """A scalar wave equation propagator object."""
    def __init__(self, model, dx, source_dt, sources, pad_width,
                 pml_width=None):
        """A scalar wave equation propagator base class.

        Arguments:
            model: An array containing wave speed
            dx: A float specifying cell size
            source_dt: A float specifying source time step size
            sources: A dictionary containing two keys:
                     'amplitude': [num_shots, num_sources_per_shot, nt]
                                  array
                     'location': [num_shots, num_sources_per_shot, ndim]
                                 array
            pad_width: An int specifying the padding to add around the
                       computational domain
            pml_width: An int specifying the number of cells of padding to be
                       added around the model for the PML
        """

        self.sources = Sources(sources, source_dt)
        self.geometry = Geometry(model.shape, dx, pad_width, pml_width,
                                 self.sources.num_shots,
                                 self.sources.num_sources_per_shot,
                                 self.sources.ndim)
        self.sources.add_padding(self.geometry.total_pad)
        self.timestep = Timestep(np.max(model), dx, self.sources.num_steps,
                                 self.sources.dt)
        self.pml = Pml(self.geometry, np.max(model))
        self.model = Model(self.geometry, self.timestep, vp=model)
        self.wavefield = Wavefield(self.geometry)

    def step(self, num_steps=1):
        raise NotImplementedError

    def _step_setup(self, num_steps):
        sources = {'amplitude': self.sources.step_range(self.timestep.step_idx,
                                                        num_steps),
                   'locations': self.sources.padded_locations}
        return sources

    def _update_pointers(self, num_steps):
        if (num_steps * self.timestep.step_ratio)%2 != 0:
            self.wavefield.current, self.wavefield.previous = \
                    self.wavefield.previous, self.wavefield.current
            for phi in self.pml.phi:
                phi.current, phi.previous = phi.previous, phi.current


class Sources(object):
    def __init__(self, sources, source_dt):
        self.amplitude = sources['amplitude'].astype(np.float32)
        self.locations = sources['locations'].astype(np.int32)
        self.dt = source_dt
        self.num_shots = self.amplitude.shape[0]
        self.num_sources_per_shot = self.amplitude.shape[1]
        self.num_steps = self.amplitude.shape[2]
        self.ndim = self.locations.shape[2]

    def add_padding(self, total_pad):
        self.padded_locations = self.locations + total_pad

    def step_range(self, start_step, num_steps):
        return self.amplitude[..., start_step : start_step + num_steps]


class Geometry(object):
    def __init__(self, model_shape, dx, pad_width, pml_width, num_shots,
                 num_sources_per_shot, ndim):

        if pml_width is None:
            pml_width = 10

        self.dx = dx
        self.pad_width = pad_width
        self.pml_width = pml_width
        self.total_pad = self.pml_width + self.pad_width
        self.ndim = len(model_shape)
        assert self.ndim == ndim, ('{} != {}, shape of model and number of '
                                   'dimensions in source locations do not '
                                   'match.'.format(self.ndim, ndim))

        # Make a list of shot indices to use when adding sources to the
        # wavefield. [num_shots, num_sources_per_shot]. For 2 shots
        # with 3 sources per shot: [[0, 0, 0], [1, 1, 1]]
        self.shotidx = np.arange(num_shots).reshape(-1, 1)\
                                           .repeat(num_sources_per_shot,
                                                   axis=1)

        self.model_shape = list(model_shape)
        self.model_shape_padded = [d + 2 * self.total_pad
                                   for d in self.model_shape]

        self.propagation_shape = [num_shots] + self.model_shape
        self.propagation_shape_padded = ([num_shots]
                                         + self.model_shape_padded)


class Timestep(object):
    def __init__(self, max_vel, dx, num_steps, source_dt):

        self.num_steps = num_steps

        max_dt = 0.6 * dx / max_vel
        self.step_ratio = int(np.ceil(source_dt / max_dt))
        self.dt = source_dt / self.step_ratio
        self.step_idx = 0


class Pml(object):
    def __init__(self, geometry, max_vel):

        pml_width = geometry.pml_width
        profile = ((np.arange(pml_width)/pml_width)**2
                   * 3 * max_vel * np.log(1000)
                   / (2 * geometry.dx * pml_width))
        self.sigma = self._set_sigma(geometry, profile)
        self.phi = [Wavefield(geometry) for dim in range(geometry.ndim)]

    def _set_sigma(self, geometry, profile):
        total_pad = geometry.total_pad
        pad_width = geometry.pad_width
        sigma = []
        for dim in range(geometry.ndim):
            sigma_dim = np.zeros(geometry.model_shape_padded[dim], np.float32)
            sigma_dim[total_pad-1:pad_width-1:-1] = profile
            sigma_dim[-total_pad:-pad_width] = profile
            sigma_dim[:pad_width] = sigma_dim[pad_width]
            sigma_dim[-pad_width:] = sigma_dim[-pad_width-1]
            sigma.append(sigma_dim)
        return sigma


class Model(object):
    def __init__(self, geometry, timestep, vp):
        self.property = {}
        self.property['vp'] = vp
        self.property['vp2dt2'] = vp**2 * timestep.dt**2
        self.padded_property = {}
        for key, value in self.property.items():
            self.padded_property[key] = np.pad(value, geometry.total_pad,
                                               'edge')


class Wavefield(object):
    def __init__(self, geometry):
        shape = geometry.propagation_shape_padded
        self.current, self.previous = [np.zeros(shape, np.float32),
                                       np.zeros(shape, np.float32)]
        self._inner_slice = [slice(geometry.total_pad, -geometry.total_pad)] \
                * geometry.ndim

    @property
    def inner(self):
        return self.current[[...] + self._inner_slice]


class Scalar1D(ScalarPropagator):
    """1D scalar wave propagator."""
    def __init__(self, model, dx, source_dt, sources, pml_width=None):
        pad_width = 2
        super(Scalar1D, self).__init__(model, dx, source_dt, sources,
                                       pad_width, pml_width=pml_width)

    def step(self, num_steps=1):
        """Propagate wavefield."""

        sources = self._step_setup(num_steps)

        libscalar1d.scalar1d.step(self.wavefield.current.T,
                                  self.wavefield.previous.T,
                                  self.pml.phi[0].current.T,
                                  self.pml.phi[0].previous.T,
                                  self.pml.sigma[0].T,
                                  self.model.padded_property['vp2dt2'].T,
                                  self.geometry.dx, self.timestep.dt,
                                  sources['amplitude'].T,
                                  sources['locations'].T, num_steps,
                                  self.geometry.pml_width,
                                  self.timestep.step_ratio)

        self._update_pointers(num_steps)

        self.timestep.step_idx += num_steps

        return self.wavefield.inner


class Scalar2D(ScalarPropagator):
    """2D scalar wave propagator."""
    def __init__(self, model, dx, source_dt, sources, pml_width=None):
        pad_width = 2
        super(Scalar2D, self).__init__(model, dx, source_dt, sources,
                                       pad_width, pml_width=pml_width)

    def step(self, num_steps=1):
        """Propagate wavefield."""

        sources = self._step_setup(num_steps)

        libscalar2d.scalar2d.step(self.wavefield.current.T,
                                  self.wavefield.previous.T,
                                  self.pml.phi[1].current.T,
                                  self.pml.phi[1].previous.T,
                                  self.pml.phi[0].current.T,
                                  self.pml.phi[0].previous.T,
                                  self.pml.sigma[1].T,
                                  self.pml.sigma[0].T,
                                  self.model.padded_property['vp2dt2'].T,
                                  self.geometry.dx, self.timestep.dt,
                                  sources['amplitude'].T,
                                  sources['locations'].T, num_steps,
                                  self.geometry.pml_width,
                                  self.timestep.step_ratio)

        self._update_pointers(num_steps)

        self.timestep.step_idx += num_steps

        return self.wavefield.inner
