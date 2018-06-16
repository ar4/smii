"""Wave equation propagator.
"""
import copy
import numpy as np
from smii.modeling.propagators import (libscalar1d, libscalar1d16,
                                       libscalar2d)

class ScalarPropagator(object):
    """A scalar wave equation propagator object."""
    def __init__(self, model, dx, source_dt, pad_width, source=None,
                 num_steps=None, pml_width=None):

        self.source = Source(source, source_dt)
        self.geometry = Geometry(model.shape, dx, pad_width, pml_width)
        self.timestep = Timestep(np.max(model), dx, self.source, num_steps)
        self.pml = Pml(self.geometry, np.max(model))
        self.model = Model(self.geometry, self.timestep, vp=model)
        self.wavefield = Wavefield(self.geometry)

    def step(self):
        raise NotImplementedError

    def _step_setup(self, num_steps, source):
        if source is None:
            source = self.source.step_range(self.timestep.step_idx, num_steps)
        else:
            source = Source(source, self.source.dt)
        source.locations = source.locations + self.geometry.total_pad
        return source

    def _update_pointers(self, num_steps):
        if (num_steps * self.timestep.step_ratio)%2 != 0:
            self.wavefield.current, self.wavefield.previous = \
                    self.wavefield.previous, self.wavefield.current
            for phi in self.pml.phi:
                phi.current, phi.previous = phi.previous, phi.current


class Source(object):
    def __init__(self, source, source_dt):
        self.amplitude = source['amplitude']
        self.locations = source['locations']
        self.dt = source_dt

    def step_range(self, start_step, num_steps):
        source = copy.copy(self)
        source.amplitude = \
                source.amplitude[:, start_step : start_step + num_steps]
        return source


class Geometry(object):
    def __init__(self, shape, dx, pad_width, pml_width):

        if pml_width is None:
            pml_width = 10

        self.shape = shape
        self.dx = dx
        self.pad_width = pad_width
        self.pml_width = pml_width
        self.total_pad = self.pml_width + self.pad_width
        self.ndim = len(self.shape)
        self.shape_padded = np.array(self.shape) + 2 * self.total_pad


class Timestep(object):
    def __init__(self, max_vel, dx, source, num_steps):

        self.num_steps = num_steps
        if num_steps is None and source.amplitude is not None:
            self.num_steps = source.amplitude.shape[-1]

        max_dt = 0.6 * dx / max_vel
        self.step_ratio = int(np.ceil(source.dt / max_dt))
        self.dt = source.dt / self.step_ratio
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
        sigma = np.zeros(geometry.ndim, np.object)
        total_pad = geometry.total_pad
        pad_width = geometry.pad_width
        for dim in range(len(sigma)):
            sigma[dim] = np.zeros(geometry.shape_padded[dim], np.float32)
            sigma[dim][total_pad-1:pad_width-1:-1] = profile
            sigma[dim][-total_pad:-pad_width] = profile
            sigma[dim][:pad_width] = sigma[dim][pad_width]
            sigma[dim][-pad_width:] = sigma[dim][-pad_width-1]
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
        shape = geometry.shape_padded
        self.current, self.previous = [np.zeros(shape, np.float32),
                                       np.zeros(shape, np.float32)]
        self._inner_slice = [slice(geometry.total_pad, -geometry.total_pad)] \
                * geometry.ndim

    @property
    def inner(self):
        return self.current[self._inner_slice]


class Scalar1D(ScalarPropagator):
    """1D scalar wave propagator."""
    def __init__(self, model, dx, source_dt, source=None, pml_width=None):
        pad_width = 3
        super(Scalar1D, self).__init__(model, dx, source_dt, pad_width,
                                       source=source, pml_width=pml_width)

    def step(self, num_steps=1, source=None):
        """Propagate wavefield."""

        source = self._step_setup(num_steps, source)

        libscalar1d.scalar1d.step(self.wavefield.current.T,
                                  self.wavefield.previous.T,
                                  self.pml.phi[0].current.T,
                                  self.pml.phi[0].previous.T,
                                  self.pml.sigma[0].T,
                                  self.model.padded_property['vp2dt2'].T,
                                  self.geometry.dx, self.timestep.dt,
                                  source.amplitude.T,
                                  source.locations.T, num_steps,
                                  self.geometry.pml_width,
                                  self.timestep.step_ratio)

        self._update_pointers(num_steps)

        self.timestep.step_idx += num_steps

        return self.wavefield.inner


class Scalar1D_16(ScalarPropagator):
    """1D scalar wave propagator."""
    def __init__(self, model, dx, source_dt, source=None, pml_width=None):
        pad_width = 8
        super(Scalar1D_16, self).__init__(model, dx, source_dt, pad_width,
                                          source=source, pml_width=pml_width)

    def step(self, num_steps=1, source=None):
        """Propagate wavefield."""

        source = self._step_setup(num_steps, source)

        libscalar1d16.scalar1d16.step(self.wavefield.current.T,
                                      self.wavefield.previous.T,
                                      self.pml.phi[0].current.T,
                                      self.pml.phi[0].previous.T,
                                      self.pml.sigma[0].T,
                                      self.model.padded_property['vp2dt2'].T,
                                      self.geometry.dx, self.timestep.dt,
                                      source.amplitude.T,
                                      source.locations.T, num_steps,
                                      self.geometry.pml_width,
                                      self.timestep.step_ratio)

        self._update_pointers(num_steps)

        self.timestep.step_idx += num_steps

        return self.wavefield.inner


class Scalar2D(ScalarPropagator):
    """2D scalar wave propagator."""
    def __init__(self, model, dx, source_dt, source=None, pml_width=None):
        pad_width = 3
        super(Scalar2D, self).__init__(model, dx, source_dt, pad_width,
                                       source=source, pml_width=pml_width)

    def step(self, num_steps=1, source=None):
        """Propagate wavefield."""

        source = self._step_setup(num_steps, source)

        libscalar2d.scalar2d.step(self.wavefield.current.T,
                                  self.wavefield.previous.T,
                                  self.pml.phi[1].current.T,
                                  self.pml.phi[1].previous.T,
                                  self.pml.phi[0].current.T,
                                  self.pml.phi[0].previous.T,
                                  self.pml.sigma[1].T,
                                  self.pml.sigma[0].T,
                                  self.model.padded_property['vp2dt2'].T,
                                  self.geometry.shape[1],
                                  self.geometry.dx, self.timestep.dt,
                                  source.amplitude.T,
                                  source.locations.T, num_steps,
                                  self.geometry.pml_width,
                                  self.timestep.step_ratio)

        self._update_pointers(num_steps)

        self.timestep.step_idx += num_steps

        return self.wavefield.inner
