"""Create constant and point scatterer models."""
import numpy as np
import scipy.special
import scipy.integrate
from scipy.ndimage.interpolation import shift
from smii.modeling.propagators.propagators import (Scalar1D, Scalar2D)
from smii.modeling.wavelets.wavelets import ricker
from smii.modeling.forward_model import forward_model
from smii.inversion.fwi import costjac

def direct_1d(x, x_s, dx, dt, c, f):
    """Use the 1D Green's function to determine the wavefield at a given
    location and time due to the given source.
    """
    r = np.abs(x - x_s)
    t_shift = (r/c) / dt + 1
    u = dx * dt * c / 2 * np.cumsum(shift(f, t_shift))
    return u


def direct_2d(x, t, x_s, dx, dt, c, f):
    """Use the 2D Green's function to determine the wavefield at a given
    location and time due to the given source.
    """
    r = np.linalg.norm(x - x_s)
    t_max = np.maximum(0, int((t - r/c) / dt))
    tmtp = t - np.arange(t_max) * dt
    summation = np.sum(f[:t_max] / np.sqrt(c**2 * tmtp**2 - r**2))
    u = dx**2 * dt * c / 2 / np.pi * summation
    return u


def direct_2d2(x, x_s, dx, dt, c, f):
    """Use the 2D Green's function to determine the wavefield at a given
    location and time due to the given source.
    """
    r = np.linalg.norm(x - x_s)
    nt = len(f)
    def func(tp, t):
        return f[int(tp / dt)] / np.sqrt(c**2 * (t - tp)**2 - r**2)
    u = np.zeros_like(f)
    t_max = int(r/c / dt)
    for t_idx in range(t_max):
        t = t_idx * dt
        u[t_idx] = scipy.integrate.quad(func, 0, t, (t+dt))[0]
    u *= dx**2 * dt * c / 2 / np.pi
    return u

def direct_2d_approx(x, x_s, dx, dt, c, f):
    """Same as direct_2d, but using an approximation to calculate the result
    for the whole time range of the source.
    """
    r = np.linalg.norm(x - x_s)
    nt = len(f)
    w = np.fft.rfftfreq(nt, dt)
    fw = np.fft.rfft(f)
    G = 1j / 4 * scipy.special.hankel1(0, -2 * np.pi * w * r / c)
    G[0] = 0
    s = G * fw * dx**2
    u = np.fft.irfft(s, nt)
    return u


def direct_3d(x, x_s, dx, dt, c, f):
    """Use the 3D Green's function to determine the wavefield at a given
    location and time due to the given source.
    """
    r = np.linalg.norm(x - x_s)
    t_shift = (r/c) / dt + 1
    u = dx**3 * dt / 4 / np.pi / r * shift(f, t_shift)
    return u


def scattered_1d(x, x_s, x_p, dx, dt, c, dc, f):
    u_p = direct_1d(x_p, x_s, dx, dt, c, f)
    du_pdt2 = np.gradient(np.gradient(u_p)) / dt**2
    u = 2 * dc / c**3 * direct_1d(x, x_p, dx, dt, c, du_pdt2)
    return u


def scattered_2d(x, x_s, x_p, dx, dt, c, dc, f):
    u_p = direct_2d_approx(x_p, x_s, dx, dt, c, f)
    du_pdt2 = np.gradient(np.gradient(u_p)) / dt**2
    u = 2 * dc / c**3 * direct_2d_approx(x, x_p, dx, dt, c, du_pdt2)
    return u


def scattered_3d(x, x_s, x_p, dx, dt, c, dc, f):
    u_p = direct_3d(x_p, x_s, dx, dt, c, f)
    du_sdt2 = np.gradient(np.gradient(u_p)) / dt**2
    u = 2 * dc / c**3 * direct_3d(x, x_p, dx, dt, c, du_pdt2)
    return u


def grad_1d(nx, x_r, x_s, x_p, dx, dt, c, dc, f):
    d = -scattered_1d(x_r, x_s, x_p, dx, dt, c, dc, f)[::-1]
    grad = np.zeros(nx, np.float32)
    for x_idx in range(nx):
        x = x_idx*dx
        u_r = direct_1d(x, x_r, dx, dt, c, d)[::-1]
        u_0 = direct_1d(x, x_s, dx, dt, c, f)
        du_0dt2 = np.gradient(np.gradient(u_0)) / dt**2
        grad[x_idx] = 2 * dt / c**3 * np.sum(u_r * du_0dt2)
    return grad


def grad_2d(nx, x_r, x_s, x_p, dx, dt, c, dc, f):
    d = -scattered_2d(x_r, x_s, x_p, dx, dt, c, dc, f)[::-1]
    grad = np.zeros(nx, np.float32)
    for z_idx in range(nx[0]):
        for x_idx in range(nx[1]):
            x = np.array([z_idx*dx, x_idx*dx])
            u_r = direct_2d_approx(x, x_r, dx, dt, c, d)[::-1]
            u_0 = direct_2d_approx(x, x_s, dx, dt, c, f)
            du_0dt2 = np.gradient(np.gradient(u_0)) / dt**2
            grad[z_idx, x_idx] = 2 * dt / c**3 * np.sum(u_r * du_0dt2)
    return grad


def grad_1d_fd(model_true, model_init, x_r, x_s, dx, dt, dc, f,
               propagator=None, prop_kwargs=None):
    x_r_idx, x_s_idx = (np.array([x_r, x_s]) / dx).astype(np.int)
    source, receiver_locations = _make_source_receiver(x_s_idx, x_r_idx, f)
    if propagator is None:
        propagator = Scalar1D
    if prop_kwargs is None:
        prop_kwargs = {}
    prop = propagator(model_true, dx, dt, source, **prop_kwargs)
    true_data, _ = forward_model(prop, receiver_locations)
    receiver = {}
    receiver['amplitude'] = true_data.receivers
    receiver['locations'] = receiver_locations
    dataset = [(source, receiver)]
    init_cost, fwi_grad = costjac(model_init, dataset, dx, dt, propagator,
                                  model_init.shape, compute_grad=True,
                                  prop_kwargs=prop_kwargs)

    nx = len(model_true)
    true_grad = np.zeros(nx, np.float32)
    for x_idx in range(nx):
        tmp_model = model_init.copy()
        tmp_model[x_idx] += dc
        new_cost, _ = costjac(tmp_model, dataset, dx, dt, propagator,
                              model_init.shape, compute_grad=False,
                              prop_kwargs=prop_kwargs)
        true_grad[x_idx] = (new_cost - init_cost) / dc
    return fwi_grad, true_grad


def grad_2d_fd(model_true, model_init, x_r, x_s, dx, dt, dc, f,
               propagator=None, prop_kwargs=None):
    x_r_idx, x_s_idx = (np.array([x_r, x_s]) / dx).astype(np.int)
    source, receiver_locations = _make_source_receiver(x_s_idx, x_r_idx, f)
    if propagator is None:
        propagator = Scalar2D
    if prop_kwargs is None:
        prop_kwargs = {}
    prop = propagator(model_true, dx, dt, source, **prop_kwargs)
    true_data, _ = forward_model(propagator, receiver_locations)
    receiver = {}
    receiver['amplitude'] = true_data.receivers
    receiver['locations'] = receiver_locations
    dataset = [(source, receiver)]
    init_cost, fwi_grad = costjac(model_init, dataset, dx, dt, propagator,
                                  model_init.shape, compute_grad=True,
                                  prop_kwargs=prop_kwargs)

    true_grad = np.zeros_like(model_true)
    for z_idx in range(model_true.shape[0]):
        for x_idx in range(model_true.shape[1]):
            tmp_model = model_init.copy()
            tmp_model[z_idx, x_idx] += dc
            new_cost, _ = costjac(tmp_model, dataset, dx, dt, propagator,
                                  model_init.shape, compute_grad=False,
                                  prop_kwargs=prop_kwargs)
            true_grad[z_idx, x_idx] = (new_cost - init_cost) / dc
    return fwi_grad, true_grad


def _make_source_receiver(x_s_idx, x_r_idx, f):
    source = {}
    source['amplitude'] = f.reshape(1, 1, -1)
    source['locations'] = x_s_idx.reshape(1, 1, -1)
    receiver_locations = x_r_idx.reshape(1, 1, -1)
    return source, receiver_locations


def _set_coords(x, dx):
    x_m = np.array(x) * dx
    x_idx = np.array(x)
    return x_m, x_idx

def model_direct_1d(c=1500, freq=25, dx=5, dt=0.0001, nx=80,
                    propagator=None, prop_kwargs=None):
    """Create a constant model, and the expected waveform at point,
       and the forward propagated wave.
    """
    model = np.ones(nx, dtype=np.float32) * c

    nt = int(2*nx*dx/c/dt)
    x_s, x_s_idx = _set_coords([[1]], dx)
    x_r, x_r_idx = _set_coords([[nx-1]], dx)
    f = ricker(freq, nt, dt, 0.05)

    expected = direct_1d(x_r, x_s, dx, dt, c, f)

    source, receiver_locations = _make_source_receiver(x_s_idx, x_r_idx, f)
    if propagator is None:
        propagator = Scalar1D
    if prop_kwargs is None:
        prop_kwargs = {}
    prop = propagator(model, dx, dt, source, **prop_kwargs)

    actual, _ = forward_model(prop, receiver_locations)

    return expected, actual.receivers.ravel()


def model_direct_2d(c=1500, freq=25, dx=5, dt=0.0001, nx=[50, 50],
                    propagator=None, prop_kwargs=None):
    """Create a constant model, and the expected waveform at point,
       and the forward propagated wave.
    """
    model = np.ones(nx, dtype=np.float32) * c

    nt = int(2*nx[0]*dx/c/dt)
    middle = int(nx[1]/2)
    x_s, x_s_idx = _set_coords([[1, middle]], dx)
    x_r, x_r_idx = _set_coords([[nx[0]-1, middle]], dx)
    #x_r, x_r_idx = _set_coords([[1, middle]], dx)
    f = ricker(freq, nt, dt, 0.05)

    expected = direct_2d_approx(x_r, x_s, dx, dt, c, f)

    source, receiver_locations = _make_source_receiver(x_s_idx, x_r_idx, f)
    if propagator is None:
        propagator = Scalar2D
    if prop_kwargs is None:
        prop_kwargs = {}
    prop = propagator(model, dx, dt, source, **prop_kwargs)

    actual, _ = forward_model(prop, receiver_locations)

    return expected, actual.receivers.ravel()


def model_scatter_1d(c=1500, dc=50, freq=25, dx=5, dt=0.0001, nx=100,
                     propagator=None, prop_kwargs=None):
    """Create a point scatterer model, and the expected waveform at point,
       and the forward propagated wave.
    """
    model = np.ones(nx, dtype=np.float32) * c

    nt = int((3*nx*dx/c + 0.05)/dt)
    x_s, x_s_idx = _set_coords([[1]], dx)
    x_r, x_r_idx = _set_coords([[1]], dx)
    x_p, x_p_idx = _set_coords([[nx-20]], dx)
    f = ricker(freq, nt, dt, 0.05)

    model[x_p_idx] += dc

    expected = scattered_1d(x_r, x_s, x_p, dx, dt, c, dc, f)

    source, receiver_locations = _make_source_receiver(x_s_idx, x_r_idx, f)
    if propagator is None:
        propagator = Scalar1D
    if prop_kwargs is None:
        prop_kwargs = {}
    prop = propagator(model, dx, dt, source, **prop_kwargs)

    actual, _ = forward_model(prop, receiver_locations)

    return expected, actual.receivers.ravel()


def model_scatter_2d(c=1500, dc=150, freq=25, dx=5, dt=0.0001, nx=[50, 50],
                     propagator=None, prop_kwargs=None):
    """Create a point scatterer model, and the expected waveform at point,
       and the forward propagated wave.
    """
    nx = np.array(nx)
    model = np.ones(nx, dtype=np.float32) * c

    nt = int((3*nx[0]*dx/c + 0.05)/dt)
    middle = int(nx[1]/2)
    x_s, x_s_idx = _set_coords([[1, middle]], dx)
    x_r, x_r_idx = _set_coords([[1, middle]], dx)
    x_p, x_p_idx = _set_coords([[nx[0]-10, middle]], dx)
    f = ricker(freq, nt, dt, 0.05)

    model[x_p_idx[0, 0], x_p_idx[0, 1]] += dc

    expected = scattered_2d(x_r, x_s, x_p, dx, dt, c, dc, f)

    source, receiver_locations = _make_source_receiver(x_s_idx, x_r_idx, f)
    if propagator is None:
        propagator = Scalar2D
    if prop_kwargs is None:
        prop_kwargs = {}
    prop = propagator(model, dx, dt, source, **prop_kwargs)

    actual, _ = forward_model(prop, receiver_locations)

    return expected, actual.receivers.ravel()


def model_grad_const_1d(c=1500, dc=1, freq=25, dx=5, dt=0.0001, nx=100,
                        propagator=None, prop_kwargs=None):
    """Create a point scatterer model, and the gradient.
    """

    nt = int((3*nx*dx/c + 0.1)/dt)
    x_s, x_s_idx = _set_coords([[1]], dx)
    x_r, x_r_idx = _set_coords([[1]], dx)
    x_p, x_p_idx = _set_coords([[nx-20]], dx)
    f = ricker(freq, nt, dt, 0.05)

    model_init = np.ones(nx, dtype=np.float32) * c
    model_true = model_init.copy()
    model_true[x_p_idx] += dc

    expected = grad_1d(nx, x_r, x_s, x_p, dx, dt, c, dc, f)
    fwi_grad, true_grad = grad_1d_fd(model_true, model_init, x_r, x_s, dx, dt,
                                     dc, f, propagator, prop_kwargs)

    return expected, fwi_grad, true_grad


def model_grad_const_2d(c=1500, dc=1, freq=25, dx=5, dt=0.0001, nx=[20, 20],
                        propagator=None, prop_kwargs=None):
    """Create a point scatterer model, and the gradient.
    """

    nt = int((3*nx[0]*dx/c + 0.1)/dt)
    middle = int(nx[1]/2)
    x_s, x_s_idx = _set_coords([[1, middle]], dx)
    x_r, x_r_idx = _set_coords([[1, middle]], dx)
    x_p, x_p_idx = _set_coords([[nx[0]-5, middle]], dx)
    f = ricker(freq, nt, dt, 0.05)

    model_init = np.ones(nx, dtype=np.float32) * c
    model_true = model_init.copy()
    model_true[x_p_idx[0, 0], x_p_idx[0, 1]] += dc

    expected = grad_2d(nx, x_r, x_s, x_p, dx, dt, c, dc, f)
    fwi_grad, true_grad = grad_2d_fd(model_true, model_init, x_r, x_s, dx, dt,
                                     dc, f, propagator, prop_kwargs)

    return expected, fwi_grad, true_grad


def model_grad_rand_1d(c=2000, randc=100, dc=1, freq=25, dx=5, dt=0.0001,
                       nx=100, propagator=None, prop_kwargs=None):
    """Create a point scatterer model, and the gradient.
    """

    nt = int((3*nx*dx/c + 0.1)/dt)
    x_s, x_s_idx = _set_coords([[1]], dx)
    x_r, x_r_idx = _set_coords([[1]], dx)
    x_p, x_p_idx = _set_coords([[nx-20]], dx)
    f = ricker(freq, nt, dt, 0.05)

    model_init = (np.random.rand(nx).astype(np.float32) * randc) + c
    model_true = model_init.copy()
    model_true += np.random.rand(nx).astype(np.float32) * dc

    fwi_grad, true_grad = grad_1d_fd(model_true, model_init, x_r, x_s, dx, dt,
                                     dc, f, propagator, prop_kwargs)

    return fwi_grad, true_grad


def model_grad_rand_2d(c=2000, randc=100, dc=1, freq=25, dx=5, dt=0.0001,
                       nx=[20, 20], propagator=None, prop_kwargs=None):
    """Create a point scatterer model, and the gradient.
    """

    nt = int((3*nx[0]*dx/c + 0.1)/dt)
    middle = int(nx[1]/2)
    x_s, x_s_idx = _set_coords([[1, middle]], dx)
    x_r, x_r_idx = _set_coords([[1, middle]], dx)
    x_p, x_p_idx = _set_coords([[nx[0]-5, middle]], dx)
    f = ricker(freq, nt, dt, 0.05)

    model_init = (np.random.rand(nx[0], nx[1]).astype(np.float32) * randc) + c
    model_true = model_init.copy()
    model_true += np.random.rand(nx[0], nx[1]).astype(np.float32) * dc

    fwi_grad, true_grad = grad_2d_fd(model_true, model_init, x_r, x_s, dx, dt,
                                     dc, f, propagator, prop_kwargs)

    return fwi_grad, true_grad



