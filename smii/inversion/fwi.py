import numpy as np
import scipy.optimize
from smii.modeling.propagators.propagators import (Scalar1D, Scalar2D)
from smii.modeling.imaging_condition.zero_lag_xcor import ZeroLagXcor
from smii.modeling.forward_model import forward_model
from smii.modeling.backpropagate import backpropagate

def fwi(model, dataset, dx, dt, maxiter=1, propagator=None,
        loss_file=None, true_model=None):

    if propagator is None:
        if model.ndim == 1:
            propagator = Scalar1D
        elif model.ndim == 2:
            propagator = Scalar2D
        else:
            raise ValueError
    
    if loss_file is not None:
        loss_file = open(loss_file, 'w')
        loss_file.write('data_cost, model_cost\n')
    
    # Optimisation
    opt = scipy.optimize.minimize(costjac,
                                  model.ravel(),
                                  args=(dataset, dx, dt, propagator,
                                        model.shape, loss_file, true_model),
                                  jac=True,
                                  bounds=[(1450, 5000)] * len(model.ravel()),
                                  options={'maxiter': maxiter,
                                           'disp': True},
                                  tol=1e-20,
                                  method='TNC')

    if loss_file is not None:
        loss_file.close()
    return opt


def costjac(x, dataset, dx, dt, propagator, shape, loss_file=None,
            true_model=None, compute_grad=True, scale_cost=False):

    model = x.reshape(shape)
    
    imaging_condition = ZeroLagXcor
    
    jac = np.zeros(model.shape, np.float32)
    total_cost = 0.0
    
    for source, receivers in dataset:
        # Forward
        prop = propagator(model, dx, dt, source, pml_width=30)
        recorded_receivers, stored_source_wavefield = \
                forward_model(prop, receivers['locations'],
                              record_receivers=True,
                              store_wavefield=compute_grad)
    
        residual = {}
        residual['amplitude'] = \
                recorded_receivers.receivers - receivers['amplitude']
        residual['locations'] = receivers['locations'].copy()
        if scale_cost: # scale residual by time**2
            nt = residual['amplitude'].shape[1]
            cost = float(dt * 0.5
                         * np.linalg.norm((np.arange(0, nt*dt, dt)**2
                                           * residual['amplitude']).ravel())**2)
        else:
            cost = float(dt * 0.5
                         * np.linalg.norm(residual['amplitude'].ravel())**2)
        total_cost += cost
    
        if compute_grad:
            # Calculate second time derivative of source
            _second_time_derivative(stored_source_wavefield, model, dt)
        
            # Backpropagation
            residual['amplitude'] = residual['amplitude'][:, ::-1]
            prop = propagator(model, dx, dt, residual, pml_width=30)
            shot_jac = imaging_condition(prop.geometry.shape, dt)
            jac += backpropagate(prop, shot_jac,
                                 stored_source_wavefield)
    
    jac *= 2 / model**3

    model_cost = 0.0
    if true_model is not None:
        model_cost = np.linalg.norm(model - true_model)
    
    if loss_file is not None:
        loss_file.write('{}, {}\n'.format(total_cost, model_cost))

    return np.float64(total_cost), jac.ravel().astype(np.float64)


def _second_time_derivative(stored_source_wavefield, model, dt):
    """Calculate second time derivative of source."""
    wavefield = stored_source_wavefield.wavefield
    wavefield[:] = np.gradient(np.gradient(wavefield, axis=0), axis=0) / dt**2
