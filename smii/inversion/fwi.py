import numpy as np
import scipy.optimize
from smii.modeling.propagators.propagators import (Scalar1D, Scalar2D)
from smii.modeling.imaging_condition.zero_lag_xcor import ZeroLagXcor
from smii.modeling.forward_model import forward_model
from smii.modeling.backpropagate import backpropagate

def fwi(model, dataset, dx, dt, propagator, maxiter=1,
        loss_file=None, true_model=None, bounds=None):
    """Update model using FWI to better match dataset.

    Arguments:
        model: An array containing wave speed.
        dataset: An iterable that returns a batch of sources
                 and receivers.
                 Both of these must be dictionaries that contain:
                    'amplitude': A [batch_size, num_per_shot, nt] array
                    'locations': A [batch_size, num_per_shot, ndim] array
        dx: A float specifying cell size.
        dt: A float specifying source time step size.
        propagator: The propagator class to use.
        maxiter: An int specifying the number of optimizer iterations to
                 perform. Optional - default 1.
        loss_file: The path to a file where the data and model cost will be
                   stored each time they are evaluated. Optional.
        true_model: An array containing the true wave speed model,
                    which will be used to calculate the model error for each
                    model considered. Optional.
        bounds: A tuple containing the minimum and maximum allowable wave
                speeds. Optional.

    Returns:
        The final model
    """
    
    if loss_file is not None:
        loss_file = open(loss_file, 'w')
        loss_file.write('data_cost, model_cost\n')

    if bounds is not None:
        bounds = [bounds] * len(model.ravel())
    
    # Optimisation
    opt = scipy.optimize.minimize(costjac,
                                  model.ravel(),
                                  args=(dataset, dx, dt, propagator,
                                        model.shape, loss_file, true_model),
                                  jac=True,
                                  bounds=bounds,
                                  options={'maxiter': maxiter,
                                           'disp': True},
                                  tol=1e-20,
                                  method='TNC')

    if loss_file is not None:
        loss_file.close()
    return opt.x


def costjac(x, dataset, dx, dt, propagator, shape, loss_file=None,
            true_model=None, compute_grad=True, prop_kwargs=None):

    model = x.reshape(shape)
    
    imaging_condition = ZeroLagXcor
    
    jac = np.zeros(model.shape, np.float32)
    total_cost = 0.0

    if prop_kwargs is None:
        prop_kwargs = {'pml_width': 30}
    
    for sources, receivers in dataset:
        # Forward
        prop = propagator(model, dx, dt, sources, **prop_kwargs)
        recorded_receivers, stored_source_wavefield = \
                forward_model(prop, receivers['locations'],
                              record_receivers=True,
                              store_wavefield=compute_grad)
    
        residual = {}
        residual['amplitude'] = \
                recorded_receivers.receivers - receivers['amplitude']
        residual['locations'] = receivers['locations'].copy()
        cost = float(dt * 0.5
                     * np.linalg.norm(residual['amplitude'].ravel())**2)
        total_cost += cost
    
        if compute_grad:
            # Calculate second time derivative of source
            _second_time_derivative(stored_source_wavefield, model, dt)
        
            # Backpropagation
            residual['amplitude'] = residual['amplitude'][..., ::-1]
            prop = propagator(model, dx, dt, residual, **prop_kwargs)
            batch_jac = imaging_condition(prop.geometry.propagation_shape, dt)
            jac += backpropagate(prop, batch_jac,
                                 stored_source_wavefield).sum(axis=0)
    
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
