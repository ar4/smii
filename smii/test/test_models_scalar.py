"""Create constant and point scatterer models."""
import numpy as np
from smii.test.models_scalar import (model_direct_1d, model_direct_2d,
                                     model_scatter_1d, model_scatter_2d)
from smii.modeling.propagators.propagators import Scalar1D as Scalar1Dcpu
from smii.modeling.propagators.propagators import Scalar2D as Scalar2Dcpu
from smii.modeling.propagators.scalar1d_gpu import Scalar1D as Scalar1Dgpu
from smii.modeling.propagators.scalar2d_gpu import Scalar2D as Scalar2Dgpu

def test_direct_1d_cpu(cpuprop_kwargs):
    expected, actual = model_direct_1d(propagator=Scalar1Dcpu,
                                       prop_kwargs=cpuprop_kwargs)
    assert np.linalg.norm(expected - actual) < 14.5


def test_direct_1d_gpu(gpuprop_kwargs):
    expected, actual = model_direct_1d(propagator=Scalar1Dgpu,
                                       prop_kwargs=gpuprop_kwargs)
    assert np.linalg.norm(expected - actual) < 14.5


def test_direct_2d_cpu(cpuprop_kwargs):
    expected, actual = model_direct_2d(propagator=Scalar2Dcpu,
                                       prop_kwargs=cpuprop_kwargs)
    assert np.linalg.norm(expected - actual) < 0.6


def test_direct_2d_gpu(gpuprop_kwargs):
    expected, actual = model_direct_2d(propagator=Scalar2Dgpu,
                                       prop_kwargs=gpuprop_kwargs)
    assert np.linalg.norm(expected - actual) < 0.6


def test_scatter_1d_cpu(cpuprop_kwargs):
    expected, actual = model_scatter_1d(propagator=Scalar1Dcpu,
                                        prop_kwargs=cpuprop_kwargs)
    t = 2000
    assert np.linalg.norm(expected[t:] - actual[t:]) < 1.1


def test_scatter_1d_gpu(gpuprop_kwargs):
    expected, actual = model_scatter_1d(propagator=Scalar1Dgpu,
                                        prop_kwargs=gpuprop_kwargs)
    t = 2000
    assert np.linalg.norm(expected[t:] - actual[t:]) < 1.1


def test_scatter_2d_cpu(cpuprop_kwargs):
    expected, actual = model_scatter_2d(propagator=Scalar2Dcpu,
                                        prop_kwargs=cpuprop_kwargs)
    t = 2000
    assert np.linalg.norm(expected[t:] - actual[t:]) < 0.081


def test_scatter_2d_gpu(gpuprop_kwargs):
    expected, actual = model_scatter_2d(propagator=Scalar2Dgpu,
                                        prop_kwargs=gpuprop_kwargs)
    t = 2000
    assert np.linalg.norm(expected[t:] - actual[t:]) < 0.081
