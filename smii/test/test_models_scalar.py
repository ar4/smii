"""Create constant and point scatterer models."""
import numpy as np
from smii.test.models_scalar import (model_direct_1d, model_direct_2d,
                                     model_scatter_1d, model_scatter_2d)

def test_direct_1d():
    expected, actual = model_direct_1d()
    #assert np.allclose(expected, actual, atol=0.21)
    assert np.linalg.norm(expected - actual) < 14.5


def test_direct_2d():
    expected, actual = model_direct_2d()
    assert np.linalg.norm(expected - actual) < 0.6


def test_scatter_1d():
    expected, actual = model_scatter_1d()
    t = 2000
    assert np.linalg.norm(expected[t:] - actual[t:]) < 1.1


def test_scatter_2d():
    expected, actual = model_scatter_2d()
    t = 2000
    assert np.linalg.norm(expected[t:] - actual[t:]) < 0.08
