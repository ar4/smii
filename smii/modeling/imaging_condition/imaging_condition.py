import numpy as np

class ImagingCondition(object):
    def __init__(self, shape, dt):
        self.image = np.zeros(shape, np.float32)
        self.dt = dt

    def add(self, source_wavefield, receiver_wavefield):
        raise NotImplementedError
