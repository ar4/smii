import numpy as np
from smii.modeling.store_wavefield.store_wavefield import StoreWavefield

class InMemory(StoreWavefield):
    def __init__(self, propagator):
        self.wavefield = np.zeros(np.append(propagator.timestep.num_steps,
                                            propagator.geometry.shape),
                                  np.float32)

    def store(self, wavefield, step):
        self.wavefield[step] = wavefield

    def restore(self, step):
        return self.wavefield[step]
