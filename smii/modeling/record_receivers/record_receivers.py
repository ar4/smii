import numpy as np

class RecordReceivers(object):

    def __init__(self, propagator, receiver_locations):
        if receiver_locations is None:
            receiver_locations = np.array([], np.int)
        self.receiver_locations = receiver_locations
        num_steps = propagator.timestep.num_steps
        num_receivers = receiver_locations.shape[:-1]
        self.receivers = np.zeros(num_receivers + (num_steps, ), np.float32)
        self.step = 0

    def record(self, wavefield, step):
        pass
