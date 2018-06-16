import numpy as np
from smii.modeling.record_receivers.record_receivers import RecordReceivers

class NearestCell(RecordReceivers):

    def __init__(self, propagator, receiver_locations):
        super(NearestCell, self).__init__(propagator, receiver_locations)
        self.receiver_locations = \
                np.rint(self.receiver_locations).astype(np.int)

    def record(self, wavefield, step=None):
        if step is None:
            step = self.step

        if self.receiver_locations.shape[-1] == 1:
            for shotidx in range(wavefield.shape[0]):
                self.receivers[shotidx, :, step] = \
                        wavefield[shotidx,
                                  self.receiver_locations[shotidx, :, 0]]

        elif self.receiver_locations.shape[-1] == 2:
            for shotidx in range(wavefield.shape[0]):
                self.receivers[shotidx, :, step] = \
                        wavefield[shotidx,
                                  self.receiver_locations[shotidx, :, 0],
                                  self.receiver_locations[shotidx, :, 1]]
        self.step = step + 1
