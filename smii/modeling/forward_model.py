from smii.modeling.store_wavefield.in_memory import InMemory
from smii.modeling.record_receivers.nearest_cell import NearestCell

def forward_model(propagator, receiver_locations,
                  record_receivers=True,
                  store_wavefield=False):

        recorder = _record_receivers(propagator, receiver_locations,
                                     record_receivers)
        storer = _store_wavefield(propagator, store_wavefield)

        num_steps = propagator.timestep.num_steps
        for step in range(1, num_steps):

            wavefield = propagator.step()

            if record_receivers:
                recorder.record(wavefield, step)

            if store_wavefield:
                storer.store(wavefield, step)

        return recorder, storer


def _record_receivers(propagator, receiver_locations, record_receivers):

        if record_receivers is True:
            record_receivers = NearestCell

        if record_receivers:
            return record_receivers(propagator, receiver_locations)
 

def _store_wavefield(propagator, store_wavefield):

        if store_wavefield is True:
            store_wavefield = InMemory

        if store_wavefield:
            return store_wavefield(propagator)
