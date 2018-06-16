class StoreWavefield(object):
    def __init__(self, propagator):
        raise NotImplementedError

    def store(self, wavefield, step):
        raise NotImplementedError

    def restore(self, step):
        raise NotImplementedError
