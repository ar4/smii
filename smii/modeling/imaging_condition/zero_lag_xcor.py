from smii.modeling.imaging_condition.imaging_condition import ImagingCondition

class ZeroLagXcor(ImagingCondition):
    def __init__(self, shape, dt):
        super(ZeroLagXcor, self).__init__(shape, dt)

    def add(self, source_wavefield, receiver_wavefield):
        self.image += source_wavefield * receiver_wavefield * self.dt
