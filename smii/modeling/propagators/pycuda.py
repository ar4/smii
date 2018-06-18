import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from smii.modeling.propagators.propagators import ScalarPropagator


class PycudaPropagator(ScalarPropagator):
    """PyCUDA implementations."""
    def __init__(self, jitfunc1, jitfunc2, fd1_d, fd2_d, model, dx, source_dt,
                 sources, pad_width, pml_width=None):
        super(PycudaPropagator, self).__init__(model.astype(np.float32),
                                               np.float32(dx),
                                               np.float32(source_dt),
                                               sources,
                                               np.int32(pad_width),
                                               pml_width=pml_width)
        self.jitfunc1 = jitfunc1
        self.jitfunc2 = jitfunc2

        # allocate and copy model to GPU

        self.model.padded_property_gpu = {}
        self.model.padded_property_gpu['vp2dt2'] = \
                drv.mem_alloc(self.model.padded_property['vp2dt2'].nbytes)
        drv.memcpy_htod(self.model.padded_property_gpu['vp2dt2'],
                self.model.padded_property['vp2dt2'])

        # allocate and initialize wavefields
        self.wavefield.current_gpu = \
                drv.mem_alloc(self.wavefield.current.nbytes)
        drv.memset_d32(self.wavefield.current_gpu, 0,
                       self.wavefield.current.size)
        self.wavefield.previous_gpu = \
                drv.mem_alloc(self.wavefield.previous.nbytes)
        drv.memset_d32(self.wavefield.previous_gpu, 0,
                       self.wavefield.previous.size)

        # allocate and initialize PML arrays
        self.pml.sigma_gpu = []
        for dim in range(self.geometry.ndim):
            self.pml.phi[dim].current_gpu = \
                    drv.mem_alloc(self.pml.phi[dim].current.nbytes)
            drv.memset_d32(self.pml.phi[dim].current_gpu, 0,
                           self.pml.phi[dim].current.size)
            self.pml.phi[dim].previous_gpu = \
                    drv.mem_alloc(self.pml.phi[dim].previous.nbytes)
            drv.memset_d32(self.pml.phi[dim].previous_gpu, 0,
                           self.pml.phi[dim].previous.size)
            self.pml.sigma_gpu.append(drv.mem_alloc(self.pml.sigma[dim].nbytes))
            drv.memcpy_htod(self.pml.sigma_gpu[dim], self.pml.sigma[dim])

        # allocate and copy sources arrays
        self.sources.amplitude_gpu \
                = drv.mem_alloc(self.sources.amplitude.nbytes)
        drv.memcpy_htod(self.sources.amplitude_gpu,
                        self.sources.amplitude)

        self.sources.padded_locations_gpu \
                = drv.mem_alloc(self.sources.padded_locations.nbytes)
        drv.memcpy_htod(self.sources.padded_locations_gpu,
                        self.sources.padded_locations)


        # create and copy finite difference coeffs to constant memory
        self.fd1_d = fd1_d
        fd1 = np.array([8/12, -1/12], np.float32) / dx
        drv.memcpy_htod(self.fd1_d, fd1)

        self.fd2_d = fd2_d
        if self.geometry.ndim == 1:
            fd2 = np.array([-5/2, 4/3, -1/12], np.float32) / dx**2
        elif self.geometry.ndim == 2:
            fd2 = np.array([-10/2, 4/3, -1/12], np.float32) / dx**2
        drv.memcpy_htod(self.fd2_d, fd2)

        # set block and grid dimensions
        threadsperblockx = 32
        blockspergridx = ((self.geometry.propagation_shape_padded[-1]
                           + (threadsperblockx - 1))
                          // threadsperblockx)
        if self.geometry.ndim == 1:
            threadsperblockz = 1
            blockspergridz = self.sources.num_shots
        elif self.geometry.ndim == 2:
            threadsperblockz = 32
            blockspergridz = ((self.geometry.propagation_shape_padded[-2]
                               + (threadsperblockz - 1))
                              // threadsperblockz) * self.sources.num_shots

        self.griddim = int(blockspergridx), int(blockspergridz)
        self.blockdim = int(threadsperblockx), int(threadsperblockz), 1

    def finalise(self):
        del (self.model.padded_property_gpu['vp2dt2'],
             self.wavefield.current_gpu,
             self.wavefield.previous_gpu,
             self.sources.amplitude_gpu, self.sources.padded_locations_gpu,
             self.jitfunc1, self.jitfunc2)

        for dim in range(self.geometry.ndim):
            del(self.pml.sigma_gpu[dim], self.pml.phi[dim].current_gpu,
                self.pml.phi[dim].previous_gpu)
