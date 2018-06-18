import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from smii.modeling.propagators.pycuda import PycudaPropagator

class Scalar1D(PycudaPropagator):
    def __init__(self, model, dx, source_dt, sources, pml_width=None,
                 nvcc_options=None):

        source = """
__constant__ float fd1_d[2];
__constant__ float fd2_d[3];

__global__ void step_d(const float *const wfc,
                float *wfp,
                const float *const phix,
                float *phixp,
                const float *const sigmax,
                const float *const model2_dt2,
                const float dt,
                const int nx)
{
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.y;
        int bx = b * nx + x;
        float wfc_xx;
        float wfc_x;
        float phix_x;
        bool in_domain = (x > 1) && (x < nx - 2);

        if (in_domain)
        {
                wfc_xx = (fd2_d[0] * wfc[bx] +
                                fd2_d[1] *
                                (wfc[bx + 1] +
                                 wfc[bx - 1]) +
                                fd2_d[2] *
                                (wfc[bx + 2] +
                                 wfc[bx - 2]));

                wfc_x = (fd1_d[0] *
                                (wfc[bx + 1] -
                                 wfc[bx - 1]) + 
                                 fd1_d[1] * 
                                 (wfc[bx + 2] -
                                 wfc[bx - 2]));

                phix_x = (fd1_d[0] *
                                (phix[bx + 1] -
                                 phix[bx - 1]) + 
                                 fd1_d[1] * 
                                 (phix[bx + 2] -
                                 phix[bx - 2]));

                wfp[bx] = 1 / (1 + dt * sigmax[x] / 2) *
                        (model2_dt2[x] *
                        (wfc_xx + phix_x) +
                        dt * sigmax[x] * wfp[bx] / 2 +
                        (2 * wfc[bx] - wfp[bx]));

                phixp[bx] = phix[bx] -
                        dt * sigmax[x] * (wfc_x + phix[bx]);

        }
}

__global__ void add_sources_d(float *wfp,
                const float *const model2_dt2,
                const float *const source_amplitude,
                const int *const sources_x,
                const int step,
                const int nx,
                const int nt, const int ns)
{
        int s = threadIdx.x;
        int b = blockIdx.x;
        int x = sources_x[b * ns + s];
        int bx = b * nx + x;

        wfp[bx] += source_amplitude[b * ns * nt + s * nt + step] *
            model2_dt2[x];
}
"""

        if nvcc_options is None:
            nvcc_options = ['--restrict', '--use_fast_math', '-O3']

        mod = SourceModule(source, options=nvcc_options)

        jitfunc1 = mod.get_function('step_d')
        jitfunc2 = mod.get_function('add_sources_d')
        fd1_d = mod.get_global('fd1_d')[0]
        fd2_d = mod.get_global('fd2_d')[0]

        pad_width = 2
        super(Scalar1D, self).__init__(jitfunc1, jitfunc2, fd1_d, fd2_d,
                                       model, dx, source_dt, sources,
                                       pad_width, pml_width=pml_width)

    def step(self, num_steps=1):

        for it in range(num_steps):
            for inner_it in range(self.timestep.step_ratio):
                self.jitfunc1(self.wavefield.current_gpu,
                              self.wavefield.previous_gpu,
                              self.pml.phi[0].current_gpu,
                              self.pml.phi[0].previous_gpu,
                              self.pml.sigma_gpu[0],
                              self.model.padded_property_gpu['vp2dt2'],
                              np.float32(self.timestep.dt),
                              np.int32(self.geometry.model_shape_padded[0]),
                              grid=self.griddim, block=self.blockdim)
                self.jitfunc2(self.wavefield.previous_gpu,
                              self.model.padded_property_gpu['vp2dt2'],
                              self.sources.amplitude_gpu,
                              self.sources.padded_locations_gpu,
                              np.int32(self.timestep.step_idx),
                              np.int32(self.geometry.model_shape_padded[0]),
                              np.int32(self.timestep.num_steps),
                              np.int32(self.sources.num_sources_per_shot),
                              grid=(self.sources.num_shots, 1),
                              block=(self.sources.num_sources_per_shot, 1, 1))
                self.wavefield.current_gpu, self.wavefield.previous_gpu = \
                    self.wavefield.previous_gpu, self.wavefield.current_gpu
                self.pml.phi[0].current_gpu, self.pml.phi[0].previous_gpu = \
                    self.pml.phi[0].previous_gpu, self.pml.phi[0].current_gpu

            self.timestep.step_idx += 1

        drv.memcpy_dtoh(self.wavefield.current, self.wavefield.current_gpu)

        return self.wavefield.inner
