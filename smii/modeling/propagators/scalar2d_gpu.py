import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from smii.modeling.propagators.pycuda import PycudaPropagator

class Scalar2D(PycudaPropagator):
    def __init__(self, model, dx, source_dt, sources, pml_width=None,
                 nvcc_options=None):

        source = """
__constant__ float fd1_d[2];
__constant__ float fd2_d[3];

__global__ void step_d(const float * const wfc,
                float * const wfp,
                const float * const phiz,
                float * const phizp,
                const float * const phix,
                float * const phixp,
                const float * const sigmaz,
                const float * const sigmax,
                const float * const model2_dt2,
                const float dt,
                const int nb,
                const int nz,
                const int nx)
{
        int zblocks_per_shot = gridDim.y / nb;
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int z = blockDim.y * (blockIdx.y % zblocks_per_shot) + threadIdx.y;
        int b = blockIdx.y / zblocks_per_shot;
        int i = z * nx + x;
        int bi = b * nz * nx + i;
        float lap;
        float wfc_x;
        float wfc_z;
        float phix_x;
        float phiz_z;
        bool in_domain = (x > 1) && (x < nx - 2)
                && (z > 1) && (z < nz - 2);

        if (in_domain)
        {
                lap = (fd2_d[0] * wfc[bi] +
                                fd2_d[1] *
                                (wfc[bi + 1] +
                                 wfc[bi - 1] +
                                 wfc[bi + nx] +
                                 wfc[bi - nx]) +
                                fd2_d[2] *
                                (wfc[bi + 2] +
                                 wfc[bi - 2] +
                                 wfc[bi + 2 * nx] +
                                 wfc[bi - 2 * nx]));

                wfc_x = (fd1_d[0] *
                                (wfc[bi + 1] -
                                 wfc[bi - 1]) + 
                                 fd1_d[1] * 
                                 (wfc[bi + 2] -
                                 wfc[bi - 2]));

                wfc_z = (fd1_d[0] *
                                (wfc[bi + nx] -
                                 wfc[bi - nx]) + 
                                 fd1_d[1] * 
                                 (wfc[bi + 2 * nx] -
                                 wfc[bi - 2 * nx]));

                phix_x = (fd1_d[0] *
                                (phix[bi + 1] -
                                 phix[bi - 1]) + 
                                 fd1_d[1] * 
                                 (phix[bi + 2] -
                                 phix[bi - 2]));

                phiz_z = (fd1_d[0] *
                                (phiz[bi + nx] -
                                 phiz[bi - nx]) + 
                                 fd1_d[1] * 
                                 (phiz[bi + 2 * nx] -
                                 phiz[bi - 2 * nx]));

                wfp[bi] = 1 / (1 + dt * (sigmaz[z] + sigmax[x]) / 2) *
                        (model2_dt2[i] * (lap + phix_x + phiz_z) +
                        dt * (sigmaz[z] + sigmax[x]) * wfp[bi] / 2 +
                        (2 * wfc[bi] - wfp[bi]) -
                        dt * dt * sigmaz[z] * sigmax[x] * wfc[bi]);

                phizp[bi] = phiz[bi] -
                        dt * (sigmaz[z] * phiz[bi] +
                        (sigmaz[z] - sigmax[x]) * wfc_z);
                phixp[bi] = phix[bi] -
                        dt * (sigmax[x] * phix[bi] +
                        (sigmax[x] - sigmaz[z]) * wfc_x);

        }
}

__global__ void add_sources_d(float * const wfp,
                const float * const model2_dt2,
                const float * const source_amplitude,
                const int * const sources_loc,
                const int step,
                const int nz,
                const int nx,
                const int nt, const int ns)
{
        int s = threadIdx.x;
        int b = blockIdx.x;
        int i = sources_loc[b * ns * 2 + s * 2 + 0] * nx +
            sources_loc[b * ns * 2 + s * 2 + 1];
        int bi = b * nz * nx + i;

        wfp[bi] += source_amplitude[b * ns * nt + s * nt + step] *
            model2_dt2[i];
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
        super(Scalar2D, self).__init__(jitfunc1, jitfunc2, fd1_d, fd2_d,
                                       model, dx, source_dt, sources,
                                       pad_width, pml_width=pml_width)

    def step(self, num_steps=1):

        for it in range(num_steps):
            for inner_it in range(self.timestep.step_ratio):
                self.jitfunc1(self.wavefield.current_gpu,
                              self.wavefield.previous_gpu,
                              self.pml.phi[0].current_gpu,
                              self.pml.phi[0].previous_gpu,
                              self.pml.phi[1].current_gpu,
                              self.pml.phi[1].previous_gpu,
                              self.pml.sigma_gpu[0],
                              self.pml.sigma_gpu[1],
                              self.model.padded_property_gpu['vp2dt2'],
                              np.float32(self.timestep.dt),
                              np.int32(self.sources.num_shots),
                              np.int32(self.geometry.model_shape_padded[0]),
                              np.int32(self.geometry.model_shape_padded[1]),
                              grid=self.griddim, block=self.blockdim)
                self.jitfunc2(self.wavefield.previous_gpu,
                              self.model.padded_property_gpu['vp2dt2'],
                              self.sources.amplitude_gpu,
                              self.sources.padded_locations_gpu,
                              np.int32(self.timestep.step_idx),
                              np.int32(self.geometry.model_shape_padded[0]),
                              np.int32(self.geometry.model_shape_padded[1]),
                              np.int32(self.timestep.num_steps),
                              np.int32(self.sources.num_sources_per_shot),
                              grid=(self.sources.num_shots, 1),
                              block=(self.sources.num_sources_per_shot, 1, 1))
                self.wavefield.current_gpu, self.wavefield.previous_gpu = \
                    self.wavefield.previous_gpu, self.wavefield.current_gpu
                self.pml.phi[0].current_gpu, self.pml.phi[0].previous_gpu = \
                    self.pml.phi[0].previous_gpu, self.pml.phi[0].current_gpu
                self.pml.phi[1].current_gpu, self.pml.phi[1].previous_gpu = \
                    self.pml.phi[1].previous_gpu, self.pml.phi[1].current_gpu

            self.timestep.step_idx += 1

        drv.memcpy_dtoh(self.wavefield.current, self.wavefield.current_gpu)

        return self.wavefield.inner
