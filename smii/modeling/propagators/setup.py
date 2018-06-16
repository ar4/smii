#!/usr/bin/env python
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('propagators', parent_package, top_path)
    config.add_extension(name='libscalar1d', sources=['scalar1d.f90'], extra_f90_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-O3', '-std=f95', '-fopenmp'], extra_link_args=['-fopenmp'])
    config.add_extension(name='libscalar2d', sources=['scalar2d.f90'], extra_f90_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-O3', '-std=f95', '-fopenmp'], extra_link_args=['-fopenmp'])
    #config.add_extension(name='libscalar1d', sources=['scalar1d.f90'], extra_f90_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-O0', '-g', '-fbounds-check', '-std=f95', '-fopenmp'], extra_link_args=['-fopenmp'])
    #config.add_extension(name='libscalar2d', sources=['scalar2d.f90'], extra_f90_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-O0', '-g', '-fbounds-check', '-std=f95', '-fopenmp'], extra_link_args=['-fopenmp'])

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
