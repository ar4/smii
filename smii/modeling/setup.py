#!/usr/bin/env python
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('modeling', parent_package, top_path)
    config.add_subpackage('propagators')
    config.add_subpackage('store_wavefield')
    config.add_subpackage('record_receivers')
    config.add_subpackage('wavelets')
    config.add_subpackage('imaging_condition')
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
