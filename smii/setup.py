#!/usr/bin/env python
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('smii', parent_package, top_path)
    config.add_subpackage('modeling')
    config.add_subpackage('imaging')
    config.add_subpackage('inversion')
    config.add_data_dir('test')
    #config.make_config_py()
    return config

if __name__ == "__main__":
    print('This is the wrong setup.py file to run')
