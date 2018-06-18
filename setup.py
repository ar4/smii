#!/usr/bin/env python
MAJOR               = 0
MINOR               = 0
MICRO               = 3
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
FULLVERSION = VERSION

def write_version_py(filename='smii/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'

version = full_version
"""

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION})
    finally:
        a.close()

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('smii')
    config.add_data_files(('smii', 'LICENSE'))
    config.add_data_files(('smii', 'README.md'))
    config.get_version('smii/version.py')
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    write_version_py()
    setup(configuration=configuration,
          author='Alan Richardson',
          author_email='alan@ausargeo.com',
          url='https://github.com/ar4/smii',
          name='smii',
          install_requires=['numpy', 'scipy', 'pycuda', 'pytest'])
