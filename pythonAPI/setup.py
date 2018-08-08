#!/usr/bin/env python
from distutils.core import setup
import os

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'


def get_hptt_root():
    """Get the root hptt folder - prefer environment variable if is set, but
    otherwise default to the parent directory of the python module - the
    shared library will be checked for in any case.
    """
    from os.path import dirname, realpath, isfile, join
    hptt_default = dirname(dirname(realpath(__file__)))
    HPTT_ROOT = os.environ.get('HPTT_ROOT', hptt_default)
    if not isfile(join(HPTT_ROOT, 'lib', 'libhptt.so')):
        raise OSError("Could not find $HPTT_ROOT/lib/libhptt.so - please make "
                      "HPTT_ROOT points to a directory where the shared "
                      "library has been built.")
    return HPTT_ROOT


def write_config():
    """Write a config file with the default shared library location to be
    installed by setup.py.
    """
    from configparser import ConfigParser
    config = ConfigParser()
    config['lib'] = {'HPTT_ROOT': get_hptt_root()}
    with open(os.path.join("hptt", "hptt.cfg"), "w") as configfile:
        config.write(configfile)


write_config()

setup(
    name="hptt",
    version="1.0.0",
    description="High-Performance Tensor Transposition",
    author="Paul Springer",
    author_email="springer@aices.rwth-aachen.de",
    packages=["hptt"],
    package_data={'hptt': ['hptt.cfg']},
    install_requires=[
        'numpy',
        'psutil',
    ]
)

print("")
output = "# "+ FAIL + "IMPORTANT"+ENDC+": execute 'export HPTT_ROOT=%s/../' #"%(os.path.dirname(os.path.realpath(__file__)))
print('#'*(len(output)-2*len(FAIL)+1))
print(output)
print('#'*(len(output)-2*len(FAIL)+1))
print("")
