#!/usr/bin/env python
from distutils.core import setup
import os

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'

setup(name="hptt",
        version="1.0.0",
        description="High-Performance Tensor Transposition",
        author="Paul Springer",
        author_email="springer@aices.rwth-aachen.de",
        packages=["hptt"]
        )

print ""
output = "# "+ FAIL + "IMPORTANT"+ENDC+": execute 'export HPTT_ROOT=%s/../' #"%(os.path.dirname(os.path.realpath(__file__)))
print '#'*(len(output)-2*len(FAIL)+1)
print output
print '#'*(len(output)-2*len(FAIL)+1)
print ""
