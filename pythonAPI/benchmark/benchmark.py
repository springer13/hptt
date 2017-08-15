import numpy as np
import hptt
import time
import argparse

perm = (1,0,2,4,3)
size = (32,96,30,40,5)
order = 'f'
alpha = 1.0
beta = 0.0
floatType = np.float32

parser = argparse.ArgumentParser(description='Benchmark for tensor transpositions using HPTT and NUMPY.')

parser.add_argument('--perm', metavar='perm', type=str, help="comma separated list of integers: permutation perm[i] = j means that A's j-th axis becomes B's i-th axis")
parser.add_argument('--size', metavar='size', type=str, help="comma separated list of integers representing the size of the input tensor A")
parser.add_argument('--rowMajor', action="store_true", help='Use row-major data layout (default: column-major)')
parser.add_argument('--floatType', metavar='floatType', type=str, help="float type can be either 'double' or 'float' (default)")
parser.add_argument('--alpha', type=float, help='alpha scalar (default: 1.0)')
parser.add_argument('--beta', type=float, help='beta scalar (default: 0.0)')
parser.add_argument('--clean', action="store_true", help='Clean output')
#parser.print_help()

args = parser.parse_args()

if( args.rowMajor ):
    order = 'c'

if( args.alpha):
    alpha = float(args.alpha)
if( args.beta):
    beta = float(args.beta)
if( args.floatType ):
    if( args.floatType == 'double' ):
        floatType = np.float64
if( args.perm ):
    perm = []
    for idx in args.perm.split(","):
        perm.append(int(idx))
if( args.size):
    size = []
    for idx in args.size.split(","):
        size.append(int(idx))
if( not args.clean ):
    print "Perm: ", perm
    print "Size: ", size
    print "flaotType", floatType
    print "alpha: ", alpha
    print "beta: ", beta
    print "Memory Layout", order
if( len(size) != len(perm) ):
    print "ERROR: size and perm are of different length"
    exit(-1)

sizeB = [size[i] for i in perm]
A = np.empty(size, order=order, dtype=np.float64)
B = np.empty(sizeB, order=order, dtype=np.float64)
hptt.randomNumaAwareInit(A)
hptt.randomNumaAwareInit(B)

Ma = np.random.rand(2500**2).astype('f')
Mb = np.random.rand(2500**2).astype('f')
timeHPTT = 1e100
for i in range(5):
   Mb = Ma *1.1 +  Mb #trash cache
   s = time.time()
   hptt.tensorTransposeAndUpdate(perm, alpha, A, beta, B)
   #B = hptt.tensorTranspose(perm,1.0,A)
   timeHPTT = min(timeHPTT, time.time() - s)

timeNP = 1e100
for i in range(5):
   Mb = Ma *1.1 +  Mb #trash cache
   s = time.time()
   B_ = np.transpose(A, perm).copy(order=order)

   timeNP = min(timeNP, time.time() - s)

if( not args.clean ):
    print "HPTT", A.size * 3 * A.itemsize / 1024. /1024. /1024. / timeHPTT, "GiB/s"
    print "numpy", A.size * 3 * A.itemsize / 1024. /1024. /1024. / timeNP, "GiB/s"
else:
    bwNumpy = A.size * 3 * A.itemsize / 1024. /1024. /1024. / timeNP
    bwHPTT = A.size * 3 * A.itemsize / 1024. /1024. /1024. / timeHPTT
    print "%.2f GiB/s %.2f GiB/s speedup: %.2f x" %(bwHPTT, bwNumpy, bwHPTT / bwNumpy)

OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'
if( beta == 0 and alpha == 1 and not hptt.equal(B, B_,1000) ):
    print B.shape
    print B_.shape
    print "validation:" + FAIL + " failed!!!" + ENDC
