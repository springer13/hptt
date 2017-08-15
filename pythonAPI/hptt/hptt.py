import numpy as np
import multiprocessing
import ctypes
from ctypes import cdll
import os
import random

HPTT_ROOT = ""
try:
    HPTT_ROOT = os.environ['HPTT_ROOT']
except:
    print "[HPTT] ERROR: HPTT_ROOT environment variable is not set. Point HPTT_ROOT to the folder which includes HPTT_ROOT/lib/libhptt.so"
    exit(-1)

# load HPTT library
HPTTlib = cdll.LoadLibrary(HPTT_ROOT+"/lib/libhptt.so")

def randomNumaAwareInit( A ):
    """ 
    initializes the passed numpy.ndarray (which have to be created with
    numpy.empty) and initializes it with random data in paralle such that the
    pages are equally distributed among the numa nodes 
    """
    HPTTlib.randomNumaAwareInit( ctypes.c_void_p(A.ctypes.data),
            ctypes.cast(A.ctypes.shape, ctypes.POINTER(ctypes.c_voidp)),
            ctypes.c_int32(A.ndim) )


def convertPermColumnMajor(perm):
    dim = len(perm)
    permR = np.empty(dim, dtype=np.int32)
    for i in xrange(dim):
        permR[i] = dim - perm[dim-i-1] - 1
    return permR

def tensorTranspose( perm, alpha, A, numThreads=-1):
    """ 
        This function computes the tensor transposition of A.
        The tensor transposition is of the form: 
           B[perm(0,1,2,...)] = alpha * A[0,1,2,...] + beta * B[perm(0,1,2,...)]

        where alpha and beta are scalors and A, and B correspond to arbitrary
        dimensional arrays (i.e., tensors). The dimensionality of A, and B
        depends on their indices (which need to be separated by commas).

        Parameter:
        -------------
        A: multi-dimensional numpy array
        alpha: scalar
        perm: tuple of ints. perm[i] = j means that A's j-th axis becomes B's i-th axis

        Example:
        ------------
        For instance, the tensor transposition B[m1,n1,m2] = 1.3 * A[m2,n1,m1];
        would be represented as: tensorTranspose([2,1,0], 1.3, A).

    """
    if( numThreads < 0 ):
        numThreads = max(1, multiprocessing.cpu_count()/2)
    order = 'C'
    if( A.flags['F_CONTIGUOUS'] ):
        order = 'F'
    Ashape = A.shape
    Bshape = [Ashape[i] for i in perm]
    B = np.empty(Bshape, dtype=A.dtype, order=order)
    if( not A.flags['F_CONTIGUOUS'] ):
        Ashape = Ashape[::-1]
        Bshape = Bshape[::-1]
        perm = convertPermColumnMajor( perm )

    permc = ctypes.cast((ctypes.c_int32 * len(perm))(*perm), ctypes.POINTER(ctypes.c_voidp))
    dataA = ctypes.c_void_p(A.ctypes.data)
    sizeA = ctypes.cast((ctypes.c_int32 * len(A.ctypes.shape))(*Ashape), ctypes.POINTER(ctypes.c_voidp))
    outerSizeA = sizeA

    dataB = ctypes.c_void_p(B.ctypes.data)
    sizeB = ctypes.cast((ctypes.c_int32 * len(B.ctypes.shape))(*Bshape), ctypes.POINTER(ctypes.c_voidp))
    outerSizeB = sizeB

    if( A.dtype == 'float32' ):
        HPTTlib.sTensorTranspose(permc, ctypes.c_int32(A.ndim),
                ctypes.c_float(alpha), dataA, sizeA, outerSizeA,
                ctypes.c_float(0.0),   dataB,        outerSizeB,
                ctypes.c_int32(numThreads))
    elif( A.dtype == 'float64' ):
        HPTTlib.dTensorTranspose(permc, ctypes.c_int32(A.ndim),
                ctypes.c_double(alpha), dataA, sizeA, outerSizeA,
                ctypes.c_double(0.0),   dataB,        outerSizeB,
                ctypes.c_int32(numThreads))
    elif( A.dtype == 'complex64' ):
        print "Data type not yet supported."
    elif( A.dtype == 'complex128' ):
        print "Data type not yet supported."
    else:
        raise ValueError('[HPTT] ERROR: unkown datatype')

    return B


def tensorTransposeAndUpdate( perm, alpha, A, beta, B, numThreads=-1):
    """ 
        This function computes the tensor transposition of A.
        The tensor transposition is of the form: 
           B[perm(0,1,2,...)] = alpha * A[0,1,2,...] + beta * B[perm(0,1,2,...)]

        where alpha and beta are scalors and A, and B correspond to arbitrary
        dimensional arrays (i.e., tensors). The dimensionality of A, and B
        depends on their indices (which need to be separated by commas).

        Parameter:
        -------------
        A [in]: multi-dimensional numpy array
        alpha: scalar
        B [out]: multi-dimensional numpy array
        beta: scalar
        perm: tuple of ints. perm[i] = j means that A's j-th axis becomes B's i-th axis

        Example:
        ------------
        For instance, the tensor transposition B[m1,n1,m2] = 1.3 * A[m2,n1,m1] + 1.2 * B[m1,n1,m2];
        would be represented as: tensorTranspose([2,1,0], 1.3, A, 1.2, B).
    """

    if(A.dtype != B.dtype ):
        raise ValueError('ERROR: the data type of A and B does not match.')

    if( numThreads < 0 ):
        numThreads = max(1, multiprocessing.cpu_count()/2)

    order = 'C'
    if( A.flags['F_CONTIGUOUS'] ):
        order = 'F'
    Ashape = A.shape
    Bshape = B.shape
    if( not A.flags['F_CONTIGUOUS'] ):
        Ashape = Ashape[::-1]
        Bshape = B.shape[::-1]
        perm = convertPermColumnMajor( perm )

    permc = ctypes.cast((ctypes.c_int32 * len(perm))(*perm), ctypes.POINTER(ctypes.c_voidp))
    dataA = ctypes.c_void_p(A.ctypes.data)
    sizeA = ctypes.cast((ctypes.c_int32 * len(A.ctypes.shape))(*Ashape), ctypes.POINTER(ctypes.c_voidp))
    outerSizeA = sizeA
    dataB = ctypes.c_void_p(B.ctypes.data)
    sizeB = ctypes.cast((ctypes.c_int32 * len(B.ctypes.shape))(*Bshape), ctypes.POINTER(ctypes.c_voidp))
    outerSizeB = sizeB

    if( A.dtype == 'float32' ):
        HPTTlib.sTensorTranspose(permc, ctypes.c_int32(A.ndim),
                ctypes.c_float(alpha), dataA, sizeA, outerSizeA,
                ctypes.c_float(beta),  dataB,        outerSizeB,
                ctypes.c_int32(numThreads))
    elif( A.dtype == 'float64' ):
        HPTTlib.dTensorTranspose(permc, ctypes.c_int32(A.ndim),
                ctypes.c_double(alpha), dataA, sizeA, outerSizeA,
                ctypes.c_double(beta),  dataB,        outerSizeB,
                ctypes.c_int32(numThreads))
    elif( A.dtype == 'complex64' ):
        print "Data type not yet supported."
    elif( A.dtype == 'complex128' ):
        print "Data type not yet supported."
    else:
        raise ValueError('[HPTT] ERROR: unkown datatype')

def equal(A, B, numSamples=-1):
    """ Ensures that alle elements of A and B are pretty much equal (due to limited machine precision) 

    Parameter:
    numSamples: number of random samples to compare (-1: all). This values is used to approximate this function and speed the result up."
    """
    threshold = 1e-4
    A = np.reshape(A, A.size)
    B = np.reshape(B, B.size)
    error = 0
    samples = range(A.size)
    if( numSamples != -1 ):
        samples = random.sample(samples, numSamples)

    for i in samples:
      Aabs = abs(A[i]);
      Babs = abs(B[i]);
      absmax = max(Aabs, Babs);
      diff = Aabs - Babs;
      if( diff < 0 ):
          diff *= -1
      if(diff > 0):
         relError = diff / absmax;
         if(relError > 4e-5 and min(Aabs,Babs) > threshold ):
            error += 1
    return error == 0
