import numpy as np
import psutil
import ctypes
from ctypes import cdll
import os
import random


if 'OMP_NUM_THREADS' in os.environ:
    DEFAULT_THREADS = int(os.environ['OMP_NUM_THREADS'])
else:
    DEFAULT_THREADS = psutil.cpu_count(logical=False)

try:
    HPTT_ROOT = os.environ['HPTT_ROOT']
except KeyError:
    try:
        import configparser
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'hptt.cfg'))
        HPTT_ROOT = config['lib']['HPTT_ROOT']
    except KeyError:
        raise OSError("[HPTT] ERROR: 'libhptt.so' can't be found. 'HPTT_ROOT' "
                      "is neither set as an environment variable or specified "
                      "in the config file. It should point to the folder which"
                      " includes '$HPTT_ROOT/lib/libhptt.so'.")


HPTTlib = cdll.LoadLibrary(os.path.join(HPTT_ROOT, "lib", "libhptt.so"))


def randomNumaAwareInit( A ):
    """
    initializes the passed numpy.ndarray (which have to be created with
    numpy.empty) and initializes it with random data in paralle such that the
    pages are equally distributed among the numa nodes
    """
    HPTTlib.randomNumaAwareInit( ctypes.c_void_p(A.ctypes.data),
            ctypes.cast(A.ctypes.shape, ctypes.POINTER(ctypes.c_voidp)),
            ctypes.c_int32(A.ndim) )


def checkContiguous(X):
    try:
        useRowMajor, order = {
            (True, False): (1, 'C'),
            (False, True): (0, 'F'),
        }[X.flags['C_CONTIGUOUS'], X.flags['F_CONTIGUOUS']]
    except KeyError:
        raise ValueError("Tensor is neither 'C' or 'F' contiguous.")

    return useRowMajor, order


def tensorTransposeAndUpdate(perm, alpha, A, beta, B, numThreads=-1):
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
    if A.dtype != B.dtype:
        raise ValueError('ERROR: the data type of A and B does not match.')

    useRowMajor, order = checkContiguous(A)
    if checkContiguous(B) != (useRowMajor, order):
        raise ValueError("Tensors do not have matching C or F orders.")

    if numThreads < 0:
        numThreads = max(1, DEFAULT_THREADS)

    # setup A ctypes
    Ashape = A.shape
    dataA = ctypes.c_void_p(A.ctypes.data)
    sizeA = ctypes.cast((ctypes.c_int32 * len(A.ctypes.shape))(*Ashape),
                        ctypes.POINTER(ctypes.c_voidp))
    outerSizeA = sizeA

    # setup B ctypes
    Bshape = B.shape
    dataB = ctypes.c_void_p(B.ctypes.data)
    sizeB = ctypes.cast((ctypes.c_int32 * len(B.ctypes.shape))(*Bshape),
                        ctypes.POINTER(ctypes.c_voidp))
    outerSizeB = sizeB

    # setup perm ctypes
    permc = ctypes.cast((ctypes.c_int32 * len(perm))(*perm),
                        ctypes.POINTER(ctypes.c_voidp))

    # dispatch to the correct dtype function
    try:
        tranpose_fn, scalar_fn = {
            'float32': (HPTTlib.sTensorTranspose, ctypes.c_float),
            'float64': (HPTTlib.dTensorTranspose, ctypes.c_double),
            'complex64': (HPTTlib.cTensorTranspose, ctypes.c_float),
            'complex128': (HPTTlib.zTensorTranspose, ctypes.c_double),
        }[str(A.dtype)]
    except KeyError:
        raise ValueError("Unsupported dtype: {}.".format(A.dtype))

    # tranpose!
    tranpose_fn(permc, ctypes.c_int32(A.ndim),
                scalar_fn(alpha), dataA, sizeA, outerSizeA,
                scalar_fn(0.0), dataB, outerSizeB,
                ctypes.c_int32(numThreads), ctypes.c_int32(useRowMajor))


def tensorTranspose(perm, alpha, A, numThreads=-1):
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
    order = 'C' if A.flags['C_CONTIGUOUS'] else 'F'
    B = np.empty([A.shape[i] for i in perm], dtype=A.dtype, order=order)

    tensorTransposeAndUpdate(perm, alpha, A, 0.0, B, numThreads=numThreads)

    return B


def transpose(a, axes=None):
    """Drop-in for ``numpy.transpose``. Permute the dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axes : list of ints, optional
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.

    Returns
    -------
    p : ndarray
        ``a`` with its axes permuted.
    """
    if axes is None:
        axes = reversed(range(a.ndim))

    return tensorTranspose(axes, 1.0, a)


def equal(A, B, numSamples=-1):
    """ Ensures that alle elements of A and B are pretty much equal (due to limited machine precision)

    Parameter:
    numSamples: number of random samples to compare (-1: all). This values is used to approximate this function and speed the result up."
    """
    threshold = 1e-4
    A = np.reshape(A, A.size)
    B = np.reshape(B, B.size)
    error = 0
    samples = list(range(A.size))
    if( numSamples != -1 ):
        samples = random.sample(samples, min(A.size,numSamples))

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
