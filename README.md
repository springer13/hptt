# High-Performance Tensor Transpose library #

HPTT is a high-performance C++ library for out-of-place tensor transpositions of the general form: 

![hptt](https://github.com/springer13/hptt/blob/master/misc/equation.png)

where A and B respectively denote the input and output tensor;
<img src=https://github.com/springer13/hptt/blob/master/misc/pi.png height=16px/> represents the user-specified
transposition, and 
<img src=https://github.com/springer13/hptt/blob/master/misc/alpha.png height=14px/> and
<img src=https://github.com/springer13/hptt/blob/master/misc/beta.png height=16px/> being scalars
(i.e., setting <img src=https://github.com/springer13/hptt/blob/master/misc/beta.png height=16px/> != 0 enables the user to update the output tensor B).

# Key Features

* Multi-threading support
* Explicit vectorization
* Auto-tuning (akin to FFTW)
    * Loop order
    * Parallelization
* Multi architecture support
    * Explicitly vectorized kernels for (AVX and ARM)
* Supports float, double, complex and double complex data types
* Supports both column-major and row-major data layouts

HPTT now also offers C- and Python-interfaces (see below).

# Requirements

You must have a working C++ compiler with c++11 support. I have tested HPTT with:

* Intel's ICPC 15.0.3, 16.0.3, 17.0.2
* GNU g++ 5.4, 6.2, 6.3
* clang++ 3.8, 3.9


# Install

Clone the repository into a desired directory and change to that location:

    git clone https://github.com/springer13/hptt.git
    cd hptt
    export CXX=<desired compiler>

Now you have several options to build the desired version of the library:

    make avx
    make arm
    make scalar

Using CMake:
    mkdir build && cd build
    cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
    #Optionally one of [-DENABLE_ARM=ON -DENABLE_AVX=ON -DENABLE_IBM=ON]    

This should create 'libhptt.so' inside the ./lib folder.


# Getting Started

Please have a look at the provided benchmark.cpp.

In general HPTT is used as follows:

    #include <hptt.h>

    // allocate tensors
    float A* = ...
    float B* = ...

    // specify permutation and size
    int dim = 6;
    int perm[dim] = {5,2,0,4,1,3};
    int size[dim] = {48,28,48,28,28};

    // create a plan (shared_ptr)
    auto plan = hptt::create_plan( perm, dim, 
                                   alpha, A, size, NULL, 
                                   beta,  B, NULL, 
                                   hptt::ESTIMATE, numThreads);

    // execute the transposition
    plan->execute();

The example above does not use any auto-tuning, but solely relies on HPTT's
performance model. To active auto-tuning, please use hptt::MEASURE, or
hptt::PATIENT instead of hptt::ESTIMATE.


## C-Interface

HPTT also offeres a C-interface. This interface is less expressive than its C++
counter part since it does not expose control over the plan.

    void sTensorTranspose( const int *perm, const int dim,
            const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
            const float beta,        float *B,                   const int *outerSizeB, 
            const int numThreads, const int useRowMajor);

    void dTensorTranspose( const int *perm, const int dim,
            const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
            const double beta,        double *B,                   const int *outerSizeB, 
            const int numThreads, const int useRowMajor);
    ...

## Python-Interface

HPTT now also offers a python-interface. The functionality offered by HPTT is comparable to [numpy.transpose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html)
with the difference being that HPTT can also update the output tensor.

    tensorTransposeAndUpdate( perm, alpha, A, beta, B, numThreads=-1)

    tensorTranspose( perm, alpha, A, numThreads=-1)

See docstring for additional information.

Installation should be straight forward via:

    cd ./pythonAPI
    python setup.py install

At this point you should be able to import the 'hptt' package within your python scripts.

The python interface also offers support for:

* Single and double precision
* Column-major and row-major data layouts
* multi-threading support (HPTT by default utilizes all cores of a system)

### Python Benchmark

You can find an elaborate example under ./pythonAPI/benchmark/benchmark.py --help

* Multi-threaded 2x Intel Haswell-EP E5-2680 v3 (24 threads)
  * Comparison again [numpy.transpose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html)

![hptt](https://github.com/springer13/hptt/blob/master/misc/hptt_vs_numpy.png)

# Documentation

You can generate the doxygen documentation via

    make doc


# Benchmark

The benchmark is the same as the original TTC benchmark [benchmark for tensor transpositions](https://github.com/HPAC/TTC/blob/master/benchmark).

You can compile the benchmark via:

    cd benchmark
    make

Before running the benchmark, please modify the number of threads and the thread
affinity within the benchmark.sh file. To run the benchmark just use:

    ./benshmark.sh

This will create hptt_benchmark.dat file containing all the runtime information
of HPTT and the reference implementation.

# Performance Results

![hptt](https://github.com/springer13/hptt/blob/master/benchmark/bw.png)

See [(pdf)](https://arxiv.org/abs/1704.04374) for details.

# TODOs

* Add explicit vectorization for IBM power
* Add explicit vectorization for complex types


# Related Projects

* Shared-Memory Tensor Contractions: 
    * [TCL](https://github.com/springer13/tcl)
    * [TBLIS](https://github.com/devinamatthews/tblis)
* Distributed-Memory Tensor Contractions:
    * [CTF](https://github.com/cyclops-community/ctf)
    * [libtensor](https://github.com/epifanovsky/libtensor)
* Tensor network codes:
    * [ITensor](http://itensor.org/)
    * [Uni10](http://yingjerkao.github.io/uni10/)

# Citation

In case you want refer to HPTT as part of a research paper, please cite the following
article [(pdf)](https://arxiv.org/abs/1704.04374):
```
@inproceedings{hptt2017,
 author = {Springer, Paul and Su, Tong and Bientinesi, Paolo},
 title = {{HPTT}: {A} {H}igh-{P}erformance {T}ensor {T}ransposition {C}++ {L}ibrary},
 booktitle = {Proceedings of the 4th ACM SIGPLAN International Workshop on Libraries, Languages, and Compilers for Array Programming},
 series = {ARRAY 2017},
 year = {2017},
 isbn = {978-1-4503-5069-3},
 location = {Barcelona, Spain},
 pages = {56--62},
 numpages = {7},
 url = {http://doi.acm.org/10.1145/3091966.3091968},
 doi = {10.1145/3091966.3091968},
 acmid = {3091968},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {High-Performance Computing, autotuning, multidimensional transposition, tensor transposition, tensors, vectorization},
}
``` 
