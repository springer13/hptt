# High-Performance Tensor Transpose library #

The High-Performance Tensor Transpose (hptt) C++ library is a prototype based on the [Tensor Transpose Compiler (TTC)](https://github.com/HPAC/TTC).


# Key Features

* Multi-threading support
* Explicit vectorization
* Auto-tuning (akin to FFTW)
    * Loop order
    * Parallelization
* Multi architecture support
    * Explicitly vectorized kernels for (AVX and ARM)
* Support float, double, complex and double complex data types

Keep in mind that this is an early prototype; a fully functional version will be
available shortly.


# Install

Clone the repository into a desired directory and change to that location:

    git clone https://github.com/springer13/hptt.git
    cd hptt
    export CXX=<desired compiler>

Now you have several options to build the desired version of the library:

    make avx
    make arm
    make scalar

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

# Requirements

You must have a working C++ compiler with c++11 support. I have tested HPTT with:

* Intel's ICPC 15.0.3, 16.0.3, 17.0.2
* GNU g++ 5.4, 6.2, 6.3

# Benchmark

The benchmark is the same as the original TTC benchmark [benchmark for tensor transpositions](https://github.com/HPAC/TTC/blob/master/benchmark/benchmark.py).

You can compile the benchmark via:

    cd benchmark
    make

Before running the benchmark, please modify the number of threads and the thread
affinity within the benchmark.sh file. To run the benchmark just use:

    ./benshmark.sh

This will create hptt_benchmark.dat file containing all the runtime information
of HPTT and the reference implementation.

# Current Limitations

* The sizes of the stride-1 indices currently needs to be a
  multiple of the vector-width. This can be easily averted ... (TODO)

# TODOs

* Add scalar remainder function in order to handle all sizes
* Add explicit vectorization for IBM power
* Add explicit vectorization for complex types

# Citation

In case you want refer to TTC as part of a research paper, please cite the following
article [(pdf)](http://arxiv.org/abs/1603.02297):
```
@article{ttc2016a,
   author      = {Paul Springer and Jeff R. Hammond and Paolo Bientinesi},
   title       = {{TTC}: TTC: A high-performance Compiler for Tensor Transpositions},
   archivePrefix = "arXiv",
   eprint = {1603.02297},
   primaryClass = "quant-ph",
   journal     = {CoRR},
   year        = {2016},
   issue_date  = {March 2016},
   url         = {http://arxiv.org/abs/1603.02297}
}
``` 
