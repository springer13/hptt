# High-Performance Tensor Transpose library#

The High-Performance Tensor Transpose (hptt) C++ library is a prototype based on the [Tensor Transpose Compiler (TTC)](https://github.com/HPAC/TTC).


# Key Features
--------------

* Multi-threading support
* Explicit vectorization
* Auto-tuning (akin to FFTW)

Keep in mind that this is an early prototype; a fully functional version will be
available shortly.


# Install
---------

1. Clone the repository into a desired directory and change to that location:

    git clone https://github.com/springer13/hptt.git
    cd hptt

2. Compiler HPTT:

   make

This should create 'libhptt.so' insdide the lib folder.


# Getting Started
-----------------

Please have a look at the provided benchmark.cpp.

In general HPTT is used as follows:

    // allocate tensors
    float A* = ...
    float B* = ...

    // specify permutation and size
    int dim = 6;
    int perm[dim] = {5,2,0,4,1,3};
    int size[dim] = {48,28,48,28,28};

    // create a plan
    hptt::Transpose transpose( size, perm, NULL, NULL, dim, A, alpha, B, beta, hptt::MEASURE, numThreads );
    transpose.createPlan();

    // execute the transposition
    transpose.execute();

# Requirements
--------------

You must have a working C++ compiler. I have tested HPTT with:

* Intel's ICC (>= 15.0)

# Benchmark
-----------

The benchmark is the same as the original TTC benchmark [benchmark for tensor transpositions](https://github.com/HPAC/TTC/blob/master/benchmark/benchmark.py).


# Citation
-----------

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
