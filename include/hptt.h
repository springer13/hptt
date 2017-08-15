/*
 *   Copyright (C) 2017  Paul Springer (springer@aices.rwth-aachen.de)
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * \mainpage High-Performance Tensor Transposition Library 
 *
 * \section intro Introduction
 * HPTT supports tensor transpositions of the general form:
 *   
 *       \f[ \mathcal{B}_{\pi(i_0,i_1,...,i_{d-1})} \gets \alpha * \mathcal{A}_{i_0,i_1,...,i_{d-1}} + \beta * \mathcal{B}_{\pi(i_0,i_1,...,i_{d-1})}, \f]
 * where \f$\alpha\f$ and \f$\beta\f$ are scalars and \f$\mathcal{A}\f$ and
 * \f$\mathcal{B}\f$ are d-dimensional tensors (i.e., multi-dimensional arrays).
 *
 * HPTT assumes a column-major data layout, thus indices are stored from left to right (e.g., \f$i_0\f$ is the stride-1 index in \f$\mathcal{A}_{i_0,i_1,...,i_{d-1}}\f$).
 *
 * \section features Key Features 
 * * Multi-threading support
 * * Explicit vectorization
 * * Auto-tuning (akin to FFTW)
 *     * Loop order
 *     * Parallelization
 * * Multi-architecture support
 *     * Explicitly vectorized kernels for (AVX and ARM)
 * * Support for float, double, complex and double complex data types
 * * Can operate on sub-tensors
 *
 * \section Requirements
 * 
 * You must have a working C++ compiler with c++11 support. I have tested HPTT with:
 * 
 * * Intel's ICPC 15.0.3, 16.0.3, 17.0.2
 * * GNU g++ 5.4, 6.2, 6.3
 * * clang++ 3.8, 3.9
 *
 * \section Install
 *
 * Clone the repository into a desired directory and change to that location:
 * 
 *     git clone https://github.com/springer13/hptt.git
 *     cd hptt
 *     export CXX=<desired compiler>
 * 
 * Now you have several options to build the desired version of the library:
 * 
 *     make avx
 *     make arm
 *     make scalar
 * 
 * This should create 'libhptt.so' inside the ./lib folder.
 * 
 * 
 * \section start Getting Started
 * 
 * In general HPTT is used as follows:
 * 
 *     #include <hptt.h>
 * 
 *     // allocate tensors
 *     float A* = ...
 *     float B* = ...
 * 
 *     // specify permutation and size
 *     int dim = 6;
 *     int perm[dim] = {5,2,0,4,1,3};
 *     int size[dim] = {48,28,48,28,28};
 * 
 *     // create a plan (shared_ptr)
 *     auto plan = hptt::create_plan( perm, dim, 
 *                                    alpha, A, size, NULL, 
 *                                    beta,  B, NULL, 
 *                                    hptt::ESTIMATE, numThreads);
 * 
 *     // execute the transposition
 *     plan->execute();
 * 
 * The example above does not use any auto-tuning, but solely relies on HPTT's
 * performance model. To active auto-tuning, please use hptt::MEASURE, or
 * hptt::PATIENT instead of hptt::ESTIMATE.
 *
 * Please refer to the hptt::Transpose class for additional information or to hptt::create_plan().
 *
 * An extensive example is provided here: ./benchmark/benchmark.cpp.
 * 
 * \section Benchmark
 * 
 * The benchmark is the same as the original TTC benchmark [benchmark for tensor transpositions](https://github.com/HPAC/TTC/blob/master/benchmark).
 * 
 * You can compile the benchmark via:
 * 
 *     cd benchmark
 *     make
 * 
 * Before running the benchmark, please modify the number of threads and the thread
 * affinity within the benchmark.sh file. To run the benchmark just use:
 * 
 *     ./benshmark.sh
 * 
 * This will create hptt_benchmark.dat file containing all the runtime information
 * of HPTT and the reference implementation.
 *
 * \section Citation
 * 
 * In case you want refer to HPTT as part of a research paper, please cite the following
 * article [(pdf)](https://arxiv.org/abs/1704.04374):
 * ```
 * @inproceedings{hptt2017,
 *    author = {Springer, Paul and Su, Tong and Bientinesi, Paolo},
 *    title = {{HPTT}: {A} {H}igh-{P}erformance {T}ensor {T}ransposition {C}++ {L}ibrary},
 *    booktitle = {Proceedings of the 4th ACM SIGPLAN International Workshop on Libraries, Languages, and Compilers for Array Programming},
 *    series = {ARRAY 2017},
 *    year = {2017},
 *    isbn = {978-1-4503-5069-3},
 *    location = {Barcelona, Spain},
 *    pages = {56--62},
 *    numpages = {7},
 *    url = {http://doi.acm.org/10.1145/3091966.3091968},
 *    doi = {10.1145/3091966.3091968},
 *    acmid = {3091968},
 *    publisher = {ACM},
 *    address = {New York, NY, USA},
 *    keywords = {High-Performance Computing, autotuning, multidimensional transposition, tensor transposition, tensors, vectorization},
 * }
 * ``` 
 * 
 */

#pragma once

#include <vector>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif


#include "transpose.h"


namespace hptt {

/**
 * \brief Creates a Tensor Transposition plan 
 *
 * A tensor transposition plan is a data structure that encodes the execution of the tensor transposition.
 * HPTT supports tensor transpositions of the form: 
 * \f[ B_{\pi(i_0,i_1,...)} = \alpha * A_{i_0,i_1,...} + \beta * B_{\pi(i_0,i_1,...)}. \f]
 * The plan can be reused over several transpositions.
 *
 * \param[in] perm dim-dimensional array representing the permutation of the indices. 
 *                 * For instance, perm[] = {1,0,2} denotes the following transposition: \f$B_{i1,i0,i2} \gets A_{i0,i1,i2}\f$.
 * \param[in] dim Dimensionality of the tensors
 * \param[in] alpha scaling factor for A
 * \param[in] A Pointer to the raw-data of the input tensor A
 * \param[in] sizeA dim-dimensional array that stores the sizes of each dimension of A 
 * \param[in] outerSizeA dim-dimensional array that stores the outer-sizes of each dimension of A.
 *                       * This parameter may be NULL, indicating that the outer-size is equal to sizeA. 
 *                       * If outerSizeA is not NULL, outerSizeA[i] >= sizeA[i] for all 0 <= i < dim must hold.
 *                       * This option enables HPTT to operate on sub-tensors.
 * \param[in] beta scaling factor for B
 * \param[inout] B Pointer to the raw-data of the output tensor B
 * \param[in] outerSizeB dim-dimensional array that stores the outer-sizes of each dimension of B.
 *                       * This parameter may be NULL, indicating that the outer-size is equal to the perm(sizeA). 
 *                       * If outerSizeA is not NULL, outerSizeB[i] >= perm(sizeA)[i] for all 0 <= i < dim must hold.
 *                       * This option enables HPTT to operate on sub-tensors.
 * \param[in] selectionMethod Determines if auto-tuning should be used. See hptt::SelectionMethod for details.
 *                            ATTENTION: If you enable auto-tuning (e.g., hptt::MEASURE)
 *                            then the output data will be used during the
 *                            auto-tuning process. The original data (i.e., A and B), however, is preserved
 *                            after this function call completes -- unless your input
 *                            data (i.e. A) has invalid data (e.g., NaN, inf).
 * \param[in] numThreads number of threads that participate in this tensor transposition.
 * \param[in] threadIds Array of OpenMP threadIds that participate in this
 *            tensor transposition. This parameter is only important if you want to call
 *            HPTT from within a parallel region (i.e., via execute_expert()).
 */
std::shared_ptr<hptt::Transpose<float> > create_plan( const int *perm, const int dim,
                 const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                 const float beta, float *B, const int *outerSizeB, 
                 const SelectionMethod selectionMethod,
                 const int numThreads, const int *threadIds = nullptr);

std::shared_ptr<hptt::Transpose<double> > create_plan( const int *perm, const int dim,
                 const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                 const double beta, double *B, const int *outerSizeB, 
                 const SelectionMethod selectionMethod,
                 const int numThreads, const int *threadIds = nullptr);

std::shared_ptr<hptt::Transpose<FloatComplex> > create_plan( const int *perm, const int dim,
                 const FloatComplex alpha, const FloatComplex *A, const int *sizeA, const int *outerSizeA, 
                 const FloatComplex beta, FloatComplex *B, const int *outerSizeB, 
                 const SelectionMethod selectionMethod,
                 const int numThreads, const int *threadIds = nullptr);

std::shared_ptr<hptt::Transpose<DoubleComplex> > create_plan( const int *perm, const int dim,
                 const DoubleComplex alpha, const DoubleComplex *A, const int *sizeA, const int *outerSizeA, 
                 const DoubleComplex beta, DoubleComplex *B, const int *outerSizeB, 
                 const SelectionMethod selectionMethod,
                 const int numThreads, const int *threadIds = nullptr);


std::shared_ptr<hptt::Transpose<float> > create_plan( const std::vector<int> &perm, const int dim,
                 const float alpha, const float *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                 const float beta, float *B, const std::vector<int> &outerSizeB, 
                 const SelectionMethod selectionMethod,
                 const int numThreads, const std::vector<int> &threadIds = {});

std::shared_ptr<hptt::Transpose<double> > create_plan( const std::vector<int> &perm, const int dim,
                 const double alpha, const double *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                 const double beta, double *B, const std::vector<int> &outerSizeB, 
                 const SelectionMethod selectionMethod,
                 const int numThreads, const std::vector<int> &threadIds = {});

std::shared_ptr<hptt::Transpose<FloatComplex> > create_plan( const std::vector<int> &perm, const int dim,
                 const FloatComplex alpha, const FloatComplex *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                 const FloatComplex beta, FloatComplex *B, const std::vector<int> &outerSizeB, 
                 const SelectionMethod selectionMethod,
                 const int numThreads, const std::vector<int> &threadIds = {});

std::shared_ptr<hptt::Transpose<DoubleComplex> > create_plan( const std::vector<int> &perm, const int dim,
                 const DoubleComplex alpha, const DoubleComplex *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                 const DoubleComplex beta, DoubleComplex *B, const std::vector<int> &outerSizeB, 
                 const SelectionMethod selectionMethod,
                 const int numThreads, const std::vector<int> &threadIds = {});



std::shared_ptr<hptt::Transpose<float> > create_plan( const int *perm, const int dim,
                 const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                 const float beta, float *B, const int *outerSizeB, 
                 const int maxAutotuningCandidates,
                 const int numThreads, const int *threadIds = nullptr);

std::shared_ptr<hptt::Transpose<double> > create_plan( const int *perm, const int dim,
                 const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                 const double beta, double *B, const int *outerSizeB, 
                 const int maxAutotuningCandidates,
                 const int numThreads, const int *threadIds = nullptr);

std::shared_ptr<hptt::Transpose<FloatComplex> > create_plan( const int *perm, const int dim,
                 const FloatComplex alpha, const FloatComplex *A, const int *sizeA, const int *outerSizeA, 
                 const FloatComplex beta, FloatComplex *B, const int *outerSizeB, 
                 const int maxAutotuningCandidates,
                 const int numThreads, const int *threadIds = nullptr);

std::shared_ptr<hptt::Transpose<DoubleComplex> > create_plan( const int *perm, const int dim,
                 const DoubleComplex alpha, const DoubleComplex *A, const int *sizeA, const int *outerSizeA, 
                 const DoubleComplex beta, DoubleComplex *B, const int *outerSizeB, 
                 const int maxAutotuningCandidates,
                 const int numThreads, const int *threadIds = nullptr);
}

extern "C"
{
/**
 * \brief Computes the out-of-place tensor transposition of A into B
 *
 * A tensor transposition plan is a data structure that encodes the execution of the tensor transposition.
 * HPTT supports tensor transpositions of the form: 
 * \f[ B_{\pi(i_0,i_1,...)} = \alpha * A_{i_0,i_1,...} + \beta * B_{\pi(i_0,i_1,...)}. \f]
 * The plan can be reused over several transpositions.
 *
 * \param[in] perm dim-dimensional array representing the permutation of the indices. 
 *                 * For instance, perm[] = {1,0,2} denotes the following transposition: \f$B_{i1,i0,i2} \gets A_{i0,i1,i2}\f$.
 * \param[in] dim Dimensionality of the tensors
 * \param[in] alpha scaling factor for A
 * \param[in] A Pointer to the raw-data of the input tensor A
 * \param[in] sizeA dim-dimensional array that stores the sizes of each dimension of A 
 * \param[in] outerSizeA dim-dimensional array that stores the outer-sizes of each dimension of A.
 *                       * This parameter may be NULL, indicating that the outer-size is equal to sizeA. 
 *                       * If outerSizeA is not NULL, outerSizeA[i] >= sizeA[i] for all 0 <= i < dim must hold.
 *                       * This option enables HPTT to operate on sub-tensors.
 * \param[in] beta scaling factor for B
 * \param[inout] B Pointer to the raw-data of the output tensor B
 * \param[in] outerSizeB dim-dimensional array that stores the outer-sizes of each dimension of B.
 *                       * This parameter may be NULL, indicating that the outer-size is equal to the perm(sizeA). 
 *                       * If outerSizeA is not NULL, outerSizeB[i] >= perm(sizeA)[i] for all 0 <= i < dim must hold.
 *                       * This option enables HPTT to operate on sub-tensors.
 * \param[in] numThreads number of threads that participate in this tensor transposition.
 */
void sTensorTranspose( const int *perm, const int dim,
                 const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                 const float beta,        float *B,                   const int *outerSizeB, 
                 const int numThreads);

void dTensorTranspose( const int *perm, const int dim,
                 const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                 const double beta,        double *B,                   const int *outerSizeB, 
                 const int numThreads);

void cTensorTranspose( const int *perm, const int dim,
                 const float _Complex alpha, const float _Complex *A, const int *sizeA, const int *outerSizeA, 
                 const float _Complex beta,        float _Complex *B,                   const int *outerSizeB, 
                 const int numThreads);

void zTensorTranspose( const int *perm, const int dim,
                 const double _Complex alpha, const double _Complex *A, const int *sizeA, const int *outerSizeA, 
                 const double _Complex beta,        double _Complex *B,                   const int *outerSizeB, 
                 const int numThreads);
}
