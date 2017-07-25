/**
 *   High-Performance Tensor Transposition Library for general tensor transpositions of the form:
 *   
 *       B_{\pi(i_0,i_1,...)} = \alpha * A_{i_0,i_1,...} + \beta * B_{\pi(i_0,i_1,...)}
 *
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

#pragma once

#include <vector>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif


#include "transpose.h"


namespace hptt {

/**
 * Creates Transpose plan for a transposition of the form: B_{\pi(i_0,i_1,...)} = \alpha * A_{i_0,i_1,...} + \beta * B_{\pi(i_0,i_1,...)}. 
 * This plan can be reused over several transpositions.
 *
 * \param[in] perm Permutation of the indices. For instance, perm[] = {1,0,2} dontes the following transposition: B[i1,i0,i2] = A[i0,i1,i2].
 * \param[in] threadIds Array of OpenMP threadIds that participate in this
 *            tensor transposition. This parameter is only important if you want to call
 *            HPTT from within a parallel region (i.e., via execute_expert<..., spawnThreads=false, ...>().
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
