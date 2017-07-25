/**
 *   High-Performance Tensor Transposition Library
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

#include <tuple>
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <chrono>

#include <float.h>
#include <stdio.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <hptt.h>
#include <hptt_utils.h>
#include <macros.h>

#define NDEBUG

//#define HPTT_TIMERS

namespace hptt {

std::shared_ptr<hptt::Transpose<float> > create_plan( const int *perm, const int dim,
                  const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                  const float beta, float *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds));
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<double> > create_plan( const int *perm, const int dim,
                  const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                  const double beta, double *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds ));
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex> > create_plan( const int *perm, const int dim,
                  const FloatComplex alpha, const FloatComplex *A, const int *sizeA, const int *outerSizeA, 
                  const FloatComplex beta, FloatComplex *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<FloatComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds));
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex> > create_plan( const int *perm, const int dim,
                  const DoubleComplex alpha, const DoubleComplex *A, const int *sizeA, const int *outerSizeA, 
                  const DoubleComplex beta, DoubleComplex *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<DoubleComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds ));
   plan->createPlan();
   return plan;
}



std::shared_ptr<hptt::Transpose<float> > create_plan( const std::vector<int> &perm, const int dim,
                  const float alpha, const float *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const float beta, float *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr ));
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<double> > create_plan( const std::vector<int> &perm, const int dim,
                  const double alpha, const double *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const double beta, double *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr ));
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex> > create_plan( const std::vector<int> &perm, const int dim,
                  const FloatComplex alpha, const FloatComplex *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const FloatComplex beta, FloatComplex *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<FloatComplex> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr ));
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex> > create_plan( const std::vector<int> &perm, const int dim,
                  const DoubleComplex alpha, const DoubleComplex *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const DoubleComplex beta, DoubleComplex *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<DoubleComplex> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr ));
   plan->createPlan();
   return plan;
}


std::shared_ptr<hptt::Transpose<float> > create_plan( const int *perm, const int dim,
                  const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                  const float beta, float *B, const int *outerSizeB, 
                  const int maxAutotuningCandidates,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE, numThreads, threadIds));
   plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<double> > create_plan( const int *perm, const int dim,
                  const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                  const double beta, double *B, const int *outerSizeB, 
                  const int maxAutotuningCandidates,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE, numThreads, threadIds ));
   plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex> > create_plan( const int *perm, const int dim,
                  const FloatComplex alpha, const FloatComplex *A, const int *sizeA, const int *outerSizeA, 
                  const FloatComplex beta, FloatComplex *B, const int *outerSizeB, 
                  const int maxAutotuningCandidates,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<FloatComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE, numThreads, threadIds));
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex> > create_plan( const int *perm, const int dim,
                  const DoubleComplex alpha, const DoubleComplex *A, const int *sizeA, const int *outerSizeA, 
                  const DoubleComplex beta, DoubleComplex *B, const int *outerSizeB, 
                  const int maxAutotuningCandidates,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<DoubleComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, MEASURE, numThreads, threadIds ));
   plan->setMaxAutotuningCandidates(maxAutotuningCandidates);
   plan->createPlan();
   return plan;
}

}
















