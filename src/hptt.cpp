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

#include <vector>
#include <memory>

#include <transpose.h>

namespace hptt {

std::shared_ptr<hptt::Transpose<float> > create_plan( const int *perm, const int dim,
                  const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                  const float beta, float *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds));
   return plan;
}

std::shared_ptr<hptt::Transpose<double> > create_plan( const int *perm, const int dim,
                  const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                  const double beta, double *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds ));
   return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex> > create_plan( const int *perm, const int dim,
                  const FloatComplex alpha, const FloatComplex *A, const int *sizeA, const int *outerSizeA, 
                  const FloatComplex beta, FloatComplex *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<FloatComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds));
   return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex> > create_plan( const int *perm, const int dim,
                  const DoubleComplex alpha, const DoubleComplex *A, const int *sizeA, const int *outerSizeA, 
                  const DoubleComplex beta, DoubleComplex *B, const int *outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const int *threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<DoubleComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds ));
   return plan;
}



std::shared_ptr<hptt::Transpose<float> > create_plan( const std::vector<int> &perm, const int dim,
                  const float alpha, const float *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const float beta, float *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr ));
   return plan;
}

std::shared_ptr<hptt::Transpose<double> > create_plan( const std::vector<int> &perm, const int dim,
                  const double alpha, const double *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const double beta, double *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr ));
   return plan;
}

std::shared_ptr<hptt::Transpose<FloatComplex> > create_plan( const std::vector<int> &perm, const int dim,
                  const FloatComplex alpha, const FloatComplex *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const FloatComplex beta, FloatComplex *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<FloatComplex> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr ));
   return plan;
}

std::shared_ptr<hptt::Transpose<DoubleComplex> > create_plan( const std::vector<int> &perm, const int dim,
                  const DoubleComplex alpha, const DoubleComplex *A, const std::vector<int> &sizeA, const std::vector<int> &outerSizeA, 
                  const DoubleComplex beta, DoubleComplex *B, const std::vector<int> &outerSizeB, 
                  const SelectionMethod selectionMethod,
                  const int numThreads, const std::vector<int> &threadIds)
{
   auto plan(std::make_shared<hptt::Transpose<DoubleComplex> >(&sizeA[0], &perm[0], &outerSizeA[0], &outerSizeB[0], dim, A, alpha, B, beta, selectionMethod, numThreads, (threadIds.size() > 0 ) ? &threadIds[0] : nullptr ));
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


extern "C"{
void sTensorTranspose( const int *perm, const int dim,
                 const float alpha, const float *A, const int *sizeA, const int *outerSizeA, 
                 const float beta,        float *B,                   const int *outerSizeB, 
                 const int numThreads)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, hptt::ESTIMATE, numThreads));
   plan->execute();
}

void dTensorTranspose( const int *perm, const int dim,
                 const double alpha, const double *A, const int *sizeA, const int *outerSizeA, 
                 const double beta,        double *B,                   const int *outerSizeB, 
                 const int numThreads)
{
   auto plan(std::make_shared<hptt::Transpose<double> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, hptt::ESTIMATE, numThreads));
   plan->execute();
}

void cTensorTranspose( const int *perm, const int dim,
                 const float _Complex alpha, const float _Complex *A, const int *sizeA, const int *outerSizeA, 
                 const float _Complex beta,        float _Complex *B,                   const int *outerSizeB, 
                 const int numThreads)
{
   auto plan(std::make_shared<hptt::Transpose<hptt::FloatComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, 
                         (const hptt::FloatComplex*) A, (hptt::FloatComplex) alpha, (hptt::FloatComplex*) B, (hptt::FloatComplex) beta, hptt::ESTIMATE, numThreads));
   plan->execute();
}

void zTensorTranspose( const int *perm, const int dim,
                 const double _Complex alpha, const double _Complex *A, const int *sizeA, const int *outerSizeA, 
                 const double _Complex beta,        double _Complex *B,                   const int *outerSizeB, 
                 const int numThreads)
{
   auto plan(std::make_shared<hptt::Transpose<hptt::DoubleComplex> >(sizeA, perm, outerSizeA, outerSizeB, dim, 
                         (const hptt::DoubleComplex*) A, (hptt::DoubleComplex) alpha, (hptt::DoubleComplex*) B, (hptt::DoubleComplex) beta, hptt::ESTIMATE, numThreads));
   plan->execute();
}
}
















