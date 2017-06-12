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

#ifndef HPTT_H
#define HPTT_H

#include <list>
#include <vector>
#include <memory>
#include <complex>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>

#include "hptt_utils.h"

#define REGISTER_BITS 256 // AVX

#ifdef HPTT_ARCH_ARM
#undef REGISTER_BITS 
#define REGISTER_BITS 128 // ARM
#endif

namespace hptt {

#ifdef DEBUG
#define HPTT_ERROR_INFO(str) fprintf(stdout, "[INFO] %s:%d : %s\n", __FILE__, __LINE__, str); exit(-1);
#else
#define HPTT_ERROR_INFO(str)
#endif

using FloatComplex = std::complex<float>;
using DoubleComplex = std::complex<double>;

class ComputeNode{
   public:
      ComputeNode() : start(-1), end(-1), inc(-1), lda(-1), ldb(-1), next(nullptr) {}

      ~ComputeNode() {
         if ( next != nullptr )
            delete next;
      }

   size_t start;
   size_t end;
   size_t inc;
   size_t lda;
   size_t ldb;
   ComputeNode *next;
};

class Plan{
   public:
      Plan() : rootNodes_(nullptr), numTasks_(0) { }

      Plan(std::vector<int>loopOrder, std::vector<int>numThreadsAtLoop) : rootNodes_(nullptr), loopOrder_(loopOrder), numThreadsAtLoop_(numThreadsAtLoop) {
         numTasks_ = 1;
         for(auto nt : numThreadsAtLoop)
            numTasks_ *= nt;
         rootNodes_ = new ComputeNode[numTasks_];
      }

      ~Plan() {
         if ( rootNodes_ != nullptr )
            delete[] rootNodes_;
      }

      void print() const {
         printVector(loopOrder_,"LoopOrder");
         printVector(numThreadsAtLoop_,"Parallelization");
      }
      const ComputeNode* getRootNode_const(int threadId) const { return &rootNodes_[threadId]; }
      ComputeNode* getRootNode(int threadId) const { return &rootNodes_[threadId]; }
      int getNumTasks() const { return numTasks_; } 

   private:
      int numTasks_;
      std::vector<int> loopOrder_;
      std::vector<int> numThreadsAtLoop_;
      ComputeNode *rootNodes_;
};

enum SelectionMethod { ESTIMATE, MEASURE, PATIENT, CRAZY };

template<typename floatType>
class Transpose{

   public:
                  
      /***************************************************
       * Cons, Decons, Copy, ...
       ***************************************************/
      Transpose( const int *sizeA, 
                 const int *perm, 
                 const int *outerSizeA, 
                 const int *outerSizeB, 
                 const int dim,
                 const floatType *A,
                 const floatType alpha,
                 floatType *B,
                 const floatType beta,
                 const SelectionMethod selectionMethod,
                 const int numThreads, 
                 const int *threadIds = nullptr) : 
         A_(A),
         B_(B),
         alpha_(alpha),
         beta_(beta),
         dim_(-1),
         numThreads_(numThreads), 
         masterPlan_(nullptr),
         blocking_constStride1_(1), //TODO
         selectionMethod_(selectionMethod),
         maxAutotuningCandidates_(-1),
         selectedParallelStrategyId_(-1)
      {
#ifdef _OPENMP
         omp_init_lock(&writelock);
#endif
         sizeA_.resize(dim);
         perm_.resize(dim);
         outerSizeA_.resize(dim);
         outerSizeB_.resize(dim);
         lda_.resize(dim);
         ldb_.resize(dim);
         if(threadIds){
            // compact threadIds. E.g., 1, 7, 5 -> local_id(1) = 0, local_id(7) = 2, local_id(5) = 1
            for(int i=0; i < numThreads; ++i)
               threadIds_.push_back(threadIds[i]);
            std::sort(threadIds_.begin(), threadIds_.end());
         }else{
            for(int i=0; i < numThreads; ++i)
               threadIds_.push_back(i);
         }

         verifyParameter(sizeA, perm, outerSizeA, outerSizeB, dim);

         // initializes dim_, outerSizeA, outerSizeB, sizeA and perm 
         skipIndices(sizeA, perm, outerSizeA, outerSizeB, dim);
         fuseIndices();

         // initializes lda_ and ldb_
         computeLeadingDimensions();

#ifdef DEBUG
         if( blocking_constStride1_ != 1 )
            fprintf(stderr, "[HPTT] ERROR: blocking for this case needs to be one. also look into _constStride1()\n");
#endif
      }

      Transpose(const Transpose &other) : A_(other.A_), B_(other.B_), 
                                          alpha_(other.alpha_),
                                          beta_(other.beta_),
                                          dim_(other.dim_),
                                          numThreads_(other.numThreads_),
                                          masterPlan_(other.masterPlan_),
                                          selectionMethod_(other.selectionMethod_),
                                          selectedParallelStrategyId_(other.selectedParallelStrategyId_),
                                          maxAutotuningCandidates_(other.maxAutotuningCandidates_),
                                          sizeA_(other.sizeA_),
                                          perm_(other.perm_),
                                          outerSizeA_(other.outerSizeA_),
                                          outerSizeB_(other.outerSizeB_),
                                          lda_(other.lda_),
                                          ldb_(other.ldb_),
                                          threadIds_(other.threadIds_) 
      { 
#ifdef _OPENMP
         omp_init_lock(&writelock);
#endif
      }

      ~Transpose() { 
#ifdef _OPENMP
         omp_destroy_lock(&writelock); 
#endif
      }

      /***************************************************
       * Getter & Setter
       ***************************************************/
      int getNumThreads() const noexcept { return numThreads_; }
      void setNumThreads(int numThreads) noexcept { numThreads_ = numThreads; }
      void setParallelStrategy(int id) noexcept { selectedParallelStrategyId_ = id; }
      floatType getAlpha() const noexcept { return alpha_; }
      floatType getBeta() const noexcept { return beta_; }
      void setAlpha(floatType alpha) noexcept { alpha_ = alpha; }
      void setBeta(floatType beta) noexcept { beta_ = beta; }
      const floatType* getInputPtr() const noexcept { return A_; }
      floatType* getOutputPtr() const noexcept { return B_; }
      void setInputPtr(const floatType *A) noexcept { A_ = A; }
      void setOutputPtr(floatType *B) noexcept { B_ = B; }
      void resetThreadIds() noexcept { threadIds_.clear(); }
      void printThreadIds() const noexcept { for( auto id : threadIds_) printf("%d, ",id); printf("\n"); } 
      int getMasterThreadId() const noexcept { return threadIds_[0]; } 
      /**
       * setMaxAutotuningCandidates() enables users to specify the number of
       * candidates that should be tested during the autotuning phase
      */
      void setMaxAutotuningCandidates (int num) { maxAutotuningCandidates_ = num; } 
      void addThreadId(int threadId) noexcept { 
#ifdef _OPENMP
         omp_set_lock(&writelock);
         threadIds_.push_back(threadId); 
         std::sort(threadIds_.begin(), threadIds_.end()); 
         omp_unset_lock(&writelock);
#endif
      }

      /***************************************************
       * Public Methods
       ***************************************************/
      void createPlan();

      template<bool useStreamingStores=true, bool spawnThreads=true, bool betaIsZero>
      void execute_expert() noexcept;
      void execute() noexcept;

   private:
      /***************************************************
       * Private Methods
       ***************************************************/
      void createPlans( std::vector<std::shared_ptr<Plan> > &plans ) const;
      std::shared_ptr<Plan> selectPlan( const std::vector<std::shared_ptr<Plan> > &plans );
      void fuseIndices();
      void skipIndices(const int *_sizeA, const int* _perm, const int *_outerSizeA, const int *_outerSizeB, const int dim);
      void computeLeadingDimensions();
      double loopCostHeuristic( const std::vector<int> &loopOrder ) const;
      double parallelismCostHeuristic( const std::vector<int> &loopOrder ) const;
      int getLocalThreadId(int myThreadId) const;
      template<bool spawnThreads>
      void getStartEnd(int n, int &myStart, int &myEnd) const;

      /***************************************************
       * Helper Methods
       ***************************************************/
      // parallelizes the loops by changing the value of parallelismStrategy
      void parallelize( std::vector<int> &parallelismStrategy,
                                        std::vector<int> &availableParallelismAtLoop, 
                                        int &totalTasks,
                                        std::list<int> &primeFactors, 
                                        const float minBalancing,
                                        const std::vector<int> &loopsAllowed) const;
      float getLoadBalance( const std::vector<int> &parallelismStrategy ) const;
      float estimateExecutionTime( const std::shared_ptr<Plan> plan); //execute just a few iterations and extrapolate the result
      void verifyParameter(const int *size, const int* perm, const int* outerSizeA, const int* outerSizeB, const int dim) const;
      void getBestParallelismStrategy ( std::vector<int> &bestParallelismStrategy ) const;
      void getBestLoopOrder( std::vector<int> &loopOrder ) const;
      void getLoopOrders(std::vector<std::vector<int> > &loopOrders) const;
      void getParallelismStrategies(std::vector<std::vector<int> > &parallelismStrategies) const;
      void getAllParallelismStrategies( std::list<int> &primeFactorsToMatch, 
            std::vector<int> &availableParallelismAtLoop, 
            std::vector<int> &achievedParallelismAtLoop, 
            std::vector<std::vector<int> > &parallelismStrategies) const;
      void getAvailableParallelism( std::vector<int> &numTasksPerLoop ) const;
      int getIncrement( int loopIdx ) const;
      void executeEstimate(const Plan *plan) noexcept; // almost identical to execute, but it just executes few iterations and then extrapolates
      double getTimeLimit() const;

      const floatType* __restrict__ A_;
      floatType* __restrict__ B_;
      floatType alpha_;
      floatType beta_;
      int dim_;
      std::vector<size_t> sizeA_;
      std::vector<int> perm_; 
      std::vector<size_t> outerSizeA_; 
      std::vector<size_t> outerSizeB_; 
      std::vector<size_t> lda_; 
      std::vector<size_t> ldb_; 
      std::vector<int> threadIds_; 
      int numThreads_;
      int selectedParallelStrategyId_;
#ifdef _OPENMP
      omp_lock_t writelock;
#endif

      std::shared_ptr<Plan> masterPlan_; 
      SelectionMethod selectionMethod_;
      int maxAutotuningCandidates_;
      static constexpr int blocking_micro_ = REGISTER_BITS / 8 / sizeof(floatType);
      static constexpr int blocking_ = blocking_micro_ * 4;
      int blocking_constStride1_; //blocking for perm[0] == 0, block in the next two leading dimensions

      static constexpr int infoLevel_ = 0; // determines which auxiliary messages should be printed
};

void trashCache(double *A, double *B, int n);

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




extern template class Transpose<float>;
extern template class Transpose<double>;
extern template class Transpose<FloatComplex>;
extern template class Transpose<DoubleComplex>;
}

#endif
