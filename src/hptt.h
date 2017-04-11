#ifndef HPTT_H
#define HPTT_H

#include <list>
#include <vector>
#include <memory>

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
                 const int numThreads) : 
         A_(A),
         B_(B),
         alpha_(alpha),
         beta_(beta),
         dim_(-1),
         numThreads_(numThreads), 
         masterPlan_(nullptr),
         blocking_constStride1_(1), //TODO
         selectionMethod_(selectionMethod),
         selectedParallelStrategyId_(-1)
      {
         sizeA_.resize(dim);
         perm_.resize(dim);
         outerSizeA_.resize(dim);
         outerSizeB_.resize(dim);
         lda_.resize(dim);
         ldb_.resize(dim);

         verifyParameter(sizeA, perm, outerSizeA, outerSizeB, dim);

         // initializes dim_, outerSizeA, outerSizeB, sizeA and perm 
         fuseIndices(sizeA, perm, outerSizeA, outerSizeB, dim);

         // initializes lda_ and ldb_
         computeLeadingDimensions();

#ifdef DEBUG
         if( blocking_constStride1_ != 1 )
            fprintf(stderr, "ERROR: blocking for this case needs to be one. also look into _constStride1()\n");
#endif
      }

      ~Transpose() { 
         if ( masterPlan_!= nullptr ){
            delete masterPlan_;
         }
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

      /***************************************************
       * Public Methods
       ***************************************************/
      void createPlan();

      void execute() noexcept;

   private:
      /***************************************************
       * Private Methods
       ***************************************************/
      void createPlans( std::vector<Plan*> &plans ) const;
      Plan* selectPlan( const std::vector<Plan*> &plans );
      void fuseIndices(const int *sizeA, const int* perm, const int *outerSizeA, const int *outerSizeB, const int dim);
      void computeLeadingDimensions();
      double loopCostHeuristic( const std::vector<int> &loopOrder ) const;
      double parallelismCostHeuristic( const std::vector<int> &loopOrder ) const;

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
      float estimateExecutionTime( const Plan *plan); //execute just a few iterations and exterpolate the result
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
      void executeEstimate(const Plan *plan) noexcept; // almost identical to execute, but it just executes few iterations and then exterpolates
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
      int numThreads_;
      int selectedParallelStrategyId_;

      Plan *masterPlan_; 
      SelectionMethod selectionMethod_;
      static constexpr int blocking_micro_ = REGISTER_BITS / 8 / sizeof(floatType);
      static constexpr int blocking_ = blocking_micro_ * 4;
      int blocking_constStride1_; //blocking for perm[0] == 0, block in the next two leading dimensions

      static constexpr int infoLevel_ = 0; // determines which auxiliary messages should be printed
};

void trashCache(double *A, double *B, int n);

std::shared_ptr<hptt::Transpose<float> > create_plan(const int *sizeA, 
                 const int *perm, 
                 const int *outerSizeA, 
                 const int *outerSizeB, 
                 const int dim,
                 const float *A,
                 const float alpha,
                 float *B,
                 const float beta,
                 const SelectionMethod selectionMethod,
                 const int numThreads)
{
   auto plan(std::make_shared<hptt::Transpose<float> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads));
   plan->createPlan();
   return plan;
}

std::shared_ptr<hptt::Transpose<double> > create_plan(const int *sizeA, 
                 const int *perm, 
                 const int *outerSizeA, 
                 const int *outerSizeB, 
                 const int dim,
                 const double *A,
                 const double alpha,
                 double *B,
                 const double beta,
                 const SelectionMethod selectionMethod,
                 const int numThreads )
{
   auto plan(std::make_shared<hptt::Transpose<double> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads ));
   plan->createPlan();
   return plan;
}

extern template class Transpose<float>;
extern template class Transpose<double>;
}

#endif
