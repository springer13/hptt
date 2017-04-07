#ifndef HPTT_H
#define HPTT_H

#include <list>
#include <vector>
#include <memory>

#include <stdio.h>

#include "hptt_utils.h"

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
      Plan(int numThreads, std::vector<int>loopOrder, std::vector<int>numThreadsAtLoop) : numThreads_(numThreads), rootNodes_(nullptr), loopOrder_(loopOrder), numThreadsAtLoop_(numThreadsAtLoop) {
         rootNodes_ = new ComputeNode[numThreads];
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

   private:
      int numThreads_;
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
                 const int numThreads ) : 
         A_(A),
         B_(B),
         alpha_(alpha),
         beta_(beta),
         dim_(-1),
         numThreads_(numThreads), 
         masterPlan_(nullptr),
         blocking_constStride1_(1), //TODO
         infoLevel_(0),
         selectionMethod_(selectionMethod)
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
      floatType getAlpha() const noexcept { return alpha_; }
      floatType getBeta() const noexcept { return beta_; }
      void setAlpha(floatType alpha) noexcept { alpha_ = alpha; }
      void setBeta(floatType beta) noexcept { beta_ = beta; }
      void setInfoLevel(int infoLevel) noexcept { infoLevel_ = infoLevel; }
      int getInfoLevel() noexcept { return infoLevel_; }

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
      float estimateExecutionTime( const Plan *plan); //execute just a few iterations and exterpolate the result
      void verifyParameter(const int *size, const int* perm, const int* outerSizeA, const int* outerSizeB, const int dim) const;
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

      Plan *masterPlan_; 
      SelectionMethod selectionMethod_;
      static constexpr int blocking_ = 128 / sizeof(floatType);
      static constexpr int blocking_micro_ = 256 / 8 / sizeof(floatType);
      int blocking_constStride1_; //blocking for perm[0] == 0, block in the next two leading dimensions

      int infoLevel_; // determines which auxiliary messages should be printed
};

void trashCache(double *A, double *B, int n);

auto create_plan(const int *sizeA, 
                 const int *perm, 
                 const int *outerSizeA, 
                 const int *outerSizeB, 
                 const int dim,
                 const float *A,
                 const float alpha,
                 float *B,
                 const float beta,
                 const SelectionMethod selectionMethod,
                 const int numThreads )
{
   auto plan(std::make_shared<hptt::Transpose<float> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads ));
   plan->createPlan();
   return plan;
}

//auto create_plan(const int *sizeA, 
//                 const int *perm, 
//                 const int *outerSizeA, 
//                 const int *outerSizeB, 
//                 const int dim,
//                 const double *A,
//                 const double alpha,
//                 double *B,
//                 const double beta,
//                 const SelectionMethod selectionMethod,
//                 const int numThreads )
//{
//   auto plan(std::make_shared<hptt::Transpose<double> >(sizeA, perm, outerSizeA, outerSizeB, dim, A, alpha, B, beta, selectionMethod, numThreads ));
//   plan->createPlan();
//   return plan;
//}

extern template class Transpose<float>;
}

#endif
