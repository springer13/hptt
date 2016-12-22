#ifndef TTC_C_H
#define TTC_C_H

#include <list>
#include <vector>

#include <stdio.h>

#include "ttc_utils.h"

namespace ttc {

#ifdef DEBUG
#define TTC_ERROR_INFO(str) fprintf(stdout, "[INFO] %s:%d : %s\n", __FILE__, __LINE__, str); exit(-1);
#else
#define TTC_ERROR_INFO(str)
#endif

extern float *trash1, *trash2;
extern int trashSize;

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
                 const float *A,
                 const float alpha,
                 float *B,
                 const float beta,
                 const SelectionMethod selectionMethod,
                 const int numThreads ) : 
         A_(A),
         B_(B),
         alpha_(alpha),
         beta_(beta),
         dim_(-1),
         numThreads_(numThreads), 
         masterPlan_(nullptr),
         blocking_(32),
         blocking_constStride1_(1), //TODO
         trash1_(nullptr),
         trash2_(nullptr),
         infoLevel_(1),
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

         trashSize_ = 42 * 1024 * 1024 / sizeof(float); //42 MiB
         trash1_ = new float[trashSize_];
         trash2_ = new float[trashSize_];

#pragma omp parallel for num_threads(numThreads_)
         for(int i=0;i < trashSize_ ; ++i){
            trash1_[i] = (((i+1) * 13) % 10000) / 10000.;
            trash2_[i] = (((i+1) * 13) % 10000) / 10000.;
         }
      }

      ~Transpose() { 
         if ( masterPlan_!= nullptr ){
            delete masterPlan_;
         }
         if( trash1_ != nullptr )
            delete[] trash1_;
         if( trash2_ != nullptr )
            delete[] trash2_;
      }

      /***************************************************
       * Getter & Setter
       ***************************************************/
      int getNumThreads() const noexcept { return numThreads_; }
      void setNumThreads(int numThreads) noexcept { numThreads_ = numThreads; }
      float getAlpha() const noexcept { return alpha_; }
      float getBeta() const noexcept { return beta_; }
      void setAlpha(float alpha) noexcept { alpha_ = alpha; }
      void setBeta(float beta) noexcept { beta_ = beta; }
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
      void trashCaches();
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

      const float* __restrict__ A_;
      float* __restrict__ B_;
      float alpha_;
      float beta_;
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
      int blocking_;
      int blocking_constStride1_; //blocking for perm[0] == 0, block in the next two leading dimensions

      float *trash1_, *trash2_;
      int trashSize_;

      int infoLevel_; // determines which auxiliary messages should be printed
};


void trashCache(double *A, double *B, int n);

}

#endif
