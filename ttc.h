#ifndef TTC_C_H
#define TTC_C_H

#include <list>
#include <vector>

#include <stdio.h>

namespace ttc {

extern float *trash1, *trash2;
extern int trashSize;

class ComputeNode{
   public:
      ComputeNode() : start(-1), end(-1), inc(-1), lda(-1), ldb(-1), next(nullptr) {}

      ~ComputeNode() {
         if ( next != nullptr )
            delete next;
      }

   int start;
   int end;
   int inc;
   int lda;
   int ldb;
   ComputeNode *next;
};

enum SelectionMethod { ESTIMATE, MEASURE, PATIENT, CRAZY };

class Transpose{

   public:
                  
      /***************************************************
       * Cons, Decons, Copy, ...
       ***************************************************/
      Transpose( const int *sizeA, 
                 const int* perm, 
                 const int* outerSizeA, 
                 const int* outerSizeB, 
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
         rootNodes_(nullptr),
         blocking_(32),
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
         if ( rootNodes_ != nullptr ){
            delete[] rootNodes_;
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
      void createPlans( std::vector<ComputeNode*> &plans ) const;
      ComputeNode* selectPlan( const std::vector<ComputeNode*> &plans );
      void fuseIndices(const int *sizeA, const int* perm, const int *outerSizeA, const int *outerSizeB, const int dim);
      void computeLeadingDimensions();
      void trashCaches();
      double loopCostHeuristic( const std::vector<int> &loopOrder ) const;
      double parallelismCostHeuristic( const std::vector<int> &loopOrder ) const;

      /***************************************************
       * Helper Methods
       ***************************************************/
      float estimateExecutionTime( const ComputeNode *rootNodes ); //execute just a few iterations and exterpolate the result
      void verifyParameter(const int *size, const int* perm, const int* outerSizeA, const int* outerSizeB, const int dim) const;
      void getLoopOrders(std::vector<std::vector<int> > &loopOrders) const;
      void getParallelismStrategies(std::list<std::vector<int> > &parallelismStrategies) const;
      void getAllParallelismStrategies( std::list<int> &primeFactorsToMatch, 
            std::vector<int> &availableParallelismAtLoop, 
            std::vector<int> &achievedParallelismAtLoop, 
            std::list<std::vector<int> > &parallelismStrategies) const;
      void getAvailableParallelism( std::vector<int> &numTasksPerLoop ) const;
      void executeEstimate(const ComputeNode *rootNodes) noexcept; // almost identical to execute, but it just executes few iterations and then exterpolates
      double getTimeLimit() const;

      const float* __restrict__ A_;
      float* __restrict__ B_;
      float alpha_;
      float beta_;
      int dim_;
      std::vector<int> sizeA_;
      std::vector<int> perm_; 
      std::vector<int> outerSizeA_; 
      std::vector<int> outerSizeB_; 
      std::vector<int> lda_; 
      std::vector<int> ldb_; 
      int numThreads_;

      ComputeNode *rootNodes_; //one for each thread
      SelectionMethod selectionMethod_;
      int blocking_;

      float *trash1_, *trash2_;
      int trashSize_;

      int infoLevel_; // determines which auxiliary messages should be printed
};


void trashCache(double *A, double *B, int n);

}

#endif
