#ifndef TTC_C_H
#define TTC_C_H

namespace ttc {

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
      }

      ~Transpose() { 
         if ( rootNodes_ != nullptr ){
            delete[] rootNodes_;
         }
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
      ComputeNode* selectPlan( const std::vector<ComputeNode*> &plans ) const;
      void fuseIndices(const int *sizeA, const int* perm, const int *outerSizeA, const int *outerSizeB, const int dim);
      void computeLeadingDimensions();

      /***************************************************
       * Helper Methods
       ***************************************************/
      float estimateExecutionTime( const ComputeNode *rootNodes ) const; //execute just a few iterations and exterpolate the result
      void verifyParameter(const int *size, const int* perm, const int* outerSizeA, const int* outerSizeB, const int dim) const;
      void getLoopOrder(std::vector<int> &loopOrder) const;
      void getParallelismStrategy(std::vector<int> &numThreadsAtLoop) const;
      void preferedLoopOrderForParallelization( std::vector<int> &loopOrder ) const;
      void getAvailableParallelism( std::vector<int> &numTasksPerLoop ) const;

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
};


void trashCache(double *A, double *B, int n);

}

#endif
