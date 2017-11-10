#pragma once

#include <list>
#include <vector>
#include <memory>
#include <algorithm>

#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "hptt_types.h"

namespace hptt {

   class Plan;

/**
 * \brief The Transpose class encodes all information related to the execution of the tensor transposition.
 *
 * Once a transpose (henceforth referred to as plan) t has been created it can be
 * executed via t->execute().
 * Moreover, a plan can be reused multiple times. For this purpose you might
 * want to have a look at the functions: 
 * * setInputPtr()
 * * setOutputPtr()
 *
 * In addition to the normal execute() function, this class also offers the
 * execute_expert() interface. This interface is intended for the expert user
 * and offers more flexibility than execute(). If you want to use the expert
 * interface, then you might want to checkout the following functions as well:
 * * resetThreadIds()
 * * addThreadId()
 */
template<typename floatType>
class Transpose
{

   public:
                  
      /***************************************************
       * Cons, Decons, Copy, ...
       ***************************************************/
      /**
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
       * \param[in] useRowMajor This flag indicates whether a row-major memory layout should be used (default: off = column-major).
       *            Column-Major: indices are stored from left to right (leftmost = stride-1 index)
       *            Row-Major: indices are stored from right to left (right = stride-1 index)
       */
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
                 const int *threadIds = nullptr,
                 const bool useRowMajor = false );

      Transpose(const Transpose &other);

      ~Transpose();

      /***************************************************
       * Getter & Setter
       ***************************************************/
      bool getConjA() noexcept { return conjA_; }
      void setConjA(bool conjA) noexcept { conjA_ = conjA; }
      int getNumThreads() const noexcept { return numThreads_; }
      void setNumThreads(int numThreads) noexcept { numThreads_ = numThreads; }
      floatType getAlpha() const noexcept { return alpha_; }
      floatType getBeta() const noexcept { return beta_; }
      /**
       * \brief set the scaling factor for A
       */
      void setAlpha(floatType alpha) noexcept { alpha_ = alpha; }
      /**
       * \brief set the scaling factor for B
       */
      void setBeta(floatType beta) noexcept { beta_ = beta; }
      /**
       * \brief Set the pointer for A
       *
       * This features is especially useful if one wants to reuse the
       * transposition over multiple invocations.
       */
      void setInputPtr(const floatType *A) noexcept { A_ = A; }
      /**
       * \brief Set the pointer for B
       *
       * This features is especially useful if one wants to reuse the
       * transposition over multiple invocations.
       */
      void setOutputPtr(floatType *B) noexcept { B_ = B; }
      /**
       * \brief Get raw-data pointer to A
       */
      const floatType* getInputPtr() const noexcept { return A_; }
      /**
       * \brief Get raw-data pointer to B
       */
      floatType* getOutputPtr() const noexcept { return B_; }

      /**
       * \brief Clears the array that stores the OpenMP threadIds. This function
       *        should only be used in conjuction with addThreadId().
       */
      void resetThreadIds() noexcept { threadIds_.clear(); }

      /**
       * setMaxAutotuningCandidates() enables users to specify the number of
       * candidates that should be tested during the autotuning phase
      */
      void setMaxAutotuningCandidates (int num) { maxAutotuningCandidates_ = num; } 

      /**
       * This thread-safe function adds an OpenMP threadId to the set of threads
       * that will participate in this tensor transposition. This function is
       * only required in conjunction with the execute_expert() interface where
       * the transposition is executed from within a parallel region (i.e.,~HPTT
       * does not spawn the threads). It is the programmers responsibility to
       * specify the correct thread IDs that participate in this call.  
       *
       * \param[in] threadId An OpenMP threadId 
       */
      void addThreadId(int threadId) noexcept { 
#ifdef _OPENMP
         omp_set_lock(&writelock);
         threadIds_.push_back(threadId); 
         std::sort(threadIds_.begin(), threadIds_.end()); 
         omp_unset_lock(&writelock);
#endif
      }

      void printThreadIds() const noexcept { for( auto id : threadIds_) printf("%d, ",id); printf("\n"); } 
      int getMasterThreadId() const noexcept { return threadIds_[0]; } 

      /***************************************************
       * Public Methods
       ***************************************************/
      /**
       * \brief Creates the plan that encodes the execution of the tensor transposition.
       */
      void createPlan();

      /**
       * Executes the transposition. This functions requires that the plan has
       * already been created via the createPlan() function.
       * This function behaves similarly to the execute() function but it
       * offers additional template parameters to improve performance for very
       * small tensor transpositions. Moreover it adds more flexibility.
       *
       * \param[in] useStreamingStores Iff this variable is set, HPTT will use
       *                         streaming stores which improves performance because they avoid the 
       *                         write-allocate traffic incurred by the write to B. However, sometimes
       *                         the user might want to avoid streaming stores
       *                         because the packed data fits int cache and is
       *                         reused shortly (e.g., within BLAS packing
       *                         routines).
       * \param[in] spawnThreads If the variable is set, the threads will be
       *                         spawned from within this call, otherwise it is
       *                         expected that this function call executes from
       *                         within a parallel region.
       * \param[in] betaIsZero   Only set this variable if beta is zero.
       */
      template<bool useStreamingStores=true, bool spawnThreads=true, bool betaIsZero>
      void execute_expert() noexcept;

      /**
       * Executes the transposition. This functions requires that the plan has
       * already been created via the createPlan() function.
       */
      void execute() noexcept;

      void print() noexcept;

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
      void setParallelStrategy(int id) noexcept { selectedParallelStrategyId_ = id; }
      void setLoopOrder(int id) noexcept { selectedLoopOrderId_ = id; }

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
      void getBestLoopOrder( std::vector<int> &loopOrder ) const; //innermost loop idx is stored at dim_-1
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

      const floatType* __restrict__ A_; //!< rawdata pointer for A
      floatType* __restrict__ B_; //!< rawdata pointer for B
      floatType alpha_; //!< scaling factor for A
      floatType beta_; //!< scaling factor for B
      int dim_; //!< dimension of the tensor
      std::vector<size_t> sizeA_; //!< size of A
      std::vector<int> perm_; //!< permutation 
      std::vector<size_t> outerSizeA_; //!< outer sizes of A
      std::vector<size_t> outerSizeB_;  //!< outer sizes of B
      std::vector<size_t> lda_;  //!< strides for all dimensions of A (first dimension has a stride of 1)
      std::vector<size_t> ldb_;  //!< strides for all dimensions of B (first dimension has a stride of 1)
      std::vector<int> threadIds_; //!< OpenMP threadIds of the threads involed in the transposition
      int numThreads_;
      int selectedParallelStrategyId_;
      int selectedLoopOrderId_;
      bool conjA_;
#ifdef _OPENMP
      omp_lock_t writelock;
#endif

      std::shared_ptr<Plan> masterPlan_; 
      SelectionMethod selectionMethod_;
      int maxAutotuningCandidates_;
      static constexpr int blocking_micro_ = REGISTER_BITS / 8 / sizeof(floatType);
      static constexpr int blocking_ = blocking_micro_ * 4;

      static constexpr int infoLevel_ = 0; // determines which auxiliary messages should be printed
};


extern template class Transpose<float>;
extern template class Transpose<double>;
extern template class Transpose<FloatComplex>;
extern template class Transpose<DoubleComplex>;

}
