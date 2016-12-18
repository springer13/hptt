#include <tuple>
#include <vector>
#include <list>
#include <algorithm>
#include <iostream>

#include <float.h>
#include <stdio.h>
#include <assert.h>

#include <omp.h>

#include "ttc.h"
#include "ttc_utils.h"

#if defined(__ICC) || defined(__INTEL_COMPILER)
#define INLINE __forceinline
#elif defined(__GNUC__) || defined(__GNUG__)
#define INLINE __attribute__((always_inline))
#endif

namespace ttc {

//B_ji = alpha * A_ij + beta * B_ji
static INLINE void sTranspose8x8(const float* __restrict__ A, const int lda, float* __restrict__ B, const int ldb  ,const __m256 &reg_alpha ,const __m256 &reg_beta)
{
   //Load A
   __m256 rowA0 = _mm256_load_ps((A + 0 +0*lda));
   __m256 rowA1 = _mm256_load_ps((A + 0 +1*lda));
   __m256 rowA2 = _mm256_load_ps((A + 0 +2*lda));
   __m256 rowA3 = _mm256_load_ps((A + 0 +3*lda));
   __m256 rowA4 = _mm256_load_ps((A + 0 +4*lda));
   __m256 rowA5 = _mm256_load_ps((A + 0 +5*lda));
   __m256 rowA6 = _mm256_load_ps((A + 0 +6*lda));
   __m256 rowA7 = _mm256_load_ps((A + 0 +7*lda));

   //8x8 transpose micro kernel
   __m256 r121, r139, r120, r138, r71, r89, r70, r88, r11, r1, r55, r29, r10, r0, r54, r28;
   r28 = _mm256_unpacklo_ps( rowA4, rowA5 );
   r54 = _mm256_unpacklo_ps( rowA6, rowA7 );
    r0 = _mm256_unpacklo_ps( rowA0, rowA1 );
   r10 = _mm256_unpacklo_ps( rowA2, rowA3 );
   r29 = _mm256_unpackhi_ps( rowA4, rowA5 );
   r55 = _mm256_unpackhi_ps( rowA6, rowA7 );
    r1 = _mm256_unpackhi_ps( rowA0, rowA1 );
   r11 = _mm256_unpackhi_ps( rowA2, rowA3 );
   r88 = _mm256_shuffle_ps( r28, r54, 0x44 );
   r70 = _mm256_shuffle_ps( r0, r10, 0x44 );
   r89 = _mm256_shuffle_ps( r28, r54, 0xee );
   r71 = _mm256_shuffle_ps( r0, r10, 0xee );
   r138 = _mm256_shuffle_ps( r29, r55, 0x44 );
   r120 = _mm256_shuffle_ps( r1, r11, 0x44 );
   r139 = _mm256_shuffle_ps( r29, r55, 0xee );
   r121 = _mm256_shuffle_ps( r1, r11, 0xee );
   rowA0 = _mm256_permute2f128_ps( r88, r70, 0x2 );
   rowA1 = _mm256_permute2f128_ps( r89, r71, 0x2 );
   rowA2 = _mm256_permute2f128_ps( r138, r120, 0x2 );
   rowA3 = _mm256_permute2f128_ps( r139, r121, 0x2 );
   rowA4 = _mm256_permute2f128_ps( r88, r70, 0x13 );
   rowA5 = _mm256_permute2f128_ps( r89, r71, 0x13 );
   rowA6 = _mm256_permute2f128_ps( r138, r120, 0x13 );
   rowA7 = _mm256_permute2f128_ps( r139, r121, 0x13 );

   //Scale A
   rowA0 = _mm256_mul_ps(rowA0, reg_alpha);
   rowA1 = _mm256_mul_ps(rowA1, reg_alpha);
   rowA2 = _mm256_mul_ps(rowA2, reg_alpha);
   rowA3 = _mm256_mul_ps(rowA3, reg_alpha);
   rowA4 = _mm256_mul_ps(rowA4, reg_alpha);
   rowA5 = _mm256_mul_ps(rowA5, reg_alpha);
   rowA6 = _mm256_mul_ps(rowA6, reg_alpha);
   rowA7 = _mm256_mul_ps(rowA7, reg_alpha);

   //Load B
   __m256 rowB0 = _mm256_load_ps((B + 0 +0*ldb));
   __m256 rowB1 = _mm256_load_ps((B + 0 +1*ldb));
   __m256 rowB2 = _mm256_load_ps((B + 0 +2*ldb));
   __m256 rowB3 = _mm256_load_ps((B + 0 +3*ldb));
   __m256 rowB4 = _mm256_load_ps((B + 0 +4*ldb));
   __m256 rowB5 = _mm256_load_ps((B + 0 +5*ldb));
   __m256 rowB6 = _mm256_load_ps((B + 0 +6*ldb));
   __m256 rowB7 = _mm256_load_ps((B + 0 +7*ldb));

   rowB0 = _mm256_add_ps( _mm256_mul_ps(rowB0, reg_beta), rowA0);
   rowB1 = _mm256_add_ps( _mm256_mul_ps(rowB1, reg_beta), rowA1);
   rowB2 = _mm256_add_ps( _mm256_mul_ps(rowB2, reg_beta), rowA2);
   rowB3 = _mm256_add_ps( _mm256_mul_ps(rowB3, reg_beta), rowA3);
   rowB4 = _mm256_add_ps( _mm256_mul_ps(rowB4, reg_beta), rowA4);
   rowB5 = _mm256_add_ps( _mm256_mul_ps(rowB5, reg_beta), rowA5);
   rowB6 = _mm256_add_ps( _mm256_mul_ps(rowB6, reg_beta), rowA6);
   rowB7 = _mm256_add_ps( _mm256_mul_ps(rowB7, reg_beta), rowA7);
   //Store B
   _mm256_store_ps((B + 0 + 0 * ldb), rowB0);
   _mm256_store_ps((B + 0 + 1 * ldb), rowB1);
   _mm256_store_ps((B + 0 + 2 * ldb), rowB2);
   _mm256_store_ps((B + 0 + 3 * ldb), rowB3);
   _mm256_store_ps((B + 0 + 4 * ldb), rowB4);
   _mm256_store_ps((B + 0 + 5 * ldb), rowB5);
   _mm256_store_ps((B + 0 + 6 * ldb), rowB6);
   _mm256_store_ps((B + 0 + 7 * ldb), rowB7);
}

template<int blockingA, int blockingB>
static INLINE void sTranspose(const float* __restrict__ A, const int lda, float* __restrict__ B, const int ldb  ,const __m256 &reg_alpha ,const __m256 &reg_beta)
{
   //invoke micro-transpose
   if(blockingA > 0 && blockingB > 0 )
   sTranspose8x8(A, lda, B, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 0 && blockingB > 8 )
   sTranspose8x8(A + 8 * lda, lda, B + 8, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 0 && blockingB > 16 )
   sTranspose8x8(A + 16 * lda, lda, B + 16, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 0 && blockingB > 24 )
   sTranspose8x8(A + 24 * lda, lda, B + 24, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 8 && blockingB > 0 )
   sTranspose8x8(A + 8, lda, B + 8 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 8 && blockingB > 8 )
   sTranspose8x8(A + 8 + 8 * lda, lda, B + 8 + 8 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 8 && blockingB > 16 )
   sTranspose8x8(A + 8 + 16 * lda, lda, B + 16 + 8 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 8 && blockingB > 24 )
   sTranspose8x8(A + 8 + 24 * lda, lda, B + 24 + 8 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 16 && blockingB > 0 )
   sTranspose8x8(A + 16, lda, B + 16 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 16 && blockingB > 8 )
   sTranspose8x8(A + 16 + 8 * lda, lda, B + 8 + 16 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 16 && blockingB > 16 )
   sTranspose8x8(A + 16 + 16 * lda, lda, B + 16 + 16 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 16 && blockingB > 24 )
   sTranspose8x8(A + 16 + 24 * lda, lda, B + 24 + 16 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 24 && blockingB > 0 )
   sTranspose8x8(A + 24, lda, B + 24 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 24 && blockingB > 8 )
   sTranspose8x8(A + 24 + 8 * lda, lda, B + 8 + 24 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 24 && blockingB > 16 )
   sTranspose8x8(A + 24 + 16 * lda, lda, B + 16 + 24 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   if(blockingA > 24 && blockingB > 24 )
   sTranspose8x8(A + 24 + 24 * lda, lda, B + 24 + 24 * ldb, ldb  , reg_alpha , reg_beta);
}

void sTranspose_int( const float* __restrict__ A, float* __restrict__ B, const __m256 alpha, const __m256 beta, const ComputeNode* plan)
{
   const int end = plan->end - (plan->inc - 1);
   const int inc = plan->inc;
   const int lda_ = plan->lda;
   const int ldb_ = plan->ldb;

   if( plan->next != nullptr )
      for(int i = plan->start; i < end; i+= inc)
         // recurse
         sTranspose_int( &A[i*lda_], &B[i*ldb_], alpha, beta, plan->next);
   else 
      // invoke macro-kernel
      sTranspose<32,32>(A, lda_, B, ldb_, alpha, beta);
}

void Transpose::trashCaches()
{
#pragma omp parallel for num_threads(numThreads_)
   for(int i=0;i < trashSize_ ; ++i){
      trash1_[i] += 1.01 * trash2_[i];
   }
}
void Transpose::createPlan()
{
   std::vector<ComputeNode*> allPlans;
   createPlans(allPlans);

   rootNodes_ = selectPlan( allPlans );
   
   //delete all other plans
   for( int i=0; i < allPlans.size(); i++ ){
      if( allPlans[i] != nullptr && allPlans[i] != rootNodes_ )
      {
         delete[] allPlans[i];
         allPlans[i] = nullptr;
      }
   }
}
void Transpose::executeEstimate(const ComputeNode *rootNodes) noexcept
{
   if( rootNodes == nullptr ) {
      printf("ERROR: plan has not yet been created.\n");
      exit(-1);
   }
   
   //broadcast reg_alpha
   __m256 reg_alpha = _mm256_set1_ps(0.0); // do not alter the content of B
   //broadcast reg_beta
   __m256 reg_beta = _mm256_set1_ps(1.0); // do not alter the content of B

#pragma omp parallel num_threads(numThreads_)
   sTranspose_int( A_, B_, reg_alpha, reg_beta, &rootNodes[omp_get_thread_num()] );
}

void Transpose::execute() noexcept
{
   if( rootNodes_ == nullptr ) {
      printf("ERROR: plan has not yet been created.\n");
      exit(-1);
   }
   
   //broadcast reg_alpha
   __m256 reg_alpha = _mm256_set1_ps(alpha_);
   //broadcast reg_beta
   __m256 reg_beta = _mm256_set1_ps(beta_);

#pragma omp parallel num_threads(numThreads_)
   sTranspose_int( A_, B_, reg_alpha, reg_beta, &rootNodes_[omp_get_thread_num()] );
}

void Transpose::getAvailableParallelism( std::vector<int> &numTasksPerLoop ) const
{
   numTasksPerLoop.resize(dim_);
   for(int loopIdx=0; loopIdx < dim_; ++loopIdx){
      int inc = 1;
      if( loopIdx == 0 || loopIdx == perm_[0] )
         inc = blocking_;
      numTasksPerLoop[loopIdx] = (sizeA_[loopIdx] + inc - 1) / inc;
      //std::cout << "parallelsim available in loop "<< loopIdx << ": "<<numTasksPerLoop[loopIdx] << std::endl;
   }
}

void Transpose::preferedLoopOrderForParallelization( std::vector<int> &loopOrder ) const
{
   // prefer non-stride-1 loops for parallelization
   for(int loopIdx=1; loopIdx < dim_; ++loopIdx){
      if( loopIdx != perm_[0] && loopIdx != perm_[1] )
         loopOrder.push_back(loopIdx);
   }
   if( sizeA_[perm_[0]] < 150 )
   {
      if( !hasItem(loopOrder, 0) )
         loopOrder.push_back(0);
      if( !hasItem(loopOrder, perm_[1]) )
         loopOrder.push_back(perm_[1]);
   } else {
      if( !hasItem(loopOrder, perm_[1]) )
         loopOrder.push_back(perm_[1]);
      if( !hasItem(loopOrder, 0) )
         loopOrder.push_back(0);
   }
   if( !hasItem(loopOrder, perm_[0]) )
      loopOrder.push_back(perm_[0]);

   assert( loopOrder.size() == dim_ );
}

void Transpose::getParallelismStrategy(std::vector<int> &numThreadsAtLoop) const
{
   numThreadsAtLoop.resize(dim_, 1);
   for( int i = 0; i < dim_; i++ ) //BUGFIX, this is requriered for whatever reason
      numThreadsAtLoop[i] = 1;
   printVector<int>(numThreadsAtLoop, "numThreadsAtLoop");
   if( numThreads_ == 1 )
      return;

   std::vector<int> primeFactors;
   getPrimeFactors( numThreads_, primeFactors );
   printVector(primeFactors , "primeFactors");

   std::vector<int> numTasksPerLoop;
   this->getAvailableParallelism( numTasksPerLoop );
   printVector(numTasksPerLoop, "numTasksPerLoop");

   std::vector<int> loopOrder; 
   this->preferedLoopOrderForParallelization( loopOrder );
   printVector(loopOrder, "loopOrder");

   std::vector<int> unmatchedPrimeFactors;
   for( auto p : primeFactors ) {
      int done = 0;
      //find a match for every primefactor
      for( auto loopIdx : loopOrder ){
         if( numTasksPerLoop[loopIdx] % p == 0 ){
            numTasksPerLoop[loopIdx] /= p;
            numThreadsAtLoop[loopIdx] *= p;
            done = 1;
            break;
         }
      }
      if( !done )
         unmatchedPrimeFactors.push_back(p);
   }
   if( unmatchedPrimeFactors.size() > 0 )
   {
      printf("unmactched: \n");
      for( auto p : unmatchedPrimeFactors) 
         std::cout<< p << " ";
      std::cout<< "\n";
      for( auto loopIdx : loopOrder )
         printf("%d: %d\n",loopIdx , numThreadsAtLoop[loopIdx]); 
      exit(-1);
   }
}

void Transpose::verifyParameter(const int *size, const int* perm, const int* outerSizeA, const int* outerSizeB, const int dim) const
{
   if ( dim < 2 ) {
      printf("ERROR: dim invalid\n");
      exit(-1);
   }

   std::vector<int> found(dim, 0);

   for(int i=0;i < dim ; ++i)
   {
      if( size[i] <= 0 ) {
         printf("ERROR: size invalid\n");
         exit(-1);
      }
      found[ perm[i] ] = 1;
   }

   for(int i=0;i < dim ; ++i)
      if( found[i] <= 0 ) {
         printf("ERROR: permutation invalid\n");
         exit(-1);
      }

   if ( outerSizeA != NULL )
      for(int i=0;i < dim ; ++i)
         if ( outerSizeA[i] < size[i] ) {
            printf("ERROR: outerSizeA invalid\n");
            exit(-1);
         }

   if ( outerSizeB != NULL )
      for(int i=0;i < dim ; ++i)
         if ( outerSizeB[i] < size[perm[i]] ) {
            printf("ERROR: outerSizeB invalid\n");
            exit(-1);
         }
}

void Transpose::computeLeadingDimensions()
{
   lda_[0] = 1;
   if( outerSizeA_[0] == -1 )
      for(int i=1;i < dim_ ; ++i)
         lda_[i] = lda_[i-1] * sizeA_[i-1];
   else
      for(int i=1;i < dim_ ; ++i)
         lda_[i] = outerSizeA_[i-1] * lda_[i-1];

   ldb_[0] = 1;
   if( outerSizeB_[0] == -1 )
      for(int i=1;i < dim_ ; ++i)
         ldb_[i] = ldb_[i-1] * sizeA_[perm_[i-1]];
   else
      for(int i=1;i < dim_ ; ++i)
         ldb_[i] = outerSizeB_[i-1] * ldb_[i-1];
}

/**
  * \brief fuses indices whenever possible 
  * \detailed For instance:
  *           perm=3,1,2,0 & size=10,11,12,13  becomes: perm=2,1,0 & size=10,11*12,13
  * \return This function will initialize sizeA_, perm_, outerSizeA_, outersize_ and dim_
*/
void Transpose::fuseIndices(const int *sizeA, const int* perm, const int *outerSizeA, const int *outerSizeB, const int dim)
{
   std::list< std::tuple<int, int> > fusedIndices;

   dim_ = 0;
   for(int i=0;i < dim ; ++i)
   {
      sizeA_[i] = sizeA[i];
      if( outerSizeA != NULL && outerSizeA != nullptr)
         outerSizeA_[i] = outerSizeA[i];
      else
         outerSizeA_[i] = -1;
      if( outerSizeB != NULL && outerSizeB != nullptr )
         outerSizeB_[i] = outerSizeB[i];
      else
         outerSizeB_[i] = -1;
   }

   // correct perm
   for(int i=0;i < dim ; ++i){
      perm_[dim_] = perm[i];
      // merge indices if the two consecutive entries are identical
      while(i+1 < dim && perm[i] + 1 == perm[i+1] 
            && ((outerSizeA == NULL && outerSizeA != nullptr) || sizeA[perm[i]] == outerSizeA[perm[i]]) 
            && ((outerSizeB == NULL && outerSizeB != nullptr) || sizeA[perm[i]] == outerSizeB[i]) ){ 
#ifdef DEBUG
         printf("MERGING indices %d and %d\n",perm[i], perm[i+1]); 
#endif
         fusedIndices.push_back( std::make_tuple(perm_[dim_],perm[i+1]) );
         i++;
      }
      dim_++;
   }

   // correct sizes and outer-sizes
   for( auto tup : fusedIndices )
   {
      sizeA_[std::get<0>(tup)] *= sizeA[std::get<1>(tup)];
      if( outerSizeA != NULL && outerSizeA != nullptr )
         outerSizeA_[std::get<0>(tup)] *= outerSizeA[std::get<1>(tup)];
      if( outerSizeB != NULL && outerSizeB != nullptr){
         int pos1 = findPos(std::get<0>(tup), perm, dim);
         int pos2 = findPos(std::get<1>(tup), perm, dim);
         outerSizeB_[pos1] *= outerSizeB[pos2];
      }
   }

   // remove gaps in the perm, if requried (e.g., perm=3,1,0 -> 2,1,0)
   if ( dim_ != dim ) {
      int currentValue = 0;
      for(int i=0;i < dim_; ++i){
         //find smallest element in perm_ and rename it to currentValue
         int minValue = 1000000;
         int minPos = -1;
         for(int pos=0; pos < dim_; ++pos){
            if ( perm_[pos] >= currentValue && perm_[pos] < minValue) {
               minValue = perm_[pos];
               minPos = pos;
            }
         }
#ifdef DEBUG
         printf("perm[%d]: %d -> %d\n",minPos, perm_[minPos], currentValue);
#endif
         perm_[minPos] = currentValue; // minValue renamed to currentValue
         sizeA_[currentValue] = sizeA_[minValue];
         currentValue++;
      }
      outerSizeA_.resize(dim_);
      outerSizeB_.resize(dim_);
      sizeA_.resize(dim_);
      perm_.resize(dim_);

#ifdef DEBUG
      printf("perm: ");
      for(int i=0;i < dim ; ++i)
         printf("%d ",perm[i]);
      printf("perm_new: ");
      for(int i=0;i < dim_ ; ++i)
         printf("%d ",perm_[i]);
      printf("sizes_new: ");
      for(int i=0;i < dim_ ; ++i)
         printf("%d ",sizeA_[i]);
#endif
   }
}

double Transpose::loopCostHeuristic( const std::vector<int> &loopOrder ) const
{
   // penalize different loop-oders differently
   double loopPenalty[dim_];
   loopPenalty[dim_-1] = 0;
   double penalty = 10;
   for(int i=dim_ - 2;i >= 0; i--){
      loopPenalty[i] = penalty;
      penalty *= 2;
   }
   // loopPenalty looks like this: [...,40,20,10,0]
   // Rationale: as inner-most indices move towards the outer-most loops, they
   // should be penalized more

   double loopCost = 0.0;
   double importance = 1.0;
   for(int i=0;i < dim_ ; ++i){
      int posA = findPos(i, loopOrder);
      int posB = findPos(perm_[i], loopOrder);
      loopCost += (loopPenalty[posA] + loopPenalty[posB] * 1.01 ) * importance; // B is slighly more important than A
      importance /= 2; // indices become less and less important as we go towards the outer-most indices
   }

   return loopCost;
}

void Transpose::getLoopOrders(std::vector<std::vector<int> > &loopOrders) const
{
   loopOrders.clear();
   std::vector<int> loopOrder;
   for(int i = 0; i < dim_; i++)
      loopOrder.push_back(i);

   // create all loopOrders
   do {
      loopOrders.push_back(loopOrder);
   } while(std::next_permutation(loopOrder.begin(), loopOrder.end()));


   // sort according to loop heuristic
   std::sort(loopOrders.begin(), loopOrders.end(), 
         [this](const std::vector<int> loopOrder1, const std::vector<int> loopOrder2)
         { 
            return this->loopCostHeuristic(loopOrder1) < this->loopCostHeuristic(loopOrder2); 
         });

   if( loopOrders.size() != factorial(dim_) ){
      printf("Internal error: number of loop-orders incorrect.\n");
      exit(-1);
   }
#ifdef DEBUG
   for(auto loopOrder : loopOrders)
   {
      printVector(loopOrder,"loop");
      printf("penalty: %f\n",loopCostHeuristic(loopOrder));
   }
#endif
}

void Transpose::createPlans( std::vector<ComputeNode*> &plans ) const
{
   std::vector<int> numThreadsAtLoop(dim_);
   this->getParallelismStrategy( numThreadsAtLoop );
   printVector(numThreadsAtLoop , "numThreadsAtLoop");

   std::vector<std::vector<int> > loopOrders;
   this->getLoopOrders(loopOrders);

   for( auto loopOrder : loopOrders)
   {
      ComputeNode *rootNodes = new ComputeNode[numThreads_];

#pragma omp parallel num_threads(numThreads_)
      {
         int threadId = omp_get_thread_num();
         ComputeNode *currentNode = &rootNodes[threadId];

         int posStride1A_inB = findPos(0, perm_);
         int posStride1B_inA = perm_[0];

         if( perm_[0] == 0 ){ printf("TODO\n"); exit(-1); } // TODO

         int numThreadsPerComm = numThreads_; //global communicator
         int threadIdComm = threadId;
         // create loops
         for(int i=0; i < dim_; ++i){
            int index = loopOrder[i];

            if( index == 0 || index == perm_[0] )
               currentNode->inc = blocking_;
            else
               currentNode->inc = 1;

            const int numSubCommunicators = numThreadsAtLoop[index];

            const int numParallelismAvailable = (sizeA_[index] + currentNode->inc - 1) / currentNode->inc;
            const int workPerThread = numParallelismAvailable / numSubCommunicators;

            numThreadsPerComm /= numSubCommunicators; //numThreads in next comminicator
            const int commId = (threadIdComm/numThreadsPerComm);
            threadIdComm = threadIdComm % numThreadsPerComm; // local threadId in next Comminicator
            if( numParallelismAvailable  % numSubCommunicators != 0 ){
               printf("ERROR: TODO: parallelism not devisible\n");
               exit(-1);
            }
            //printf("%d: loop %d uses %d, CommId: %d, localThreadId: %d\n",threadId, index, numSubCommunicators, commId, threadIdComm );

            currentNode->start = commId * workPerThread * currentNode->inc;
            currentNode->end = std::min( sizeA_[index], (commId+1) * workPerThread * currentNode->inc );

            currentNode->lda = lda_[index];
            currentNode->ldb = ldb_[findPos(index, perm_)];
            currentNode->next = new ComputeNode;

            currentNode = currentNode->next;
         }

         //macro-kernel
         currentNode->start = -1;
         currentNode->end = -1;
         currentNode->inc = -1;
         currentNode->lda = lda_[ posStride1B_inA ];
         currentNode->ldb = ldb_[ posStride1A_inB ];
         currentNode->next = nullptr;
      }
      plans.push_back(rootNodes);
   }
}

/**
 * Estimates the time in seconds for the given computeTree
 */
float Transpose::estimateExecutionTime( const ComputeNode *rootNodes )
{
   this->trashCaches();

   double startTime = omp_get_wtime();
   this->executeEstimate(rootNodes);
   double elapsedTime = omp_get_wtime() - startTime;

   const double minMeasurementTime = 0.1; // in seconds

   // do aleast 3 repetitions or spent at least 'minMeasurementTime' seconds for each candidate
   int nRepeat = std::min(3, (int) std::ceil(minMeasurementTime / elapsedTime));

   //execute just a few iterations and exterpolate the result
   startTime = omp_get_wtime();
   for(int i=0;i < nRepeat ; ++i) //ATTENTION: we are not clearing the caches inbetween runs
      this->executeEstimate( rootNodes );
   elapsedTime = omp_get_wtime() - startTime;
   elapsedTime /= nRepeat;

#ifdef DEBUG
   printf("Estimated time: %.3e ms.\n",elapsedTime * 1000); 
#endif
   return elapsedTime; 
}

double Transpose::getTimeLimit() const
{
   if( selectionMethod_ == ESTIMATE )
      return 0.0;
   else if( selectionMethod_ == MEASURE)
      return 10.;   // 10s
   else if( selectionMethod_ == PATIENT)
      return 60.;   // 1m
   else if( selectionMethod_ == CRAZY )
      return 3600.; // 1h
   else{
      printf("ERROR: sectionMethod unknown.\n");
      exit(-1);
   }
   return -1;
}

ComputeNode* Transpose::selectPlan( const std::vector<ComputeNode*> &plans)
{
   if( plans.size() <= 0 ){
      printf("Internal error: not enough plans generated.\n");
      exit(-1);
   }

   double timeLimit = this->getTimeLimit(); //in seconds

   float minTime = FLT_MAX;
   ComputeNode* bestPlan = plans[0];

   if( plans.size() > 1 )
   {
      double startTime = omp_get_wtime();
      for( auto p : plans )
      {
         if( omp_get_wtime() - startTime >= timeLimit ) // timelimit reached
            break;

         float estimatedTime = this->estimateExecutionTime( p );

         if( estimatedTime < minTime ){
            bestPlan = p;
            minTime = estimatedTime;
         }
      }
   }
   return bestPlan;
}

}
















