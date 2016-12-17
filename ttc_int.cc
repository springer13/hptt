#include <list>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>

#include <stdio.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <complex.h>
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

void getAvailableParallelism(const std::vector<int> &size, const std::vector<int>&perm, const int dim, const int blocking, std::vector<int> &numTasksPerLoop)
{
   numTasksPerLoop.resize(dim);
   for(int loopIdx=0; loopIdx < dim; ++loopIdx){
      int inc = 1;
      if( loopIdx == 0 || loopIdx == perm[0] )
         inc = blocking;
      numTasksPerLoop[loopIdx] = (size[loopIdx] + inc - 1) / inc;
      //std::cout << "parallelsim available in loop "<< loopIdx << ": "<<numTasksPerLoop[loopIdx] << std::endl;
   }
}

void preferedLoopOrderForParallelization( const std::vector<int> &size, const std::vector<int> &perm, const int dim, std::vector<int> &loopOrder)
{
   // prefer non-stride-1 loops for parallelization
   for(int loopIdx=1; loopIdx < dim; ++loopIdx){
      if( loopIdx != perm[0] and loopIdx != perm[1] )
         loopOrder.push_back(loopIdx);
   }
   if( size[perm[0]] < 150 )
   {
      if( !hasItem(loopOrder, 0) )
         loopOrder.push_back(0);
      if( !hasItem(loopOrder, perm[1]) )
         loopOrder.push_back(perm[1]);
   } else {
      if( !hasItem(loopOrder, perm[1]) )
         loopOrder.push_back(perm[1]);
      if( !hasItem(loopOrder, 0) )
         loopOrder.push_back(0);
   }
   if( !hasItem(loopOrder, perm[0]) )
      loopOrder.push_back(perm[0]);

   assert( loopOrder.size() == dim );
}

void getParallelismStrategy(const std::vector<int> &size, const std::vector<int>&perm, 
      const int dim, const int numThreads, const int blocking, std::vector<int> &numThreadsAtLoop)
{
   std::vector<int> primeFactors;
   getPrimeFactors( numThreads, primeFactors );

   std::vector<int> numTasksPerLoop;
   getAvailableParallelism(size, perm, dim, blocking, numTasksPerLoop);

   std::vector<int> loopOrder; 
   preferedLoopOrderForParallelization( size, perm, dim, loopOrder);

   numThreadsAtLoop.resize(dim);
   for(int i=0; i < dim; ++i)
      numThreadsAtLoop[i] = 1;

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

int verifyParameter(const int *size, const int* perm, const int* outerSizeA, const int* outerSizeB, const int dim)
{
   if ( dim < 2 )
      return 1;

   int found[dim];
   for(int i=0;i < dim ; ++i)
      found[i] = 0;

   for(int i=0;i < dim ; ++i)
   {
      if( size[i] <= 0 )
         return 2;
      found[ perm[i] ] = 1;
   }

   for(int i=0;i < dim ; ++i)
      if( found[i] <= 0 )
         return 3;

   if ( outerSizeA != NULL )
      for(int i=0;i < dim ; ++i)
         if ( outerSizeA[i] < size[i] )
            return 4;

   if ( outerSizeB != NULL )
      for(int i=0;i < dim ; ++i)
         if ( outerSizeB[i] < size[perm[i]] )
            return 5;

   return 0;
}

void computeLeadingDimensions( const std::vector<int> &size, const std::vector<int> &perm, int dim,
                               const std::vector<int> &outerSizeA, const std::vector<int> &outerSizeB, 
                               std::vector<int> &lda, std::vector<int> &ldb )
{
   lda[0] = 1;
   if( outerSizeA[0] == -1 )
      for(int i=1;i < dim ; ++i)
         lda[i] = lda[i-1] * size[i-1];
   else
      for(int i=1;i < dim ; ++i)
         lda[i] = outerSizeA[i-1] * lda[i-1];

   ldb[0] = 1;
   if( outerSizeB[0] == -1 )
      for(int i=1;i < dim ; ++i)
         ldb[i] = ldb[i-1] * size[perm[i-1]];
   else
      for(int i=1;i < dim ; ++i)
         ldb[i] = outerSizeB[i-1] * ldb[i-1];
}

int fuseIndices(const int *outerSizeA, const int *outerSizeB, const int *size, const int* perm, const int dim,
                 std::vector<int> &outerSizeA_, std::vector<int> &outerSizeB_, std::vector<int> &size_, std::vector<int> &perm_)
{
   // fuses indices whenever possible 
   // For instance:
   // perm=3,1,2,0 & size=10,11,12,13  becomes: perm=2,1,0 & size=10,11*12,13

   std::list< std::tuple<int, int> > fusedIndices;

   int dim_ = 0;
   for(int i=0;i < dim ; ++i)
   {
      size_[i] = size[i];
      if( outerSizeA != NULL )
         outerSizeA_[i] = outerSizeA[i];
      else
         outerSizeA_[i] = -1;
      if( outerSizeB != NULL )
         outerSizeB_[i] = outerSizeB[i];
      else
         outerSizeB_[i] = -1;
   }

   // correct perm
   for(int i=0;i < dim ; ++i){
      perm_[dim_] = perm[i];
      // merge indices if the two consecutive entries are identical
      while(i+1 < dim && perm[i] + 1 == perm[i+1] 
            && (outerSizeA == NULL || size[perm[i]] == outerSizeA[perm[i]]) 
            && (outerSizeB == NULL || size[perm[i]] == outerSizeB[i]) ){ 
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
      size_[std::get<0>(tup)] *= size[std::get<1>(tup)];
      if( outerSizeA != NULL )
         outerSizeA_[std::get<0>(tup)] *= outerSizeA[std::get<1>(tup)];
      if( outerSizeB != NULL ){
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
         size_[currentValue] = size_[minValue];
         currentValue++;
      }
#ifdef DEBUG
      printf("perm: ");
      for(int i=0;i < dim ; ++i)
         printf("%d ",perm[i]);
      printf("perm_new: ");
      for(int i=0;i < dim_ ; ++i)
         printf("%d ",perm_[i]);
      printf("sizes_new: ");
      for(int i=0;i < dim_ ; ++i)
         printf("%d ",size_[i]);
#endif
   }

   return dim_;
}

void getLoopOrder(const std::vector<int> &perm, const int dim, std::vector<int> &loopOrder)
{
   // TODO use heuristics
   for(int i=0; i < dim; ++i)
      loopOrder[i] = perm[dim-1-i]; // loop-order according to: reversed(perm_)
}

void sTranspose_int( const float* __restrict__ A, float* __restrict__ B, const __m256 alpha, const __m256 beta, const node_t* plan)
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

void createPlans(const std::vector<int> &outerSizeA, const std::vector<int> &outerSizeB, const std::vector<int> &size, 
                 const std::vector<int> &perm, const int dim, const int numThreads, std::vector<plan_t*> &plans)
{
   int emitCode = 0; // only for DEBUGGING

   const int blocking = 32;
   std::vector<int> lda(dim);
   std::vector<int> ldb(dim);
   computeLeadingDimensions( size, perm, dim, outerSizeA, outerSizeB, lda, ldb );
   std::vector<int> numThreadsAtLoop(dim);
   getParallelismStrategy(size, perm, dim, numThreads, blocking, numThreadsAtLoop);

   std::vector<int> loopOrder(dim);
   getLoopOrder(perm, dim, loopOrder);

   plan_t *plan = (plan_t*) malloc(sizeof(plan_t));
   plan->numThreads = numThreads;
   plan->localPlans = (node_t **) malloc(sizeof(node_t*) * numThreads);

#pragma omp parallel num_threads(numThreads)
   {
      int threadId = omp_get_thread_num();
      node_t *localPlan = (node_t*) malloc(sizeof(node_t));
      node_t *currentPtr = localPlan;

      int posStride1A_inB = findPos(0, perm);
      int posStride1B_inA = perm[0];

      if( perm[0] == 0 ){ printf("TODO\n"); exit(-1); } // TODO

      int numThreadsPerComm = numThreads; //global communicator
      int threadIdComm = threadId;
      // create loops
      for(int i=0; i < dim; ++i){
         int index = loopOrder[i];

         if( index == 0 || index == perm[0] )
            currentPtr->inc = blocking;
         else
            currentPtr->inc = 1;

         const int numSubCommunicators = numThreadsAtLoop[index];

         const int numParallelismAvailable = (size[index] + currentPtr->inc - 1) / currentPtr->inc;
         const int workPerThread = numParallelismAvailable / numSubCommunicators;
         
         numThreadsPerComm /= numSubCommunicators; //numThreads in next comminicator
         const int commId = (threadIdComm/numThreadsPerComm);
         threadIdComm = threadIdComm % numThreadsPerComm; // local threadId in next Comminicator
         if( numParallelismAvailable  % numSubCommunicators != 0 ){
            printf("ERROR: TODO: parallelism not devisible\n");
            exit(-1);
         }
         //printf("%d: loop %d uses %d, CommId: %d, localThreadId: %d\n",threadId, index, numSubCommunicators, commId, threadIdComm );

         currentPtr->start = commId * workPerThread * currentPtr->inc;
         currentPtr->end = std::min( size[index], (commId+1) * workPerThread * currentPtr->inc );

         currentPtr->lda = lda[index];
         currentPtr->ldb = ldb[findPos(index, perm)];
         currentPtr->next = (node_t*) malloc(sizeof(node_t));

         if ( emitCode ){
            char underscores_old[10];
            for(int j=0;j < i ; ++j)
               underscores_old[j] = '_';
            underscores_old[i] = '\0';
            char underscores[10];
            for(int j=0;j <= i ; ++j)
               underscores[j] = '_';
            underscores[i+1] = '\0';

            printf("for(int i%d = %d; i%d < %d; i%d += %d){\n",index,currentPtr->start,index,currentPtr->end,index,currentPtr->inc);
            printf("  const float *A%s = &A%s[i%d * lda%d]; float *B%s = &B%s[i%d * ldb%d];\n",underscores,underscores_old,index,index,underscores,underscores_old,index,findPos(index, perm));
         }

         currentPtr = currentPtr->next;
      }

      //macro-kernel
      currentPtr->start = -1;
      currentPtr->end = -1;
      currentPtr->inc = -1;
      currentPtr->lda = lda[ posStride1B_inA ];
      currentPtr->ldb = ldb[ posStride1A_inB ];
      currentPtr->next = nullptr;
      if ( emitCode ){
         char underscores[20];
         for(int j=0;j < dim ; ++j)
            underscores[j] = '_';
         underscores[dim] = '\0';

         printf("  sTranspose16x16(A%s, lda%d, B%s, ldb%d, reg_alpha, reg_beta);\n",underscores,posStride1B_inA, underscores, posStride1A_inB);
         for(int j=0;j < dim ; ++j)
            printf("  }\n");
      }
      plan->localPlans[omp_get_thread_num()] = localPlan;
   }

   plans.push_back(plan);
}

plan_t* selectPlan(std::vector<plan_t*> &plans)
{
   return plans[0]; // TODO
}


}

