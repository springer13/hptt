#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>

#include <stdio.h>
#include <assert.h>

#include <omp.h>

#include "ttc.h"
#include "ttc_int.h"
#include "ttc_utils.h"

namespace ttc {

plan_t* createPlan(const int *outerSizeA, const int *outerSizeB, const int *size, const int* perm, const int dim, int numThreads)
{
   int emitCode = 0; // only for DEBUGGING

   int errorCode = verifyParameter(size, perm, outerSizeA, outerSizeB, dim);
   if( errorCode > 0 ) {
      printf("Error code: %d\n", errorCode);
      exit(-1);
   }

   const int blocking = 32;
   std::vector<int> outerSizeA_(dim);
   std::vector<int> outerSizeB_(dim);
   std::vector<int> size_(dim);
   std::vector<int> perm_(dim);
   int dim_ = fuseIndices(outerSizeA, outerSizeB, size, perm, dim, 
         outerSizeA_, outerSizeB_, size_, perm_);

   std::vector<int> lda(dim_);
   std::vector<int> ldb(dim_);
   computeLeadingDimensions( size_, perm_, dim_, outerSizeA_, outerSizeB_, lda, ldb );

   std::vector<int> loopOrder(dim_);
   getLoopOrder(perm_, dim_, loopOrder);
   std::vector<int> numThreadsAtLoop(dim_);
   getParallelismStrategy(size_, perm_, dim_, numThreads, blocking, numThreadsAtLoop);

   plan_t *plan = (plan_t*) malloc(sizeof(plan_t));
   plan->numThreads = numThreads;
   plan->localPlans = (node_t **) malloc(sizeof(node_t*) * numThreads);

#pragma omp parallel num_threads(numThreads)
   {
      int threadId = omp_get_thread_num();
      node_t *localPlan = (node_t*) malloc(sizeof(node_t));
      node_t *currentPtr = localPlan;

      int posStride1A_inB = findPos(0, perm_);
      int posStride1B_inA = perm_[0];

      if( perm_[0] == 0 ){ printf("TODO\n"); exit(-1); } // TODO

      int numThreadsPerComm = numThreads; //global communicator
      int threadIdComm = threadId;
      // create loops
      for(int i=0; i < dim_; ++i){
         int index = loopOrder[i];

         if( index == 0 || index == perm_[0] )
            currentPtr->inc = blocking;
         else
            currentPtr->inc = 1;

         const int numSubCommunicators = numThreadsAtLoop[index];

         const int numParallelismAvailable = (size_[index] + currentPtr->inc - 1) / currentPtr->inc;
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
         currentPtr->end = std::min( size_[index], (commId+1) * workPerThread * currentPtr->inc );

         currentPtr->lda = lda[index];
         currentPtr->ldb = ldb[findPos(index, perm_)];
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
            printf("  const float *A%s = &A%s[i%d * lda%d]; float *B%s = &B%s[i%d * ldb%d];\n",underscores,underscores_old,index,index,underscores,underscores_old,index,findPos(index, perm_));
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
         for(int j=0;j < dim_ ; ++j)
            underscores[j] = '_';
         underscores[dim_] = '\0';

         printf("  sTranspose16x16(A%s, lda%d, B%s, ldb%d, reg_alpha, reg_beta);\n",underscores,posStride1B_inA, underscores, posStride1A_inB);
         for(int j=0;j < dim_ ; ++j)
            printf("  }\n");
      }
      plan->localPlans[omp_get_thread_num()] = localPlan;
   }
   return plan;
}

void sTranspose( const float* __restrict__ A, float* __restrict__ B, const float alpha, const float beta, plan_t* plan)
{
   //broadcast reg_alpha
   __m256 reg_alpha = _mm256_set1_ps(alpha);
   //broadcast reg_beta
   __m256 reg_beta = _mm256_set1_ps(beta);

#pragma omp parallel num_threads(plan->numThreads)
   sTranspose_int( A, B, reg_alpha, reg_beta, plan->localPlans[omp_get_thread_num()] );
}

}

