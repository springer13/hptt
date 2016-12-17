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
   int errorCode = verifyParameter(size, perm, outerSizeA, outerSizeB, dim);
   if( errorCode > 0 ) {
      printf("Error code: %d\n", errorCode);
      exit(-1);
   }

   std::vector<int> outerSizeA_(dim);
   std::vector<int> outerSizeB_(dim);
   std::vector<int> size_(dim);
   std::vector<int> perm_(dim);
   int dim_ = fuseIndices(outerSizeA, outerSizeB, size, perm, dim, 
         outerSizeA_, outerSizeB_, size_, perm_);

   std::vector<plan_t*> allPlans;
   createPlans(outerSizeA_, outerSizeB_, size_, perm_, dim_, numThreads, allPlans);

   return selectPlan( allPlans );
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

