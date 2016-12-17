#include <list>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>

#include <stdio.h>
#include <complex.h>

#include "ttc.h"

#if defined(__ICC) || defined(__INTEL_COMPILER)
#define INLINE __forceinline
#elif defined(__GNUC__) || defined(__GNUG__)
#define INLINE __attribute__((always_inline))
#endif

namespace ttc {

void getAvailableParallelism(const std::vector<int> &size, const std::vector<int>&perm, const int dim, const int blocking, std::vector<int> &numTasksPerLoop);


void preferedLoopOrderForParallelization( const std::vector<int> &size, const std::vector<int> &perm, const int dim, std::vector<int> &loopOrder);

void getParallelismStrategy(const std::vector<int> &size, const std::vector<int>&perm, 
      const int dim, const int numThreads, const int blocking, std::vector<int> &numThreadsAtLoop);


int verifyParameter(const int *size, const int* perm, const int* outerSizeA, const int* outerSizeB, const int dim);

void computeLeadingDimensions( const std::vector<int> &size, const std::vector<int> &perm, int dim,
                               const std::vector<int> &outerSizeA, const std::vector<int> &outerSizeB, 
                               std::vector<int> &lda, std::vector<int> &ldb );


int fuseIndices(const int *outerSizeA, const int *outerSizeB, const int *size, const int* perm, const int dim,
                 std::vector<int> &outerSizeA_, std::vector<int> &outerSizeB_, std::vector<int> &size_, std::vector<int> &perm_);


void getLoopOrder(const std::vector<int> &perm, const int dim, std::vector<int> &loopOrder);


void sTranspose_int( const float* __restrict__ A, float* __restrict__ B, const __m256 alpha, const __m256 beta, const node_t* plan);

}

