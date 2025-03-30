#include <stdlib.h>

template<typename floatType>
void transpose_ref( uint32_t *size, uint32_t *perm, int dim, 
      const floatType* __restrict__ A, floatType alpha, int *outerSizeA, int *offsetA, int innerStrideA, 
      floatType* __restrict__ B, floatType beta, int *outerSizeB, int *offsetB, int innerStrideB, const bool conjA);
