#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>

#include <memory>
#include <vector>
#include <numeric>
#include <string>
#include <algorithm>
#include <iostream>
#include <complex>

#include "defines.h"


template<typename floatType>
void transpose_ref( uint32_t *size, uint32_t *perm, int dim, 
      const floatType* __restrict__ A, floatType alpha, int *outerSizeA, int *offsetA, int innerStrideA,  
      floatType* __restrict__ B, floatType beta, int *outerSizeB, int *offsetB, int innerStrideB, const bool conjA)
{
   std::vector<int> tempOuterSizeA, tempOuterSizeB, tempOffsetA, tempOffsetB, tempPointerB;

   // Stride One is location of 0 in perm. Perm[0] may not be B stride one unless perm[0] == 0
   // perm provided yields positions in A data from a B order index
   // perm calculated below relates positions in B data to an A order index
   tempPointerB.resize(dim);
   for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
         if (i == perm[j])
            tempPointerB[i] = j;

   // Use default values if any of the pointers are nullptr
   if (outerSizeA == nullptr) {
      tempOuterSizeA.resize(dim);
      for (int i = 0; i < dim; i++) tempOuterSizeA[i] = size[i];
      outerSizeA = tempOuterSizeA.data();
   }
    
   if (outerSizeB == nullptr) {
      tempOuterSizeB.resize(dim);
      for (int i = 0; i < dim; i++) tempOuterSizeB[i] = size[perm[i]];
      outerSizeB = tempOuterSizeB.data();
   }

   if (offsetA == nullptr) {
      tempOffsetA.resize(dim);  // Default to zeros
      for (int i = 0; i < dim; i++) tempOffsetA[i] = 0;
      offsetA = tempOffsetA.data();
   }

   if (offsetB == nullptr) {
      tempOffsetB.resize(dim);  // Default to zeros
      for (int i = 0; i < dim; i++) tempOffsetB[i] = 0;
      offsetB = tempOffsetB.data();
   }

   // compute stride for all dimensions w.r.t. A (like lda)
   uint32_t strideA[dim];
   strideA[0] = innerStrideA;
   for(int i=1; i < dim; ++i) 
      strideA[i] = strideA[i-1] * outerSizeA[i-1];

   // compute stride for all dimensions w.r.t. B (like ldb)
   uint32_t strideB[dim];
   strideB[0] = innerStrideB;
   for(int i=1; i < dim; ++i)
      strideB[i] = strideB[i-1] * outerSizeB[i-1];

   // combine all non-stride-one dimensions of B into a single dimension for
   // maximum parallelism
   uint32_t sizeOuter = 1;
   for(int i=0; i < dim; ++i)
      if( i != perm[0] )
         sizeOuter *= size[i];

   uint32_t sizeInner = size[perm[0]];
   uint32_t strideAinner = strideA[perm[0]];

   // This implementation traverses the output tensor in a linear fashion
   
#pragma omp parallel for
   for(uint32_t j=0; j < sizeOuter; ++j)
   {
      uint32_t indexOffsetA = 0;
      uint32_t indexOffsetB = 0;
      uint32_t j_tmp_A = j;
      uint32_t j_tmp_B = j;
      for(int i=1; i < dim; ++i)
      {
         int current_index_A = j_tmp_A % size[perm[i]];
         j_tmp_A /= size[perm[i]];
         j_tmp_B /= size[perm[i]];
         indexOffsetA += (current_index_A + offsetA[perm[i]]) * strideA[perm[i]];
         indexOffsetB += (j_tmp_B + 1) * offsetB[i] * strideB[i];
         indexOffsetB += j_tmp_B * (outerSizeB[i] - offsetB[i] - size[perm[i]]) * strideB[i];
      }

      const floatType* __restrict__ A_ = A + indexOffsetA;
      floatType* __restrict__ B_ = B + indexOffsetB + (offsetB[0] * innerStrideB) + (j * outerSizeB[0] * innerStrideB);

      if( beta == (floatType) 0 )
         for(int i=0; i < sizeInner; ++i) {
#ifdef DEBUG
            //printf("A[%d] = %e -> B[%d] = %e\n", ((i + offsetA[perm[0]]) * strideAinner) + indexOffsetA, A_[(i + offsetA[perm[0]]) * strideAinner], (i * innerStrideB) + indexOffsetB + (offsetB[0] * innerStrideB) + (j * outerSizeB[0] * innerStrideB), B_[i * innerStrideB]);
#endif
            if( conjA )
               B_[i * innerStrideB] = alpha * std::conj(A_[(i + offsetA[perm[0]]) * strideAinner]).real();
            else
               B_[i * innerStrideB] = alpha * A_[(i + offsetA[perm[0]]) * strideAinner];}
      else
         for(int i=0; i < sizeInner; ++i) {
#ifdef DEBUG
            //printf("A[%d] = %e -> B[%d] = %e\n", ((i + offsetA[perm[0]]) * strideAinner) + indexOffsetA, A_[(i + offsetA[perm[0]]) * strideAinner], (i * innerStrideB) + indexOffsetB + (offsetB[0] * innerStrideB) + (j * outerSizeB[0] * innerStrideB), B_[i * innerStrideB]);
#endif
            if( conjA )
               B_[i * innerStrideB] = alpha * std::conj(A_[(i + offsetA[perm[0]]) * strideAinner]).real() + beta * B_[i * innerStrideB];
            else
               B_[i * innerStrideB] = alpha * A_[(i + offsetA[perm[0]]) * strideAinner] + beta * B_[i * innerStrideB];}
   }
}

template void transpose_ref<float>( uint32_t *size, uint32_t *perm, int dim, 
      const float* __restrict__ A, float alpha, int *outerSizeA, int *offsetA, int innerStrideA,
      float* __restrict__ B, float beta, int *outerSizeB, int *offsetB, int innerStrideB, const bool conjA);
template void transpose_ref<FloatComplex>( uint32_t *size, uint32_t *perm, int dim, 
      const FloatComplex* __restrict__ A, FloatComplex alpha, int *outerSizeA, int *offsetA, int innerStrideA,
      FloatComplex* __restrict__ B, FloatComplex beta, int *outerSizeB, int *offsetB, int innerStrideB, const bool conjA);
template void transpose_ref<double>( uint32_t *size, uint32_t *perm, int dim, 
      const double* __restrict__ A, double alpha, int *outerSizeA, int *offsetA, int innerStrideA,
      double* __restrict__ B, double beta, int *outerSizeB, int *offsetB, int innerStrideB, const bool conjA);
template void transpose_ref<DoubleComplex>( uint32_t *size, uint32_t *perm, int dim, 
      const DoubleComplex* __restrict__ A, DoubleComplex alpha, int *outerSizeA, int *offsetA, int innerStrideA,
      DoubleComplex* __restrict__ B, DoubleComplex beta, int *outerSizeB, int *offsetB, int innerStrideB, const bool conjA);

