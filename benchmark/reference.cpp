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

// Check if 'floatType' is complex
template <typename floatType>
class is_complex {
  template <
      template <typename> typename _floatType, typename T,
      typename = typename std::enable_if<
          std::is_same<_floatType<T>, std::complex<T>>::value, bool>::type>
  bool static constexpr _is_complex(_floatType<T>) {
    return true;
  }
  template <typename _floatType>
  bool static constexpr _is_complex(_floatType) {
    return false;
  }

 public:
  static constexpr auto value =
      _is_complex(typename std::remove_reference<
                  typename std::remove_cv<floatType>::type>::type());
};


// Re-define 'conj' to make sure it return a floating point if
// argument is a floating point
template <typename floatType, typename = typename std::enable_if<
                                  !is_complex<floatType>::value, bool>::type>
floatType conj(floatType &&x) {
  return x;
}

template<typename floatType>
void transpose_ref( uint32_t *size, uint32_t *perm, int dim, 
      const floatType* __restrict__ A, floatType alpha, 
      floatType* __restrict__ B, floatType beta, const bool conjA)
{
   // compute stride for all dimensions w.r.t. A
   uint32_t strideA[dim];
   strideA[0] = 1;
   for(int i=1; i < dim; ++i)
      strideA[i] = strideA[i-1] * size[i-1];

   // combine all non-stride-one dimensions of B into a single dimension for
   // maximum parallelism
   uint32_t sizeOuter = 1;
   for(int i=0; i < dim; ++i)
      if( i != perm[0] )
         sizeOuter *= size[i]; 

   uint32_t sizeInner = size[perm[0]];

   // This implementation traverses the output tensor in a linear fashion
   
#pragma omp parallel for
   for(uint32_t j=0; j < sizeOuter; ++j)
   {
      uint32_t offsetA = 0;
      uint32_t offsetB = 0;
      uint32_t j_tmp = j;
      for(int i=1; i < dim; ++i)
      {
         int current_index = j_tmp % size[perm[i]];
         j_tmp /= size[perm[i]];
         offsetA += current_index * strideA[perm[i]];
      }

      const floatType* __restrict__ A_ = A + offsetA;
      floatType* __restrict__ B_ = B + j*sizeInner;

      uint32_t strideAinner = strideA[perm[0]];

      if( beta == (floatType) 0 )
         for(int i=0; i < sizeInner; ++i)
            if( conjA )
               B_[i] = alpha * conj(A_[i * strideAinner]);
            else
               B_[i] = alpha * A_[i * strideAinner];
      else
         for(int i=0; i < sizeInner; ++i)
            if( conjA )
               B_[i] = alpha * conj(A_[i * strideAinner]) + beta * B_[i];
            else
               B_[i] = alpha * A_[i * strideAinner] + beta * B_[i];
   }
}

template void transpose_ref<float>( uint32_t *size, uint32_t *perm, int dim, 
      const float* __restrict__ A, float alpha, 
      float* __restrict__ B, float beta, const bool conjA);
template void transpose_ref<FloatComplex>( uint32_t *size, uint32_t *perm, int dim, 
      const FloatComplex* __restrict__ A, FloatComplex alpha, 
      FloatComplex* __restrict__ B, FloatComplex beta, const bool conjA);
template void transpose_ref<double>( uint32_t *size, uint32_t *perm, int dim, 
      const double* __restrict__ A, double alpha, 
      double* __restrict__ B, double beta, const bool conjA);
template void transpose_ref<DoubleComplex>( uint32_t *size, uint32_t *perm, int dim, 
      const DoubleComplex* __restrict__ A, DoubleComplex alpha, 
      DoubleComplex* __restrict__ B, DoubleComplex beta, const bool conjA);
