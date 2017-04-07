#include <tuple>
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <iostream>
#include <cmath>

#include <float.h>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>

#include <omp.h>

#include "hptt.h"
#include "hptt_utils.h"

#if defined(__ICC) || defined(__INTEL_COMPILER)
#define INLINE __forceinline
#elif defined(__GNUC__) || defined(__GNUG__)
#define INLINE __attribute__((always_inline))
#endif

namespace hptt {

template<typename floatType>
static INLINE void prefetch(const floatType* A, const int lda)
{
   constexpr int blocking_micro_ = 256 / 8 / sizeof(floatType);
   for(int i=0;i < blocking_micro_; ++i)
      _mm_prefetch((char*)(A + i * lda), _MM_HINT_T2);
}


template <typename floatType, int betaIsZero>
struct micro_kernel{};

template <int betaIsZero>
struct micro_kernel<double, betaIsZero>
{
    static void execute(const double* __restrict__ A, const size_t lda, double* __restrict__ B, const size_t ldb, const double alpha ,const double beta)
    {
       __m256d reg_alpha = _mm256_set1_pd(alpha); // do not alter the content of B
       __m256d reg_beta = _mm256_set1_pd(beta); // do not alter the content of B
       //Load A
       __m256d rowA0 = _mm256_load_pd((A + 0 +0*lda));
       __m256d rowA1 = _mm256_load_pd((A + 0 +1*lda));
       __m256d rowA2 = _mm256_load_pd((A + 0 +2*lda));
       __m256d rowA3 = _mm256_load_pd((A + 0 +3*lda));

       //4x4 transpose micro kernel
       __m256d r4, r34, r3, r33;
       r33 = _mm256_shuffle_pd( rowA2, rowA3, 0x3 );
       r3 = _mm256_shuffle_pd( rowA0, rowA1, 0x3 );
       r34 = _mm256_shuffle_pd( rowA2, rowA3, 0xc );
       r4 = _mm256_shuffle_pd( rowA0, rowA1, 0xc );
       rowA0 = _mm256_permute2f128_pd( r34, r4, 0x2 );
       rowA1 = _mm256_permute2f128_pd( r33, r3, 0x2 );
       rowA2 = _mm256_permute2f128_pd( r33, r3, 0x13 );
       rowA3 = _mm256_permute2f128_pd( r34, r4, 0x13 );

       //Scale A
       rowA0 = _mm256_mul_pd(rowA0, reg_alpha);
       rowA1 = _mm256_mul_pd(rowA1, reg_alpha);
       rowA2 = _mm256_mul_pd(rowA2, reg_alpha);
       rowA3 = _mm256_mul_pd(rowA3, reg_alpha);

       //Load B
       if( !betaIsZero )
       {
          __m256d rowB0 = _mm256_load_pd((B + 0 +0*ldb));
          __m256d rowB1 = _mm256_load_pd((B + 0 +1*ldb));
          __m256d rowB2 = _mm256_load_pd((B + 0 +2*ldb));
          __m256d rowB3 = _mm256_load_pd((B + 0 +3*ldb));

          rowB0 = _mm256_add_pd( _mm256_mul_pd(rowB0, reg_beta), rowA0);
          rowB1 = _mm256_add_pd( _mm256_mul_pd(rowB1, reg_beta), rowA1);
          rowB2 = _mm256_add_pd( _mm256_mul_pd(rowB2, reg_beta), rowA2);
          rowB3 = _mm256_add_pd( _mm256_mul_pd(rowB3, reg_beta), rowA3);
          //Store B
          _mm256_store_pd((B + 0 + 0 * ldb), rowB0);
          _mm256_store_pd((B + 0 + 1 * ldb), rowB1);
          _mm256_store_pd((B + 0 + 2 * ldb), rowB2);
          _mm256_store_pd((B + 0 + 3 * ldb), rowB3);
       } else {
          //Store B
          _mm256_store_pd((B + 0 + 0 * ldb), rowA0);
          _mm256_store_pd((B + 0 + 1 * ldb), rowA1);
          _mm256_store_pd((B + 0 + 2 * ldb), rowA2);
          _mm256_store_pd((B + 0 + 3 * ldb), rowA3);
       }
    }
};

template <int betaIsZero>
struct micro_kernel<float, betaIsZero>
{
    static void execute(const float* __restrict__ A, const size_t lda, float* __restrict__ B, const size_t ldb, const float alpha ,const float beta)
    {
       __m256 reg_alpha = _mm256_set1_ps(alpha); // do not alter the content of B
       __m256 reg_beta = _mm256_set1_ps(beta); // do not alter the content of B
       //Load A
       __m256 rowA0 = _mm256_load_ps((A +0*lda));
       __m256 rowA1 = _mm256_load_ps((A +1*lda));
       __m256 rowA2 = _mm256_load_ps((A +2*lda));
       __m256 rowA3 = _mm256_load_ps((A +3*lda));
       __m256 rowA4 = _mm256_load_ps((A +4*lda));
       __m256 rowA5 = _mm256_load_ps((A +5*lda));
       __m256 rowA6 = _mm256_load_ps((A +6*lda));
       __m256 rowA7 = _mm256_load_ps((A +7*lda));

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
       if( !betaIsZero )
       {
          __m256 rowB0 = _mm256_load_ps((B +0*ldb));
          __m256 rowB1 = _mm256_load_ps((B +1*ldb));
          __m256 rowB2 = _mm256_load_ps((B +2*ldb));
          __m256 rowB3 = _mm256_load_ps((B +3*ldb));
          __m256 rowB4 = _mm256_load_ps((B +4*ldb));
          __m256 rowB5 = _mm256_load_ps((B +5*ldb));
          __m256 rowB6 = _mm256_load_ps((B +6*ldb));
          __m256 rowB7 = _mm256_load_ps((B +7*ldb));

          rowB0 = _mm256_add_ps( _mm256_mul_ps(rowB0, reg_beta), rowA0);
          rowB1 = _mm256_add_ps( _mm256_mul_ps(rowB1, reg_beta), rowA1);
          rowB2 = _mm256_add_ps( _mm256_mul_ps(rowB2, reg_beta), rowA2);
          rowB3 = _mm256_add_ps( _mm256_mul_ps(rowB3, reg_beta), rowA3);
          rowB4 = _mm256_add_ps( _mm256_mul_ps(rowB4, reg_beta), rowA4);
          rowB5 = _mm256_add_ps( _mm256_mul_ps(rowB5, reg_beta), rowA5);
          rowB6 = _mm256_add_ps( _mm256_mul_ps(rowB6, reg_beta), rowA6);
          rowB7 = _mm256_add_ps( _mm256_mul_ps(rowB7, reg_beta), rowA7);
          //Store B
          _mm256_store_ps((B + 0 * ldb), rowB0);
          _mm256_store_ps((B + 1 * ldb), rowB1);
          _mm256_store_ps((B + 2 * ldb), rowB2);
          _mm256_store_ps((B + 3 * ldb), rowB3);
          _mm256_store_ps((B + 4 * ldb), rowB4);
          _mm256_store_ps((B + 5 * ldb), rowB5);
          _mm256_store_ps((B + 6 * ldb), rowB6);
          _mm256_store_ps((B + 7 * ldb), rowB7);
       } else {
          _mm256_store_ps((B + 0 * ldb), rowA0);
          _mm256_store_ps((B + 1 * ldb), rowA1);
          _mm256_store_ps((B + 2 * ldb), rowA2);
          _mm256_store_ps((B + 3 * ldb), rowA3);
          _mm256_store_ps((B + 4 * ldb), rowA4);
          _mm256_store_ps((B + 5 * ldb), rowA5);
          _mm256_store_ps((B + 6 * ldb), rowA6);
          _mm256_store_ps((B + 7 * ldb), rowA7);
       }
    }
};


template<typename floatType>
static void streamingStore( floatType* out, const floatType *in );

template<>
void streamingStore<float>( float* out, const float*in ){
   _mm256_stream_ps(out, _mm256_load_ps(in));
}
template<>
void streamingStore<double>( double* out, const double*in ){
   _mm256_stream_pd(out, _mm256_load_pd(in));
}

template<int blockingA, int blockingB, int betaIsZero, typename floatType>
static INLINE void macro_kernel(const floatType* __restrict__ A, const floatType* __restrict__ Anext, const size_t lda, 
                                   floatType* __restrict__ B, const floatType* __restrict__ Bnext, const size_t ldb,
                                   const floatType alpha ,const floatType beta)
{
   constexpr int blocking_ = 128 / sizeof(floatType);
   constexpr int blocking_micro_ = 256 / 8 / sizeof(floatType);

   bool useStreamingStores = betaIsZero && (blockingB*sizeof(floatType))%64 == 0 && ((uint64_t)B)%32 == 0 && (ldb*sizeof(floatType))%32 == 0;

   floatType *Btmp = B;
   size_t ldb_tmp = ldb;
   floatType buffer[blockingA * blockingB];// __attribute__((aligned(64)));
   if( useStreamingStores ){
      Btmp = buffer;
      ldb_tmp = blockingB;
   }

   if( blockingA == blocking_ && blockingB == blocking_ )
   {
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
      prefetch<floatType>(Anext + (0 * lda + 0), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (0 * lda + 0), lda, Btmp + (0 * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (0 * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      prefetch<floatType>(Anext + (2*blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (2*blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (3*blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (3*blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      micro_kernel<floatType,betaIsZero>::execute(A + (0 * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero>::execute(A + (blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      micro_kernel<floatType,betaIsZero>::execute(A + (2*blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero>::execute(A + (3*blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (2*blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      prefetch<floatType>(Anext + (0 * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (0 * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (blocking_micro_ * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (blocking_micro_ * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (2*blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      prefetch<floatType>(Anext + (2*blocking_micro_ * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (2*blocking_micro_ * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (3*blocking_micro_ * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (3*blocking_micro_ * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (3*blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      micro_kernel<floatType,betaIsZero>::execute(A + (0 * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero>::execute(A + (blocking_micro_ * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (3*blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      micro_kernel<floatType,betaIsZero>::execute(A + (2*blocking_micro_ * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero>::execute(A + (3*blocking_micro_ * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
   }else if( blockingA == 2*blocking_micro_ && blockingB == blocking_ ) {
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
      prefetch<floatType>(Anext + (0 * lda + 0), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (0 * lda + 0), lda, Btmp + (0 * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (0 * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      prefetch<floatType>(Anext + (2*blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (2*blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (3*blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (3*blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      micro_kernel<floatType,betaIsZero>::execute(A + (0 * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero>::execute(A + (blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      micro_kernel<floatType,betaIsZero>::execute(A + (2*blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero>::execute(A + (3*blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
   }else if( blockingA == blocking_ && blockingB == 2*blocking_micro_) {
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
      prefetch<floatType>(Anext + (0 * lda + 0), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (0 * lda + 0), lda, Btmp + (0 * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      micro_kernel<floatType,betaIsZero>::execute(A + (0 * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero>::execute(A + (blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (2*blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      prefetch<floatType>(Anext + (0 * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (0 * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (blocking_micro_ * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero>::execute(A + (blocking_micro_ * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !useStreamingStores )
         prefetch<floatType>(Bnext + (3*blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      micro_kernel<floatType,betaIsZero>::execute(A + (0 * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero>::execute(A + (blocking_micro_ * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
   } else {
      //invoke micro-transpose
      if(blockingA > 0 && blockingB > 0 )
         micro_kernel<floatType,betaIsZero>::execute(A, lda, Btmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 0 && blockingB > blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + blocking_micro_ * lda, lda, Btmp + blocking_micro_, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 0 && blockingB > 2*blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + 2*blocking_micro_ * lda, lda, Btmp + 2*blocking_micro_, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 0 && blockingB > 3*blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + 3*blocking_micro_ * lda, lda, Btmp + 3*blocking_micro_, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > blocking_micro_ && blockingB > 0 )
         micro_kernel<floatType,betaIsZero>::execute(A + blocking_micro_, lda, Btmp + blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > blocking_micro_ && blockingB > blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + blocking_micro_ + blocking_micro_ * lda, lda, Btmp + blocking_micro_ + blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > blocking_micro_ && blockingB > 2*blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + blocking_micro_ + 2*blocking_micro_ * lda, lda, Btmp + 2*blocking_micro_ + blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > blocking_micro_ && blockingB > 3*blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + blocking_micro_ + 3*blocking_micro_ * lda, lda, Btmp + 3*blocking_micro_ + blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 2*blocking_micro_ && blockingB > 0 )
         micro_kernel<floatType,betaIsZero>::execute(A + 2*blocking_micro_, lda, Btmp + 2*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 2*blocking_micro_ && blockingB > blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + 2*blocking_micro_ + blocking_micro_ * lda, lda, Btmp + blocking_micro_ + 2*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 2*blocking_micro_ && blockingB > 2*blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + 2*blocking_micro_ + 2*blocking_micro_ * lda, lda, Btmp + 2*blocking_micro_ + 2*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 2*blocking_micro_ && blockingB > 3*blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + 2*blocking_micro_ + 3*blocking_micro_ * lda, lda, Btmp + 3*blocking_micro_ + 2*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 3*blocking_micro_ && blockingB > 0 )
         micro_kernel<floatType,betaIsZero>::execute(A + 3*blocking_micro_, lda, Btmp + 3*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 3*blocking_micro_ && blockingB > blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + 3*blocking_micro_ + blocking_micro_ * lda, lda, Btmp + blocking_micro_ + 3*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 3*blocking_micro_ && blockingB > 2*blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + 3*blocking_micro_ + 2*blocking_micro_ * lda, lda, Btmp + 2*blocking_micro_ + 3*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 3*blocking_micro_ && blockingB > 3*blocking_micro_ )
         micro_kernel<floatType,betaIsZero>::execute(A + 3*blocking_micro_ + 3*blocking_micro_ * lda, lda, Btmp + 3*blocking_micro_ + 3*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);
   }

   // write buffer to main-memory via non-temporal stores
   if( useStreamingStores )
      for( int i = 0; i < blockingA; i++){
         for( int j = 0; j < blockingB; j+=blocking_micro_)
            streamingStore<floatType>(B + i * ldb + j, buffer + i * ldb_tmp + j);
      }
}

template<int blockingA, int blockingB, int betaIsZero, typename floatType>
void sTranspose_int( const floatType* __restrict__ A, const floatType* __restrict__ Anext, 
                     floatType* __restrict__ B, const floatType* __restrict__ Bnext, const floatType alpha, const floatType beta, const ComputeNode* plan)
{
   const int32_t end = plan->end - (plan->inc - 1);
   const int32_t inc = plan->inc;
   const size_t lda_ = plan->lda;
   const size_t ldb_ = plan->ldb;

   const int32_t remainder = (plan->end - plan->start) % inc;

   constexpr int blocking_ = 128 / sizeof(floatType);
   constexpr int blocking_micro_ = 256 / 8 / sizeof(floatType);

   if( plan->next->next != nullptr ){
      // recurse
      int32_t i;
      for(i = plan->start; i < end; i+= inc)
      {
         if( i + inc < end )
            sTranspose_int<blockingA, blockingB, betaIsZero>( &A[i*lda_], &A[(i+1)*lda_], &B[i*ldb_], &B[(i+1)*ldb_], alpha, beta, plan->next);
         else
            sTranspose_int<blockingA, blockingB, betaIsZero>( &A[i*lda_], Anext, &B[i*ldb_], Bnext, alpha, beta, plan->next);
      }
      // remainder
      if( blocking_/2 >= blocking_micro_ && (i + blocking_/2) <= plan->end ){
         if( lda_ == 1)
            sTranspose_int<blocking_/2, blockingB, betaIsZero>( &A[i*lda_], Anext, &B[i*ldb_], Bnext, alpha, beta, plan->next);
         else if( ldb_ == 1)
            sTranspose_int<blockingA, blocking_/2, betaIsZero>( &A[i*lda_], Anext, &B[i*ldb_], Bnext, alpha, beta, plan->next);
         i+=blocking_/2;
      }
//      if( blocking_/4 >= blocking_micro_ && (i + blocking_/4) <= plan->end ){
//         if( lda_ == 1)
//            sTranspose_int<blocking_/4, blockingB, betaIsZero>( &A[i*lda_], Anext, &B[i*ldb_], Bnext, alpha, beta, plan->next);
//         else if( ldb_ == 1)
//            sTranspose_int<blockingA, blocking_/4, betaIsZero>( &A[i*lda_], Anext, &B[i*ldb_], Bnext, alpha, beta, plan->next);
//         i+=blocking_/4;
//      }

   } else {
      const size_t lda_macro_ = plan->next->lda;
      const size_t ldb_macro_ = plan->next->ldb;
      // invoke macro-kernel
      
      int32_t i;
      for(i = plan->start; i < end; i+= inc)
         if( i + inc < end )
            macro_kernel<blockingA, blockingB, betaIsZero,floatType>(&A[i*lda_], &A[(i+1)*lda_], lda_macro_, &B[i*ldb_], &B[(i+1)*ldb_], ldb_macro_, alpha, beta);
         else
            macro_kernel<blockingA, blockingB, betaIsZero,floatType>(&A[i*lda_], Anext, lda_macro_, &B[i*ldb_], Bnext, ldb_macro_, alpha, beta);
      // remainder
      if( blocking_/2 >= blocking_micro_ && (i + blocking_/2) <= plan->end ){
         if( lda_ == 1)
            macro_kernel<blocking_/2, blockingB, betaIsZero,floatType>(&A[i*lda_], Anext, lda_macro_, &B[i*ldb_], Bnext, ldb_macro_, alpha, beta);
         else if( ldb_ == 1)
            macro_kernel<blockingA, blocking_/2, betaIsZero,floatType>(&A[i*lda_], Anext, lda_macro_, &B[i*ldb_], Bnext, ldb_macro_, alpha, beta);
         i+=blocking_/2;
      }
//      if( blocking_/4 >= blocking_micro_ && (i + blocking_/4) <= plan->end ){
//         if( lda_ == 1)
//            macro_kernel<blocking_/4, blockingB, betaIsZero,floatType>(&A[i*lda_], Anext, lda_macro_, &B[i*ldb_], Bnext, ldb_macro_, alpha, beta);
//         else if( ldb_ == 1)
//            macro_kernel<blockingA, blocking_/4, betaIsZero,floatType>(&A[i*lda_], Anext, lda_macro_, &B[i*ldb_], Bnext, ldb_macro_, alpha, beta);
//         i+=blocking_/4;
//      }
   }
}

template<int betaIsZero, typename floatType>
void sTranspose_int_constStride1( const floatType* __restrict__ A, floatType* __restrict__ B, const floatType alpha, const floatType beta, const ComputeNode* plan)
{
   const int32_t end = plan->end - (plan->inc - 1);
   constexpr int32_t inc = 1; // TODO
   const size_t lda_ = plan->lda;
   const size_t ldb_ = plan->ldb;

   if( plan->next != nullptr )
      for(int i = plan->start; i < end; i+= inc)
         // recurse
         sTranspose_int_constStride1<betaIsZero>( &A[i*lda_], &B[i*ldb_], alpha, beta, plan->next);
   else 
      if( !betaIsZero )
      {
         for(int32_t i = plan->start; i < end; i+= inc)
            B[i] = alpha * A[i] + beta * B[i];
      } else {
         for(int32_t i = plan->start; i < end; i+= inc)
            B[i] = alpha * A[i];
      }
}


template<typename floatType>
void Transpose<floatType>::executeEstimate(const Plan *plan) noexcept
{
   if( plan == nullptr ) {
      fprintf(stderr,"ERROR: plan has not yet been created.\n");
      exit(-1);
   }
   
#pragma omp parallel num_threads(numThreads_)
   if ( perm_[0] != 0 ) {
      auto rootNode = plan->getRootNode_const( omp_get_thread_num() );
      if( std::fabs(beta_) < 1e-17 ) {
         sTranspose_int<blocking_,blocking_,1,floatType>( A_,A_, B_, B_, 0.0, 1.0, rootNode );
      } else {
         sTranspose_int<blocking_,blocking_,0,floatType>( A_,A_, B_, B_, 0.0, 1.0, rootNode );
      }
   } else {
      auto rootNode = plan->getRootNode_const( omp_get_thread_num() );
      if( std::fabs(beta_) < 1e-17 ) {
         sTranspose_int_constStride1<1,floatType>( A_, B_, 0.0, 1.0, rootNode);
      }else{
         sTranspose_int_constStride1<0,floatType>( A_, B_, 0.0, 1.0, rootNode);
      }
   }
}

template<typename floatType>
void Transpose<floatType>::execute() noexcept
{
   if( masterPlan_ == nullptr ) {
      fprintf(stderr,"ERROR: master plan has not yet been created.\n");
      exit(-1);
   }
   
#pragma omp parallel num_threads(numThreads_)
   if ( perm_[0] != 0 ) {
      auto rootNode = masterPlan_->getRootNode_const( omp_get_thread_num() );
      if( std::fabs(beta_) < 1e-17 ) {
         sTranspose_int<blocking_,blocking_,1,floatType>( A_, A_, B_, B_, alpha_, beta_, rootNode );
      } else {
         sTranspose_int<blocking_,blocking_,0,floatType>( A_, A_, B_, B_, alpha_, beta_, rootNode );
      }
   } else {
      auto rootNode = masterPlan_->getRootNode_const( omp_get_thread_num() );
      if( std::fabs(beta_) < 1e-17 ) {
         sTranspose_int_constStride1<1,floatType>( A_, B_, alpha_, beta_, rootNode);
      } else {
         sTranspose_int_constStride1<0,floatType>( A_, B_, alpha_, beta_, rootNode);
      }
   }
}

template<typename floatType>
int Transpose<floatType>::getIncrement( int loopIdx ) const
{
   int inc = 1;
   if( perm_[0] != 0 ) {
      if( loopIdx == 0 || loopIdx == perm_[0] )
         inc = blocking_;
   } else {
      if( loopIdx == 1 || loopIdx == perm_[1] )
         inc = blocking_constStride1_;
   }
   return inc;
}

template<typename floatType>
void Transpose<floatType>::getAvailableParallelism( std::vector<int> &numTasksPerLoop ) const
{
   numTasksPerLoop.resize(dim_);
   for(int loopIdx=0; loopIdx < dim_; ++loopIdx){
      int inc = this->getIncrement(loopIdx);
      numTasksPerLoop[loopIdx] = (sizeA_[loopIdx] + inc - 1) / inc;
   }
}

template<typename floatType>
void Transpose<floatType>::getAllParallelismStrategies( std::list<int> &primeFactorsToMatch, 
                                             std::vector<int> &availableParallelismAtLoop, 
                                             std::vector<int> &achievedParallelismAtLoop, 
                                             std::vector<std::vector<int> > &parallelismStrategies) const
{
   if( primeFactorsToMatch.size() > 0 ){
      for( auto p : primeFactorsToMatch )
      {
         for( int i = 0; i < dim_; i++ )
         {
            std::list<int> primeFactorsToMatch_(primeFactorsToMatch); 
            std::vector<int> availableParallelismAtLoop_(availableParallelismAtLoop); 
            std::vector<int> achievedParallelismAtLoop_(achievedParallelismAtLoop);

            primeFactorsToMatch_.erase( std::find(primeFactorsToMatch_.begin(), primeFactorsToMatch_.end(), p) );
            availableParallelismAtLoop_[i] = (availableParallelismAtLoop_[i] + p - 1) / p;
            achievedParallelismAtLoop_[i] *= p;

            this->getAllParallelismStrategies( primeFactorsToMatch_, 
                  availableParallelismAtLoop_, 
                  achievedParallelismAtLoop_, 
                  parallelismStrategies);
         }
      }
   } else {
      // avoid duplicates
      if( parallelismStrategies.end() == std::find(parallelismStrategies.begin(), parallelismStrategies.end(), achievedParallelismAtLoop) )
         parallelismStrategies.push_back(achievedParallelismAtLoop);
   }
}

template<typename floatType>
double Transpose<floatType>::parallelismCostHeuristic( const std::vector<int> &achievedParallelismAtLoop ) const
{
   std::vector<int> availableParallelismAtLoop;
   this->getAvailableParallelism( availableParallelismAtLoop);

   double cost = 1;
   // penalize load-imbalance
   for(int loopIdx=0; loopIdx < dim_ ; ++loopIdx){
      if ( achievedParallelismAtLoop[loopIdx] <= 1 ) 
         continue;

      const int blocksPerThread = (availableParallelismAtLoop[loopIdx] + achievedParallelismAtLoop[loopIdx] -1) / achievedParallelismAtLoop[loopIdx];
      int inc = this->getIncrement( loopIdx );
      const int effectiveSize = blocksPerThread * inc * achievedParallelismAtLoop[loopIdx];
      cost *= ((double)(effectiveSize ) / sizeA_[loopIdx]);
   } 

   // penalize parallelization of stride-1 loops
   if( perm_[0] == 0 )
      cost *= std::pow(1.01, achievedParallelismAtLoop[0] - 1); //strongly penalize this case

   cost *= std::pow(1.00010, std::min(16,achievedParallelismAtLoop[0] - 1));        // if at all, prefer ...
   cost *= std::pow(1.00015, std::min(16,achievedParallelismAtLoop[perm_[0]] - 1)); // parallelization in stride-1 of A
   

   const int workPerThread = (availableParallelismAtLoop[perm_[0]] + achievedParallelismAtLoop[perm_[0]] -1) / achievedParallelismAtLoop[perm_[0]];
   if( workPerThread * sizeof(floatType) % 64 != 0 && achievedParallelismAtLoop[perm_[0]] > 1 ){ //avoid false-sharing
      cost *= std::pow(1.00015, std::min(16,achievedParallelismAtLoop[perm_[0]] - 1)); // penalize this parallelization again
   }
   return cost;
}

template<typename floatType>
void Transpose<floatType>::getParallelismStrategies(std::vector<std::vector<int> > &parallelismStrategies) const
{
   parallelismStrategies.clear();
   if( numThreads_ == 1 ){
      parallelismStrategies.push_back(std::vector<int>(dim_, 1));
      return;
   }

   // ATTENTION: we don't care about the case where numThreads_ is a large prime number!!!
   // (sorry, KNC)
   //
   // we factorize numThreads into its prime factors because we have to match
   // every one to a certain loop. In principle every loop could be used to
   // match every primefactor, but some choices are preferable over others.
   // E.g., we want to achive good load-balancing _and_ try to avoid the
   // stride-1 index of B (due to false sharing)
   std::list<int> primeFactors;
   getPrimeFactors( numThreads_, primeFactors );
   if( this->infoLevel_ > 0 )
      printVector(primeFactors,"primes");

   std::vector<int> availableParallelismAtLoop;
   this->getAvailableParallelism( availableParallelismAtLoop);
   if( this->infoLevel_ > 0 )
      printVector(availableParallelismAtLoop,"available Parallelism");

   std::vector<int> achievedParallelismAtLoop (dim_, 1);

   this->getAllParallelismStrategies( primeFactors, 
         availableParallelismAtLoop, 
         achievedParallelismAtLoop, 
         parallelismStrategies);
   
   // sort according to loop heuristic
   std::sort(parallelismStrategies.begin(), parallelismStrategies.end(), 
         [this](const std::vector<int> loopOrder1, const std::vector<int> loopOrder2)
         { 
            return this->parallelismCostHeuristic(loopOrder1) < this->parallelismCostHeuristic(loopOrder2); 
         });

   if( this->infoLevel_ > 1 )
      for( auto strat : parallelismStrategies ){
         printVector(strat,"parallelization");
         printf("cost: %f\n", this->parallelismCostHeuristic( strat ));
      }
}

template<typename floatType>
void Transpose<floatType>::verifyParameter(const int *size, const int* perm, const int* outerSizeA, const int* outerSizeB, const int dim) const
{
   if ( dim < 2 ) {
      fprintf(stderr,"ERROR: dim invalid\n");
      exit(-1);
   }

   std::vector<int> found(dim, 0);

   for(int i=0;i < dim ; ++i)
   {
      if( size[i] <= 0 ) {
         fprintf(stderr,"ERROR: size invalid\n");
         exit(-1);
      }
      found[ perm[i] ] = 1;
   }

   for(int i=0;i < dim ; ++i)
      if( found[i] <= 0 ) {
         fprintf(stderr,"ERROR: permutation invalid\n");
         exit(-1);
      }

   if ( outerSizeA != NULL )
      for(int i=0;i < dim ; ++i)
         if ( outerSizeA[i] < size[i] ) {
            fprintf(stderr,"ERROR: outerSizeA invalid\n");
            exit(-1);
         }

   if ( outerSizeB != NULL )
      for(int i=0;i < dim ; ++i)
         if ( outerSizeB[i] < size[perm[i]] ) {
            fprintf(stderr,"ERROR: outerSizeB invalid\n");
            exit(-1);
         }
}

template<typename floatType>
void Transpose<floatType>::computeLeadingDimensions()
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
template<typename floatType>
void Transpose<floatType>::fuseIndices(const int *sizeA, const int* perm, const int *outerSizeA, const int *outerSizeB, const int dim)
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
            && (outerSizeA == NULL || outerSizeA == nullptr || sizeA[perm[i]] == outerSizeA[perm[i]]) 
            && (outerSizeB == NULL || outerSizeB == nullptr || sizeA[perm[i]] == outerSizeB[i]) ){ 
#ifdef DEBUG
         fprintf(stderr,"MERGING indices %d and %d\n",perm[i], perm[i+1]); 
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
   if( dim_ < 2 || (dim_ == 2 && perm_[0] == 0) ){
      fprintf(stderr,"TODO: support dimension too small: map to copy()\n");
      exit(-1);
   }
}

template<typename floatType>
double Transpose<floatType>::loopCostHeuristic( const std::vector<int> &loopOrder ) const
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

template<typename floatType>
void Transpose<floatType>::getLoopOrders(std::vector<std::vector<int> > &loopOrders) const
{
//   std::vector<int> loopOrder1 { 5, 3, 4, 1, 0, 2 };
//   loopOrders.push_back(loopOrder1 );
//   return;
   loopOrders.clear();
   std::vector<int> loopOrder;
   for(int i = 0; i < dim_; i++)
      loopOrder.push_back(i);

   // create all loopOrders
   do {
      if ( perm_[0] == 0 && loopOrder[dim_-1] != 0 )
         continue; // ATTENTION: we skipp all loop-orders where the stride-1 index is not the inner-most loop iff perm[0] == 0 (both for perf & correctness)

      loopOrders.push_back(loopOrder);
   } while(std::next_permutation(loopOrder.begin(), loopOrder.end()));


   // sort according to loop heuristic
   std::sort(loopOrders.begin(), loopOrders.end(), 
         [this](const std::vector<int> loopOrder1, const std::vector<int> loopOrder2)
         { 
            return this->loopCostHeuristic(loopOrder1) < this->loopCostHeuristic(loopOrder2); 
         });

   if( this->infoLevel_ > 1 )
      for(auto loopOrder : loopOrders) {
         printVector(loopOrder,"loop");
         printf("penalty: %f\n",loopCostHeuristic(loopOrder));
      }
}

template<typename floatType>
void Transpose<floatType>::createPlan()
{
//   printf("entering createPlan()\n");
   std::vector<Plan*> allPlans;
   createPlans(allPlans);

   masterPlan_ = selectPlan( allPlans );
   
   //delete all other plans
   for( int i=0; i < allPlans.size(); i++ ){
      if( allPlans[i] != nullptr && allPlans[i] != masterPlan_ )
      {
         delete allPlans[i];
         allPlans[i] = nullptr;
      }
   }
}

template<typename floatType>
void Transpose<floatType>::createPlans( std::vector<Plan*> &plans ) const
{
   std::vector<std::vector<int> > parallelismStrategies;
   this->getParallelismStrategies(parallelismStrategies);

   std::vector<std::vector<int> > loopOrders;
   this->getLoopOrders(loopOrders);

   if( perm_[0] != 0 && ( sizeA_[0] % 16 != 0 || sizeA_[perm_[0]] % 16 != 0 ) ) {
      fprintf(stderr, "Error/TODO: vectorization of remainder\n");
      exit(-1);
   }

   // combine the loopOrder and parallelismStragegies according to their
   // heuristics, search the space with a growing rectanle (from best to worst,
   // see line marked with ***)
   for( int start= 0; start< std::max( parallelismStrategies.size(), loopOrders.size() ); start++ )
   for( int i = 0; i < parallelismStrategies.size(); i++)
   {
      for( int j = 0; j < loopOrders.size(); j++)
      {
         if( i > start || j > start || (i != start && j != start) ) continue; //these are already done ***

         auto numThreadsAtLoop = parallelismStrategies[i];
         auto loopOrder = loopOrders[j];
         Plan *plan = new Plan(numThreads_, loopOrder, numThreadsAtLoop );

#pragma omp parallel num_threads(numThreads_)
         {
            int threadId = omp_get_thread_num();
            ComputeNode *currentNode = plan->getRootNode(threadId);

            int posStride1A_inB = findPos(0, perm_);
            int posStride1B_inA = perm_[0];


            int numThreadsPerComm = numThreads_; //global communicator
            int threadIdComm = threadId;
            // create loops
            for(int i=0; i < dim_; ++i){
               int index = loopOrder[i];
               currentNode->inc = this->getIncrement( index );

               const int numSubCommunicators = numThreadsAtLoop[index];
               const int numParallelismAvailable = (sizeA_[index] + currentNode->inc - 1) / currentNode->inc;
               const int workPerThread = (numParallelismAvailable + numSubCommunicators -1) / numSubCommunicators;

               numThreadsPerComm /= numSubCommunicators; //numThreads in next comminicator
               const int commId = (threadIdComm/numThreadsPerComm);
               threadIdComm = threadIdComm % numThreadsPerComm; // local threadId in next Comminicator

               currentNode->start = std::min( sizeA_[index], commId * workPerThread * currentNode->inc );
               currentNode->end = std::min( sizeA_[index], (commId+1) * workPerThread * currentNode->inc );

               currentNode->lda = lda_[index];
               currentNode->ldb = ldb_[findPos(index, perm_)];

               if( perm_[0] != 0 || i != dim_-1 ){
                  currentNode->next = new ComputeNode;
                  currentNode = currentNode->next;
               }
            }

            //macro-kernel
            if( perm_[0] != 0 )
            {
               currentNode->start = -1;
               currentNode->end = -1;
               currentNode->inc = -1;
               currentNode->lda = lda_[ posStride1B_inA ];
               currentNode->ldb = ldb_[ posStride1A_inB ];
               currentNode->next = nullptr;
            }
         }
         plans.push_back(plan);
      }
   }
//   printf("#plans: %d\n", plans.size());
   if( plans.size() != parallelismStrategies.size() * loopOrders.size() )
   {
      fprintf(stderr,"Internal error: number of plans does not fit\n");
      exit(-1);
   }
}

/**
 * Estimates the time in seconds for the given computeTree
 */
template<typename floatType>
float Transpose<floatType>::estimateExecutionTime( const Plan *plan)
{
   double startTime = omp_get_wtime();
   this->executeEstimate(plan);
   double elapsedTime = omp_get_wtime() - startTime;

   const double minMeasurementTime = 0.1; // in seconds

   // do at least 3 repetitions or spent at least 'minMeasurementTime' seconds for each candidate
   int nRepeat = std::min(3, (int) std::ceil(minMeasurementTime / elapsedTime));

   //execute just a few iterations and exterpolate the result
   startTime = omp_get_wtime();
   for(int i=0;i < nRepeat ; ++i) //ATTENTION: we are not clearing the caches inbetween runs
      this->executeEstimate( plan );
   elapsedTime = omp_get_wtime() - startTime;
   elapsedTime /= nRepeat;

#ifdef DEBUG
   printf("Estimated time: %.3e ms.\n",elapsedTime * 1000); 
#endif
   return elapsedTime; 
}

template<typename floatType>
double Transpose<floatType>::getTimeLimit() const
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
      fprintf(stderr,"ERROR: sectionMethod unknown.\n");
      exit(-1);
   }
   return -1;
}

template<typename floatType>
Plan* Transpose<floatType>::selectPlan( const std::vector<Plan*> &plans)
{
   if( plans.size() <= 0 ){
      fprintf(stderr,"Internal error: not enough plans generated.\n");
      exit(-1);
   }

   double timeLimit = this->getTimeLimit(); //in seconds

   float minTime = FLT_MAX;
   int bestPlan_id = 0;

   if( plans.size() > 1 )
   {
      int plansEvaluated = 0;
      double startTime = omp_get_wtime();
      for( int plan_id = 0; plan_id < plans.size(); plan_id++ )
      {
         auto p = plans[plan_id];
         if( omp_get_wtime() - startTime >= timeLimit ) // timelimit reached
            break;

         float estimatedTime = this->estimateExecutionTime( p );
         plansEvaluated++;

         if( estimatedTime < minTime ){
            bestPlan_id = plan_id;
            minTime = estimatedTime;
         }
         if( this->infoLevel_ > 1 ){
            printf("Plan %d will take roughly %f ms.\n", plan_id, estimatedTime * 1000.);
            plans[plan_id]->print();
         }
      }
      if( this->infoLevel_ > 0 )
         printf("We evaluated %d/%d candidates and selected candidate %d.\n", plansEvaluated, plans.size(), bestPlan_id); 
   }
   if( this->infoLevel_ > 0 ){
      printf("Configuration of best plan:\n");
      plans[bestPlan_id]->print();
   }
   return plans[bestPlan_id];
}

template class Transpose<float>;
template class Transpose<double>;

}
















