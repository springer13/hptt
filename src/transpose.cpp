/**
 * \file
 * Compute the tensor transposition
 */

#include <tuple>
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <chrono>

#include <float.h>
#include <stdio.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../include/utils.h"
#include "../include/macros.h"
#include "../include/compute_node.h"
#include "../include/utils.h"
#include "../include/hptt_types.h"
#include "../include/plan.h"
#include "../include/transpose.h"

namespace hptt {

template <typename floatType, int betaIsZero, bool conjA>
struct micro_kernel
{
    static void execute(const floatType* __restrict__ A, const size_t lda, floatType* __restrict__ B, const size_t ldb, const floatType alpha, const floatType beta)
    {
       constexpr int n = (REGISTER_BITS/8) / sizeof(floatType);

       if( betaIsZero )
          for(int j=0; j < n; ++j)
             for(int i=0; i < n; ++i)
                if( conjA )
                   B[i + j * ldb] = alpha * conj(A[i * lda + j]);
                else
                   B[i + j * ldb] = alpha * A[i * lda + j];
       else
          for(int j=0; j < n; ++j)
             for(int i=0; i < n; ++i)
                if( conjA )
                   B[i + j * ldb] = alpha * conj(A[i * lda + j]) + beta * B[i + j * ldb];
                else
                   B[i + j * ldb] = alpha * A[i * lda + j] + beta * B[i + j * ldb];
    }
};

template<typename floatType>
static void streamingStore( floatType* out, const floatType *in )
{
   constexpr int n = REGISTER_BITS/8/sizeof(floatType);
   for(int i=0; i < n; ++i)
      out[i] = in[i];
}


#ifdef HPTT_ARCH_AVX
#include <immintrin.h>

template<typename floatType>
static INLINE void prefetch(const floatType* A, const int lda)
{
   constexpr int blocking_micro_ = REGISTER_BITS/8 / sizeof(floatType);
   for(int i=0;i < blocking_micro_; ++i)
      _mm_prefetch((char*)(A + i * lda), _MM_HINT_T2);
}
template <int betaIsZero, bool conjA>
struct micro_kernel<double, betaIsZero, conjA>
{
    static void execute(const double* __restrict__ A, const size_t lda, double* __restrict__ B, const size_t ldb, const double alpha ,const double beta)
    {
       __m256d reg_alpha = _mm256_set1_pd(alpha); // do not alter the content of B
       __m256d reg_beta = _mm256_set1_pd(beta); // do not alter the content of B
       //Load A
       __m256d rowA0 = _mm256_loadu_pd((A +0*lda));
       __m256d rowA1 = _mm256_loadu_pd((A +1*lda));
       __m256d rowA2 = _mm256_loadu_pd((A +2*lda));
       __m256d rowA3 = _mm256_loadu_pd((A +3*lda));

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
          __m256d rowB0 = _mm256_loadu_pd((B +0*ldb));
          __m256d rowB1 = _mm256_loadu_pd((B +1*ldb));
          __m256d rowB2 = _mm256_loadu_pd((B +2*ldb));
          __m256d rowB3 = _mm256_loadu_pd((B +3*ldb));

          rowB0 = _mm256_add_pd( _mm256_mul_pd(rowB0, reg_beta), rowA0);
          rowB1 = _mm256_add_pd( _mm256_mul_pd(rowB1, reg_beta), rowA1);
          rowB2 = _mm256_add_pd( _mm256_mul_pd(rowB2, reg_beta), rowA2);
          rowB3 = _mm256_add_pd( _mm256_mul_pd(rowB3, reg_beta), rowA3);
          //Store B
          _mm256_storeu_pd((B + 0 * ldb), rowB0);
          _mm256_storeu_pd((B + 1 * ldb), rowB1);
          _mm256_storeu_pd((B + 2 * ldb), rowB2);
          _mm256_storeu_pd((B + 3 * ldb), rowB3);
       } else {
          //Store B
          _mm256_storeu_pd((B + 0 * ldb), rowA0);
          _mm256_storeu_pd((B + 1 * ldb), rowA1);
          _mm256_storeu_pd((B + 2 * ldb), rowA2);
          _mm256_storeu_pd((B + 3 * ldb), rowA3);
       }
    }
};

template <int betaIsZero, bool conjA>
struct micro_kernel<float, betaIsZero, conjA>
{
    static void execute(const float* __restrict__ A, const size_t lda, float* __restrict__ B, const size_t ldb, const float alpha ,const float beta)
    {
       __m256 reg_alpha = _mm256_set1_ps(alpha); // do not alter the content of B
       __m256 reg_beta = _mm256_set1_ps(beta); // do not alter the content of B
       //Load A
       __m256 rowA0 = _mm256_loadu_ps((A +0*lda));
       __m256 rowA1 = _mm256_loadu_ps((A +1*lda));
       __m256 rowA2 = _mm256_loadu_ps((A +2*lda));
       __m256 rowA3 = _mm256_loadu_ps((A +3*lda));
       __m256 rowA4 = _mm256_loadu_ps((A +4*lda));
       __m256 rowA5 = _mm256_loadu_ps((A +5*lda));
       __m256 rowA6 = _mm256_loadu_ps((A +6*lda));
       __m256 rowA7 = _mm256_loadu_ps((A +7*lda));

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
          __m256 rowB0 = _mm256_loadu_ps((B +0*ldb));
          __m256 rowB1 = _mm256_loadu_ps((B +1*ldb));
          __m256 rowB2 = _mm256_loadu_ps((B +2*ldb));
          __m256 rowB3 = _mm256_loadu_ps((B +3*ldb));
          __m256 rowB4 = _mm256_loadu_ps((B +4*ldb));
          __m256 rowB5 = _mm256_loadu_ps((B +5*ldb));
          __m256 rowB6 = _mm256_loadu_ps((B +6*ldb));
          __m256 rowB7 = _mm256_loadu_ps((B +7*ldb));

          rowB0 = _mm256_add_ps( _mm256_mul_ps(rowB0, reg_beta), rowA0);
          rowB1 = _mm256_add_ps( _mm256_mul_ps(rowB1, reg_beta), rowA1);
          rowB2 = _mm256_add_ps( _mm256_mul_ps(rowB2, reg_beta), rowA2);
          rowB3 = _mm256_add_ps( _mm256_mul_ps(rowB3, reg_beta), rowA3);
          rowB4 = _mm256_add_ps( _mm256_mul_ps(rowB4, reg_beta), rowA4);
          rowB5 = _mm256_add_ps( _mm256_mul_ps(rowB5, reg_beta), rowA5);
          rowB6 = _mm256_add_ps( _mm256_mul_ps(rowB6, reg_beta), rowA6);
          rowB7 = _mm256_add_ps( _mm256_mul_ps(rowB7, reg_beta), rowA7);
          //Store B
          _mm256_storeu_ps((B + 0 * ldb), rowB0);
          _mm256_storeu_ps((B + 1 * ldb), rowB1);
          _mm256_storeu_ps((B + 2 * ldb), rowB2);
          _mm256_storeu_ps((B + 3 * ldb), rowB3);
          _mm256_storeu_ps((B + 4 * ldb), rowB4);
          _mm256_storeu_ps((B + 5 * ldb), rowB5);
          _mm256_storeu_ps((B + 6 * ldb), rowB6);
          _mm256_storeu_ps((B + 7 * ldb), rowB7);
       } else {
          _mm256_storeu_ps((B + 0 * ldb), rowA0);
          _mm256_storeu_ps((B + 1 * ldb), rowA1);
          _mm256_storeu_ps((B + 2 * ldb), rowA2);
          _mm256_storeu_ps((B + 3 * ldb), rowA3);
          _mm256_storeu_ps((B + 4 * ldb), rowA4);
          _mm256_storeu_ps((B + 5 * ldb), rowA5);
          _mm256_storeu_ps((B + 6 * ldb), rowA6);
          _mm256_storeu_ps((B + 7 * ldb), rowA7);
       }
    }
};

template<>
void streamingStore<float>( float* out, const float*in ){
   _mm256_stream_ps(out, _mm256_loadu_ps(in));
}
template<>
void streamingStore<double>( double* out, const double*in ){
   _mm256_stream_pd(out, _mm256_loadu_pd(in));
}
#else
template<typename floatType>
static INLINE void prefetch(const floatType* A, const int lda) { }
#endif

#ifdef HPTT_ARCH_ARM
#include <arm_neon.h>

template <int betaIsZero, bool conjA>
struct micro_kernel<float, betaIsZero>
{
    static void execute(const float* __restrict__ A, const size_t lda, float* __restrict__ B, const size_t ldb, const float alpha ,const float beta)
    {
       float32x4_t reg_alpha = vdupq_n_f32(alpha);
       float32x4_t reg_beta = vdupq_n_f32(beta);

       //Load A
       float32x4_t rowA0 = vld1q_f32((A +0*lda));
       float32x4_t rowA1 = vld1q_f32((A +1*lda));
       float32x4_t rowA2 = vld1q_f32((A +2*lda));
       float32x4_t rowA3 = vld1q_f32((A +3*lda));

       //4x4 transpose micro kernel
       float32x4x2_t t0,t1,t2,t3;
       t0 = vuzpq_f32(rowA0, rowA2);
       t1 = vuzpq_f32(rowA1, rowA3);
       t2 = vtrnq_f32(t0.val[0], t1.val[0]);
       t3 = vtrnq_f32(t0.val[1], t1.val[1]);

       //Scale A
       rowA0 = vmulq_f32(t2.val[0], reg_alpha);
       rowA1 = vmulq_f32(t3.val[0], reg_alpha);
       rowA2 = vmulq_f32(t2.val[1], reg_alpha);
       rowA3 = vmulq_f32(t3.val[1], reg_alpha);

       //Load B
       if( !betaIsZero )
       {
          float32x4_t rowB0 = vld1q_f32((B +0*ldb));
          float32x4_t rowB1 = vld1q_f32((B +1*ldb));
          float32x4_t rowB2 = vld1q_f32((B +2*ldb));
          float32x4_t rowB3 = vld1q_f32((B +3*ldb));

          rowB0 = vaddq_f32( vmulq_f32(rowB0, reg_beta), rowA0);
          rowB1 = vaddq_f32( vmulq_f32(rowB1, reg_beta), rowA1);
          rowB2 = vaddq_f32( vmulq_f32(rowB2, reg_beta), rowA2);
          rowB3 = vaddq_f32( vmulq_f32(rowB3, reg_beta), rowA3);
          //Store B
          vst1q_f32((B + 0 * ldb), rowB0);
          vst1q_f32((B + 1 * ldb), rowB1);
          vst1q_f32((B + 2 * ldb), rowB2);
          vst1q_f32((B + 3 * ldb), rowB3);
       } else {
          //Store B
          vst1q_f32((B + 0 * ldb), rowA0);
          vst1q_f32((B + 1 * ldb), rowA1);
          vst1q_f32((B + 2 * ldb), rowA2);
          vst1q_f32((B + 3 * ldb), rowA3);
       }
    }
};
#endif

#ifdef HPTT_ARCH_IBM
//#include <altivec.h> //vector conflicts with std::vector (TODO)
//
//template <int betaIsZero>
//struct micro_kernel<float, betaIsZero>
//{
//    static void execute(const float* __restrict__ A, const size_t lda, float* __restrict__ B, const size_t ldb, const float alpha ,const float beta)
//    {
//       vector float reg_alpha = vec_splats(alpha);
//
//       //Load A
//       vector float rowA0 = vec_ld(0,const_cast<float*>(A+0*lda));
//       vector float rowA1 = vec_ld(0,const_cast<float*>(A+1*lda));
//       vector float rowA2 = vec_ld(0,const_cast<float*>(A+2*lda));
//       vector float rowA3 = vec_ld(0,const_cast<float*>(A+3*lda));
//
//       //4x4 transpose micro kernel
//       vector float aa = (vector float) {2, 3, 2.5, 3.5};
//       vector float bb = (vector float) {2.25, 3.25, 2.75, 3.75};
//       vector float cc = (vector float) {2, 2.25, 3, 3.25};
//       vector float dd = (vector float) {2.5, 2.75, 3.5, 3.75};
//
//       vector float r010 = vec_perm(rowA0,rowA1, aa); //0,4,2,6
//       vector float r011 = vec_perm(rowA0,rowA1, bb); //1,5,3,7
//       vector float r230 = vec_perm(rowA2,rowA3, aa); //8,12,10,14
//       vector float r231 = vec_perm(rowA2,rowA3, bb); //9,13,11,15
//
//       rowA0 = vec_perm(r010, r230, cc); //0,4,8,12
//       rowA1 = vec_perm(r011, r231, cc); //1,5,9,13
//       rowA2 = vec_perm(r010, r230, dd); //2,6,10,14
//       rowA3 = vec_perm(r011, r231, dd); //3,7,11,15
//
//       //Scale A
//       rowA0 = vec_mul(rowA0, reg_alpha);
//       rowA1 = vec_mul(rowA1, reg_alpha);
//       rowA2 = vec_mul(rowA2, reg_alpha);
//       rowA3 = vec_mul(rowA3, reg_alpha);
//
//       if( !betaIsZero )
//       {
//          vector float reg_beta = vec_splats(beta);
//          //Load B
//          vector float rowB0 = vec_ld(0,const_cast<float*>(B+0*ldb));
//          vector float rowB1 = vec_ld(0,const_cast<float*>(B+1*ldb));
//          vector float rowB2 = vec_ld(0,const_cast<float*>(B+2*ldb));
//          vector float rowB3 = vec_ld(0,const_cast<float*>(B+3*ldb));
//
//          rowB0 = vec_madd( rowB0, reg_beta, rowA0);
//          rowB1 = vec_madd( rowB1, reg_beta, rowA1);
//          rowB2 = vec_madd( rowB2, reg_beta, rowA2);
//          rowB3 = vec_madd( rowB3, reg_beta, rowA3);
//
//          //Store B
//          vec_st(rowB0, 0, B + 0 * ldb);
//          vec_st(rowB1, 0, B + 1 * ldb);
//          vec_st(rowB2, 0, B + 2 * ldb);
//          vec_st(rowB3, 0, B + 3 * ldb);
//       } else {             
//          //Store B         
//          vec_st(rowA0, 0, B + 0 * ldb);
//          vec_st(rowA1, 0, B + 1 * ldb);
//          vec_st(rowA2, 0, B + 2 * ldb);
//          vec_st(rowA3, 0, B + 3 * ldb);
//       }
//    }
//};
#endif


template<int betaIsZero, typename floatType, bool conjA>
static INLINE void macro_kernel_scalar(const floatType* __restrict__ A, const size_t lda, int blockingA,  
                                             floatType* __restrict__ B, const size_t ldb, int blockingB,
                                             const floatType alpha ,const floatType beta)
{
#ifdef DEBUG
   assert( blockingA > 0 && blockingB > 0);
#endif

   if( betaIsZero )
      for(int j=0; j < blockingA; ++j)
         for(int i=0; i < blockingB; ++i)
            if( conjA )
               B[i + j * ldb] = alpha * conj(A[i * lda + j]);
            else
               B[i + j * ldb] = alpha * A[i * lda + j];
   else
      for(int j=0; j < blockingA; ++j)
         for(int i=0; i < blockingB; ++i)
            if( conjA )
               B[i + j * ldb] = alpha * conj(A[i * lda + j]) + beta * B[i + j * ldb];
            else
               B[i + j * ldb] = alpha * A[i * lda + j] + beta * B[i + j * ldb];
}

template<int blockingA, int blockingB, int betaIsZero, typename floatType, bool useStreamingStores_, bool conjA>
static INLINE void macro_kernel(const floatType* __restrict__ A, const floatType* __restrict__ Anext, const size_t lda, 
                                   floatType* __restrict__ B, const floatType* __restrict__ Bnext, const size_t ldb,
                                   const floatType alpha ,const floatType beta)
{
   constexpr int blocking_micro_ = REGISTER_BITS/8 / sizeof(floatType);
   constexpr int blocking_ = blocking_micro_ * 4;

   const bool useStreamingStores = useStreamingStores_ && betaIsZero && (blockingB*sizeof(floatType))%64 == 0 && ((uint64_t)B)%32 == 0 && (ldb*sizeof(floatType))%32 == 0;

   floatType *Btmp = B;
   size_t ldb_tmp = ldb;
   floatType buffer[blockingA * blockingB];// __attribute__((aligned(64)));
   if( (useStreamingStores_ && useStreamingStores) ){
      Btmp = buffer;
      ldb_tmp = blockingB;
   }

   if( blockingA == blocking_ && blockingB == blocking_ )
   {
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
      prefetch<floatType>(Anext + (0 * lda + 0), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (0 * lda + 0), lda, Btmp + (0 * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (0 * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      prefetch<floatType>(Anext + (2*blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (2*blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (3*blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (3*blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (0 * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (2*blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (3*blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (2*blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      prefetch<floatType>(Anext + (0 * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (0 * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (blocking_micro_ * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (2*blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      prefetch<floatType>(Anext + (2*blocking_micro_ * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (2*blocking_micro_ * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (3*blocking_micro_ * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (3*blocking_micro_ * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (3*blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (0 * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (3*blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (2*blocking_micro_ * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (3*blocking_micro_ * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
   }else if( blockingA == 2*blocking_micro_ && blockingB == blocking_ ) {
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
      prefetch<floatType>(Anext + (0 * lda + 0), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (0 * lda + 0), lda, Btmp + (0 * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (0 * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      prefetch<floatType>(Anext + (2*blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (2*blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (3*blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (3*blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (0 * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (2*blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 2*blocking_micro_), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (3*blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 3*blocking_micro_), ldb_tmp  , alpha , beta);
   }else if( blockingA == blocking_ && blockingB == 2*blocking_micro_) {
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
      prefetch<floatType>(Anext + (0 * lda + 0), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (0 * lda + 0), lda, Btmp + (0 * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + 0), lda, Btmp + (0 * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (0 * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + blocking_micro_), lda, Btmp + (blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (2*blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      prefetch<floatType>(Anext + (0 * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (0 * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      prefetch<floatType>(Anext + (blocking_micro_ * lda + 2*blocking_micro_), lda);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + 2*blocking_micro_), lda, Btmp + (2*blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
      if( !(useStreamingStores_ && useStreamingStores) )
         prefetch<floatType>(Bnext + (3*blocking_micro_ * ldb_tmp + 0), ldb_tmp);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (0 * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + 0), ldb_tmp  , alpha , beta);
      micro_kernel<floatType,betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + 3*blocking_micro_), lda, Btmp + (3*blocking_micro_ * ldb_tmp + blocking_micro_), ldb_tmp  , alpha , beta);
   } else {
      //invoke micro-transpose
      if(blockingA > 0 && blockingB > 0 )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A, lda, Btmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 0 && blockingB > blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + blocking_micro_ * lda, lda, Btmp + blocking_micro_, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 0 && blockingB > 2*blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + 2*blocking_micro_ * lda, lda, Btmp + 2*blocking_micro_, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 0 && blockingB > 3*blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + 3*blocking_micro_ * lda, lda, Btmp + 3*blocking_micro_, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > blocking_micro_ && blockingB > 0 )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + blocking_micro_, lda, Btmp + blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > blocking_micro_ && blockingB > blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + blocking_micro_ + blocking_micro_ * lda, lda, Btmp + blocking_micro_ + blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > blocking_micro_ && blockingB > 2*blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + blocking_micro_ + 2*blocking_micro_ * lda, lda, Btmp + 2*blocking_micro_ + blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > blocking_micro_ && blockingB > 3*blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + blocking_micro_ + 3*blocking_micro_ * lda, lda, Btmp + 3*blocking_micro_ + blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 2*blocking_micro_ && blockingB > 0 )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + 2*blocking_micro_, lda, Btmp + 2*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 2*blocking_micro_ && blockingB > blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + 2*blocking_micro_ + blocking_micro_ * lda, lda, Btmp + blocking_micro_ + 2*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 2*blocking_micro_ && blockingB > 2*blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + 2*blocking_micro_ + 2*blocking_micro_ * lda, lda, Btmp + 2*blocking_micro_ + 2*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 2*blocking_micro_ && blockingB > 3*blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + 2*blocking_micro_ + 3*blocking_micro_ * lda, lda, Btmp + 3*blocking_micro_ + 2*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 3*blocking_micro_ && blockingB > 0 )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + 3*blocking_micro_, lda, Btmp + 3*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 3*blocking_micro_ && blockingB > blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + 3*blocking_micro_ + blocking_micro_ * lda, lda, Btmp + blocking_micro_ + 3*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 3*blocking_micro_ && blockingB > 2*blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + 3*blocking_micro_ + 2*blocking_micro_ * lda, lda, Btmp + 2*blocking_micro_ + 3*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);

      //invoke micro-transpose
      if(blockingA > 3*blocking_micro_ && blockingB > 3*blocking_micro_ )
         micro_kernel<floatType,betaIsZero, conjA>::execute(A + 3*blocking_micro_ + 3*blocking_micro_ * lda, lda, Btmp + 3*blocking_micro_ + 3*blocking_micro_ * ldb_tmp, ldb_tmp  , alpha , beta);
   }

   // write buffer to main-memory via non-temporal stores
   if( (useStreamingStores_ && useStreamingStores) )
      for( int i = 0; i < blockingA; i++){
         for( int j = 0; j < blockingB; j+=blocking_micro_)
            streamingStore<floatType>(B + i * ldb + j, buffer + i * ldb_tmp + j);
      }
}

template<int betaIsZero, typename floatType, bool conjA>
void transpose_int_scalar( const floatType* __restrict__ A, int sizeStride1A, 
                                  floatType* __restrict__ B, int sizeStride1B, const floatType alpha, const floatType beta, const ComputeNode* plan)
{
   const int32_t end = plan->end;
   const size_t lda = plan->lda;
   const size_t ldb = plan->ldb;
   if( plan->next->next != nullptr ){
      // recurse
      int i = plan->start;
      if( lda == 1)
         transpose_int_scalar<betaIsZero, floatType, conjA>( &A[i*lda], end - plan->start, &B[i*ldb], sizeStride1B, alpha, beta, plan->next);
      else if( ldb == 1)
         transpose_int_scalar<betaIsZero, floatType, conjA>( &A[i*lda], sizeStride1A, &B[i*ldb], end - plan->start, alpha, beta, plan->next);
      else
         for(; i < end; i++)
            transpose_int_scalar<betaIsZero, floatType, conjA>( &A[i*lda], sizeStride1A, &B[i*ldb], sizeStride1B, alpha, beta, plan->next);
   }else{
      // macro-kernel
      const size_t lda_macro = plan->next->lda;
      const size_t ldb_macro = plan->next->ldb;
      int i = plan->start;
      const size_t scalarRemainder = plan->end - plan->start;
      if( scalarRemainder > 0 ){
         if( lda == 1)
            macro_kernel_scalar<betaIsZero,floatType, conjA>(&A[i*lda], lda_macro, scalarRemainder, &B[i*ldb], ldb_macro, sizeStride1B, alpha, beta);
         else if( ldb == 1)
            macro_kernel_scalar<betaIsZero,floatType, conjA>(&A[i*lda], lda_macro, sizeStride1A, &B[i*ldb], ldb_macro, scalarRemainder, alpha, beta);
      }
   }
}
template<int blockingA, int blockingB, int betaIsZero, typename floatType, bool useStreamingStores, bool conjA>
void transpose_int( const floatType* __restrict__ A, const floatType* __restrict__ Anext, 
                     floatType* __restrict__ B, const floatType* __restrict__ Bnext, const floatType alpha, const floatType beta, 
                     const ComputeNode* plan)
{
   const size_t end = plan->end - (plan->inc - 1);
   const size_t inc = plan->inc;
   const size_t lda = plan->lda;
   const size_t ldb = plan->ldb;

   constexpr size_t blocking_micro_ = REGISTER_BITS/8 / sizeof(floatType);
   constexpr size_t blocking_ = blocking_micro_ * 4;

   if( plan->next->next != nullptr ){
      // recurse
      size_t i;
      for(i = plan->start; i < end; i+= inc)
      {
         if( i + inc < end )
            transpose_int<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>( &A[i*lda], &A[(i+1)*lda], &B[i*ldb], &B[(i+1)*ldb], alpha, beta, plan->next);
         else
            transpose_int<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>( &A[i*lda], Anext, &B[i*ldb], Bnext, alpha, beta, plan->next);
      }
      // remainder
      if( blocking_/2 >= blocking_micro_ && (i + blocking_/2) <= plan->end ){
         if( lda == 1)
            transpose_int<blocking_/2, blockingB, betaIsZero, floatType, useStreamingStores, conjA>( &A[i*lda], Anext, &B[i*ldb], Bnext, alpha, beta, plan->next);
         else if( ldb == 1)
            transpose_int<blockingA, blocking_/2, betaIsZero, floatType, useStreamingStores, conjA>( &A[i*lda], Anext, &B[i*ldb], Bnext, alpha, beta, plan->next);
         i+=blocking_/2;
      }
      if( blocking_/4 >= blocking_micro_ && (i + blocking_/4) <= plan->end ){
         if( lda == 1)
            transpose_int<blocking_/4, blockingB, betaIsZero, floatType, useStreamingStores, conjA>( &A[i*lda], Anext, &B[i*ldb], Bnext, alpha, beta, plan->next);
         else if( ldb == 1)
            transpose_int<blockingA, blocking_/4, betaIsZero, floatType, useStreamingStores, conjA>( &A[i*lda], Anext, &B[i*ldb], Bnext, alpha, beta, plan->next);
         i+=blocking_/4;
      }
      const size_t scalarRemainder = plan->end - i;
      if( scalarRemainder > 0 ){
         if( lda == 1)
            transpose_int_scalar<betaIsZero, floatType, conjA>( &A[i*lda], scalarRemainder, &B[i*ldb], -1, alpha, beta, plan->next);
         else
            transpose_int_scalar<betaIsZero, floatType, conjA>( &A[i*lda], -1, &B[i*ldb], scalarRemainder, alpha, beta, plan->next);
      }
   } else {
      const size_t lda_macro = plan->next->lda;
      const size_t ldb_macro = plan->next->ldb;
      // invoke macro-kernel
      
      size_t i;
      for(i = plan->start; i < end; i+= inc)
         if( i + inc < end )
            macro_kernel<blockingA, blockingB, betaIsZero,floatType, useStreamingStores, conjA>(&A[i*lda], &A[(i+1)*lda], lda_macro, &B[i*ldb], &B[(i+1)*ldb], ldb_macro, alpha, beta);
         else
            macro_kernel<blockingA, blockingB, betaIsZero,floatType, useStreamingStores, conjA>(&A[i*lda], Anext, lda_macro, &B[i*ldb], Bnext, ldb_macro, alpha, beta);
      // remainder
      if( blocking_/2 >= blocking_micro_ && (i + blocking_/2) <= plan->end ){
         if( lda == 1)
            macro_kernel<blocking_/2, blockingB, betaIsZero,floatType, useStreamingStores, conjA>(&A[i*lda], Anext, lda_macro, &B[i*ldb], Bnext, ldb_macro, alpha, beta);
         else if( ldb == 1)
            macro_kernel<blockingA, blocking_/2, betaIsZero,floatType, useStreamingStores, conjA>(&A[i*lda], Anext, lda_macro, &B[i*ldb], Bnext, ldb_macro, alpha, beta);
         i+=blocking_/2;
      }
      if( blocking_/4 >= blocking_micro_ && (i + blocking_/4) <= plan->end ){
         if( lda == 1)
            macro_kernel<blocking_/4, blockingB, betaIsZero,floatType, useStreamingStores, conjA>(&A[i*lda], Anext, lda_macro, &B[i*ldb], Bnext, ldb_macro, alpha, beta);
         else if( ldb == 1)
            macro_kernel<blockingA, blocking_/4, betaIsZero,floatType, useStreamingStores, conjA>(&A[i*lda], Anext, lda_macro, &B[i*ldb], Bnext, ldb_macro, alpha, beta);
         i+=blocking_/4;
      }
      const size_t scalarRemainder = plan->end - i;
      if( scalarRemainder > 0 ){
         if( lda == 1)
            macro_kernel_scalar<betaIsZero,floatType, conjA>(&A[i*lda], lda_macro, scalarRemainder, &B[i*ldb], ldb_macro, blockingB, alpha, beta);
         else if( ldb == 1)
            macro_kernel_scalar<betaIsZero,floatType, conjA>(&A[i*lda], lda_macro, blockingA, &B[i*ldb], ldb_macro, scalarRemainder, alpha, beta);
      }
   }
}

template<int betaIsZero, typename floatType, bool useStreamingStores, bool conjA>
void transpose_int_constStride1( const floatType* __restrict__ A, floatType* __restrict__ B, const floatType alpha, const floatType beta, 
      const ComputeNode* plan)
{
   const int32_t end = plan->end - (plan->inc - 1);
   constexpr int32_t inc = 1; // TODO
   const size_t lda = plan->lda;
   const size_t ldb = plan->ldb;

   if( plan->next != nullptr )
      for(int i = plan->start; i < end; i+= inc)
         // recurse
         transpose_int_constStride1<betaIsZero, floatType, useStreamingStores, conjA>( &A[i*lda], &B[i*ldb], alpha, beta, plan->next);
   else 
      if( !betaIsZero )
      {
         for(int32_t i = plan->start; i < end; i+= inc)
            if( conjA )
               B[i] = alpha * conj(A[i]) + beta * B[i];
            else
               B[i] = alpha * A[i] + beta * B[i];
      } else {
         if( useStreamingStores)
            if( conjA )
// #pragma vector nontemporal
               for(int32_t i = plan->start; i < end; i+= inc)
                  B[i] = alpha * conj(A[i]);
            else
// #pragma vector nontemporal
               for(int32_t i = plan->start; i < end; i+= inc)
                  B[i] = alpha * A[i];
         else
            if( conjA )
               for(int32_t i = plan->start; i < end; i+= inc)
                  B[i] = alpha * conj(A[i]);
            else
               for(int32_t i = plan->start; i < end; i+= inc)
                  B[i] = alpha * A[i];
      }
}


   template<typename floatType>
   Transpose<floatType>::Transpose( const int *sizeA, 
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
                 const int *threadIds, 
                 const bool useRowMajor) : 
         A_(A),
         B_(B),
         alpha_(alpha),
         beta_(beta),
         dim_(-1),
         numThreads_(numThreads), 
         selectedParallelStrategyId_(-1),
         selectedLoopOrderId_(-1),
         conjA_(false),
         masterPlan_(nullptr),
         selectionMethod_(selectionMethod),
         maxAutotuningCandidates_(-1)
      {
#ifdef _OPENMP
         omp_init_lock(&writelock);
#endif
         int tmpPerm[dim];
         int tmpSizeA[dim];
         int tmpOuterSizeA[dim];
         int tmpOuterSizeB[dim];
         accountForRowMajor(sizeA, outerSizeA, outerSizeB, perm, 
               tmpSizeA, tmpOuterSizeA, tmpOuterSizeB, tmpPerm, dim, useRowMajor); 

         sizeA_.resize(dim);
         perm_.resize(dim);
         outerSizeA_.resize(dim);
         outerSizeB_.resize(dim);
         lda_.resize(dim);
         ldb_.resize(dim);
         if(threadIds){
            // compact threadIds. E.g., 1, 7, 5 -> local_id(1) = 0, local_id(7) = 2, local_id(5) = 1
            for(int i=0; i < numThreads; ++i)
               threadIds_.push_back(threadIds[i]);
            std::sort(threadIds_.begin(), threadIds_.end());
         }else{
            for(int i=0; i < numThreads; ++i)
               threadIds_.push_back(i);
         }

         verifyParameter(tmpSizeA, tmpPerm, tmpOuterSizeA, tmpOuterSizeB, dim);

         // initializes dim_, outerSizeA, outerSizeB, sizeA and perm 
         skipIndices(tmpSizeA, tmpPerm, tmpOuterSizeA, tmpOuterSizeB, dim);
         fuseIndices();

         // initializes lda_ and ldb_
         computeLeadingDimensions();

         // create plan
         this->createPlan();
      }

      template<typename floatType>
      Transpose<floatType>::Transpose(const Transpose<floatType> &other) : A_(other.A_), B_(other.B_), 
                                          alpha_(other.alpha_),
                                          beta_(other.beta_),
                                          dim_(other.dim_),
                                          sizeA_(other.sizeA_),
                                          perm_(other.perm_),
                                          outerSizeA_(other.outerSizeA_),
                                          outerSizeB_(other.outerSizeB_),
                                          lda_(other.lda_),
                                          ldb_(other.ldb_),
                                          threadIds_(other.threadIds_),
                                          numThreads_(other.numThreads_),
                                          selectedParallelStrategyId_(other.selectedParallelStrategyId_),
                                          selectedLoopOrderId_(other.selectedLoopOrderId_),
                                          conjA_(other.conjA_),
                                          masterPlan_(other.masterPlan_),
                                          selectionMethod_(other.selectionMethod_),
                                          maxAutotuningCandidates_(other.maxAutotuningCandidates_)

      { 
#ifdef _OPENMP
         omp_init_lock(&writelock);
#endif
      }

      template<typename floatType>
      Transpose<floatType>::~Transpose() { 
#ifdef _OPENMP
         omp_destroy_lock(&writelock); 
#endif
      }

template<typename floatType>
void Transpose<floatType>::executeEstimate(const Plan *plan) noexcept
{
   if( plan == nullptr ) {
      fprintf(stderr,"[HPTT] ERROR: plan has not yet been created.\n");
      exit(-1);
   }
   
   constexpr bool useStreamingStores = false;

   const int numTasks = plan->getNumTasks();
#ifdef _OPENMP
#pragma omp parallel for num_threads(numThreads_)  if(numThreads_ > 1)
#endif
   for( int taskId = 0; taskId < numTasks; taskId++)
      if ( perm_[0] != 0 ) {
         auto rootNode = plan->getRootNode_const( taskId );
         if( std::abs(beta_) < getZeroThreshold<floatType>() ) {
            if( conjA_ )
               transpose_int<blocking_,blocking_,1,floatType, useStreamingStores, true>( A_,A_, B_, B_, 0.0, 1.0, rootNode);
            else
               transpose_int<blocking_,blocking_,1,floatType, useStreamingStores, false>( A_,A_, B_, B_, 0.0, 1.0, rootNode);
         } else {
            if( conjA_ )
               transpose_int<blocking_,blocking_,0,floatType, useStreamingStores, true>( A_,A_, B_, B_, 0.0, 1.0, rootNode);
            else
               transpose_int<blocking_,blocking_,0,floatType, useStreamingStores, false>( A_,A_, B_, B_, 0.0, 1.0, rootNode);
         }
      } else {
         auto rootNode = plan->getRootNode_const( taskId );
         if( std::abs(beta_) < getZeroThreshold<floatType>() ) {
            if( conjA_ )
               transpose_int_constStride1<1,floatType, useStreamingStores, true>( A_, B_, 0.0, 1.0, rootNode);
            else
               transpose_int_constStride1<1,floatType, useStreamingStores, false>( A_, B_, 0.0, 1.0, rootNode);
         }else{
            if( conjA_ )
               transpose_int_constStride1<0,floatType, useStreamingStores, true>( A_, B_, 0.0, 1.0, rootNode);
            else
               transpose_int_constStride1<0,floatType, useStreamingStores, false>( A_, B_, 0.0, 1.0, rootNode);
         }
      }
}



template<int betaIsZero, typename floatType, bool useStreamingStores, bool spawnThreads, bool conjA>
static void axpy_1D( const floatType* __restrict__ A, floatType* __restrict__ B, const int myStart, const int myEnd, const floatType alpha, const floatType beta, int numThreads)
{
   if( !betaIsZero )
   {
      HPTT_DUPLICATE(spawnThreads,
         for(int32_t i = myStart; i < myEnd; i++)
            if( conjA )
               B[i] = alpha * conj(A[i]) + beta * B[i];
            else
               B[i] = alpha * A[i] + beta * B[i];
      )
   } else {
      if( useStreamingStores)
// #pragma vector nontemporal
         HPTT_DUPLICATE(spawnThreads,
            for(int32_t i = myStart; i < myEnd; i++)
               if( conjA )
                  B[i] = alpha * conj(A[i]);
               else
                  B[i] = alpha * A[i];
         )
      else
         HPTT_DUPLICATE(spawnThreads,
            for(int32_t i = myStart; i < myEnd; i++)
               if( conjA )
                  B[i] = alpha * conj(A[i]);
               else
                  B[i] = alpha * A[i];
         )
   }
}

template<int betaIsZero, typename floatType, bool useStreamingStores, bool spawnThreads, bool conjA>
static void axpy_2D( const floatType* __restrict__ A, const int lda, 
                        floatType* __restrict__ B, const int ldb, 
                        const int n0, const int myStart, const int myEnd, const floatType alpha, const floatType beta, int numThreads)
{
   if( !betaIsZero )
   {
      HPTT_DUPLICATE(spawnThreads,
         for(int32_t j = myStart; j < myEnd; j++)
            for(int32_t i = 0; i < n0; i++)
               if( conjA )
                  B[i + j * ldb] = alpha * conj(A[i + j * lda]) + beta * B[i + j * ldb];
               else
                  B[i + j * ldb] = alpha * A[i + j * lda] + beta * B[i + j * ldb];
      )
   } else {
      if( useStreamingStores)
         HPTT_DUPLICATE(spawnThreads,
            for(int32_t j = myStart; j < myEnd; j++)
// _Pragma("vector nontemporal")
            for(int32_t i = 0; i < n0; i++)
               if( conjA )
                  B[i + j * ldb] = alpha * conj(A[i + j * lda]);
               else
                  B[i + j * ldb] = alpha * A[i + j * lda];
         )
      else
         HPTT_DUPLICATE(spawnThreads,
            for(int32_t j = myStart; j < myEnd; j++)
               for(int32_t i = 0; i < n0; i++)
                  if( conjA )
                     B[i + j * ldb] = alpha * conj(A[i + j * lda]);
                  else
                     B[i + j * ldb] = alpha * A[i + j * lda];
         )
   }
}

template<typename floatType> 
template<bool spawnThreads>
void Transpose<floatType>::getStartEnd(int n, int &myStart, int &myEnd) const
{
#ifdef _OPENMP
   int myLocalThreadId = getLocalThreadId(omp_get_thread_num());
#else
   int myLocalThreadId = 0;
#endif

   if(myLocalThreadId == -1 ) // skip those threads which do not participate in this plan
   {
      myStart = n;
      myEnd = n;
      return;
   }
   if(spawnThreads){ // worksharing will be handled by the OpenMP runtime
      myStart = 0;
      myEnd = n;
      return;
   }

   const int workPerThread = (n + numThreads_-1)/numThreads_;
   myStart = std::min(n, myLocalThreadId * workPerThread);
   myEnd = std::min(n, (myLocalThreadId+1) * workPerThread);
}


template<typename floatType> 
int Transpose<floatType>::getLocalThreadId(int myThreadId) const
{
   int myLocalId = -1;
   for(int i=0; i < numThreads_; ++i)
      if(myThreadId == threadIds_[i])
         myLocalId = i;
   return myLocalId;
}

template<typename floatType> 
template<bool useStreamingStores, bool spawnThreads, bool betaIsZero>
void Transpose<floatType>::execute_expert() noexcept
{
   if( masterPlan_ == nullptr ) {
      fprintf(stderr,"[HPTT] ERROR: master plan has not yet been created.\n");
      exit(-1);
   }

   int myStart = 0;
   int myEnd = 0;

   if( dim_ == 1)
   {
      getStartEnd<spawnThreads>(sizeA_[0], myStart, myEnd);
      if( conjA_ )
         axpy_1D<betaIsZero, floatType, useStreamingStores, spawnThreads, true>( A_, B_, myStart, myEnd, alpha_, beta_, numThreads_ );
      else
         axpy_1D<betaIsZero, floatType, useStreamingStores, spawnThreads, false>( A_, B_, myStart, myEnd, alpha_, beta_, numThreads_ );
      return;
   }
   else if( dim_ == 2 && perm_[0] == 0)
   {
      getStartEnd<spawnThreads>(sizeA_[1], myStart, myEnd);
      if( conjA_ )
         axpy_2D<betaIsZero, floatType, useStreamingStores, spawnThreads, true>( A_, lda_[1], B_, ldb_[1], sizeA_[0], myStart, myEnd, alpha_, beta_, numThreads_ );
      else
         axpy_2D<betaIsZero, floatType, useStreamingStores, spawnThreads, false>( A_, lda_[1], B_, ldb_[1], sizeA_[0], myStart, myEnd, alpha_, beta_, numThreads_ );
      return;
   }

   const int numTasks = masterPlan_->getNumTasks();
   // const int numThreads = numThreads_;
   getStartEnd<spawnThreads>(numTasks, myStart, myEnd);

   HPTT_DUPLICATE(spawnThreads,
      for( int taskId = myStart; taskId < myEnd; taskId++)
         if ( perm_[0] != 0 ) {
            auto rootNode = masterPlan_->getRootNode_const( taskId );
            if( conjA_ )
               transpose_int<blocking_,blocking_,betaIsZero,floatType, useStreamingStores, true>( A_, A_, B_, B_, alpha_, beta_, rootNode);
            else
               transpose_int<blocking_,blocking_,betaIsZero,floatType, useStreamingStores, false>( A_, A_, B_, B_, alpha_, beta_, rootNode);
         } else {
            auto rootNode = masterPlan_->getRootNode_const( taskId );
            if( conjA_ )
               transpose_int_constStride1<betaIsZero,floatType, useStreamingStores, true>( A_, B_, alpha_, beta_, rootNode);
            else
               transpose_int_constStride1<betaIsZero,floatType, useStreamingStores, false>( A_, B_, alpha_, beta_, rootNode);
         }
   )
}
template<typename floatType> 
void Transpose<floatType>::execute() noexcept
{
   if( masterPlan_ == nullptr ) {
      fprintf(stderr,"[HPTT] ERROR: master plan has not yet been created.\n");
      exit(-1);
   }

   bool spawnThreads = numThreads_ > 1;
   bool betaIsZero = (beta_ == (floatType) 0.0);
   constexpr bool useStreamingStores = true;
   if( spawnThreads )
   {
      if( betaIsZero) {
         this->execute_expert<useStreamingStores, true, true>();
      } else {
         this->execute_expert<useStreamingStores, true, false>();
      }
   } else {
      if( betaIsZero) {
         this->execute_expert<useStreamingStores, false, true>();
      } else {
         this->execute_expert<useStreamingStores, false, false>();
      }
   }
}

template<typename floatType> 
void Transpose<floatType>::print() noexcept 
{ 
   masterPlan_->print(); 
}

template<typename floatType>
int Transpose<floatType>::getIncrement( int loopIdx ) const
{
   int inc = 1;
   if( perm_[0] != 0 ) {
      if( loopIdx == 0 || loopIdx == perm_[0] )
         inc = blocking_;
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
      // match every primefactor ...
      for( auto p : primeFactorsToMatch )
      {
         // ... with every loop
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

// balancing if one tries to parallelize avail many tasks with req many threads
// e.g., balancing(3,4) = 0.75
static float getBalancing(int avail, int req)
{
   return ((float)(avail))/(((avail+req-1)/req)*req);
}

template<typename floatType>
float Transpose<floatType>::getLoadBalance( const std::vector<int> &parallelismStrategy ) const
{
   float load_balance = 1.0;
   int totalTasks = 1;
   for(int i=0; i < dim_; ++i){

      size_t inc = this->getIncrement(i);
      while(sizeA_[i] < inc)
         inc /= 2;
      int availableParallelism = (sizeA_[i] + inc - 1) / inc;

      if( i == 0 || perm_[i] == 0) 
         // account for the load-imbalancing due to blocking
         load_balance *= getBalancing(sizeA_[i], inc); 
      load_balance *= getBalancing(availableParallelism, parallelismStrategy[i]); 
      totalTasks *= parallelismStrategy[i];
   }

   //how well can these tasks be distributed among numThreads_?
   // e.g., totalTasks = 3, numThreads = 8 => 3./8
   // e.g., totalTasks = 5, numThreads = 8 => 5./8
   // e.g., totalTasks = 15, numThreads = 8 => 15./16
   // e.g., totalTasks = 17, numThreads = 8 => 17./24
   float workDistribution = ((float)totalTasks) / (((totalTasks + numThreads_ - 1)/numThreads_)*numThreads_);

   load_balance *= workDistribution ;
   return load_balance; 
}

template<typename floatType>
void Transpose<floatType>::getBestParallelismStrategy ( std::vector<int> &bestParallelismStrategy ) const
{
   std::vector<int> availableParallelismAtLoop;
   this->getAvailableParallelism( availableParallelismAtLoop);
   int totalAvailableParallelism = std::accumulate(availableParallelismAtLoop.begin(), 
                                 availableParallelismAtLoop.end(), 1, std::multiplies<int>());

   // reduce the probability of parallelizing the stride-1 index
   // if this loops would be parallelized, these two statements ensure that each
   // thread would have at least two macro-kernels of work at this loop-level
   //
   // However, if the total available parallelism is too small, then we do not
   // artificially limit the available parallelism further
   int reduceParallelismB = 4; //avoid parallelization in stride-1 B more strongly
   int reduceParallelismA = 2;
   if( totalAvailableParallelism < 2 * numThreads_ )
      reduceParallelismB = 1;
   else if( totalAvailableParallelism < 4 * numThreads_ )
      reduceParallelismB = 2;
   totalAvailableParallelism = (totalAvailableParallelism / availableParallelismAtLoop[perm_[0]]) * 
                                (availableParallelismAtLoop[perm_[0]]/reduceParallelismB);
   if( totalAvailableParallelism < 2 * numThreads_ )
      reduceParallelismA = 1;
   availableParallelismAtLoop[perm_[0]] = std::max(1, availableParallelismAtLoop[perm_[0]] 
                                                   / reduceParallelismB ); 
   availableParallelismAtLoop[0] = std::max(1, availableParallelismAtLoop[0] / reduceParallelismA );

   // Objectives: 1) load-balancing 
   //             2) avoid parallelizing stride-1 loops (rational: less consecutive memory accesses)
   //             3) avoid false sharing

   std::vector<int> loopsAllowed;
   for(int i=dim_-1; i >= 1; i--)
      if(perm_[i] != 0)
         loopsAllowed.push_back(perm_[i]);
   std::vector<int> loopsAllowedStride1 {0,perm_[0]};

   int totalTasks = 1; // goal: totalTasks should be a close multiple of numTasks_
   std::list<int> primeFactors;
   getPrimeFactors( numThreads_, primeFactors );

   // 1. parallelize using 100% load balancing
   parallelize( bestParallelismStrategy, availableParallelismAtLoop, totalTasks, primeFactors, 1.0, loopsAllowed);
 
   if( totalTasks != numThreads_ )
   { // no perfect match has been found

      // Option 1: keep parallelizing non-stride-1 loops only, but allowing load-imbalance
      std::vector<int> strat1(bestParallelismStrategy);
      std::vector<int> avail1(availableParallelismAtLoop);
      std::list<int> primes1(primeFactors);
      int totalTasks1 = totalTasks;
      parallelize( strat1, avail1, totalTasks1, primes1, 0.92, loopsAllowed);
      if( getLoadBalance(strat1) > 0.90 )
      {
         std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
         return;
      }

      if(perm_[0] != 0)
      {
         // Option 2: also parallelize stride-1 loops, enforcing perfect loop balancing
         std::vector<int> strat2(bestParallelismStrategy);
         std::vector<int> avail2(availableParallelismAtLoop);
         std::list<int> primes2(primeFactors);
         int totalTasks2 = totalTasks;
         parallelize( strat2, avail2, totalTasks2, primes2, 1.0, loopsAllowedStride1 );
         if( getLoadBalance(strat2) > 0.92 )
         {
            std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
            return;
         }

         //keep on going based on strat1
         parallelize( strat1, avail1, totalTasks1, primes1, 1.0, loopsAllowedStride1 );
         if( getLoadBalance(strat1) > 0.90 )
         {
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
            return;
         }

         //keep on going based on strat2
         parallelize( strat2, avail2, totalTasks2, primes2, 0.92, loopsAllowed);
         if( getLoadBalance(strat2) > 0.92 )
         {
            std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
            return;
         }

         if( getLoadBalance(strat1) > 0.80 ) //reduced threshold
         {
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
            return;
         }
         if( getLoadBalance(strat2) > 0.82 ) //reduced threshold
         {
            std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
            return;
         }

         parallelize( strat1, avail1, totalTasks1, primes1, 0.9, loopsAllowedStride1 );
         parallelize( strat2, avail2, totalTasks2, primes2, 0.8, loopsAllowed );
         float lb1 = getLoadBalance(strat1);
         float lb2 = getLoadBalance(strat2);
//         printVector(strat2,"strat2");
//         printf("strat2: %f\n",getLoadBalance(strat2));
         if( (lb1 > 0.8 && lb2 < 0.85) || (lb1 > lb2 && lb1 > 0.75) )
         {
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
            return;
         }
         if(lb2 >= 0.85)
         {
            std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
            return;
         }

         //fallback
         std::vector<int> allLoops;
         for(int i=dim_-1; i >= 1; i--)
            allLoops.push_back(perm_[i]);
         allLoops.push_back(0);
         allLoops.push_back(perm_[0]);
         parallelize( strat1, avail1, totalTasks1, primes1, 0., allLoops);
         std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());

      }else{
         parallelize( strat1, avail1, totalTasks1, primes1, 0.0, loopsAllowed);
         std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
      }
   }
}

template<typename floatType>
void Transpose<floatType>::parallelize( std::vector<int> &parallelismStrategy,
                                        std::vector<int> &availableParallelismAtLoop, 
                                        int &totalTasks,
                                        std::list<int> &primeFactors, 
                                        const float minBalancing, const std::vector<int> &loopsAllowed ) const
   
{
   bool suboptimalParallelizationUsed = false;
   // find loop which minimizes load imbalance for the given prime factor
   for(auto it = primeFactors.begin(); it != primeFactors.end(); it++)
   {
      int suitedLoop = -1;
      float bestBalancing = 0;

      for(auto idx : loopsAllowed )
      {
         float balancing = getBalancing(availableParallelismAtLoop[idx], *it);
         if(balancing > bestBalancing){
            bestBalancing = balancing;
            suitedLoop = idx;
         }
      }
      //allow up to one slightly less optimal splitting to prefer parallelizing
      //idx=0 over idx=perm[0]
      if( suboptimalParallelizationUsed == false &&
            suitedLoop == perm_[0] && getBalancing(availableParallelismAtLoop[0], *it) >= 0.949 ){
         suitedLoop = 0;
         suboptimalParallelizationUsed = true;
      }
      if( suitedLoop != -1 && bestBalancing >= minBalancing ){
         availableParallelismAtLoop[suitedLoop] /= *it;
         parallelismStrategy[suitedLoop] *= *it;
         totalTasks *= *it;
         it = primeFactors.erase(it);
         it--;
      }
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
      parallelismStrategies.emplace_back(std::vector<int>(dim_, 1));
      return;
   }
   std::vector<int> bestParallelismStrategy(dim_, 1);
   getBestParallelismStrategy(bestParallelismStrategy);
   if( this->infoLevel_ > 0 )
      printf("Loadbalancing: %f\n", getLoadBalance(bestParallelismStrategy));

   if( selectionMethod_ == ESTIMATE )
   {
      parallelismStrategies.push_back(bestParallelismStrategy);
      return;
   }

   // ATTENTION: we don't care about the case where numThreads_ is a large prime number...
   // (sorry, KNC)
   //
   // we factorize numThreads into its prime factors because we have to match
   // every one to a certain loop. In principle every loop could be used to
   // match every primefactor, but some choices are preferable over others.
   // E.g., we want to achieve good load-balancing _and_ try to avoid the
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

   parallelismStrategies.insert(parallelismStrategies.begin(), bestParallelismStrategy);

   if( this->infoLevel_ > 1 )
      for( auto strat : parallelismStrategies ){
         printVector(strat,"parallelization");
         printf("cost: %f\n", this->parallelismCostHeuristic( strat ));
      }
}

template<typename floatType>
void Transpose<floatType>::verifyParameter(const int *size, const int* perm, const int* outerSizeA, const int* outerSizeB, const int dim) const
{
   if ( dim < 1 ) {
      fprintf(stderr,"[HPTT] ERROR: dimensionality too low.\n");
      exit(-1);
   }

   std::vector<int> found(dim, 0);

   for(int i=0;i < dim ; ++i)
   {
      if( size[i] <= 0 ) {
         fprintf(stderr,"[HPTT] ERROR: size at position %d is invalid\n", i);
         exit(-1);
      }
      found[ perm[i] ] = 1;
   }

   for(int i=0;i < dim ; ++i)
      if( found[i] <= 0 ) {
         fprintf(stderr,"[HPTT] ERROR: permutation invalid\n");
         exit(-1);
      }

   if ( outerSizeA != NULL )
      for(int i=0;i < dim ; ++i)
         if ( outerSizeA[i] < size[i] ) {
            fprintf(stderr,"[HPTT] ERROR: outerSizeA invalid\n");
            exit(-1);
         }

   if ( outerSizeB != NULL )
      for(int i=0;i < dim ; ++i)
         if ( outerSizeB[i] < size[perm[i]] ) {
            fprintf(stderr,"[HPTT] ERROR: outerSizeB invalid\n");
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

template<typename floatType>
void Transpose<floatType>::skipIndices(const int *sizeA, const int* perm, const int *outerSizeA, const int *outerSizeB, const int dim)
{
   for(int i=0;i < dim ; ++i)
   {
      perm_[i] = perm[i];
      sizeA_[i] = sizeA[i];
      if(outerSizeA)
         outerSizeA_[i] = outerSizeA[i];
      else
         outerSizeA_[i] = sizeA[i];
      if(outerSizeB)
         outerSizeB_[i] = outerSizeB[i];
      else
         outerSizeB_[i] = sizeA[perm[i]];
   }

   int skipped = 0;
   for(int i=0;i < dim ; ++i)
   {
      int idxB = 0;
      for(;idxB < dim ; ++idxB)
         if(perm[idxB] == i)
            break;
      if( sizeA[i] == 1 && (!outerSizeA || outerSizeA[i] == 1) && (!outerSizeB || outerSizeB[idxB] == 1) )
      {
         sizeA_[i] = -1;
         outerSizeA_[i] = -1;
         outerSizeB_[idxB] = -1;
         perm_[idxB] = -1;
         skipped++;
      }
   }
   // compact arrays (remove -1)
   for(int i=0;i < dim ; ++i)
      if( (int)sizeA_[i] == -1 )
      {
         int j=i+1;
         for(;j < dim ; ++j)
            if( (int)sizeA_[j] != -1 )
               break;
         if( j < dim )
            std::swap(sizeA_[i], sizeA_[j]);
      }
   for(int i=0;i < dim ; ++i)
      if( outerSizeA_[i] == -1 )
      {
         int j=i+1;
         for(;j < dim ; ++j)
            if( outerSizeA_[j] != -1 )
               break;
         if( j < dim )
            std::swap(outerSizeA_[i], outerSizeA_[j]);
      }
   for(int i=0;i < dim ; ++i)
      if( outerSizeB_[i] == -1 )
      {
         int j=i+1;
         for(;j < dim ; ++j)
            if( outerSizeB_[j] != -1 )
               break;
         if( j < dim )
            std::swap(outerSizeB_[i], outerSizeB_[j]);
      }
   for(int i=0;i < dim ; ++i)
      if( perm_[i] == -1 )
      {
         int j=i+1;
         for(;j < dim ; ++j)
            if( perm_[j] != -1 )
               break;
         if( j < dim )
            std::swap(perm_[i], perm_[j]);
      }

   dim_ = dim - skipped;
   if( dim_ == 0 ){
      dim_ = 1;
      perm_.resize(dim_);
      sizeA_.resize(dim_);
      outerSizeA_.resize(dim_);
      outerSizeB_.resize(dim_);
      perm_[0] = 0;
      sizeA_[0] = 1;
      outerSizeA_[0] = 1;
      outerSizeB_[0] = 1;
   } else {
      perm_.resize(dim_);
      sizeA_.resize(dim_);
      outerSizeA_.resize(dim_);
      outerSizeB_.resize(dim_);

      // remove gaps in the perm, if requried (e.g., perm=3,1,0 -> 2,1,0)
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
         perm_[minPos] = currentValue; // minValue renamed to currentValue
         currentValue++;
      }
   }

#ifdef DEBUG
   printVector(perm_,"perm");
   printVector(sizeA_,"sizeA");
   printVector(outerSizeA_,"outerSizeA");
   printVector(outerSizeB_,"outerSizeB");
#endif
}

/**
  * \brief fuses indices whenever possible 
  * \detailed For instance:
  *           perm=3,1,2,0 & size=10,11,12,13  becomes: perm=2,1,0 & size=10,11*12,13
  * \return This function will initialize sizeA_, perm_, outerSizeA_, outersize_ and dim_
*/
template<typename floatType>
void Transpose<floatType>::fuseIndices()
{
   std::list< std::tuple<int, int> > fusedIndices;

   std::vector<int> perm;
   // correct perm
   for(int i=0;i < dim_; ++i){
      // merge indices if the two consecutive entries are identical
      int toMerge = i;
      perm.push_back(perm_[i]);
      while(i+1 < dim_ && perm_[i] + 1 == perm_[i+1] 
            && ((int)sizeA_[perm_[i]] == outerSizeA_[perm_[i]]) 
            && ((int)sizeA_[perm_[i]] == outerSizeB_[i]) ){ 
#ifdef DEBUG
         fprintf(stderr,"[HPTT] MERGING indices %d and %d\n",perm_[i], perm_[i+1]); 
#endif
         fusedIndices.emplace_back( std::make_tuple(perm_[toMerge],perm_[i+1]) );
         i++;
      }
   }

   // correct sizes and outer-sizes
   for( auto tup : fusedIndices )
   {
      sizeA_[std::get<0>(tup)] *= sizeA_[std::get<1>(tup)];
      outerSizeA_[std::get<0>(tup)] *= outerSizeA_[std::get<1>(tup)];
      outerSizeA_[std::get<1>(tup)] = -1;

      auto pos1 = std::find(perm_.begin(), perm_.end(), std::get<0>(tup)) - perm_.begin();
      auto pos2 = std::find(perm_.begin(), perm_.end(), std::get<1>(tup)) - perm_.begin();
      outerSizeB_[pos1] *= outerSizeB_[pos2];
      outerSizeB_[pos2] = -1;
   }

   if ( fusedIndices.size() > 0 ) {
      perm_ = perm;
      // remove gaps in the perm, if requried (e.g., perm=3,1,0 -> 2,1,0)
      int currentValue = 0;
      for(size_t i = 0;i < perm_.size(); ++i){
         //find smallest element in perm_ and rename it to currentValue
         int minValue = 1000000;
         int minPos = -1;
         for(int pos = 0; pos < (int)perm_.size(); ++pos){
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


      // compact outer size (e.g.: outerSizeA_[] = {24,-1,5,-1,13} -> {24,5,13,-1,-1} -> {24,5,13}
      for(int i=0;i < dim_ ; ++i)
         if( outerSizeA_[i] == -1 )
         {
            int j=i+1;
            for(;j < dim_ ; ++j)
               if( outerSizeA_[j] != -1 )
                  break;
            if( j < dim_ )
               std::swap(outerSizeA_[i], outerSizeA_[j]);
         }
      for(int i=0;i < dim_ ; ++i)
         if( outerSizeB_[i] == -1 )
         {
            int j=i+1;
            for(;j < dim_ ; ++j)
               if( outerSizeB_[j] != -1 )
                  break;
            if( j < dim_ )
               std::swap(outerSizeB_[i], outerSizeB_[j]);
         }
      dim_ -= fusedIndices.size();
      outerSizeA_.resize(dim_);
      outerSizeB_.resize(dim_);
      sizeA_.resize(dim_);
      perm_.resize(dim_);

#ifdef DEBUG
      printf("\nperm_new: ");
      for(int i=0;i < dim_ ; ++i)
         printf("%d ",perm_[i]);
      printf("\nsizes_new: ");
      for(int i=0;i < dim_ ; ++i)
         printf("%d ",sizeA_[i]);
      printf("\nouterSizeA_new: ");
      for(int i=0;i < dim_ ; ++i)
         printf("%d ",outerSizeA_[i]);
      printf("\nouterSizeB_new: ");
      for(int i=0;i < dim_ ; ++i)
         printf("%d ",outerSizeB_[i]);
      printf("\n");
#endif
   } 
}

// returns the best loop order (same as the best one with exhaustive search)
template<typename floatType>
void Transpose<floatType>::getBestLoopOrder( std::vector<int> &loopOrder ) const
{
   auto totalOuterSizeA = std::accumulate(outerSizeA_.begin(), outerSizeA_.end(), 1, std::multiplies<size_t>()) * sizeof(floatType);
   auto totalOuterSizeB = std::accumulate(outerSizeB_.begin(), outerSizeB_.end(), 1, std::multiplies<size_t>()) * sizeof(floatType);
   if( totalOuterSizeA > totalOuterSizeB && totalOuterSizeB <= 22 * 1024. * 1024. ) // B is likely to fit into L3 cache
   {
      // prefer accesses to A over those to B (Rationale: reduce TLB misses)
      for(int i=0;i < dim_ ; ++i) 
         loopOrder[dim_-1-i] = i; //innermost loop idx is stored at dim_-1
      return;
   }else if( totalOuterSizeB > totalOuterSizeA && totalOuterSizeA <= 22 * 1024. * 1024. ) // B is likely to fit into L3 cache
   {
      // prefer accesses to B over those to A (Rationale: reduce TLB misses)
      for(int i=0;i < dim_ ; ++i) 
         loopOrder[dim_-1-i] = dim_-1-i; //innermost loop idx is stored at dim_-1
      return;
   }

   // create cost matrix; cost[i,idx] === cost for idx being at loop-level i
   double costs[dim_*dim_];
   for(int i=0;i < dim_ ; ++i){
      for(int idx=0;idx < dim_ ; ++idx){ //idx is at loop i
         double cost = 0;
         if( i != 0 ){
            const int posB = findPos(idx, perm_);
            const int importanceA = (1<<(dim_ - idx)); // stride-1 has the most importance ...
            const int importanceB = (1<<(dim_ - posB));                // subsequent indices are half as important
            const int penalty = 10 * (1<<(i-1));
            constexpr double bias = 1.01;
            cost = (importanceA + importanceB * bias) * penalty;
         }
         costs[i + idx * dim_] = cost;
      }
   }
   std::list<int> availLoopLevels; //available rows
   std::list<int> availIndices;
   for(int i=0;i < dim_ ; ++i){
      availLoopLevels.push_back(i);
      availIndices.push_back(i);
   }

   // create best loop order constructively without generating all
   for(int i=0;i < dim_ ; ++i){
      //find column with maximum cost
      int selectedIdx= 0;
      double maxValueAll = 0;
      for(auto c : availIndices)
      {
         double maxValue = 0;
         for(auto r : availLoopLevels){
            const double val = costs[c * dim_ + r];
            maxValue = (val > maxValue) ? val : maxValue;
         }

         if(maxValue > maxValueAll){
            maxValueAll = maxValue;
            selectedIdx= c;
         }
      }
      //find minimum in that column
      int selectedLoopLevel = 0;
      double minValue = 1e100;
      for(auto r : availLoopLevels){
         const double val = costs[selectedIdx * dim_ + r];
         if(val < minValue){
            minValue = val;
            selectedLoopLevel = r;
         }
      }
      // update loop order
      loopOrder[dim_-1-i] = selectedIdx; //innermost loop idx is stored at dim_-1
      // remove selected row
      for( auto it = availLoopLevels.begin(); it != availLoopLevels.end(); it++ )
         if(*it == selectedLoopLevel){
            availLoopLevels.erase(it);
            break;
         }
      // remove selected col
      for( auto it = availIndices.begin(); it != availIndices.end(); it++ )
         if(*it == selectedIdx){
            availIndices.erase(it);
            break;
         }
   }
}

template<typename floatType>
double Transpose<floatType>::loopCostHeuristic( const std::vector<int> &loopOrder ) const
{
   double loopCost= 0.0;
   for(int i=1;i < dim_ ; ++i){
      const int idx = loopOrder[dim_-1-i];
      const int posB = findPos(idx, perm_);
      const int importanceA = (1<<(dim_ - idx)); // stride-1 has the most importance ...
      const int importanceB = (1<<(dim_ - posB));                // subsequent indices are half as important
      const int penalty = 10 * (1<<(i-1));
      double bias = 1.01;
      loopCost += (importanceA + importanceB * bias) * penalty;
   } 

   return loopCost;
}

template<typename floatType>
void Transpose<floatType>::getLoopOrders(std::vector<std::vector<int> > &loopOrders) const
{
   loopOrders.clear();
   if( selectionMethod_ == ESTIMATE ){
      loopOrders.emplace_back(std::vector<int>(dim_));
      getBestLoopOrder( loopOrders[0] );
      return;
   }

   std::vector<int> loopOrder;
   for(int i = 0; i < dim_; i++)
      loopOrder.push_back(i);

   // create all loopOrders
   do {
      if ( perm_[0] == 0 && loopOrder[dim_-1] != 0 )
         continue; // ATTENTION: we skip all loop-orders where the stride-1 index is not the inner-most loop iff perm[0] == 0 (both for perf & correctness)

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
#ifdef HPTT_TIMERS
   double timeStart = omp_get_wtime();
#endif
   
   std::vector<std::shared_ptr<Plan> > allPlans;
   createPlans(allPlans);

#ifdef HPTT_TIMERS
   printf("createPlans() took %f ms\n",(omp_get_wtime()-timeStart)*1000);
   timeStart = omp_get_wtime();
#endif
   masterPlan_ = selectPlan( allPlans );
   if( this->infoLevel_ > 0 ){
      printf("Configuration of best plan:\n");
      masterPlan_->print();
   }
#ifdef HPTT_TIMERS
   printf("SelectPlan() took %f ms\n",(omp_get_wtime()-timeStart)*1000);
#endif
}

template<typename floatType>
void Transpose<floatType>::createPlans( std::vector<std::shared_ptr<Plan> > &plans ) const
{
   if( dim_ == 1 || (dim_ == 2 && perm_[0] == 0)) {
      plans.emplace_back(new Plan); //create dummy plan
      return; // handled within execute()
   }
#ifdef HPTT_TIMERS
   double parallelStrategiesTime = omp_get_wtime();
#endif
   std::vector<std::vector<int> > parallelismStrategies;
   this->getParallelismStrategies(parallelismStrategies);
#ifdef HPTT_TIMERS
   printf("There exists %d parallel strategies. Time: %f ms\n",parallelismStrategies.size(), (omp_get_wtime()-parallelStrategiesTime)*1000);

   double loopOrdersTime = omp_get_wtime();
#endif
   std::vector<std::vector<int> > loopOrders;
   this->getLoopOrders(loopOrders);
#ifdef HPTT_TIMERS
   printf("There exists %d loop orders. Time: %f ms\n",loopOrders.size(), (omp_get_wtime()-loopOrdersTime)*1000);
#endif

   if( selectedParallelStrategyId_ != -1 ){
      int selectedParallelStrategyId = std::min((int)parallelismStrategies.size()-1, selectedParallelStrategyId_);
      std::vector<int>parStrategy(parallelismStrategies[selectedParallelStrategyId]);
      printVector(parStrategy,"selected parallel: ");
      parallelismStrategies.clear();
      parallelismStrategies.push_back(parStrategy);
   }
   if( selectedLoopOrderId_ != -1 ){
      int selectedLoopOrderId = std::min((int)loopOrders.size()-1, selectedLoopOrderId_);
      std::vector<int>loopOrder(loopOrders[selectedLoopOrderId]);
      printVector(loopOrder,"selected loopOrder: ");
      loopOrders.clear();
      loopOrders.push_back(loopOrder);
   }

   const int posStride1A_inB = findPos(0, perm_);
   const int posStride1B_inA = perm_[0];

   // combine the loopOrder and parallelismStrategies according to their
   // heuristics, search the space with a growing rectangle (from best to worst,
   // see line marked with ***)
   bool done = false;
   for( size_t start= 0; start< std::max( parallelismStrategies.size(), loopOrders.size() ) && !done; start++ )
      for( size_t i = 0; i < parallelismStrategies.size() && !done; i++)
      {
         for( size_t j = 0; j < loopOrders.size() && !done; j++)
         {
            if( i > start || j > start || (i != start && j != start) ) continue; //these are already done ***

            auto numThreadsAtLoop = parallelismStrategies[i];
            auto loopOrder = loopOrders[j];
            auto plan = std::make_shared<Plan>(loopOrder, numThreadsAtLoop );
            const int numTasks = plan->getNumTasks();

#ifdef _OPENMP
#pragma omp parallel for num_threads(numThreads_) if(numThreads_ > 1)
#endif
            for( int taskId = 0; taskId < numTasks; taskId++)
            {
               ComputeNode *currentNode = plan->getRootNode(taskId);

               int numThreadsPerComm = numTasks; //global communicator // e.g., 6
               int taskIdComm = taskId; // e.g., 0,1,2,3,4,5
               // divide each loop-level l, corresponding to index loopOrder[l], into numThreadsAtLoop[index] chunks
               for(int l=0; l < dim_; ++l){
                  const int index = loopOrder[l];
                  currentNode->inc = this->getIncrement( index );

                  const int numTasksAtLevel = numThreadsAtLoop[index]; //  e.g., 3
                  const int numParallelismAvailable = (sizeA_[index] + currentNode->inc - 1) / currentNode->inc; // e.g., 5
                  const int workPerThread = (numParallelismAvailable + numTasksAtLevel -1) / numTasksAtLevel; // ceil(5/3) = 2

                  numThreadsPerComm /= numTasksAtLevel; //numThreads in next communicator // 6/3 = 2
                  const int commId = (taskIdComm/numThreadsPerComm); //  = 0,0,1,1,2,2
                  taskIdComm = taskIdComm % numThreadsPerComm; // local taskId in next communicator // 0,1,0,1,0,1

                  currentNode->start = std::min( sizeA_[index], commId * workPerThread * currentNode->inc );
                  currentNode->end = std::min( sizeA_[index], (commId+1) * workPerThread * currentNode->inc );

                  currentNode->lda = lda_[index];
                  currentNode->ldb = ldb_[findPos(index, perm_)];

                  if( perm_[0] != 0 || l != dim_-1 ){
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
            if( (selectionMethod_ == ESTIMATE) || 
                (selectionMethod_ == MEASURE && plans.size() > 200) || 
                (selectionMethod_ == PATIENT && plans.size() > 400) || 
                (selectionMethod_ == CRAZY && plans.size() > 800) )
               done = true;
         }
      }
}

/**
 * Estimates the time in seconds for the given computeTree
 */
template<typename floatType>
float Transpose<floatType>::estimateExecutionTime( const std::shared_ptr<Plan> plan)
{
   auto startTime = std::chrono::high_resolution_clock::now();
   this->executeEstimate(plan.get());
   double elapsedTime = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >
      (std::chrono::high_resolution_clock::now() - startTime).count();

   const double minMeasurementTime = 0.1; // in seconds

   // do at least 3 repetitions or spent at least 'minMeasurementTime' seconds for each candidate
   int nRepeat = std::min(3, (int) std::ceil(minMeasurementTime / elapsedTime));

   //execute just a few iterations and exterpolate the result
   startTime = std::chrono::high_resolution_clock::now();
   for(int i=0;i < nRepeat ; ++i) //ATTENTION: we are not clearing the caches inbetween runs
      this->executeEstimate( plan.get() );
   elapsedTime = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >
      (std::chrono::high_resolution_clock::now() - startTime).count();
   elapsedTime /= nRepeat;

#ifdef DEBUG
   printf("Estimated time: %.3e ms.\n",elapsedTime); 
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
      fprintf(stderr,"[HPTT] ERROR: selectionMethod unknown.\n");
      exit(-1);
   }
   return -1;
}

template<typename floatType>
std::shared_ptr<Plan> Transpose<floatType>::selectPlan( const std::vector<std::shared_ptr<Plan> > &plans)
{
   if( plans.size() <= 0 ){
      fprintf(stderr,"[HPTT] Internal error: not enough plans generated.\n");
      exit(-1);
   }
   if( selectionMethod_ == ESTIMATE ) // fast return
      return plans[0];

   double timeLimit = this->getTimeLimit() * 1000; //in ms
   int maxAutotuningCandidates = plans.size();
   if( maxAutotuningCandidates_ != -1 ){
      maxAutotuningCandidates = maxAutotuningCandidates_;
      timeLimit = 1e9;
   }

   float minTime = FLT_MAX;
   int bestPlan_id = 0;

   if( plans.size() > 1 )
   {
      int plansEvaluated = 0;
      auto startTime = std::chrono::high_resolution_clock::now();
      for( int plan_id = 0; plan_id < maxAutotuningCandidates; plan_id++ )
      {
         auto p = plans[plan_id];

         double elapsedTime = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >
            (std::chrono::high_resolution_clock::now() - startTime).count();
         if( elapsedTime >= timeLimit ) // timelimit reached
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
         printf("We evaluated %d/%ld candidates and selected candidate %d.\n", plansEvaluated, plans.size(), bestPlan_id); 
   }
   return plans[bestPlan_id];
}

template class Transpose<float>;
template class Transpose<double>;
template class Transpose<FloatComplex>;
template class Transpose<DoubleComplex>;

template void Transpose<float>::execute_expert<true, true, true>();
template void Transpose<float>::execute_expert<true, false, true>();
template void Transpose<float>::execute_expert<false, true, true>();
template void Transpose<float>::execute_expert<false, false, true>();
template void Transpose<float>::execute_expert<true, true, false>();
template void Transpose<float>::execute_expert<true, false, false>();
template void Transpose<float>::execute_expert<false, true, false>();
template void Transpose<float>::execute_expert<false, false, false>();

template void Transpose<double>::execute_expert<true,true, true>();
template void Transpose<double>::execute_expert<false,true, true>();
template void Transpose<double>::execute_expert<true,false, true>();
template void Transpose<double>::execute_expert<false,false, true>();
template void Transpose<double>::execute_expert<true,true, false>();
template void Transpose<double>::execute_expert<false,true, false>();
template void Transpose<double>::execute_expert<true,false, false>();
template void Transpose<double>::execute_expert<false,false, false>();

template void Transpose<FloatComplex>::execute_expert<true,true, true>();
template void Transpose<FloatComplex>::execute_expert<false,true, true>();
template void Transpose<FloatComplex>::execute_expert<true,false, true>();
template void Transpose<FloatComplex>::execute_expert<false,false, true>();
template void Transpose<FloatComplex>::execute_expert<true,true, false>();
template void Transpose<FloatComplex>::execute_expert<false,true, false>();
template void Transpose<FloatComplex>::execute_expert<true,false, false>();
template void Transpose<FloatComplex>::execute_expert<false,false, false>();

template void Transpose<DoubleComplex>::execute_expert<true,true, true>();
template void Transpose<DoubleComplex>::execute_expert<false,true, true>();
template void Transpose<DoubleComplex>::execute_expert<true,false, true>();
template void Transpose<DoubleComplex>::execute_expert<false,false, true>();
template void Transpose<DoubleComplex>::execute_expert<true,true, false>();
template void Transpose<DoubleComplex>::execute_expert<false,true, false>();
template void Transpose<DoubleComplex>::execute_expert<true,false, false>();
template void Transpose<DoubleComplex>::execute_expert<false,false, false>();


}
