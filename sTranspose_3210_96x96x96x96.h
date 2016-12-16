#ifndef STRANSPOSE_3210_96X96X96X96_H
#define STRANSPOSE_3210_96X96X96X96_H
#include <xmmintrin.h>
#include <immintrin.h>
#include <complex.h>
#if defined(__ICC) || defined(__INTEL_COMPILER)
#define INLINE __forceinline
#elif defined(__GNUC__) || defined(__GNUG__)
#define INLINE __attribute__((always_inline))
#endif

#ifndef _TTC_STRANSPOSE8X8
#define _TTC_STRANSPOSE8X8
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
#endif
#ifndef _TTC_STRANSPOSE16X16
#define _TTC_STRANSPOSE16X16
//B_ji = alpha * A_ij + beta * B_ji
static INLINE void sTranspose16x16(const float* __restrict__ A, const int lda, float* __restrict__ B, const int ldb  ,const __m256 &reg_alpha ,const __m256 &reg_beta)
{
   //invoke micro-transpose
   sTranspose8x8(A, lda, B, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 8 * lda, lda, B + 8, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 8, lda, B + 8 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 8 + 8 * lda, lda, B + 8 + 8 * ldb, ldb  , reg_alpha , reg_beta);

}
#endif
/**
 * B(i3,i2,i1,i0) <- alpha * A(i0,i1,i2,i3) + beta * B(i3,i2,i1,i0);
 */
template<int size0, int size1, int size2, int size3>
void sTranspose_3210_96x96x96x96( const float* __restrict__ A, float* __restrict__ B, const float alpha, const float beta, const int *lda, const int *ldb)
{
   int lda1;
   int lda2;
   int lda3;
   if( lda == NULL ){
      lda1 = size0;
      lda2 = size1 * lda1;
      lda3 = size2 * lda2;
   }else{
      lda1 = lda[0];
      lda2 = lda[1] * lda1;
      lda3 = lda[2] * lda2;
   }
   int ldb1;
   int ldb2;
   int ldb3;
   if( ldb == NULL ){
      ldb1 = size3;
      ldb2 = size2 * ldb1;
      ldb3 = size1 * ldb2;
   }else{
      ldb1 = ldb[0];
      ldb2 = ldb[1] * ldb1;
      ldb3 = ldb[2] * ldb2;
   }
   const int remainder0 = size0 % 16;
   const int remainder3 = size3 % 16;
   //broadcast reg_alpha
   __m256 reg_alpha = _mm256_set1_ps(alpha);
   //broadcast reg_beta
   __m256 reg_beta = _mm256_set1_ps(beta);
   for(int i0 = 0; i0 < size0 - 15; i0+= 16)
      for(int i1 = 0; i1 < size1; i1+= 1)
         for(int i2 = 0; i2 < size2; i2+= 1)
            for(int i3 = 0; i3 < size3 - 15; i3+= 16)
               sTranspose16x16(&A[i0 + i1*lda1 + i2*lda2 + i3*lda3], lda3, &B[i0*ldb3 + i1*ldb2 + i2*ldb1 + i3], ldb3  ,reg_alpha ,reg_beta);

   //Remainder loop
   for(int i0 = size0 - remainder0; i0 < size0; i0 += 1)
      for(int i1 = 0; i1 < size1; i1 += 1)
         for(int i2 = 0; i2 < size2; i2 += 1)
            for(int i3 = 0; i3 < size3 - remainder3; i3 += 1)
               B[i0*ldb3 + i1*ldb2 + i2*ldb1 + i3] = alpha*A[i0 + i1*lda1 + i2*lda2 + i3*lda3] + beta*B[i0*ldb3 + i1*ldb2 + i2*ldb1 + i3];
   //Remainder loop
   for(int i0 = 0; i0 < size0; i0 += 1)
      for(int i1 = 0; i1 < size1; i1 += 1)
         for(int i2 = 0; i2 < size2; i2 += 1)
            for(int i3 = size3 - remainder3; i3 < size3; i3 += 1)
               B[i0*ldb3 + i1*ldb2 + i2*ldb1 + i3] = alpha*A[i0 + i1*lda1 + i2*lda2 + i3*lda3] + beta*B[i0*ldb3 + i1*ldb2 + i2*ldb1 + i3];
}
#endif
