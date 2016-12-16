#ifndef STRANSPOSE_43210_64X64X64X64X64_H
#define STRANSPOSE_43210_64X64X64X64X64_H
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
 * B(i4,i3,i2,i1,i0) <- alpha * A(i0,i1,i2,i3,i4) + beta * B(i4,i3,i2,i1,i0);
 */
template<int size0, int size1, int size2, int size3, int size4>
void sTranspose_43210_64x64x64x64x64( const float* __restrict__ A, float* __restrict__ B, const float alpha, const float beta, const int *lda, const int *ldb)
{
   int lda1;
   int lda2;
   int lda3;
   int lda4;
   if( lda == NULL ){
      lda1 = size0;
      lda2 = size1 * lda1;
      lda3 = size2 * lda2;
      lda4 = size3 * lda3;
   }else{
      lda1 = lda[0];
      lda2 = lda[1] * lda1;
      lda3 = lda[2] * lda2;
      lda4 = lda[3] * lda3;
   }
   int ldb1;
   int ldb2;
   int ldb3;
   int ldb4;
   if( ldb == NULL ){
      ldb1 = size4;
      ldb2 = size3 * ldb1;
      ldb3 = size2 * ldb2;
      ldb4 = size1 * ldb3;
   }else{
      ldb1 = ldb[0];
      ldb2 = ldb[1] * ldb1;
      ldb3 = ldb[2] * ldb2;
      ldb4 = ldb[3] * ldb3;
   }
   const int remainder0 = size0 % 16;
   const int remainder4 = size4 % 16;
   //broadcast reg_alpha
   __m256 reg_alpha = _mm256_set1_ps(alpha);
   //broadcast reg_beta
   __m256 reg_beta = _mm256_set1_ps(beta);
   for(int i0 = 0; i0 < size0 - 15; i0+= 16)
      for(int i1 = 0; i1 < size1; i1+= 1)
         for(int i2 = 0; i2 < size2; i2+= 1)
            for(int i3 = 0; i3 < size3; i3+= 1)
               for(int i4 = 0; i4 < size4 - 15; i4+= 16)
                  sTranspose16x16(&A[i0 + i1*lda1 + i2*lda2 + i3*lda3 + i4*lda4], lda4, &B[i0*ldb4 + i1*ldb3 + i2*ldb2 + i3*ldb1 + i4], ldb4  ,reg_alpha ,reg_beta);
   //Remainder loop
   for(int i0 = size0 - remainder0; i0 < size0; i0 += 1)
      for(int i1 = 0; i1 < size1; i1 += 1)
         for(int i2 = 0; i2 < size2; i2 += 1)
            for(int i3 = 0; i3 < size3; i3 += 1)
               for(int i4 = 0; i4 < size4 - remainder4; i4 += 1)
                  B[i0*ldb4 + i1*ldb3 + i2*ldb2 + i3*ldb1 + i4] = alpha*A[i0 + i1*lda1 + i2*lda2 + i3*lda3 + i4*lda4] + beta*B[i0*ldb4 + i1*ldb3 + i2*ldb2 + i3*ldb1 + i4];
   //Remainder loop
   for(int i0 = 0; i0 < size0; i0 += 1)
      for(int i1 = 0; i1 < size1; i1 += 1)
         for(int i2 = 0; i2 < size2; i2 += 1)
            for(int i3 = 0; i3 < size3; i3 += 1)
               for(int i4 = size4 - remainder4; i4 < size4; i4 += 1)
                  B[i0*ldb4 + i1*ldb3 + i2*ldb2 + i3*ldb1 + i4] = alpha*A[i0 + i1*lda1 + i2*lda2 + i3*lda3 + i4*lda4] + beta*B[i0*ldb4 + i1*ldb3 + i2*ldb2 + i3*ldb1 + i4];
}
#endif
