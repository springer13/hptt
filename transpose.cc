#include "ttc_c.h"
#include <stdio.h>
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



int findPos(int value, const int *array, int n)
{
   for(int i=0;i < n ; ++i)
      if( array[i] == value )
         return i;
   return -1;
}

int verifyParameter(const int *size, const int* perm, const int dim)
{
   if ( dim < 2 )
      return 1;

   int found[dim];
   for(int i=0;i < dim ; ++i)
      found[i] = 0;

   for(int i=0;i < dim ; ++i)
   {
      if( size[i] <= 0 )
         return 2;
      found[ perm[i] ] = 1;
   }

   for(int i=0;i < dim ; ++i)
      if( found[i] <= 0 )
         return 3;
   return 0;
}

void computeLeadingDimensions( const int* size, const int *perm, int dim,
                               const int *outerSizeA, const int*outerSizeB, 
                               int *lda, int *ldb )
{
   lda[0] = 1;
   if( outerSizeA == NULL )
      for(int i=1;i < dim ; ++i)
         lda[i] = lda[i-1] * size[i-1];
   else
      for(int i=1;i < dim ; ++i)
         lda[i] = outerSizeA[i-1] * lda[i-1];

   ldb[0] = 1;
   if( outerSizeB == NULL )
      for(int i=1;i < dim ; ++i)
         ldb[i] = ldb[i-1] * size[perm[i-1]];
   else
      for(int i=1;i < dim ; ++i)
         ldb[i] = outerSizeB[i-1] * ldb[i-1];
}

node_t* createPlan(const int *outerSizeA, const int *outerSizeB, const int *size, const int* perm, const int dim)
{
   int emitCode = 1; // only for DEBUGGING

   int errorCode = verifyParameter(size, perm, dim);
   if( errorCode > 0 ) {
      printf("Error: %d\n", errorCode);
      exit(-1);
   }

   int lda[dim];
   int ldb[dim];
   computeLeadingDimensions( size, perm, dim, outerSizeA, outerSizeB, lda, ldb );
   
   node_t *plan = (node_t*) malloc(sizeof(node_t));
   node_t *currentPtr = plan;

   int posStride1A_inB = findPos(0, perm, dim);
   int posStride1B_inA = perm[0];

   if( perm[0] == 0 ){ printf("TODO\n"); exit(-1); } // TODO

   // create loops
   for(int i=0; i < dim; ++i) {
      int index = perm[dim-1-i]; // loop-order according to: reversed(perm)

      currentPtr->start = 0;
      if( index == 0 || index == perm[0] )
         currentPtr->inc = 16;
      else
         currentPtr->inc = 1;
      currentPtr->end = size[index];
      currentPtr->lda = lda[index];
      currentPtr->ldb = ldb[findPos(index, perm, dim)];
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
         printf("  const float *A%s = &A%s[i%d * lda%d]; float *B%s = &B%s[i%d * ldb%d];\n",underscores,underscores_old,index,index,underscores,underscores_old,index,findPos(index, perm, dim));
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
      for(int j=0;j < dim ; ++j)
         underscores[j] = '_';
      underscores[dim] = '\0';

      printf("  sTranspose16x16(A%s, lda%d, B%s, ldb%d, reg_alpha, reg_beta);\n",underscores,posStride1B_inA, underscores, posStride1A_inB);
      for(int j=0;j < dim ; ++j)
         printf("  }\n");
   }

   return plan;
}

void ttc_sTranspose_int( const float* __restrict__ A, float* __restrict__ B, const __m256 alpha, const __m256 beta, const node_t* plan)
{
   const int end = plan->end - (plan->inc - 1);
   const int inc = plan->inc;
   const int lda_ = plan->lda;
   const int ldb_ = plan->ldb;

   if( plan->next != nullptr )
      for(int i = plan->start; i < end; i+= inc)
         // recurse
         ttc_sTranspose_int( &A[i*lda_], &B[i*ldb_], alpha, beta, plan->next);
   else 
      // invoke macro-kernel
      sTranspose16x16(A, lda_, B, ldb_, alpha, beta);
}

/**
 * B(i2,i1,i0) <- alpha * A(i0,i1,i2) + beta * B(i2,i1,i0);
 */
void ttc_sTranspose( const float* __restrict__ A, float* __restrict__ B, const float alpha, const float beta, const node_t*plan)
{
   //broadcast reg_alpha
   __m256 reg_alpha = _mm256_set1_ps(alpha);
   //broadcast reg_beta
   __m256 reg_beta = _mm256_set1_ps(beta);

   ttc_sTranspose_int( A, B, reg_alpha, reg_beta, plan);
}

void trashCache(double *A, double *B, int n)
{
   for(int i = 0; i < n; i++)
      A[i] += 0.999 * B[i];
}
