// include header


#include <memory>
#include <vector>
#include <numeric>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <stdlib.h>


#include <ttc_c.h>
#include "ttc.h"


void equal_(const float *A, const float*B, int total_size){
  int error = 0;
   const float *Atmp= A;
   const float *Btmp= B;
   for(int i=0;i < total_size ; ++i){
      double Aabs = (Atmp[i] < 0) ? -Atmp[i] : Atmp[i];
      double Babs = (Btmp[i] < 0) ? -Btmp[i] : Btmp[i];
      double max = (Aabs < Babs) ? Babs : Aabs;
      double diff = (Aabs - Babs);
      diff = (diff < 0) ? -diff : diff;
      if(diff > 0){
         double relError = (diff / max);
         if(relError > 4e-5){
            //printf("i: %d relError: %.8e\n",i,relError);
            error += 1;
         }
      }
   }
   if( error > 0 ) 
     printf("ERROR\n");
  else
     printf("SUCCESS\n");
}

void restore(const float* A, float* B, size_t n)
{
   for(size_t i=0;i < n ; ++i)
      B[i] = A[i];
}

int main(int argc, char *argv[]) 
{
  int numThreads = 1;
  if( getenv("OMP_NUM_THREADS") != NULL )
     numThreads = atoi(getenv("OMP_NUM_THREADS"));
  printf("numThreads: %d\n",numThreads);
  float alpha = 2.1;
  float beta = 4.0;

  if( argc < 2 ){
     printf("Usage: <dim> <permutation each index separated by ' '> <size of each index separated by ' '>\n");
     exit(-1);
  }
  int dim = atoi(argv[1]);
  if( argc < 2 + 2*dim ){
     printf("Error: not enough indices for permutation provided.");
     exit(-1);
  }
  uint32_t perm[dim];
  printf("permutation: ");
  for(int i=0;i < dim ; ++i){
     perm[i] = atoi(argv[2+i]);
     printf("%d,", perm[i]);
  }
  printf("\nsize: ");
  uint32_t size[dim];
  size_t total_size = 1;
  for(int i=0;i < dim ; ++i){
     size[i] = atoi(argv[2+dim+i]);
     printf("%d,", size[i]);
     total_size *= size[i];
  }
  printf("\n");

  int nRepeat = 5;

  // Create handle
  ttc_handler_s *ttc_handle = ttc_init();

  // Create transpose parameter
  ttc_param_s param = { .alpha.s = alpha, .beta.s = beta, .lda = NULL, .ldb = NULL, .perm = perm, .size = size, .loop_perm = NULL, .dim = dim};

  // Set TTC options (THIS IS OPTIONAL)
  int maxImplemenations = 10;
  ttc_set_opt( ttc_handle, TTC_OPT_MAX_IMPL, (void*)&maxImplemenations, 1 );
  ttc_set_opt( ttc_handle, TTC_OPT_NUM_THREADS, (void*)&numThreads, 1 );
  char affinity[] = "compact,1";
  ttc_set_opt( ttc_handle, TTC_OPT_AFFINITY, (void*)affinity, strlen(affinity) );

  // Allocating memory for tensors

  int largerThanL3 = 1024*1024*100/sizeof(double);
  float *A, *B, *B_copy, *B_ttc;
  double *trash1, *trash2;
  posix_memalign((void**) &trash1, 32, sizeof(double) * largerThanL3);
  posix_memalign((void**) &trash2, 32, sizeof(double) * largerThanL3);
  posix_memalign((void**) &B, 32, sizeof(float) * total_size);
  posix_memalign((void**) &A, 32, sizeof(float) * total_size);
  posix_memalign((void**) &B_copy, 32, sizeof(float) * total_size);
  posix_memalign((void**) &B_ttc, 32, sizeof(float) * total_size);

  // initialize data
#pragma omp parallel for
  for(int i=0;i < total_size; ++i)
     A[i] = (((i+1)*13 % 10000) - 5000.) / 10000.;
#pragma omp parallel for
  for(int i=0;i < total_size ; ++i){
     B[i] = (((i+1)*17 % 10000) - 5000.) / 10000.;
     B_copy[i] = B[i];
     B_ttc[i] = B[i];
  }

#pragma omp parallel for
  for(int i=0;i < largerThanL3; ++i)
  {
     trash1[i] = ((i+1)*13)%100000;
     trash2[i] = ((i+1)*13)%100000;
  }


  // Execute transpose
  {  //ttc-c-paul
     int perm_[dim];
     int size_[dim];
     for(int i=0;i < dim ; ++i){
        perm_[i] = perm[i];
        size_[i] = size[i];
     }

     ttc::Transpose transpose( size_, perm_, NULL, NULL, dim, A, alpha, B_ttc, beta, ttc::MEASURE, numThreads );
     transpose.createPlan();

     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        ttc::trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        transpose.execute();
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC (paul): %.2f ms. %.2f GiB/s\n", minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }
  { // original ttc
     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        ttc::trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        // Execute transpose
        ttc_transpose(ttc_handle, &param, A, B);
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC: %.2f ms. %.2f GiB/s\n", minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }
  
  // Verification
  equal_(B, B_ttc, total_size);

  return 0;
}
