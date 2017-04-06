// include header


#include <memory>
#include <vector>
#include <numeric>
#include <string>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <math.h>

#include "../src/hptt.h"

#include <hptc/hptc.h>

#include <ttc_c.h>

#define ORIG_TTC
#define RELEASE_HPTT

int equal_(const float *A, const float*B, int total_size){
  int error = 0;
   const float *Atmp= A;
   const float *Btmp= B;
   for(int i=0;i < total_size ; ++i){
      if( Atmp[i] != Atmp[i] || Btmp[i] != Btmp[i]  || isinf(Atmp[i]) || isinf(Btmp[i]) ){
         error += 1; //test for NaN or Inf
         continue;
      }
      double Aabs = (Atmp[i] < 0) ? -Atmp[i] : Atmp[i];
      double Babs = (Btmp[i] < 0) ? -Btmp[i] : Btmp[i];
      double max = (Aabs < Babs) ? Babs : Aabs;
      double diff = (Aabs - Babs);
      diff = (diff < 0) ? -diff : diff;
      if(diff > 0){
         double relError = (diff / max);
         if(relError > 4e-5){
//            printf("%.3e %.3e\n",relError, max);
            error += 1;
         }
      }
   }
   return (error == 0) ? 1 : 0;
}

void restore(const float* A, float* B, size_t n)
{
   for(size_t i=0;i < n ; ++i)
      B[i] = A[i];
}

void transpose_ref( uint32_t *size, uint32_t *perm, int dim, const float* __restrict__ A, float alpha, float* __restrict__ B, float beta)
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

      const float* __restrict__ A_ = A + offsetA;
      float* __restrict__ B_ = B + j*sizeInner;

      uint32_t strideAinner = strideA[perm[0]];

      for(int i=0; i < sizeInner; ++i)
         B_[i] = alpha * A_[i * strideAinner] + beta * B_[i];
   }
}

int main(int argc, char *argv[]) 
{
  int numThreads = 1;
  if( getenv("OMP_NUM_THREADS") != NULL )
     numThreads = atoi(getenv("OMP_NUM_THREADS"));
  printf("numThreads: %d\n",numThreads);
  float alpha = 2.2;
  float beta = 4.1;

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
  std::string perm_str = "";
  for(int i=0;i < dim ; ++i){
     perm[i] = atoi(argv[2+i]);
     perm_str += std::to_string(perm[i]) + ",";
  }
  uint32_t size[dim];
  std::string size_str = "";
  size_t total_size = 1;
  for(int i=0;i < dim ; ++i){
     size[i] = atoi(argv[2+dim+i]);
     size_str += std::to_string(size[i]) + ",";
     total_size *= size[i];
  }

  int nRepeat = 5;

  // Allocating memory for tensors
  int largerThanL3 = 1024*1024*100/sizeof(double);
  float *A, *B, *B_ref, *B_orig, *B_proto, *B_hptt;
  double *trash1, *trash2;
  int ret = posix_memalign((void**) &trash1, 32, sizeof(double) * largerThanL3);
  ret += posix_memalign((void**) &trash2, 32, sizeof(double) * largerThanL3);
  ret += posix_memalign((void**) &B, 32, sizeof(float) * total_size);
  ret += posix_memalign((void**) &A, 32, sizeof(float) * total_size);
  ret += posix_memalign((void**) &B_ref, 32, sizeof(float) * total_size);
  ret += posix_memalign((void**) &B_orig, 32, sizeof(float) * total_size);
  ret += posix_memalign((void**) &B_proto, 32, sizeof(float) * total_size);
  ret += posix_memalign((void**) &B_hptt, 32, sizeof(float) * total_size);
  if( ret ){
     printf("ALLOC ERROR\n");
     exit(-1);
  }

  // initialize data
#pragma omp parallel for
  for(int i=0;i < total_size; ++i)
     A[i] = (((i+1)*13 % 1000) - 500.) / 1000.;
#pragma omp parallel for
  for(int i=0;i < total_size ; ++i){
     B[i] = (((i+1)*17 % 1000) - 500.) / 1000.;
     B_orig[i] = B[i];
     B_ref[i]  = B[i];
     B_hptt[i] = B[i];
     B_proto[i] = B[i];
  }

#pragma omp parallel for
  for(int i=0;i < largerThanL3; ++i)
  {
     trash1[i] = ((i+1)*13)%10000;
     trash2[i] = ((i+1)*13)%10000;
  }
  
  {  //hptt prototype
     int perm_[dim];
     int size_[dim];
     for(int i=0;i < dim ; ++i){
        perm_[i] = (int)perm[i];
        size_[i] = (int)size[i];
     }

     hptt::Transpose transpose( size_, perm_, NULL, NULL, dim, A, alpha, B_proto, beta, hptt::MEASURE, numThreads );
     transpose.createPlan();

     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        restore(B, B_proto, total_size);
        hptt::trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        // Execute transpose
        transpose.execute();
        auto elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("HPTT (proto) %d %s %s: %.2f ms. %.2f GiB/s\n", dim, perm_str.c_str(), size_str.c_str(), minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }

  { // reference
     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        restore(B, B_ref, total_size);
        hptt::trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        transpose_ref( size, perm, dim, A, alpha, B_ref, beta);
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC (ref) %d %s %s: %.2f ms. %.2f GiB/s\n", dim, perm_str.c_str(), size_str.c_str(), minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }

  // Verification
  if( !equal_(B_ref, B_proto, total_size) )
     fprintf(stderr, "error in ttc_proto\n");

#ifdef ORIG_TTC
  { // original ttc
     // Create handle
     ttc_handler_s *ttc_handle = ttc_init();

     // Create transpose parameter
     ttc_param_s param = { .alpha.s = alpha, .beta.s = beta, .lda = NULL, .ldb = NULL, .perm = perm, .size = size, .loop_perm = NULL, .dim = dim};

     // Set TTC options (THIS IS OPTIONAL)
     int maxImplemenations = 100;
     ttc_set_opt( ttc_handle, TTC_OPT_MAX_IMPL, (void*)&maxImplemenations, 1 );
     ttc_set_opt( ttc_handle, TTC_OPT_NUM_THREADS, (void*)&numThreads, 1 );
     char affinity[] = "compact,1";
     ttc_set_opt( ttc_handle, TTC_OPT_AFFINITY, (void*)affinity, strlen(affinity) );

     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        restore(B, B_orig, total_size);
        hptt::trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        // Execute transpose
        ttc_transpose(ttc_handle, &param, A, B_orig);
        double elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("TTC (orig) %d %s %s: %.2f ms. %.2f GiB/s\n", dim, perm_str.c_str(), size_str.c_str(), minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
     //verify
     if( !equal_(B_orig, B_ref, total_size) )
        fprintf(stderr, "error in reference\n");
  }
#endif

#ifdef RELEASE_HPTT
  { // RELEASE HPTT
     std::vector<uint32_t> perm_;
     for(int i=0;i < dim; ++i)
        perm_.push_back(perm[i]);
     std::vector<uint32_t> size_;
     for(int i=0;i < dim; ++i)
        size_.push_back(size[i]);

     hptc::DeducedFloatType<float> alpha_ = alpha;
     hptc::DeducedFloatType<float> beta_ = beta;
     double timeout = 10;
     double minTime = 1e200;
     auto plan = hptc::create_trans_plan<float>(A, B_hptt,
            size_, perm_,
            alpha_, beta_, 
            numThreads, timeout);

     for(int i=0;i < nRepeat ; ++i){
        restore(B, B_hptt, total_size);
        hptt::trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        if (nullptr != plan)
           plan->exec();
        else
           printf("ERROR\n");
        auto elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("HPTT (release) %d %s %s: %.2f ms. %.2f GiB/s\n", dim, perm_str.c_str(), size_str.c_str(), minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
     // verify 
     if( !equal_(B_ref, B_hptt, total_size) )
        fprintf(stderr, "error in ttc_hptt\n");
  }
#endif   

  return 0;
}
