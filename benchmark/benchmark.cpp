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

//#include <ttc_c.h>
//#include "ttc.h"


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

//  // Create handle
//  ttc_handler_s *ttc_handle = ttc_init();
//
//  // Create transpose parameter
//  ttc_param_s param = { .alpha.s = alpha, .beta.s = beta, .lda = NULL, .ldb = NULL, .perm = perm, .size = size, .loop_perm = NULL, .dim = dim};
//
//  // Set TTC options (THIS IS OPTIONAL)
//  int maxImplemenations = 100;
//  ttc_set_opt( ttc_handle, TTC_OPT_MAX_IMPL, (void*)&maxImplemenations, 1 );
//  ttc_set_opt( ttc_handle, TTC_OPT_NUM_THREADS, (void*)&numThreads, 1 );
//  char affinity[] = "compact,1";
//  ttc_set_opt( ttc_handle, TTC_OPT_AFFINITY, (void*)affinity, strlen(affinity) );

  // Allocating memory for tensors

  int largerThanL3 = 1024*1024*100/sizeof(double);
  float *A, *B, *B_orig, *B_paul, *B_tong;
  double *trash1, *trash2;
  int ret = posix_memalign((void**) &trash1, 32, sizeof(double) * largerThanL3);
  ret += posix_memalign((void**) &trash2, 32, sizeof(double) * largerThanL3);
  ret += posix_memalign((void**) &B, 32, sizeof(float) * total_size);
  ret += posix_memalign((void**) &A, 32, sizeof(float) * total_size);
  ret += posix_memalign((void**) &B, 32, sizeof(float) * total_size);
  ret += posix_memalign((void**) &B_orig, 32, sizeof(float) * total_size);
  ret += posix_memalign((void**) &B_paul, 32, sizeof(float) * total_size);
  ret += posix_memalign((void**) &B_tong, 32, sizeof(float) * total_size);
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
     B_tong[i] = B[i];
     B_paul[i] = B[i];
  }

#pragma omp parallel for
  for(int i=0;i < largerThanL3; ++i)
  {
     trash1[i] = ((i+1)*13)%10000;
     trash2[i] = ((i+1)*13)%10000;
  }
  
  // Execute transpose
  {  //hptt-paul
     int perm_[dim];
     int size_[dim];
     for(int i=0;i < dim ; ++i){
        perm_[i] = (int)perm[i];
        size_[i] = (int)size[i];
     }

     hptt::Transpose transpose( size_, perm_, NULL, NULL, dim, A, alpha, B_paul, beta, hptt::MEASURE, numThreads );
     transpose.createPlan();

     double minTime = 1e200;
     for(int i=0;i < nRepeat ; ++i){
        restore(B_paul, B, total_size);
        hptt::trashCache(trash1, trash2, largerThanL3);
        auto begin_time = omp_get_wtime();
        transpose.execute();
        auto elapsed_time = omp_get_wtime() - begin_time;
        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
     }
     printf("HPTT (paul) %d %s %s: %.2f ms. %.2f GiB/s\n", dim, perm_str.c_str(), size_str.c_str(), minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
  }
//  { // original ttc
//     double minTime = 1e200;
//     for(int i=0;i < nRepeat ; ++i){
//        restore(B_orig, B, total_size);
//        ttc::trashCache(trash1, trash2, largerThanL3);
//        auto begin_time = omp_get_wtime();
//        // Execute transpose
//        ttc_transpose(ttc_handle, &param, A, B_orig);
//        double elapsed_time = omp_get_wtime() - begin_time;
//        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
//     }
//     printf("TTC (orig) %d %s %s: %.2f ms. %.2f GiB/s\n", dim, perm_str.c_str(), size_str.c_str(), minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
//  }
//  { // tong ttc
//     std::vector<uint32_t> perm_;
//     for(int i=0;i < dim; ++i)
//        perm_.push_back(perm[i]);
//     std::vector<uint32_t> size_;
//     for(int i=0;i < dim; ++i)
//        size_.push_back(size[i]);
//
//     hptc::DeducedFloatType<float> alpha_ = alpha;
//     hptc::DeducedFloatType<float> beta_ = beta;
//     double timeout = 10;
//     double minTime = 1e200;
//     auto plan = hptc::create_cgraph_trans<float>(A, B_tong,
//            size_, perm_,
//            alpha_, beta_, 
//            numThreads, timeout);
//
//     for(int i=0;i < nRepeat ; ++i){
//        restore(B_tong, B, total_size);
//        ttc::trashCache(trash1, trash2, largerThanL3);
//        auto begin_time = omp_get_wtime();
//        if (nullptr != plan)
//           plan->exec();
//        else
//           printf("ERROR\n");
//        auto elapsed_time = omp_get_wtime() - begin_time;
//        minTime = (elapsed_time < minTime) ? elapsed_time : minTime;
//     }
//     printf("TTC (tong) %d %s %s: %.2f ms. %.2f GiB/s\n", dim, perm_str.c_str(), size_str.c_str(), minTime*1000, sizeof(float)*total_size*3/1024./1024./1024 / minTime);
//  }
  
  // Verification
//  if( !equal_(B_orig, B_paul, total_size) )
//     fprintf(stderr, "error in ttc_paul\n");
//  if( !equal_(B_orig, B_tong, total_size) )
//     fprintf(stderr, "error in ttc_tong\n");

  return 0;
}
