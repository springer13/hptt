/**
 * @author: Paul Springer (springer@aices.rwth-aachen.de)
 */


#include <memory>
#include <vector>
#include <numeric>
#include <string>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <cmath>
#include <complex>

#include "../include/hptt.h"
#include "../benchmark/reference.h"
#include "../benchmark/defines.h"

#define MAX_DIM 8
#define NUM_TESTS 40

template<typename floatType>
static double getZeroThreshold();
template<>
double getZeroThreshold<double>() { return 1e-16;}
template<>
double getZeroThreshold<DoubleComplex>() { return 1e-16;}
template<>
double getZeroThreshold<float>() { return 1e-6;}
template<>
double getZeroThreshold<FloatComplex>() { return 1e-6;}


template<typename floatType>
int equal_(const floatType *A, const floatType*B, int total_size){
  int error = 0;
   for(int i=0;i < total_size ; ++i){
      if( A[i] != A[i] || B[i] != B[i]  || std::isinf(std::abs(A[i])) || std::isinf(std::abs(B[i])) ){
         error += 1; //test for NaN or Inf
         continue;
      }
      double Aabs = std::abs(A[i]);
      double Babs = std::abs(B[i]);
      double max = std::max(Aabs, Babs);
      double diff = Aabs - Babs;
      diff = (diff < 0) ? -diff : diff;
      if(diff > 0){
         double relError = diff / max;
         if(relError > 4e-5 && std::min(Aabs,Babs) > getZeroThreshold<floatType>()*5 ){
//            fprintf(stderr,"%.3e  %.3e %.3e\n",relError, A[i], B[i]);
            error += 1;
         }
      }
   }
   return (error == 0) ? 1 : 0;
}

template<typename floatType>
void restore(const floatType* A, floatType* B, size_t n)
{
   for(size_t i=0;i < n ; ++i)
      B[i] = A[i];
}

template<typename floatType>
static void getRandomTest(int &dim, uint32_t *perm, uint32_t *size, floatType &beta, 
      int &numThreads, 
      std::string &perm_str, 
      std::string &size_str, 
      const int total_size)
{
   dim = (rand() % MAX_DIM) + 1;
   uint32_t maxSizeDim = std::max(1.0, std::pow(total_size, 1.0/dim));
   std::vector<int> perm_(dim);
   for(int i=0;i < dim ; ++i){
      size[i] = std::max((((double)rand())/RAND_MAX) * maxSizeDim, 1.);
      perm_[i] = i;
   }
   std::random_shuffle(perm_.begin(), perm_.end());
   for(int i=0;i < dim ; ++i)
      perm[i] = perm_[i];

   numThreads = std::max(std::round((((double)rand())/RAND_MAX) * 24), 1.);
   if( rand() > RAND_MAX/2 )
      beta = 0.0;
   else
      beta = 4.0;

   for(int i=0;i < dim ; ++i){
      perm_str += std::to_string(perm[i]) + " ";
      size_str += std::to_string(size[i]) + " ";
   }
   printf("dim: %d\n", dim);
   printf("beta: %f\n", std::real(beta));
   printf("perm: %s\n", perm_str.c_str());
   printf("size: %s\n", size_str.c_str());
   printf("numThreads: %d\n",numThreads);
}

template<typename floatType>
void runTests()
{
   int numThreads = 1;
   floatType alpha = 2.;
   floatType beta = 4.;
   //beta = 0; 

   srand(time(NULL));
   int dim;
   uint32_t perm[MAX_DIM];
   uint32_t size[MAX_DIM];
   size_t total_size = 128*1024*1024;

   // Allocating memory for tensors
   floatType *A, *B, *B_ref, *B_hptt;
   int ret = posix_memalign((void**) &B, 64, sizeof(floatType) * total_size);
   ret += posix_memalign((void**) &A, 64, sizeof(floatType) * total_size);
   ret += posix_memalign((void**) &B_ref, 64, sizeof(floatType) * total_size);
   ret += posix_memalign((void**) &B_hptt, 64, sizeof(floatType) * total_size);
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
      B_ref[i]  = B[i];
      B_hptt[i] = B[i];
   }

   for(int j=0; j < NUM_TESTS; ++j)
   {  
      std::string perm_str = "";
      std::string size_str = "";
      getRandomTest(dim, perm, size, beta, numThreads, perm_str, size_str, total_size);
      int perm_[dim];
      int size_[dim];
      for(int i=0;i < dim ; ++i){
         perm_[i] = (int)perm[i];
         size_[i] = (int)size[i];
      }

      auto plan = hptt::create_plan( perm_, dim, 
            alpha, A, size_, NULL, 
            beta, B_hptt, NULL, 
            hptt::ESTIMATE, numThreads);

      restore(B, B_ref, total_size);
      transpose_ref<floatType>( size, perm, dim, A, alpha, B_ref, beta);

      restore(B, B_hptt, total_size);
      plan->execute();

      if( !equal_(B_ref, B_hptt, total_size) )
      {
         fprintf(stderr, "Error in HPTT.\n");
         fprintf(stderr,"%d OMP_NUM_THREADS=%d ./benchmark.exe %d  %s  %s\n",sizeof(floatType), numThreads, dim, perm_str.c_str(), size_str.c_str());
         exit(-1);
      }
   }
   free(A);
   free(B);
   free(B_ref);
   free(B_hptt);
}

int main(int argc, char *argv[]) 
{
  printf("float tests: \n");
  runTests<float>();

  printf("double tests: \n");
  runTests<double>();

  printf("float complex tests: \n");
  runTests<FloatComplex>();

  printf("double complex tests: \n");
  runTests<DoubleComplex>();

  printf("[SUCCESS] All tests have passed.\n");
  return 0;
}
