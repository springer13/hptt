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
#ifdef DEBUG
   printf("\nNumber of Errors: %d of %d\n", error, total_size);
#endif
   return (error == 0) ? 1 : 0;
}

template<typename floatType>
void restore(const floatType* A, floatType* B, size_t n)
{
   for(size_t i=0;i < n ; ++i)
      B[i] = A[i];
}

template<typename floatType>
static void getRandomTest(int &dim, uint32_t *perm, uint32_t *size, 
      uint32_t *outerSizeA, uint32_t *outerSizeB,
      uint32_t *offsetA, uint32_t *offsetB,
      int &innerStrideA, int &innerStrideB,
      floatType &beta, 
      int &numThreads, 
      std::string &perm_str, std::string &size_str, 
      std::string &outerSizeA_str, std::string &offsetA_str, 
      std::string &outerSizeB_str, std::string &offsetB_str, 
      const int total_size, bool subTensors)
{
   dim = (rand() % MAX_DIM) + 1;
   uint32_t maxSizeDim = std::max(1.0, std::pow(total_size, 1.0/dim));
   std::vector<int> perm_(dim);
   for(int i=0;i < dim ; ++i){
      outerSizeA[i] = std::max((((double)rand())/RAND_MAX) * maxSizeDim, 1.);
      size[i] = outerSizeA[i];
      if (subTensors) 
         size[i] = std::max((((double)rand())/RAND_MAX) * outerSizeA[i], 1.);
      perm_[i] = i;
   }
   std::random_shuffle(perm_.begin(), perm_.end());
   for(int i=0;i < dim ; ++i) 
   {
      perm[i] = perm_[i];
      outerSizeB[i] = outerSizeA[perm[i]];
      offsetA[i] = 0;
      offsetB[i] = 0;
      if (subTensors)
      {
         outerSizeB[i] = std::max((((double)rand())/RAND_MAX) * maxSizeDim, (double)size[perm[i]]);
         offsetA[i] = std::max((((double)rand())/RAND_MAX) * (outerSizeA[i] - size[i]), 0.);
         offsetB[i] = std::max((((double)rand())/RAND_MAX) * (outerSizeB[i] - size[perm[i]]), 0.);
      }
   }

   numThreads = std::max(std::round((((double)rand())/RAND_MAX) * 24), 1.);
   if( rand() > RAND_MAX/2 )
      beta = 0.0;
   else
      beta = 4.0;

   // Provide a larger inner stride if the tensor is less than a integer factor of the total size
   int ordinalSizeA = std::accumulate(outerSizeA, outerSizeA+dim, 1, std::multiplies<uint32_t>());
   int ordinalSizeB = std::accumulate(outerSizeB, outerSizeB+dim, 1, std::multiplies<uint32_t>());
   innerStrideA = (ordinalSizeA < (total_size / (dim*4))) ? std::max((((double)rand())/RAND_MAX) * dim, 1.) : 1;
   innerStrideB = (ordinalSizeB < (total_size / (dim*4))) ? std::max((((double)rand())/RAND_MAX) * dim, 1.) : 1;

   for(int i=0;i < dim ; ++i){
      perm_str += std::to_string(perm[i]) + " ";
      size_str += std::to_string(size[i]) + " ";
      outerSizeA_str += std::to_string(outerSizeA[i]) + " ";
      offsetA_str += std::to_string(offsetA[i]) + " ";
      outerSizeB_str += std::to_string(outerSizeB[i]) + " ";
      offsetB_str += std::to_string(offsetB[i]) + " ";
   }
   printf("dim: %d\n", dim);
   printf("beta: %f\n", std::real(beta));
   printf("perm: %s\n", perm_str.c_str());
   printf("size: %s\n", size_str.c_str());
   printf("outerSizeA: %s\n", outerSizeA_str.c_str());
   printf("outerSizeB: %s\n", outerSizeB_str.c_str());
   printf("offsetA: %s\n", offsetA_str.c_str());
   printf("offsetB: %s\n", offsetB_str.c_str());
   printf("ordinalSizeA: %d\n", ordinalSizeA);
   printf("ordinalSizeB: %d\n", ordinalSizeB);
   printf("innerStrideA: %d\n", innerStrideA);
   printf("innerStrideB: %d\n", innerStrideB);
   printf("numThreads: %d\n",numThreads);
}

template<typename floatType>
void runTests(bool subTensors = false)
{
   int numThreads = 1;
   floatType alpha = 2.;
   floatType beta = 4.;

   srand(time(NULL));
   int dim;
   uint32_t perm[MAX_DIM];
   uint32_t size[MAX_DIM];
   uint32_t outerSizeA[MAX_DIM];
   uint32_t outerSizeB[MAX_DIM];
   uint32_t offsetA[MAX_DIM];
   uint32_t offsetB[MAX_DIM];
   int innerStrideA = 2;
   int innerStrideB = 2;
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
   printf("Total size: %lu\n", total_size);
   printf("Last element of A: %f\n", A[total_size-1]);
   printf("Last element of B: %f\n", B[total_size-1]);

   for(int j=0; j < NUM_TESTS; ++j)
   {  
      std::string perm_str = "";
      std::string size_str = "";
      std::string outerSizeA_str = "";
      std::string outerSizeB_str = "";
      std::string offsetA_str = "";
      std::string offsetB_str = "";
      std::cout<<"Test "<<j<<std::endl;
      getRandomTest(dim, perm, size, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB, beta, numThreads, perm_str, size_str, outerSizeA_str, offsetA_str, outerSizeB_str, offsetB_str, total_size, subTensors);
      //dim = 8;
      //beta = 0.0;
      //int presetPerm[8] = {6, 3, 7, 2, 0, 1, 4, 5};
      //int presetSize[8] = {4, 5, 8, 1, 7, 4, 9, 9};
      //int presetOuterSizeA[8] = {4, 5, 8, 1, 7, 4, 9, 9};
      //int presetOuterSizeB[8] = {9, 1, 9, 8, 4, 5, 7, 4};
      //int presetOffsetA[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      //int presetOffsetB[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      int perm_[dim];
      int size_[dim];
      int outerSizeA_[dim];
      int outerSizeB_[dim];
      int offsetA_[dim];
      int offsetB_[dim];
      for(int i=0;i < dim ; ++i){
         perm_[i] = (int)perm[i];
         size_[i] = (int)size[i];
         outerSizeA_[i] = (int)outerSizeA[i];
         outerSizeB_[i] = (int)outerSizeB[i];
         offsetA_[i] = (int)offsetA[i];
         offsetB_[i] = (int)offsetB[i];
         //perm[i] = (int)presetPerm[i];
         //size[i] = (int)presetSize[i];
         //outerSizeA[i] = (int)presetOuterSizeA[i];
         //outerSizeB[i] = (int)presetOuterSizeB[i];
         //offsetA[i] = (int)presetOffsetA[i];
         //offsetB[i] = (int)presetOffsetB[i];
      }

      auto plan = hptt::create_plan( perm_, dim, 
            alpha, A, size_, outerSizeA_, offsetA_, innerStrideA,
            beta, B_hptt, outerSizeB_, offsetB_, innerStrideB,
            hptt::ESTIMATE, numThreads);

      restore(B, B_ref, total_size);
      transpose_ref<floatType>(size, perm, dim, A, alpha, outerSizeA_, offsetA_, innerStrideA, B_ref, beta, outerSizeB_, offsetB_, innerStrideB, false);
      restore(B, B_hptt, total_size);
      plan->execute();

      if( !equal_(B_ref, B_hptt, total_size) )
      {
         fprintf(stderr, "Error in HPTT.\n");
         fprintf(stderr,"%lu OMP_NUM_THREADS=%d ./benchmark.exe %d  %s  %s\n",sizeof(floatType), numThreads, dim, perm_str.c_str(), size_str.c_str());
         exit(-1);
      }
      std::cout << "Test " << j << " passed." << std::endl;
   }
   std::cout << "All tests passed." << std::endl;
   free(A);
   free(B);
   free(B_ref);
   free(B_hptt);
}

int main(int argc, char *argv[]) 
{
  printf("float tests: \n");
  runTests<float>();
  runTests<float>(true);

  printf("double tests: \n");
  runTests<double>();
  runTests<double>(true);

  printf("float complex tests: \n");
  runTests<FloatComplex>();
  runTests<FloatComplex>(true);

  printf("double complex tests: \n");
  runTests<DoubleComplex>();
  runTests<DoubleComplex>(true);

  printf("[SUCCESS] All tests have passed.\n");
  return 0;
}
