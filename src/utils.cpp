/**
 * @author: Paul Springer (springer@aices.rwth-aachen.de)
 */

#include <vector>
#include <list>
#include <stdio.h>
#include <stdlib.h>

namespace hptt {

void getPrimeFactors( int n, std::list<int> &primeFactors )
{
   primeFactors.clear();
   for(int i=2;i <= n ; ++i){
      while( n % i == 0 ){
         primeFactors.push_back(i);
         n /= i;
      }
   }
   if( primeFactors.size() <= 0 ){
      fprintf(stderr,"[HPTT] Internal error: primefactorization for %d did not work.\n", n);
      exit(-1);
   }
}

int findPos(int value, const int *array, int n)
{
   for(int i=0;i < n ; ++i)
      if( array[i] == value )
         return i;
   return -1;
}

void trashCache(double *A, double *B, int n)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
   for(int i = 0; i < n; i++)
      A[i] += 0.999 * B[i];
}

int factorial(int n){
   if( n == 1 ) return 1;
   return n * factorial(n-1);
}

void accountForRowMajor(const int *sizeA, const int *outerSizeA, const int *outerSizeB, const int *perm, 
      int *tmpSizeA, int *tmpOuterSizeA, int *tmpOuterSizeB, int *tmpPerm, const int dim, const bool useRowMajor)
{
   for(int i=0; i < dim; ++i){
      int idx = i;
      if( useRowMajor ){
         idx = dim - 1 - i; // reverse order
         tmpPerm[i] = dim - perm[idx] - 1;
      }else
         tmpPerm[i] = perm[i];
      tmpSizeA[i] = sizeA[idx]; 
      tmpOuterSizeA[i] = outerSizeA[idx];
      tmpOuterSizeB[i] = outerSizeB[idx];
   }
}

}

extern "C"
void randomNumaAwareInit(float *data, const long *size, int dim)
{
   long totalSize = 1;
   for(int i = 0; i < dim; i++)
      totalSize *= size[i];
#pragma omp parallel for
   for(int i=0; i < totalSize; ++i)
      data[i] = (i+1)%1000 - 500;


}
