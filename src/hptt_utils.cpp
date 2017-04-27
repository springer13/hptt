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
#pragma omp parallel
   for(int i = 0; i < n; i++)
      A[i] += 0.999 * B[i];
}

int factorial(int n){
   if( n == 1 ) return 1;
   return n * factorial(n-1);
}

}
