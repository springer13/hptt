#include <vector>
#include <stdio.h>
#include <stdlib.h>

namespace ttc {

void getPrimeFactors( int n, std::vector<int> &primeFactors )
{
   primeFactors.clear();
   for(int i=2;i <= n ; ++i){
      while( n % i == 0 ){
         primeFactors.push_back(i);
         n /= i;
      }
   }
   if( primeFactors.size() <= 0 ){
      printf("Internal error: primfactorization for %d did not work.\n", n);
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
   for(int i = 0; i < n; i++)
      A[i] += 0.999 * B[i];
}

int factorial(int n){
   if( n == 1 ) return 1;
   return n * factorial(n-1);
}

}
