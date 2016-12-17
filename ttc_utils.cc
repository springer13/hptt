#include <vector>

namespace ttc {

void getPrimeFactors( int n, std::vector<int> &primeFactors )
{
   for(int i=2;i <= n ; ++i){
      while( n % i == 0 ){
         primeFactors.push_back(i);
         n /= i;
      }
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

}
