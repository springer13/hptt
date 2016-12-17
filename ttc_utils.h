#ifndef TTC_UTILS_H
#define TTC_UTILS_H

template<typename t>
int hasItem(const std::vector<t> &vec, t value)
{
   return ( std::find(vec.begin(), vec.end(), value) != vec.end() );
}

template<typename t>
void printVector(const std::vector<t> &vec, const char* label){
   std::cout << label <<": ";
   for( auto a : vec )
      std::cout << a << ", ";
   std::cout << "\n";
}

void getPrimeFactors( int n, std::vector<int> &primeFactors )
{
   for(int i=2;i <= n ; ++i){
      while( n % i == 0 ){
         primeFactors.push_back(i);
         n /= i;
      }
   }
}

template<typename t>
int findPos(t value, const std::vector<t> &array)
{
   for(int i=0;i < array.size() ; ++i)
      if( array[i] == value )
         return i;
   return -1;
}

int findPos(int value, const int *array, int n);

void trashCache(double *A, double *B, int n);

#endif


