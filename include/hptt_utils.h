/**
 * @author: Paul Springer (springer@aices.rwth-aachen.de)
 */

#ifndef HPTT_UTILS_H
#define HPTT_UTILS_H

#include <list>
#include <vector>
#include <iostream>

namespace hptt {

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

template<typename t>
void printVector(const std::list<t> &vec, const char* label){
   std::cout << label <<": ";
   for( auto a : vec )
      std::cout << a << ", ";
   std::cout << "\n";
}


void getPrimeFactors( int n, std::list<int> &primeFactors );

template<typename t>
int findPos(t value, const std::vector<t> &array)
{
   for(int i=0;i < array.size() ; ++i)
      if( array[i] == value )
         return i;
   return -1;
}

int findPos(int value, const int *array, int n);

int factorial(int n);
}

#endif


