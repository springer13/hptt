/**
 * @author: Paul Springer (springer@aices.rwth-aachen.de)
 */

#pragma once

#include <list>
#include <vector>
#include <iostream>

#include "hptt_types.h"

namespace hptt {
  
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

void trashCache(double *A, double *B, int n);

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

void accountForRowMajor(const int *sizeA, const int *outerSizeA, const int *outerSizeB, const int *perm, 
               int *tmpSizeA, int *tmpOuterSizeA, int *tmpouterSizeB, int *tmpPerm, const int dim, const bool useRowMajor);
}


