#pragma once

#include <complex>

#define REGISTER_BITS 256 // AVX
#ifdef HPTT_ARCH_ARM
#undef REGISTER_BITS 
#define REGISTER_BITS 128 // ARM
#endif

namespace hptt {

enum SelectionMethod { ESTIMATE, MEASURE, PATIENT, CRAZY };

using FloatComplex = std::complex<float>;
using DoubleComplex = std::complex<double>;

}

