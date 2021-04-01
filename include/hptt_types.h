#pragma once

#include <complex>

#define REGISTER_BITS 256 // AVX
#ifdef HPTT_ARCH_ARM
#undef REGISTER_BITS 
#define REGISTER_BITS 128 // ARM
#endif

namespace hptt {

/**
 * \brief Determines the duration of the auto-tuning process.
 *
 * * ESTIMATE: 0 seconds (i.e., no auto-tuning)
 * * MEASURE: 10 seconds
 * * PATIENT: 60 seconds
 * * CRAZY : 3600 seconds
 */
enum SelectionMethod { ESTIMATE, MEASURE, PATIENT, CRAZY };

using FloatComplex = std::complex<float>;
using DoubleComplex = std::complex<double>;

}

