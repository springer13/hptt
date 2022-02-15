/**
  Copyright 2018 Paul Springer
  
  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
  
  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  
  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
  
  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
  
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#ifdef DEBUG
#define HPTT_ERROR_INFO(str) fprintf(stdout, "[INFO] %s:%d : %s\n", __FILE__, __LINE__, str); exit(-1);
#else
#define HPTT_ERROR_INFO(str)
#endif

#if defined(__ICC) || defined(__INTEL_COMPILER)
#define INLINE __forceinline
#elif defined(__GNUC__) || defined(__GNUG__)
#define INLINE __attribute__((always_inline)) inline
#endif

#ifdef _OPENMP

#define HPTT_DUPLICATE_2(condition, ...) \
if (condition) { _Pragma("omp parallel for num_threads(numThreads) collapse(2)") \
                 __VA_ARGS__ } \
else           { __VA_ARGS__ }

#define HPTT_DUPLICATE(condition, ...) \
if (condition) { _Pragma("omp parallel for num_threads(numThreads)") \
                 __VA_ARGS__ } \
else           { __VA_ARGS__ }

#else

#define HPTT_DUPLICATE(condition, ...) { __VA_ARGS__ }
#define HPTT_DUPLICATE_2(condition, ...) { __VA_ARGS__ } 

#endif
