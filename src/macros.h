#pragma once

#define HPTT_DUPLICATE_2(condition, ...) \
if (condition) { _Pragma("omp parallel for num_threads(numThreads) collapse(2)") \
                 __VA_ARGS__ } \
else           { __VA_ARGS__ }

#define HPTT_DUPLICATE(condition, ...) \
if (condition) { _Pragma("omp parallel for num_threads(numThreads)") \
                 __VA_ARGS__ } \
else           { __VA_ARGS__ }
