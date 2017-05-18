/**
 *   High-Performance Tensor Transposition Library
 *
 *   Copyright (C) 2017  Paul Springer (springer@aices.rwth-aachen.de)
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


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
