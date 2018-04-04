#ifndef COMMON_H
#define COMMON_H

#include<vector>
#include<algorithm>
#include<map>

#include <mmintrin.h>//MMX
#include <xmmintrin.h>//SSE
#include <emmintrin.h>//SSE2
#include <immintrin.h>//AVX and AVX2 // AVX-512

#ifdef GNU
#define _mm256_set_m128(va, vb) \
        _mm256_insertf128_ps(_mm256_castps128_ps256(vb), va, 1)
#endif

#define THREADS 16
#define ITHREADS 32 //INITIALIZATION THREADS
#define STATS_EFF true

#include <parallel/algorithm>
#include <omp.h>
#include <cstdlib>

#endif
