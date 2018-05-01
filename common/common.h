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

#define THREADS 32
#define ITHREADS 32 //INITIALIZATION THREADS
#define STATS_EFF true

#include <parallel/algorithm>
#include <omp.h>
#include <cstdlib>
#include <limits>


template <class T, class Z>
void normalize_transpose(T *&cdata, uint64_t n, uint64_t d){
	T *mmax = static_cast<T*>(aligned_alloc(32,sizeof(T)*d));
	T *mmin = static_cast<T*>(aligned_alloc(32,sizeof(T)*d));

	//Find min and max for each attribute list
	for(uint64_t m = 0; m < d; m++){
		mmax[m] = 0;
		mmin[m] = std::numeric_limits<T>::max();
		for(uint64_t i = 0; i < n; i++){
			T value = cdata[m*n + i];
			//if (m == 0) std::cout << m << " < " << value << "," << value <<"," << mmax[m] << std::endl;
			mmax[m] = std::max(mmax[m],value);
			mmin[m] = std::min(mmin[m],value);
		}
	}

	//Normalize values
	for(uint64_t m = 0; m < d; m++){
		T _max = mmax[m];
		T _min = mmin[m];
		T _mm = _max - _min;
		//if ( _mm == 0 ){ std::cout << m << " <"<< _max << " - " << _min << std::endl; }
		for(uint64_t i = 0; i < n; i++){
			T value = cdata[m*n+i];
			value = (value - _min)/_mm;
			cdata[m*n + i] = value;
		}
	}
	free(mmax);
	free(mmin);
}

template<class T, class Z>
void normalize(T *& cdata, uint64_t n, uint64_t d){
	T *mmax = static_cast<T*>(aligned_alloc(32,sizeof(T)*d));
	T *mmin = static_cast<T*>(aligned_alloc(32,sizeof(T)*d));

	for(uint64_t m = 0; m < d; m++){ mmax[m] = 0; mmin[m] = std::numeric_limits<T>::max(); }
	//Find min and max for each attribute list
	for(uint64_t i = 0; i < n; i++){
		for(uint64_t m = 0; m < d; m++){
			T value = cdata[i*d + m];
			mmax[m] = std::max(mmax[m],value);
			mmin[m] = std::min(mmin[m],value);
		}
	}

	//Normalize values
	__builtin_prefetch(mmax,0,3);
	__builtin_prefetch(mmin,0,3);
	for(uint64_t i = 0; i < n; i++){
		for(uint64_t m = 0; m < d; m++){
			T _max = mmax[m];
			T _min = mmin[m];
			T _mm = _max - _min;
			T value = cdata[i*d+m];
			value = (value - _min)/_mm;
			cdata[i*d + m] = value;
		}
	}
	free(mmax);
	free(mmin);
}


#endif
