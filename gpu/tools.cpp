#include "tools.h"
#include <parallel/algorithm>

template<class T, class Z>
void psort(gpta_pair<T,Z> *curr_pcoord,uint64_t n){
	__gnu_parallel::sort(&curr_pcoord[0],(&curr_pcoord[0])+n,cmp_gpta_pair_asc<T,Z>);
}

template void psort(gpta_pair<float,uint64_t> *curr_pcoord,uint64_t n);
template void psort(gpta_pair<float,uint32_t> *curr_pcoord,uint64_t n);

template <class T>
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

template void normalize_transpose(float *&cdata, uint64_t n, uint64_t d);
