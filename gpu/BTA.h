#ifndef BTA_H
#define BTA_H
/*
 * Bitonic Top-k Aggregation
 */

#include "GAA.h"

template<class T, class Z>
__global__ void local_sort(T *gdata, uint64_t n, uint64_t d, T *gscores){
	uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

	gdata[tid] = 0;
}

template<class T, class Z>
class BTA : public GAA<T,Z>{
	public:
		BTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "BTA";
		};

	void findTopK(uint64_t k);
};

template<class T, class Z>
void BTA<T,Z>::findTopK(uint64_t k){

	local_sort<T,Z><<<1,1>>>(this->gdata,10,10,this->gscores);
}


#endif
