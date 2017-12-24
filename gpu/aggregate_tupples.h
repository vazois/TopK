#ifndef AGGREGATE_TUPPLES_H
#define AGGREGATE_TUPPLES_H

#include "CudaHelper.h"


template<class T, uint32_t block>
__global__ void init_tupples(T *scores, T* gdata, uint64_t n , uint64_t d){
	uint64_t offset = block * blockIdx.x + threadIdx.x;
	T agg = 0;
	if(offset < n){
		for(uint64_t i = 0;i <d;i++){
			agg += gdata[offset + i * n];
		}
		scores[offset] = agg;
	}
}


#endif
