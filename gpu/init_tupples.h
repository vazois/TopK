#ifndef INIT_TUPPLES_H
#define INIT_TUPPLES_H

#include "CudaHelper.h"

template<class T>
struct Tupple{
	uint64_t *tupple_ids = NULL;
	T *scores = NULL;
	T *last = NULL;
};

template<class T>
__host__ void alloc_tupples(Tupple<T> &tupples, uint64_t n){
	cutil::safeMalloc<uint64_t,uint64_t>(&(tupples.tupple_ids),sizeof(uint64_t)*n,"tupple_ids alloc");
	cutil::safeMalloc<T,uint64_t>(&(tupples.scores),sizeof(T)*n,"scores alloc");
	//cutil::safeMalloc<T,uint64_t>(&(tupples.last),sizeof(T)*n,"nexts alloc");
}

template<class T>
__host__ void free_tupples(Tupple<T> &tupples){
	if(tupples.tupple_ids != NULL) cudaFree(tupples.tupple_ids);
	if(tupples.scores != NULL) cudaFree(tupples.scores);
	if(tupples.last != NULL) cudaFree(tupples.last);
}

template<class T, uint32_t block>
__global__ void init_tupples(uint64_t *tupple_ids, T *scores, T* gdata, uint64_t n){
	uint64_t offset = block * blockIdx.x + threadIdx.x;
	if(offset < n){
		tupple_ids[offset] = offset;
		scores[offset] = gdata[offset];
	}
}
#endif
