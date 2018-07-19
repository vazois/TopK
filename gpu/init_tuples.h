#ifndef INIT_TUPLES_H
#define INIT_TUPLES_H

#include "../tools/CudaHelper.h"

template<class T>
struct Tuple{
	uint32_t *tuple_ids = NULL;
	T *scores = NULL;
};

template<class T>
__host__ void alloc_tuples(Tuple<T> &tuples, uint64_t n){
	cutil::safeMalloc<uint32_t,uint64_t>(&(tuples.tuple_ids),sizeof(uint32_t)*n,"tupple_ids alloc");
	cutil::safeMalloc<T,uint64_t>(&(tuples.scores),sizeof(T)*n,"scores alloc");
}

template<class T>
__host__ void free_tuples(Tuple<T> &tuples){
	if(tuples.tuple_ids != NULL) cudaFree(tuples.tuple_ids);
	if(tuples.scores != NULL) cudaFree(tuples.scores);
}

template<class T, uint32_t block>
__global__ void init_tuples(uint32_t *tuple_ids, T *scores, T* gdata, uint64_t n){
	uint64_t offset = block * blockIdx.x + threadIdx.x;
	if(offset < n){
		tuple_ids[offset] = offset;
		scores[offset] = gdata[offset];
	}
}
#endif
