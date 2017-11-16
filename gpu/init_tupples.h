#ifndef INIT_TUPPLES_H
#define INIT_TUPPLES_H

template<uint32_t block>
__global__ void init_tupples_4(uint64_t *gtupple, uint64_t n){
	uint64_t offset = block * blockIdx.x + threadIdx.x;
	if( offset < n ) gtupple[offset] = offset;
}

#endif
