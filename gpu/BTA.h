#ifndef BTA_H
#define BTA_H
/*
 * Bitonic Top-k Aggregation
 */

#include "GAA.h"
#define BTA_BLOCK_SIZE 256
#define BTA_TUPLES_PER_BLOCK 4096

template<class T, class Z>
__global__ void local_sort(T *gdata, uint64_t n, uint64_t d, T *gscores){
	//__shared__ Z tuple_ids[BTA_TUPLES_PER_BLOCK];
	__shared__ T tuple_scores[BTA_TUPLES_PER_BLOCK];

	uint32_t tid = threadIdx.x;
	uint64_t offset = blockIdx.x * BTA_TUPLES_PER_BLOCK + tid;
	for(uint64_t i = tid; i < BTA_TUPLES_PER_BLOCK; i+=BTA_BLOCK_SIZE){
		T score = 0;
		for(uint64_t m = 0; m < d; m++){
			score+=gdata[m*n + offset];
		}
		tuple_scores[i] = score;
		gscores[offset] = tuple_scores[i];//Debug
		offset+= BTA_BLOCK_SIZE;
	}
	__syncthreads();
}

template<class T, class Z>
class BTA : public GAA<T,Z>{
	public:
		BTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "BTA";
		};

		void alloc();
		void init(T *weights, uint32_t *query);
		void findTopK(uint64_t k);
};

template<class T, class Z>
void BTA<T,Z>::alloc(){
	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");
	cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*this->d,"gdata alloc");

	cutil::safeMalloc<T,uint64_t>(&(this->cscores),sizeof(T)*this->n,"cscores alloc");
	cutil::safeMalloc<T,uint64_t>(&(this->gscores),sizeof(T)*this->n,"gscores alloc");
}

template<class T, class Z>
void BTA<T,Z>::init(T *weights, uint32_t *query){
	cutil::safeCopyToDevice<T,uint64_t>(this->gdata,this->cdata,sizeof(T)*this->n*this->d, " copy from cdata to gdata ");

	cutil::cudaCheckErr(cudaMemcpyToSymbol(gpu_weights, weights, sizeof(T)*MAX_ATTRIBUTES),"copy weights");
	cutil::cudaCheckErr(cudaMemcpyToSymbol(gpu_query, query, sizeof(uint32_t)*MAX_ATTRIBUTES),"copy query");
}

template<class T, class Z>
void BTA<T,Z>::findTopK(uint64_t k){
	dim3 _block(BTA_BLOCK_SIZE,1,1);
	dim3 _grid((this->n-1)/BTA_TUPLES_PER_BLOCK + 1, 1, 1);
	local_sort<T,Z><<<_grid,_block>>>(this->gdata,this->n,this->d,this->gscores);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing local_sort");
}


#endif
