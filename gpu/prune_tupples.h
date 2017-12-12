#ifndef PRUNE_TUPPLES_H
#define PRUNE_TUPPLES_H

#include <cub/cub.cuh>
#include "CudaHelper.h"

#define BLOCK_SIZE 512

template<class T>
struct PruneComparator
{
    T compare;
    CUB_RUNTIME_FUNCTION __forceinline__
	PruneComparator(T compare) : compare(compare) {}
    CUB_RUNTIME_FUNCTION __forceinline__
    bool operator()(const T &a) const {
        return (a < compare);
    }
};

class PruneConfig{
	public:
		PruneConfig(){
			this->prune = NULL;
			this->d_out = NULL;

			this->d_temp_storage = NULL;
			this->temp_storage_bytes = 0;
		}
		~PruneConfig(){
			if(this->d_temp_storage != NULL) cudaFree(this->d_temp_storage);
			if(this->d_out != NULL) cudaFree(this->d_out);
			if(this->prune != NULL) cudaFree(this->prune);
			if(this->dnum != NULL) cudaFree(this->dnum);
		}

		//void alloc(){}
		void alloc_tmp_storage(size_t temp_storage_bytes){
			if(this->d_temp_storage == NULL ){
				this->temp_storage_bytes = temp_storage_bytes;
				cutil::safeMalloc<void,uint64_t>(&(this->d_temp_storage),temp_storage_bytes,"tmp_storage alloc");
			}else{
				if(this->temp_storage_bytes < temp_storage_bytes ) std::cout << "tmp_storage_bytes increased: " << this->temp_storage_bytes << " < " << temp_storage_bytes << std::endl;
			}
		}

		void alloc_output(uint64_t n){
			if(this->d_out == NULL) cutil::safeMalloc<uint64_t,uint64_t>(&(this->d_out),sizeof(uint64_t) * n,"d_out alloc");
			if(this->prune == NULL) cutil::safeMalloc<uint8_t,uint64_t>(&(this->prune),sizeof(uint8_t) * n,"prune alloc");
			if(this->dnum == NULL ) cutil::safeMalloc<uint64_t,uint64_t>(&(this->dnum),sizeof(uint64_t),"dnum alloc");
		}

		uint8_t* get_prune(){ return this->prune; }
		uint64_t* get_dout(){ return this->d_out; }
		uint64_t* get_dnum(){ return this->dnum; }

		void* tmp_storage(){ return this->d_temp_storage; }
		size_t tmp_storage_bytes(){ return this->temp_storage_bytes; }
	private:
		uint8_t *prune;
		uint64_t *d_out;
		uint64_t *dnum;

		void *d_temp_storage;
		size_t temp_storage_bytes;

};

static PruneConfig pconf;

template<class T, uint32_t block>
__global__ void evaluate(T *score, T *last, T threshold, uint8_t *prune, uint64_t suffix_len, uint64_t n){
	uint64_t offset = block * blockIdx.x + threadIdx.x;
	if(offset < n){
		prune[offset] = (score[offset] + last[offset] * suffix_len >= threshold);
//		if(prune[offset] == 0){
//			printf("%f,%f,%d\n",score[offset] + last[offset] * suffix_len, threshold, prune[offset]);
//		}
	}
}

template<class T, uint32_t block>

template<class T>
__host__ void prune_tupples(uint64_t *tupple_ids, T *score, T *last, uint64_t n, T threshold, uint64_t suffix_len){
	dim3 grid(n/BLOCK_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);
	PruneComparator<T> pcmp(threshold);
	size_t temp_storage_bytes;

	pconf.alloc_output(n);
	evaluate<T,BLOCK_SIZE><<<grid,block>>>(score,last,threshold,pconf.get_prune(),suffix_len,n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing evaluate");

	if(pconf.tmp_storage()==NULL){
		cub::DevicePartition::Flagged(pconf.tmp_storage(),temp_storage_bytes,tupple_ids,pconf.get_prune(),pconf.get_dout(),pconf.get_dnum(), n);
		pconf.alloc_tmp_storage(temp_storage_bytes);
		cub::DevicePartition::Flagged(pconf.tmp_storage(),temp_storage_bytes,tupple_ids,pconf.get_prune(),pconf.get_dout(),pconf.get_dnum(), n);

	}else{

	}
}


#endif
