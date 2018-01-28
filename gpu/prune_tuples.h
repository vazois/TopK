#ifndef PRUNE_TUPLES_H
#define PRUNE_TUPLES_H

#include <cub/cub.cuh>
#include "CudaHelper.h"

#define BLOCK_SIZE 512

template<class T, uint32_t block>
__global__ void evaluate(uint32_t *tupple_ids,T *score, T *gdataC, T threshold, uint8_t *prune, uint64_t suffix_len, uint64_t n){
	uint64_t offset = block * blockIdx.x + threadIdx.x;
	if(offset < n){
		uint64_t index = tupple_ids[offset];
		prune[offset] = (score[offset] + gdataC[index] * suffix_len >= threshold);
	}
}

template<class T, uint32_t block>
//__global__ void compact_tupples(uint32_t *tupple_ids,T *scores, T *gdataC, T* gdataN, uint32_t *tupple_ids_out, T *scores_out, uint64_t dnum){
__global__ void compact_tuples(uint32_t *tupple_ids,T *scores, T* gdataN, uint32_t *tupple_ids_out, T *scores_out, uint64_t dnum){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	if(offset < dnum){
		uint64_t index = tupple_ids_out[offset];

		tupple_ids[offset] = index;
		scores[offset] = scores_out[offset] + gdataN[index];
	}
}

template<class T>
class CompactConfig{
	public:
		CompactConfig(){
			this->tids_out = NULL;
			this->sout = NULL;
			this->prune = NULL;

			this->d_temp_storage = NULL;
			this->temp_storage_bytes = 0;

			this->init = false;
		}
		~CompactConfig(){
			if(this->d_temp_storage != NULL) cudaFree(this->d_temp_storage);
			if(this->tids_out != NULL) cudaFree(this->tids_out);
			if(this->sout != NULL) cudaFree(this->sout);
			if(this->prune != NULL) cudaFree(this->prune);
			if(this->dnum != NULL) cudaFree(this->dnum);
			if(this->dnum_cpu != NULL) cudaFreeHost(this->dnum_cpu);
		}


		void prune_tuples(uint32_t *tupple_ids, T *score, T *gdataC, T *gdataN, uint64_t &n, T threshold, uint64_t suffix_len);

		void alloc_tmp_storage(){
			if(this->d_temp_storage == NULL ){
				std::cout << "tmp_storage(MB): " << this->temp_storage_bytes/((float)1024*1024) << std::endl;
				cutil::safeMalloc<void,uint64_t>(&(this->d_temp_storage),this->temp_storage_bytes,"tmp_storage alloc");
			}
		}

		void alloc(uint64_t n){
			cutil::safeMalloc<uint32_t,uint64_t>(&(this->tids_out),sizeof(uint64_t) * n,"tupple_ids_out alloc");
			cutil::safeMalloc<T,uint64_t>(&(this->sout),sizeof(T) * n,"scores_out alloc");
			cutil::safeMalloc<uint8_t,uint64_t>(&(this->prune),sizeof(uint8_t) * n,"prune alloc");
			cutil::safeMalloc<uint64_t,uint64_t>(&(this->dnum),sizeof(uint64_t),"dnum alloc");
			cutil::safeMallocHost<uint64_t,uint64_t>(&(this->dnum_cpu),sizeof(uint64_t),"dnum_cpu alloc");
		}

		uint64_t get_dnum_cpu(){
			cutil::safeCopyToHost<uint64_t,uint64_t>(this->dnum_cpu,this->dnum,sizeof(uint64_t), " copy from dnum to dnum_cpu ");
			return this->dnum_cpu[0];
		}

	private:
		bool init;
		uint32_t *tids_out;
		T *sout;
		uint8_t *prune;

		uint64_t *dnum;
		uint64_t *dnum_cpu;

		void *d_temp_storage;
		size_t temp_storage_bytes;
};

template<class T>
__host__ void CompactConfig<T>::prune_tuples(uint32_t *tuple_ids, T *score, T *gdataC, T *gdataN, uint64_t &n, T threshold, uint64_t suffix_len){
	dim3 grid((n-1)/BLOCK_SIZE + 1,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	if(!init) this->alloc(n);
	evaluate<T,BLOCK_SIZE><<<grid,block>>>(tuple_ids,score,gdataC,threshold,prune,suffix_len,n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing evaluate");

	if(!init){
		cub::DeviceSelect::Flagged(this->d_temp_storage,this->temp_storage_bytes,tuple_ids,this->prune,this->tids_out,this->dnum, n);
		this->alloc_tmp_storage();
		this->init = true;
	}

	cub::DeviceSelect::Flagged(this->d_temp_storage,this->temp_storage_bytes,tuple_ids,this->prune,this->tids_out,this->dnum, n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing ids partition");
	cub::DeviceSelect::Flagged(this->d_temp_storage,this->temp_storage_bytes,score,this->prune,this->sout,this->dnum, n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing scores partition");

	uint64_t dnum = this->get_dnum_cpu();
	//std::cout << "rows qualified: " << dnum << std::endl;//Debug
	dim3 ggrid((dnum-1)/BLOCK_SIZE + 1,1,1);
	dim3 gblock(BLOCK_SIZE,1,1);
	//compact_tupples<T,BLOCK_SIZE><<<ggrid,gblock>>>(tupple_ids,score,gdataC,gdataN,this->tids_out,this->sout,dnum);
	compact_tuples<T,BLOCK_SIZE><<<ggrid,gblock>>>(tuple_ids,score,gdataN,this->tids_out,this->sout,dnum);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing gather");

	n=dnum;
}

#endif
