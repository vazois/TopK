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
			if(this->dnum_cpu != NULL) cudaFreeHost(this->dnum_cpu);
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

		void free_tmp_storage(){
			cudaFree(this->d_temp_storage);
			this->d_temp_storage=NULL;
		}

		void alloc_output(uint64_t n){
			//if(this->d_out == NULL) cutil::safeMalloc<uint64_t,uint64_t>(&(this->d_out),sizeof(uint64_t) * n,"d_out alloc");
			//if(this->prune == NULL) cutil::safeMalloc<uint8_t,uint64_t>(&(this->prune),sizeof(uint8_t) * n,"prune alloc");
			if(this->dnum == NULL ) cutil::safeMalloc<uint64_t,uint64_t>(&(this->dnum),sizeof(uint64_t),"dnum alloc");
			if(this->dnum_cpu == NULL ) cutil::safeMallocHost<uint64_t,uint64_t>(&(this->dnum_cpu),sizeof(uint64_t),"dnum_cpu alloc");
		}

		uint8_t* get_prune(){ return this->prune; }
		uint64_t* get_dout(){ return this->d_out; }
		uint64_t* get_dnum(){ return this->dnum; }
		uint64_t get_dnum_cpu(){
			cutil::safeCopyToHost<uint64_t,uint64_t>(this->dnum_cpu,this->dnum,sizeof(uint64_t), " copy from dnum to dnum_cpu ");
			return this->dnum_cpu[0];
		}

		void* tmp_storage(){ return this->d_temp_storage; }
		size_t tmp_storage_bytes(){ return this->temp_storage_bytes; }
	private:
		uint8_t *prune;
		uint64_t *d_out;
		uint64_t *dnum;
		uint64_t *dnum_cpu;

		void *d_temp_storage;
		size_t temp_storage_bytes;

};

static PruneConfig pconf;

template<class T, uint32_t block>
__global__ void evaluate(uint64_t *tupple_ids,T *score, T *gdataC, T threshold, uint8_t *prune, uint64_t suffix_len, uint64_t n){
	uint64_t offset = block * blockIdx.x + threadIdx.x;
	if(offset < n){
		uint64_t index = tupple_ids[offset];
		prune[offset] = (score[offset] + gdataC[index] * suffix_len >= threshold);
	}
}

//template<class T, uint32_t block>
//__global__ void gather_tupples(uint64_t *tupple_ids,T *score, T *gdataC, T* gdataN, uint64_t *d_out, uint64_t dnum, uint64_t suffix_len){
//	uint64_t offset = block * blockIdx.x + threadIdx.x;
//
//	if(offset < dnum){
//		uint64_t index = d_out[offset];
//		T attr = gdataN[index];
//
////		if(index == 2438){
////			printf(" %d : %f + %f = %f\n",index,score[index], attr, score[index] + attr);
////		}
//		tupple_ids[offset] = index;
//		score[index]= score[index] + attr;
//	}
//
//}

//template<class T,uint32_t block>
//__global__ void update(T *score, T *gdataC, uint8_t *prune,uint64_t dnum){
//	uint64_t offset = block * blockIdx.x + threadIdx.x;
//
//	if(offset < dnum){
//		score[offset]=gdataC[offset];
//		prune[offset]=0;
//	}
//}

template<class T, uint32_t block>
__global__ void compact_tupples(uint64_t *tupple_ids,T *scores, T *gdataC, T* gdataN, uint64_t *tupple_ids_out, T *scores_out, uint64_t dnum){
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
			this->prune = NULL;
			this->d_out = NULL;

			this->d_temp_storage = NULL;
			this->temp_storage_bytes = 0;
		}
		~CompactConfig(){
			if(this->d_temp_storage != NULL) cudaFree(this->d_temp_storage);
			if(this->d_out != NULL) cudaFree(this->d_out);
			if(this->prune != NULL) cudaFree(this->prune);
			if(this->dnum != NULL) cudaFree(this->dnum);
			if(this->dnum_cpu != NULL) cudaFreeHost(this->dnum_cpu);
		}

		uint8_t *prune;
		uint64_t *d_out;
		uint64_t *dnum;
		uint64_t *dnum_cpu;

		void *d_temp_storage;
		size_t temp_storage_bytes;
};



template<class T>
__host__ void prune_tupples(uint64_t *tupple_ids, T *score, T *gdataC, T *gdataN, uint64_t &n, T threshold, uint64_t suffix_len){
	dim3 grid((n-1)/BLOCK_SIZE + 1,1,1);
	dim3 block(BLOCK_SIZE,1,1);
	PruneComparator<T> pcmp(threshold);
	size_t temp_storage_bytes;

	pconf.alloc_output(n);
	uint64_t *tupple_ids_out;
	T *scores_out;
	uint8_t *prune;

	cutil::safeMalloc<uint64_t,uint64_t>(&(tupple_ids_out),sizeof(uint64_t) * n,"tupple_ids_out alloc");
	cutil::safeMalloc<T,uint64_t>(&(scores_out),sizeof(T) * n,"scores_out alloc");
	cutil::safeMalloc<uint8_t,uint64_t>(&(prune),sizeof(uint8_t) * n,"prune alloc");

	evaluate<T,BLOCK_SIZE><<<grid,block>>>(tupple_ids,score,gdataC,threshold,prune,suffix_len,n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing evaluate");

	if(pconf.tmp_storage()==NULL){
		cub::DevicePartition::Flagged(pconf.tmp_storage(),temp_storage_bytes,tupple_ids,prune,tupple_ids_out,pconf.get_dnum(), n);
		pconf.alloc_tmp_storage(temp_storage_bytes);
	}

	cub::DevicePartition::Flagged(pconf.tmp_storage(),temp_storage_bytes,tupple_ids,prune,tupple_ids_out,pconf.get_dnum(), n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing ids partition");
	cub::DevicePartition::Flagged(pconf.tmp_storage(),temp_storage_bytes,score,prune,scores_out,pconf.get_dnum(), n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing scores partition");

	uint64_t dnum = pconf.get_dnum_cpu();
	std::cout << "rows qualified: " << dnum << std::endl;
	dim3 ggrid((dnum-1)/BLOCK_SIZE + 1,1,1);
	dim3 gblock(BLOCK_SIZE,1,1);
	compact_tupples<T,BLOCK_SIZE><<<ggrid,gblock>>>(tupple_ids,score,gdataC,gdataN,tupple_ids_out,scores_out,dnum);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing gather");

	n=dnum;

	cudaFree(tupple_ids_out);
	cudaFree(scores_out);
	cudaFree(prune);
}

//template<class T>
//__host__ void prune_tupples2(uint64_t *tupple_ids, T *score, T *gdataC, T *gdataN, uint64_t &n, T threshold, uint64_t suffix_len){
//	dim3 grid((n-1)/BLOCK_SIZE + 1,1,1);
//	dim3 block(BLOCK_SIZE,1,1);
//	PruneComparator<T> pcmp(threshold);
//	size_t temp_storage_bytes;
//
////	std::cout << "rows: " << n << std::endl;
//	pconf.alloc_output(n);
//	evaluate<T,BLOCK_SIZE><<<grid,block>>>(tupple_ids,score,gdataC,threshold,pconf.get_prune(),suffix_len,n);
//	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing evaluate");
//
//	if(pconf.tmp_storage()==NULL){
//		cub::DevicePartition::Flagged(pconf.tmp_storage(),temp_storage_bytes,tupple_ids,pconf.get_prune(),pconf.get_dout(),pconf.get_dnum(), n);
//		pconf.alloc_tmp_storage(temp_storage_bytes);
//		cub::DevicePartition::Flagged(pconf.tmp_storage(),temp_storage_bytes,tupple_ids,pconf.get_prune(),pconf.get_dout(),pconf.get_dnum(), n);
//		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing gather");
//
//		uint64_t dnum = pconf.get_dnum_cpu();
//		std::cout << "rows qualified: " << dnum << std::endl;
//
//		dim3 ggrid((dnum-1)/BLOCK_SIZE + 1,1,1);
//		dim3 gblock(BLOCK_SIZE,1,1);
//
//		gather_tupples<T,BLOCK_SIZE><<<ggrid,gblock>>>(tupple_ids,score,gdataC,gdataN,pconf.get_dout(),dnum, suffix_len);
//		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing gather");
//
////		update<T,BLOCK_SIZE><<<ggrid,gblock>>>(score,gdataC,pconf.get_prune(),dnum);
////		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing update score");
//
//		n = dnum;
//		pconf.free_tmp_storage();
//	}else{
//		cub::DevicePartition::Flagged(pconf.tmp_storage(),temp_storage_bytes,tupple_ids,pconf.get_prune(),pconf.get_dout(),pconf.get_dnum(), n);
//		pconf.alloc_tmp_storage(temp_storage_bytes);
//		cub::DevicePartition::Flagged(pconf.tmp_storage(),temp_storage_bytes,tupple_ids,pconf.get_prune(),pconf.get_dout(),pconf.get_dnum(), n);
//		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing update score");
//		uint64_t dnum = pconf.get_dnum_cpu();
//		std::cout << "rows qualified: " << dnum << std::endl;
//
//		dim3 ggrid((dnum-1)/BLOCK_SIZE + 1,1,1);
//		dim3 gblock(BLOCK_SIZE,1,1);
//
//		gather_tupples<T,BLOCK_SIZE><<<ggrid,gblock>>>(tupple_ids,score,gdataC,gdataN,pconf.get_dout(),dnum, suffix_len);
//		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing gather");
//
////		update<T,BLOCK_SIZE><<<ggrid,gblock>>>(score,gdataC,pconf.get_prune(),dnum);
////		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing update score");
//
//		n = dnum;
//		pconf.free_tmp_storage();
//	}
//}


#endif
