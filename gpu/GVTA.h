#ifndef GVTA_H
#define GVTA_H

#include "GAA.h"

/*
 * blocks: data blocks
 * n: tuple number
 * qq: number of query attributes
 * dbsize: data block size
 * pnum: partition number
 * nb: number of blocks (levels)
 */
template<class T, class Z>
__global__ void gvta_atm_16_4096(gvta_block<T,Z> *blocks, uint64_t nb, uint64_t qq, uint64_t k, T *out)
{
	uint64_t i = 0;//data block level
	uint64_t offset = blockIdx.x * nb * GVTA_BLOCK_SIZE;
	T v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0;
	T v8 = 0, v9 = 0, vA = 0, vB = 0, vC = 0, vD = 0, vE = 0, vF = 0;

	__shared__ T threshold;
	__shared__ T heap[32];
	__shared__ T buffer[256];
	T *data;
	T *tvector;
	//T tmp = 0;
	if(threadIdx.x < 32) heap[threadIdx.x] = 0;
	while(i < nb)
	{
		data = &blocks[i].data[blockIdx.x * GVTA_BLOCK_SIZE];
		if(threadIdx.x == 0){
			tvector = blocks[i].tvector;
			threshold = 0;
		}

		v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0;
		v8 = 0, v9 = 0, vA = 0, vB = 0, vC = 0, vD = 0, vE = 0, vF = 0;
		for(uint32_t m = 0; m < qq; m++)
		{
			uint64_t ai = gpu_query[m];
			uint64_t begin = GVTA_BLOCK_SIZE * GVTA_PARTITIONS * ai;
			if(threadIdx.x == 0) threshold += tvector[ai * GVTA_PARTITIONS + blockIdx.x];

			v0 += data[begin + threadIdx.x       ] * gpu_weights[ai];
			v1 += data[begin + threadIdx.x + 256 ] * gpu_weights[ai];
			v2 += data[begin + threadIdx.x + 512 ] * gpu_weights[ai];
			v3 += data[begin + threadIdx.x + 768 ] * gpu_weights[ai];
			v4 += data[begin + threadIdx.x + 1024] * gpu_weights[ai];
			v5 += data[begin + threadIdx.x + 1280] * gpu_weights[ai];
			v6 += data[begin + threadIdx.x + 1536] * gpu_weights[ai];
			v7 += data[begin + threadIdx.x + 1792] * gpu_weights[ai];
			v8 += data[begin + threadIdx.x + 2048] * gpu_weights[ai];
			v9 += data[begin + threadIdx.x + 2304] * gpu_weights[ai];
			vA += data[begin + threadIdx.x + 2560] * gpu_weights[ai];
			vB += data[begin + threadIdx.x + 2816] * gpu_weights[ai];
			vC += data[begin + threadIdx.x + 3072] * gpu_weights[ai];
			vD += data[begin + threadIdx.x + 3328] * gpu_weights[ai];
			vE += data[begin + threadIdx.x + 3584] * gpu_weights[ai];
			vF += data[begin + threadIdx.x + 3840] * gpu_weights[ai];
		}

		/*
		 * Rebuild - Reduce 4096 -> 2048
		 */
		uint32_t laneId = threadIdx.x;
		uint32_t level, step, dir;
		for(level = 1; level < k; level = level << 1){
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
				v1 = swap(v1,step,dir);
				v2 = swap(v2,step,dir);
				v3 = swap(v3,step,dir);
				v4 = swap(v4,step,dir);
				v5 = swap(v5,step,dir);
				v6 = swap(v6,step,dir);
				v7 = swap(v7,step,dir);
				v8 = swap(v8,step,dir);
				v9 = swap(v9,step,dir);
				vA = swap(vA,step,dir);
				vB = swap(vB,step,dir);
				vC = swap(vC,step,dir);
				vD = swap(vD,step,dir);
				vE = swap(vE,step,dir);
				vF = swap(vF,step,dir);
			}
		}
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
		v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
		v3 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v3, k),v3);
		v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
		v5 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v5, k),v5);
		v6 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v6, k),v6);
		v7 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v7, k),v7);
		v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
		v9 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v9, k),v9);
		vA = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vA, k),vA);
		vB = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vB, k),vB);
		vC = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vC, k),vC);
		vD = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vD, k),vD);
		vE = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vE, k),vE);
		vF = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vF, k),vF);
		v0 = (threadIdx.x & k) == 0 ? v0 : v1;
		v2 = (threadIdx.x & k) == 0 ? v2 : v3;
		v4 = (threadIdx.x & k) == 0 ? v4 : v5;
		v6 = (threadIdx.x & k) == 0 ? v6 : v7;
		v8 = (threadIdx.x & k) == 0 ? v8 : v9;
		vA = (threadIdx.x & k) == 0 ? vA : vB;
		vC = (threadIdx.x & k) == 0 ? vC : vD;
		vE = (threadIdx.x & k) == 0 ? vE : vF;

		/*
		 * Rebuild - Reduce 2048 -> 1024
		 */
		level = k >> 1;
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
			v0 = swap(v0,step,dir);
			v2 = swap(v2,step,dir);
			v4 = swap(v4,step,dir);
			v6 = swap(v6,step,dir);
			v8 = swap(v8,step,dir);
			vA = swap(vA,step,dir);
			vC = swap(vC,step,dir);
			vE = swap(vE,step,dir);
		}
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
		v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
		v6 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v6, k),v6);
		v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
		vA = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vA, k),vA);
		vC = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vC, k),vC);
		vE = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vE, k),vE);
		v0 = (threadIdx.x & k) == 0 ? v0 : v2;
		v4 = (threadIdx.x & k) == 0 ? v4 : v6;
		v8 = (threadIdx.x & k) == 0 ? v8 : vA;
		vC = (threadIdx.x & k) == 0 ? vC : vE;

		/*
		 * Rebuild - Reduce 1024 -> 512
		 */
		level = k >> 1;
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
			v0 = swap(v0,step,dir);
			v4 = swap(v4,step,dir);
			v8 = swap(v8,step,dir);
			vC = swap(vC,step,dir);
		}
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
		v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
		vC = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vC, k),vC);
		v0 = (threadIdx.x & k) == 0 ? v0 : v4;
		v8 = (threadIdx.x & k) == 0 ? v8 : vC;

		/*
		 * Rebuild - Reduce 512 -> 256
		 */
		level = k >> 1;
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
			v0 = swap(v0,step,dir);
			v8 = swap(v8,step,dir);
		}
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
		v0 = (threadIdx.x & k) == 0 ? v0 : v8;
		buffer[threadIdx.x] = v0;
		__syncthreads();

		if(threadIdx.x < 32)
		{
			v0 = buffer[threadIdx.x];
			v1 = buffer[threadIdx.x+32];
			v2 = buffer[threadIdx.x+64];
			v3 = buffer[threadIdx.x+96];
			v4 = buffer[threadIdx.x+128];
			v5 = buffer[threadIdx.x+160];
			v6 = buffer[threadIdx.x+192];
			v7 = buffer[threadIdx.x+224];

			/*
			 * Rebuild - Reduce 256 -> 128
			 */
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
				v1 = swap(v1,step,dir);
				v2 = swap(v2,step,dir);
				v3 = swap(v3,step,dir);
				v4 = swap(v4,step,dir);
				v5 = swap(v5,step,dir);
				v6 = swap(v6,step,dir);
				v7 = swap(v7,step,dir);
			}
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
			v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
			v3 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v3, k),v3);
			v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
			v5 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v5, k),v5);
			v6 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v6, k),v6);
			v7 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v7, k),v7);
			v0 = (threadIdx.x & k) == 0 ? v0 : v1;
			v2 = (threadIdx.x & k) == 0 ? v2 : v3;
			v4 = (threadIdx.x & k) == 0 ? v4 : v5;
			v6 = (threadIdx.x & k) == 0 ? v6 : v7;

			/*
			 * Rebuild - Reduce 128 -> 64
			 */
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
				v2 = swap(v2,step,dir);
				v4 = swap(v4,step,dir);
				v6 = swap(v6,step,dir);
			}
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
			v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
			v6 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v6, k),v6);
			v0 = (threadIdx.x & k) == 0 ? v0 : v2;
			v4 = (threadIdx.x & k) == 0 ? v4 : v6;

			/*
			 * Rebuild - Reduce 64 -> 32
			 */
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
				v4 = swap(v4,step,dir);
			}
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
			v0 = (threadIdx.x & k) == 0 ? v0 : v4;

			/*
			 * Rebuild - Reduce 32 -> 16
			 */
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
			}
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v0 = (threadIdx.x & k) == 0 ? v0 : 0;

			/*
			 * Sort 16
			 */
			for(level = k; level < 32; level = level << 1){
				for(step = level; step > 0; step = step >> 1){
					dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
					v0 = swap(v0,step,dir);
				}
			}

			/*
			 * Merge with Buffer
			 */
			if(i == 0)
			{
				heap[31 - threadIdx.x] = v0;
			}else{
				v1 = heap[threadIdx.x];
				v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
				v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
				v0 = (threadIdx.x & k) == 0 ? v0 : v1;

				for(step = level; step > 0; step = step >> 1){
					dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
					v0 = swap(v0,step,dir);
				}
				v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
				v0 = (threadIdx.x & k) == 0 ? v0 : 0;

				for(level = k; level < 32; level = level << 1){
					for(step = level; step > 0; step = step >> 1){
						dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
						v0 = swap(v0,step,dir);
					}
				}
				heap[31 - threadIdx.x] = v0;
			}
			//if(threadIdx.x == 0) threshold = tmp;
		}
		__syncthreads();
		/*
		 * Break if suitable threshold reached
		 */
		if(heap[k-1] >= threshold){
			//if(threadIdx.x == 0) printf("[%d]: %d < %d\n",threadIdx.x,(uint32_t)i,(uint32_t)nb);
			//if(threadIdx.x == 0) slevel[blockIdx.x] = i;
			//if(i == 1)
			break;
		}
		__syncthreads();
		i++;
	}

	/*
	 * Write-back heaps of each partition
	 */
	if(threadIdx.x < k){
		offset = blockIdx.x * k;
		if((blockIdx.x & 0x1) == 0) out[offset + (k-1) - threadIdx.x] = heap[threadIdx.x];
		else out[offset + threadIdx.x] = heap[threadIdx.x];
//		out[offset + threadIdx.x] = heap[threadIdx.x];
	}
}

template<class T, class Z>
class GVTA : public GAA<T,Z>{
	public:
		GVTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "GVTA";
			this->clear_mem_counters();
			cudaStreamCreate(&s0);
			cudaStreamCreate(&s1);
		};

		~GVTA()
		{
			cudaStreamDestroy(s0);
			cudaStreamDestroy(s1);
			if(blocks){ cudaFreeHost(this->blocks); }
			if(cout){ cudaFreeHost(cout); }
			#if USE_DEVICE_MEM
				if(this->gblocks) cudaFree(this->gblocks);
				if(gout) cudaFree(gout);
			#endif
			std::cout << "CLEAR: {" << this->algo << "}" << std::endl;
		};

		void alloc();
		void init();
		void findTopK(uint64_t k, uint64_t qq);

		void benchmark(){
			GAA<T,Z>::benchmark();
			std::cout << "gvta_mem(MB): " << this->gvta_mem/(1024*1024) << std::endl;
			std::cout << "base_mem(MB): " << this->base_mem/(1024*1024) << std::endl;
			std::cout << "reorder_mem(MB): " << this->reorder_mem/(1024*1024) << std::endl;
		}

	private:
		uint64_t tuples_per_part;
		uint64_t num_blocks;
		gvta_block<T,Z> *blocks = NULL;
		gvta_block<T,Z> *gblocks = NULL;
		T *cout = NULL;
		T *gout = NULL;

		void cclear();
		void reorder();
		void atm_16(uint64_t k, uint64_t qq);
		void atm_16_dm_driver(uint64_t k, uint64_t qq);
		void atm_16_hm_driver(uint64_t k, uint64_t qq);

		void clear_mem_counters(){
			this->gvta_mem = 0;
			this->base_mem = 0;
			this->reorder_mem = 0;
		}
		double gvta_mem;
		double reorder_mem;
		double base_mem;

		cudaStream_t s0;
		cudaStream_t s1;
};

template<class T, class Z>
void GVTA<T,Z>::alloc(){
	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");// Allocate cpu data memory
	this->base_mem += sizeof(T)*this->n*this->d;
}

template<class T, class Z>
void GVTA<T,Z>::reorder()
{
	dim3 block(256,1,1);
	dim3 grid(1,1,1);
	Z *grvector = NULL;//first seen position vector
	Z *grvector_out = NULL;
	Z *dkeys_in = NULL;//local sort vector
	Z *dkeys_out = NULL;
	T *dvalues_in = NULL;//local sort values
	T *dvalues_out = NULL;

	Z *hkeys = NULL;
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

	/////////////////////////////////////////
	//1: Allocate space for reordered blocks
	this->tuples_per_part = ((this->n - 1)/GVTA_PARTITIONS) + 1;
	this->num_blocks = ((this->tuples_per_part - 1) / GVTA_BLOCK_SIZE) + 1;
	cutil::safeMallocHost<gvta_block<T,Z>,uint64_t>(&(this->blocks),sizeof(gvta_block<T,Z>)*this->num_blocks,"alloc host gvta_blocks");
	std::cout << this->n << " = p,psz(" << GVTA_PARTITIONS <<"," << this->tuples_per_part << ") - " << "b,bsz(" << this->num_blocks << "," << GVTA_BLOCK_SIZE << ")" << std::endl;
	for(uint64_t i = 0; i< this->num_blocks; i++)//TODO:safemalloc
	{
		this->gvta_mem += sizeof(T)*GVTA_PARTITIONS*GVTA_BLOCK_SIZE*this->d;
		this->gvta_mem += sizeof(T)*GVTA_PARTITIONS*this->d;
	}

	/////////////////////////////////////////////
	//2: Allocate temporary space for reordering
	//hkeys = (Z*)malloc(sizeof(Z)*this->tuples_per_part);
	cutil::safeMallocHost<Z,uint64_t>(&hkeys,sizeof(Z)*this->tuples_per_part,"hkeys alloc");
	this->reorder_mem += sizeof(Z)*this->tuples_per_part;
	cutil::safeMallocHost<Z,uint64_t>(&(grvector),sizeof(Z)*this->tuples_per_part,"alloc rvector");
	this->reorder_mem += sizeof(Z)*this->tuples_per_part;
	cutil::safeMallocHost<Z,uint64_t>(&(grvector_out),sizeof(Z)*this->tuples_per_part,"alloc rvector");
	this->reorder_mem += sizeof(Z)*this->tuples_per_part;
	cutil::safeMallocHost<Z,uint64_t>(&(dkeys_in),sizeof(Z)*this->tuples_per_part,"alloc dkeys_in");
	this->reorder_mem += sizeof(Z)*this->tuples_per_part;
	cutil::safeMallocHost<Z,uint64_t>(&(dkeys_out),sizeof(Z)*this->tuples_per_part,"alloc dkeys_out");
	this->reorder_mem += sizeof(Z)*this->tuples_per_part;
	cutil::safeMallocHost<T,uint64_t>(&(dvalues_in),sizeof(T)*this->tuples_per_part,"alloc dvalues_in");
	this->reorder_mem += sizeof(T)*this->tuples_per_part;
	cutil::safeMallocHost<T,uint64_t>(&(dvalues_out),sizeof(T)*this->tuples_per_part,"alloc dvalues_out");
	this->reorder_mem += sizeof(T)*this->tuples_per_part;
	cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, dkeys_in, dkeys_out, dvalues_in, dvalues_out, this->tuples_per_part);
	cutil::cudaCheckErr(cudaMalloc(&d_temp_storage, temp_storage_bytes),"alloc d_temp_storage");
	///Allocation ends//

	uint64_t offset = 0;
	//a: for each partition
	for(uint64_t i = 0; i < GVTA_PARTITIONS; i++){
		//b:Find partition size
		uint64_t psize = offset + this->tuples_per_part < this->n ? this->tuples_per_part : (this->n - offset);
		grid.x = (this->tuples_per_part-1)/block.x + 1;
		//c: initialize local tuple ids for sorting attributes
		init_rvlocal<Z><<<grid,block>>>(dkeys_in,psize);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_rvlocal");

		//initialize first seen position//
		init_first_seen_position<Z><<<grid,block>>>(grvector,psize);//d: initialize local vector for first seen position
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_rvglobal");
		for(uint64_t m = 0; m < this->d; m++){
			//e: copy attributes and sort objects per attribute
			cutil::safeCopyToDevice<T,uint64_t>(dvalues_in,&this->cdata[m*this->n + offset],sizeof(T)*psize, " copy from cdata to dvalues_in");
			cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, dvalues_in, dvalues_out, dkeys_in, dkeys_out, psize);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing SortPairsDescending");

			//f: update first seen position
			max_rvglobal<Z><<<grid,block>>>(grvector, dkeys_out,this->tuples_per_part);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing max_rvglobal");
		}

		////////////////////////////////////////////////////
		//Find reordered position from first seen position//
		//initialize indices for reordering//
		//g: initialize auxiliary vectors to decide position between same first seen position objects
		init_rvlocal<Z><<<grid,block>>>(dkeys_in,psize);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_rvlocal for final reordering");
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, grvector, grvector_out, dkeys_in, dkeys_out, psize);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing SortPairsAscending for final reordering");

		//h: final data rearrangement using hkeys
		cutil::safeCopyToHost<Z,uint64_t>(hkeys,dkeys_out,sizeof(Z)*psize, " copy from dkeys_out to hkeys for final reordering");
		uint64_t bi = 0;
		T tvector[NUM_DIMS];//Vector used to keep track of the threshold attributes
		for(uint64_t j = 0; j < this->tuples_per_part; j+=GVTA_BLOCK_SIZE)
		{
			for(uint64_t m = 0; m < this->d; m++) tvector[m] = 0;
			//for(uint64_t jj = j; jj < j + GVTA_BLOCK_SIZE; jj++)
			for(uint64_t jj = 0; jj < GVTA_BLOCK_SIZE; jj++)
			{
				if(j + jj < psize){//
					uint64_t id = offset + hkeys[j+jj];
					for(uint64_t m = 0; m < this->d; m++)
					{
						T v = this->cdata[this->n * m + id];
						this->blocks[bi].data[GVTA_BLOCK_SIZE * GVTA_PARTITIONS * m + (i * GVTA_BLOCK_SIZE + jj)] = v;
						tvector[m] = std::max(tvector[m],v);
					}
				}else{//initialize to negative if block is partially full
					for(uint64_t m = 0; m < this->d; m++)
					{
						this->blocks[bi].data[GVTA_BLOCK_SIZE * GVTA_PARTITIONS * m + (i *GVTA_BLOCK_SIZE + jj)] = -1.0;
					}
				}
			}
			if(bi > 0){
				//Row store
				//std::memcpy(&this->blocks[bi-1].tvector[this->d * i], tvector,sizeof(T)*this->d);
				//Column store
				T *tvec = this->blocks[bi-1].tvector;
				for(uint64_t m = 0; m < this->d; m++){
					tvec[ GVTA_PARTITIONS * m  + i] = tvector[m];
				}
			}
			bi++;
		}
		//
		offset += this->tuples_per_part;
	}

	//DEBUG//
//	std::cout << std::fixed << std::setprecision(6);
//	if(GVTA_PARTITIONS == 2){
//		std::cout << "<<<< COLUMN MAJOR FINAL ORDERING >>>>" << std::endl;
//		for(uint64_t b = 0; b < 4; b++){//4 blocks per partition
//			T *data = this->blocks[b].data;
//			T *tvector = this->blocks[b].tvector;
//
//			std::cout <<"[" << b << "]" << std::endl;
//			for(uint64_t m = 0; m < this->d; m++)
//			{
//				uint64_t poff = 0;
//				for(uint64_t j = 0; j < GVTA_BLOCK_SIZE * GVTA_PARTITIONS; j++)
//				{
//					std::cout << data[GVTA_BLOCK_SIZE * GVTA_PARTITIONS * m + j] << " ";
//					if((j+1) % GVTA_BLOCK_SIZE == 0){
//						//std::cout << "(" << tvector[ (j % GVTA_BLOCK_SIZE)   +  (j / (GVTA_BLOCK_SIZE * GVTA_PARTITIONS))] <<")";
//						std::cout << "(" << tvector[poff + m] <<")";
//						//std::cout << "(0)";
//						std::cout << " | ";
//						poff+=this->d;
//					}
//				}
//				std::cout << std::endl;
//			}
//		}
//		std::cout << "____________________________________________________________________________________________________________________________\n";
//	}

//	for(uint64_t i = 0; i < 1; i++){
//		for(uint64_t b = 0; b < this->num_blocks; b++){
//			std::cout << std::fixed << std::setprecision(4);
//			std::cout << "[" << std::setfill('0') << std::setw(3) << b <<  "] ";
//			for(uint64_t m = 0; m < this->d; m++){
//				std::cout << this->blocks[b].tvector[ GVTA_PARTITIONS * m  + i] << " ";
//			}
//			std::cout << std::endl;
//		}
//	}

	for(uint64_t i = 0; i < GVTA_PARTITIONS; i++){
		for(uint64_t b = 1; b < this->num_blocks; b++){
			for(uint64_t m = 0; m < this->d; m++){
				if( this->blocks[b].tvector[ GVTA_PARTITIONS * m  + i] > this->blocks[b-1].tvector[ GVTA_PARTITIONS * m  + i] )
				{
					std::cout << std::fixed << std::setprecision(4);
					std::cout << "ERROR" << std::endl;
					std::cout << "[" << std::setfill('0') << std::setw(3) << i << "," << b <<  "] ";
					for(uint64_t mm = 0; mm < this->d; mm++){ std::cout << this->blocks[b-1].tvector[ GVTA_PARTITIONS * mm  + i] << " "; }
					std::cout << std::endl << "+++++++++++++++++++++" << std::endl;
					std::cout << "[" << std::setfill('0') << std::setw(3) << i << "," << b <<  "] ";
					for(uint64_t mm = 0; mm < this->d; mm++){ std::cout << this->blocks[b].tvector[ GVTA_PARTITIONS * mm  + i] << " "; }
					std::cout << std::endl;
					exit(1);
				}
			}
		}
	}

	/////////////////////////
	//Free not needed space//
	cudaFree(d_temp_storage);
	cudaFreeHost(grvector);
	cudaFreeHost(grvector_out);
	cudaFreeHost(dkeys_in);
	cudaFreeHost(dkeys_out);
	cudaFreeHost(dvalues_in);
	cudaFreeHost(dvalues_out);

	cudaFreeHost(hkeys);
	if(!VALIDATE) cudaFreeHost(this->cdata); this->cdata = NULL;//TODO: used by VAGG in findTopK
}

template<class T, class Z>
void GVTA<T,Z>::cclear()
{
	for(uint64_t i = 0; i < GVTA_PARTITIONS * KKE; i++) this->cout[i] = 0;
	#if USE_DEVICE_MEM
		cutil::safeCopyToDevice<T,uint64_t>(this->gout,this->cout,sizeof(T) * GVTA_PARTITIONS * KKE, "error copying from gout to out");
	#endif
}

template<class T, class Z>
void GVTA<T,Z>::init()
{
	normalize_transpose<T>(this->cdata, this->n, this->d);
	this->t.start();
	this->reorder();
	this->tt_init = this->t.lap();


	#if USE_DEVICE_MEM
		cutil::safeMalloc<gvta_block<T,Z>,uint64_t>(&(this->gblocks),sizeof(gvta_block<T,Z>)*this->num_blocks,"alloc gpu gvta_blocks");
		cutil::safeCopyToDevice<gvta_block<T,Z>,uint64_t>(this->gblocks,this->blocks,sizeof(gvta_block<T,Z>)*this->num_blocks,"error copying to gpu gvta_blocks");
	#else
		this->gblocks = this->blocks;
	#endif

	cutil::safeMallocHost<T,uint64_t>(&cout,sizeof(T) * GVTA_PARTITIONS * KKE,"out alloc");
	#if USE_DEVICE_MEM
		cutil::safeMalloc<T,uint64_t>(&gout,sizeof(T) * GVTA_PARTITIONS * KKE,"gout alloc");
	#else
		gout = cout;
	#endif
}

template<class T,class Z>
void GVTA<T,Z>::atm_16_dm_driver(uint64_t k, uint64_t qq){
	dim3 atm_16_block(256,1,1);
	dim3 atm_16_grid(GVTA_PARTITIONS, 1, 1);

	this->t.start();
	gvta_atm_16_4096<T,Z><<<atm_16_grid,atm_16_block,256*sizeof(T)>>>(this->gblocks,this->num_blocks,qq,k,gout);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing gvta_atm_16");
	//this->tt_processing += this->t.lap("gvta_atm_16_dm");
	this->tt_processing += this->t.lap("");

	cutil::safeCopyToHost<T,uint64_t>(cout,gout,sizeof(T) * GVTA_PARTITIONS * k, "error copying from gout to out");
	//for(int i = 0; i < GVTA_PARTITIONS * k; i+=k){ for(int j = i; j < i+k; j++){ std::cout << cout[j] << " "; } std::cout << std::endl; }
	std::sort(cout, cout + GVTA_PARTITIONS * k,std::greater<T>());
	this->gpu_threshold = cout[k-1];

	#if VALIDATE
	Time<msecs> t;
	GVAGG<T,Z> gvagg(this->blocks,this->n,this->d,GVTA_BLOCK_SIZE,GVTA_PARTITIONS,this->num_blocks);
	t.start();
	T cpu_gvagg=gvagg.findTopKgvta(k,qq, this->weights,this->query);
	//t.lap("cpu gvagg");
	this->cpu_threshold = cpu_gvagg;

	std::sort(cout, cout + GVTA_PARTITIONS * k,std::greater<T>());
	std::cout << "threshold=[" << cout[k-1] << "," << this->cpu_threshold << "," << cpu_gvagg << "]"<< std::endl;
	if(abs((double)cout[k-1] - (double)cpu_gvagg) > (double)0.00000000000001
			||
			abs((double)cout[k-1] - (double)this->cpu_threshold) > (double)0.00000000000001
			) {
		std::cout << std::fixed << std::setprecision(16);
		std::cout << "ERROR: {" << cout[k-1] << "," << this->cpu_threshold << "," << cpu_gvagg << "}" << std::endl; exit(1);
	}
	#endif
}

template<class T, class Z>
void GVTA<T,Z>::atm_16_hm_driver(uint64_t k, uint64_t qq)
{

}

template<class T,class Z>
void GVTA<T,Z>::atm_16(uint64_t k,uint64_t qq)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 atm_16_block(256,1,1);
	dim3 atm_16_grid(GVTA_PARTITIONS, 1, 1);
	cudaSetDevice(0);
#if VALIDATE
	//this->findTopKtpac(k,(uint8_t)qq,this->weights,this->query);
	VAGG<T,Z> vagg(this->cdata,this->n,this->d);
	this->t.start();
	this->cpu_threshold = vagg.findTopKtpac(k, qq,this->weights,this->query);
	this->t.lap("vagg");

	GVAGG<T,Z> gvagg(this->blocks,this->n,this->d,GVTA_BLOCK_SIZE,GVTA_PARTITIONS,this->num_blocks);
	this->t.start();
	T cpu_gvagg=gvagg.findTopKgvta(k,qq, this->weights,this->query);
	this->t.lap("gvagg");
	//gvagg.findTopKgvta2(k,qq, this->weights,this->query);
#endif

	/*
	 * Top-k With Early Stopping
	 */
	for(uint32_t i = 0; i < GVTA_PARTITIONS * k; i++) cout[i] = 0;
#if USE_DEVICE_MEM
	cutil::safeCopyToDevice<T,uint64_t>(gout,cout,sizeof(T) * GVTA_PARTITIONS * k, "error copying to gout");
#else
	//cudaMemPrefetchAsync(this->gblocks,sizeof(gvta_block<T,Z>)*this->num_blocks, 0, NULL);
#endif
	//std::cout << "[" << atm_16_grid.x << " , " << atm_16_block.x << "] ";
	this->t.start();
	cudaEventRecord(start);
	gvta_atm_16_4096<T,Z><<<atm_16_grid,atm_16_block,256*sizeof(T),s0>>>(this->gblocks,this->num_blocks,qq,k,gout);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing gvta_atm_16");
	cudaEventRecord(stop);

//	this->tt_processing += this->t.lap("gvta_atm_16_2");
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	this->tt_processing = milliseconds;
	//this->tt_processing += this->t.lap();
	//std::cout << "lap: " << this->tt_processing << std::endl;

#if USE_DEVICE_MEM
	cutil::safeCopyToHost<T,uint64_t>(cout,gout,sizeof(T) * GVTA_PARTITIONS * k, "error copying from gout to out");
#endif
//	for(uint32_t i = 0; i < 128; i+=k)
//	{
//		for(uint32_t j = i; j < i + k; j++) std::cout << out[j] << " ";
//		std::cout << "[" << std::is_sorted(&out[i],(&out[i+k])) << "]" << std::endl;
//	}

	this->gpu_threshold = cout[k-1];
#if VALIDATE
	std::sort(cout, cout + GVTA_PARTITIONS * k,std::greater<T>());
	std::cout << "threshold=[" << cout[k-1] << "," << this->cpu_threshold << "," << cpu_gvagg << "]"<< std::endl;
	if(abs((double)cout[k-1] - (double)cpu_gvagg) > (double)0.00000000000001
			||
			abs((double)cout[k-1] - (double)this->cpu_threshold) > (double)0.00000000000001
			) {
		std::cout << std::fixed << std::setprecision(16);
		std::cout << "ERROR: {" << cout[k-1] << "," << this->cpu_threshold << "," << cpu_gvagg << "}" << std::endl; exit(1);
	}
#endif
}

template<class T, class Z>
void GVTA<T,Z>::findTopK(uint64_t k, uint64_t qq){
	this->parts = GVTA_PARTITIONS;
	this->cclear();
	//this->atm_16(k,qq);
	this->atm_16_dm_driver(k,qq);
	cutil::cudaCheckErr(cudaPeekAtLastError(),"AC");
}

#endif
