#ifndef GVTA_H
#define GVTA_H

#include "GAA.h"

template<class T, class Z>
class GVTA : public GAA<T,Z>{
	public:
		GVTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "GVTA";
			this->clear_mem_counters();
		};

		~GVTA()
		{
			if(blocks != NULL){ cudaFreeHost(this->blocks); }
			#if USE_DEVICE_MEM
				cudaFree(this->gblocks);
			#endif
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
		void reorder();

		void clear_mem_counters(){
			this->gvta_mem = 0;
			this->base_mem = 0;
			this->reorder_mem = 0;
		}
		double gvta_mem;
		double reorder_mem;
		double base_mem;
};

/*
 * blocks: data blocks
 * n: tuple number
 * qq: number of query attributes
 * dbsize: data block size
 * pnum: partition number
 * nb: number of blocks (levels)
 */
template<class T, class Z>
__global__ void gvta_atm_16(gvta_block<T,Z> *blocks, uint64_t nb, uint64_t qq, uint64_t k, T *out)
{
	uint64_t i = 0;//data block level
	uint64_t offset = blockIdx.x * nb * GVTA_BLOCK_SIZE;
	T v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0;
	T v8 = 0, v9 = 0, vA = 0, vB = 0, vC = 0, vD = 0, vE = 0, vF = 0;
	//__shared__ Z ids[GVTA_BLOCK_SIZE];
	//__shared__ T scores[GVTA_BLOCK_SIZE];
	T *data;
	//T *tvector;
	while(i < nb)
	{
		data = &blocks[i].data[blockIdx.x * GVTA_BLOCK_SIZE];
		//tvector = &blocks[i].tvector[threshold_offset];

		v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0;
		v8 = 0, v9 = 0, vA = 0, vB = 0, vC = 0, vD = 0, vE = 0, vF = 0;
		//for(uint32_t j = threadIdx.x; j < GVTA_BLOCK_SIZE; j+=blockDim.x){ scores[j] = 0; }

		for(uint32_t m = 0; m < qq; m++)
		{
			uint64_t ai = gpu_query[m];
			uint64_t begin = GVTA_BLOCK_SIZE * GVTA_PARTITIONS * ai;
			v0 += data[begin + threadIdx.x] * gpu_weights[ai];
			v1 += data[begin + threadIdx.x + 256] * gpu_weights[ai];
			v2 += data[begin + threadIdx.x + 512] * gpu_weights[ai];
			v3 += data[begin + threadIdx.x + 768] * gpu_weights[ai];
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
		out[offset + threadIdx.x] = v0;
		out[offset + threadIdx.x + 256] = v1;
		out[offset + threadIdx.x + 512] = v2;
		out[offset + threadIdx.x + 768] = v3;
		out[offset + threadIdx.x + 1024] = v4;
		out[offset + threadIdx.x + 1280] = v5;
		out[offset + threadIdx.x + 1536] = v6;
		out[offset + threadIdx.x + 1792] = v7;
		out[offset + threadIdx.x + 2048] = v8;
		out[offset + threadIdx.x + 2304] = v9;
		out[offset + threadIdx.x + 2560] = vA;
		out[offset + threadIdx.x + 2816] = vB;
		out[offset + threadIdx.x + 3072] = vC;
		out[offset + threadIdx.x + 3328] = vD;
		out[offset + threadIdx.x + 3584] = vE;
		out[offset + threadIdx.x + 3840] = vF;
		offset+=GVTA_BLOCK_SIZE;
		i++;
	}
}

template<class T, class Z>
__global__ void gvta_atm_16_2(gvta_block<T,Z> *blocks, uint64_t nb, uint64_t qq, uint64_t k, T *out)
{
	uint64_t i = 0;//data block level
	uint64_t offset = blockIdx.x * nb * GVTA_BLOCK_SIZE;
	T v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0;
	T v8 = 0, v9 = 0, vA = 0, vB = 0, vC = 0, vD = 0, vE = 0, vF = 0;
	//__shared__ Z ids[GVTA_BLOCK_SIZE];
	__shared__ T threshold;
	__shared__ T heap[32];
	__shared__ T buffer[256];
	T *data;
	T *tvector;
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

			v0 += data[begin + threadIdx.x] * gpu_weights[ai];
			v1 += data[begin + threadIdx.x + 256] * gpu_weights[ai];
			v2 += data[begin + threadIdx.x + 512] * gpu_weights[ai];
			v3 += data[begin + threadIdx.x + 768] * gpu_weights[ai];
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

			//			out[offset + threadIdx.x] = v0;
			//			out[offset + threadIdx.x + 32] = v2;
			//			out[offset + threadIdx.x + 64] = v4;
			//			out[offset + threadIdx.x + 96] = v6;

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

//			out[offset + threadIdx.x] = v0;
//			out[offset + threadIdx.x + 32] = v4;

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

			//out[offset + threadIdx.x] = v0;

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
			//out[offset + 31 - threadIdx.x] = v0;
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
		}
		__syncthreads();
		/*
		 * Break if suitable threshold reached
		 */
//		if(heap[k-1] > threshold){
//			//if(threadIdx.x == 0) printf("[%d],%d < %d\n",blockIdx.x,(uint32_t)i);
//			break;
//		}
		//offset+=GVTA_BLOCK_SIZE;
		i++;
	}

	//Store Ascending Descending
	if(threadIdx.x < k){
		offset = blockIdx.x * k;
		//if((blockIdx.x & 0x1) == 0) out[offset + (k-1) - threadIdx.x] = heap[threadIdx.x];
		//else out[offset + threadIdx.x] = heap[threadIdx.x];
		out[offset + threadIdx.x] = heap[threadIdx.x];
	}
}

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
	cutil::safeMallocHost<gvta_block<T,Z>,uint64_t>(&(this->blocks),sizeof(gvta_block<T,Z>)*this->num_blocks,"alloc gvta_blocks");
	std::cout << this->n << " = p,psz(" << GVTA_PARTITIONS <<"," << this->tuples_per_part << ") - " << "b,bsz(" << this->num_blocks << "," << GVTA_BLOCK_SIZE << ")" << std::endl;
	for(uint64_t i = 0; i< this->num_blocks; i++)//TODO:safemalloc
	{
		//cutil::safeMallocHost<T,uint64_t>(&(this->blocks[i].data),sizeof(T)*GVTA_PARTITIONS*GVTA_BLOCK_SIZE*this->d,"alloc gvta_block data("+std::to_string((unsigned long long)i)+")");
		//cutil::safeMallocHost<T,uint64_t>(&(this->blocks[i].tvector),sizeof(T)*GVTA_PARTITIONS*this->d,"alloc gvta_block tvector("+std::to_string((unsigned long long)i)+")");
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
						this->blocks[bi].data[GVTA_BLOCK_SIZE * GVTA_PARTITIONS * m + (i *GVTA_BLOCK_SIZE + jj)] = v;
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

//	//DEBUG//
//	std::cout << std::fixed << std::setprecision(6);
//	//if(GVTA_PARTITIONS == 2){
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
//					//std::cout << data[GVTA_BLOCK_SIZE * GVTA_PARTITIONS * m + j] << " ";
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
//	//}

	/////////////////////////
	//Free not needed space//
	cudaFree(d_temp_storage);
	cudaFree(grvector);
	cudaFree(grvector_out);
	cudaFree(dkeys_in);
	cudaFree(dkeys_out);
	cudaFree(dvalues_in);
	cudaFree(dvalues_out);

	cudaFreeHost(hkeys);
	//cudaFreeHost(this->cdata); this->cdata = NULL;//TODO: used by VAGG in findTopK
}

template<class T, class Z>
void GVTA<T,Z>::init()
{
	normalize_transpose<T>(this->cdata, this->n, this->d);
	this->t.start();
	this->reorder();
#if USE_DEVICE_MEM
	cutil::safeMalloc<gvta_block<T,Z>,uint64_t>(&(this->gblocks),sizeof(gvta_block<T,Z>)*this->num_blocks,"alloc gpu gvta_blocks");
	for(uint64_t i = 0; i< this->num_blocks; i++)//TODO:safemalloc
	{
		//cutil::safeMalloc<T,uint64_t>(&(this->gblocks[i].data),sizeof(T)*GVTA_PARTITIONS*GVTA_BLOCK_SIZE*this->d,"alloc gpu gvta_block data("+std::to_string((unsigned long long)i)+")");
		//cutil::safeMalloc<T,uint64_t>(&(this->gblocks[i].tvector),sizeof(T)*GVTA_PARTITIONS*this->d,"alloc gpu gvta_block tvector("+std::to_string((unsigned long long)i)+")");
	}
	cutil::safeCopyToDevice<gvta_block<T,Z>,uint64_t>(this->gblocks,this->blocks,sizeof(gvta_block<T,Z>)*this->num_blocks,"error copying to gpu gvta_blocks");
#else
	this->gblocks = this->blocks;
#endif
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void GVTA<T,Z>::findTopK(uint64_t k, uint64_t qq){
	dim3 atm_16_block(256,1,1);
	dim3 atm_16_grid(GVTA_PARTITIONS, 1, 1);
	T *out, *gout;
	cutil::safeMallocHost<T,uint64_t>(&out,sizeof(T)*this->n,"out alloc");
#if USE_DEVICE_MEM
	cutil::safeMalloc<T,uint64_t>(&gout,sizeof(T)*this->n,"gout alloc");
#else
	gout = out;
#endif

	T threshold = 0;
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

	for(uint32_t i = 0; i < this->n; i++) out[i] = 0;
	#if USE_DEVICE_MEM
		cutil::safeCopyToDevice<T,uint64_t>(gout,out,sizeof(T)*this->n, "error copying to gout");
	#endif
	this->t.start();
	gvta_atm_16_2<T,Z><<<atm_16_grid,atm_16_block>>>(this->gblocks,this->num_blocks,qq,k,gout);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing gvta_atm_16");
	this->t.lap("gvta_atm_16_2");

	#if USE_DEVICE_MEM
		cutil::safeCopyToHost<T,uint64_t>(out,gout,sizeof(T)*this->n, "error copying to out");
	#endif
//	for(uint32_t i = 0; i < 128; i+=k)
//	{
//		for(uint32_t j = i; j < i + k; j++) std::cout << out[j] << " ";
//		std::cout << "[" << std::is_sorted(&out[i],(&out[i+k])) << "]" << std::endl;
//	}
	std::sort(out, out + this->num_blocks * k,std::greater<T>());
	threshold = out[k-1];
	if(abs(out[k-1] - cpu_gvagg) > 0.0000001) { std::cout << "ERROR: {" << out[k-1] << "," << this->cpu_threshold << "," << cpu_gvagg << "}" << std::endl; exit(1); }
	cudaFreeHost(out);
	std::cout << "threshold=[" << threshold << "," << this->cpu_threshold << "," << cpu_gvagg << "]"<< std::endl;
}

#endif
