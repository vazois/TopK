#ifndef BTA_H
#define BTA_H
/*
 * Bitonic Top-k Aggregation
 */

#include "GAA.h"
#define BTA_BLOCK_SIZE 512
#define BTA_TUPLES_PER_BLOCK 4096

template<class T, class Z>
__global__ void aggregate(T *gdata, uint64_t n, uint64_t d, T *gscores){
	__shared__ T tuple_scores[BTA_TUPLES_PER_BLOCK];

	uint32_t tid = threadIdx.x;
	uint64_t goffset = blockIdx.x * BTA_TUPLES_PER_BLOCK + tid;//global starting offset
	for(uint64_t loffset = tid; loffset < BTA_TUPLES_PER_BLOCK; loffset+=blockDim.x){//increment local block offset
		T score = 0;
		for(uint64_t m = 0; m < d; m++){
			uint64_t ai = gpu_query[m];
			score+=gdata[ai*n + goffset] * gpu_weights[ai];//column-store load global relation access
		}
		tuple_scores[loffset] = score;//write scores in shared memory
		gscores[goffset] = tuple_scores[loffset];//write-back scores//optional//for debug purpose
		goffset+= blockDim.x;//increase global offset
	}
}

/*
 *
 */
template<class T>
__global__ void scores_topk(T *ivec, T *ovec, uint64_t k){
	//__shared__ T mx[blockDim.x << 1];
	__shared__ T sm_vals[4096];

	uint64_t goffset = (blockIdx.x << 12) + threadIdx.x;
	for(uint64_t loffset = threadIdx.x; loffset < BTA_TUPLES_PER_BLOCK; loffset+=blockDim.x){
		sm_vals[loffset] = ivec[goffset];
		goffset+= blockDim.x;
	}
	__syncthreads();

	for(uint32_t level = 1; level < k; level = level << 1){
		uint32_t dir = level << 1;
		for(uint32_t step = level; step > 0; step = step >> 1){
			int low = threadIdx.x & (step - 1);
			int i = (threadIdx.x << 1) - low;
			bool reverse = ((dir & i) == 0);
			for(uint32_t offset = 0; offset < 4096; offset+=(blockDim.x << 1)){
				T v0 = sm_vals[offset+i];
				T v1 = sm_vals[offset+i+step];
				bool swap = reverse ^ (v0 < v1);
				T left = swap ? v0 : v1;
				v1 = swap ? v1 : v0;
				v0 = left;
				sm_vals[offset+i] = v0;
				sm_vals[offset+i+step] = v1;
			}
			__syncthreads();
		}
	}
//	int low = threadIdx.x & (k-1);
//	int i = (threadIdx.x << 1) - low;
//	sm_vals[i] =  fmaxf(sm_vals[i],sm_vals[i+k]);
	__syncthreads();

	goffset = (blockIdx.x << 12) + threadIdx.x;
	for(uint64_t loffset = threadIdx.x; loffset < BTA_TUPLES_PER_BLOCK; loffset+=blockDim.x){
		ovec[goffset] = sm_vals[loffset];//Write-back scores//
		goffset+= blockDim.x;
	}
}

template<class T>
__device__ T swap(T a, uint32_t stride, int dir){
	T b = __shfl_xor_sync(0xFFFFFFFF,a,stride);
	return ((a < b) == dir) ? b : a;
}

__device__ uint32_t bfe(uint32_t a, uint32_t i){
	return (a >> i) & 0x1;
}

/*
 * Partial sort top-k function for at most k=16
 */
template<class T, uint32_t bsize>
__global__ void lsort_atm_16(T *ivec, T *ovec, uint64_t k){
	__shared__ T sm_vals[4096];
	uint32_t goffset;
	T v0, v1, v2, v3, v4, v5, v6, v7;
	T v8, v9, vA, vB, vC, vD, vE, vF;

	goffset = (blockIdx.x << 12) + threadIdx.x;
	v0 = ivec[goffset];
	v1 = ivec[goffset + 256];
	v2 = ivec[goffset + 512];
	v3 = ivec[goffset + 768];
	v4 = ivec[goffset + 1024];
	v5 = ivec[goffset + 1280];
	v6 = ivec[goffset + 1536];
	v7 = ivec[goffset + 1792];
	v8 = ivec[goffset + 2048];
	v9 = ivec[goffset + 2304];
	vA = ivec[goffset + 2560];
	vB = ivec[goffset + 2816];
	vC = ivec[goffset + 3072];
	vD = ivec[goffset + 3328];
	vE = ivec[goffset + 3584];
	vF = ivec[goffset + 3840];

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

	sm_vals[threadIdx.x] = v0;
	sm_vals[threadIdx.x + 256] = v1;
	sm_vals[threadIdx.x + 512] = v2;
	sm_vals[threadIdx.x + 768] = v3;
	sm_vals[threadIdx.x + 1024] = v4;
	sm_vals[threadIdx.x + 1280] = v5;
	sm_vals[threadIdx.x + 1536] = v6;
	sm_vals[threadIdx.x + 1792] = v7;
	sm_vals[threadIdx.x + 2048] = v8;
	sm_vals[threadIdx.x + 2304] = v9;
	sm_vals[threadIdx.x + 2560] = vA;
	sm_vals[threadIdx.x + 2816] = vB;
	sm_vals[threadIdx.x + 3072] = vC;
	sm_vals[threadIdx.x + 3328] = vD;
	sm_vals[threadIdx.x + 3584] = vE;
	sm_vals[threadIdx.x + 3840] = vF;
	__syncthreads();
//

	//merge-rebuild//
	uint32_t i;
	uint32_t rw = 256;
	laneId = threadIdx.x;
	level = k >> 1;
	for(uint32_t size = 2048; size > 256; size = size >> 1){//
		if ( threadIdx.x < rw ){

			i = threadIdx.x;
			v0 = sm_vals[i]; i += rw;
			v1 = sm_vals[i]; i += rw;
			v2 = sm_vals[i]; i += rw;
			v3 = sm_vals[i]; i += rw;
			v4 = sm_vals[i]; i += rw;
			v5 = sm_vals[i]; i += rw;
			v6 = sm_vals[i]; i += rw;
			v7 = sm_vals[i]; i += rw;
			v8 = sm_vals[i]; i += rw;
			v9 = sm_vals[i]; i += rw;
			vA = sm_vals[i]; i += rw;
			vB = sm_vals[i]; i += rw;
			vC = sm_vals[i]; i += rw;
			vD = sm_vals[i]; i += rw;
			vE = sm_vals[i]; i += rw;
			vF = sm_vals[i];

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
			i = threadIdx.x;
			sm_vals[i] = v0; i += rw;
			sm_vals[i] = v2; i += rw;
			sm_vals[i] = v4; i += rw;
			sm_vals[i] = v6; i += rw;
			sm_vals[i] = v8; i += rw;
			sm_vals[i] = vA; i += rw;
			sm_vals[i] = vC; i += rw;
			sm_vals[i] = vE;
		}
		__syncthreads();

		rw = rw >> 1;
		if (threadIdx.x < rw){
			i = threadIdx.x;
			v0 = sm_vals[i]; i += rw;
			v1 = sm_vals[i]; i += rw;
			v2 = sm_vals[i]; i += rw;
			v3 = sm_vals[i]; i += rw;
			v4 = sm_vals[i]; i += rw;
			v5 = sm_vals[i]; i += rw;
			v6 = sm_vals[i]; i += rw;
			v7 = sm_vals[i]; i += rw;
			v8 = sm_vals[i]; i += rw;
			v9 = sm_vals[i]; i += rw;
			vA = sm_vals[i]; i += rw;
			vB = sm_vals[i]; i += rw;
			vC = sm_vals[i]; i += rw;
			vD = sm_vals[i]; i += rw;
			vE = sm_vals[i]; i += rw;
			vF = sm_vals[i];

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

			i = threadIdx.x;
			sm_vals[i] = v0; i += rw;
			sm_vals[i] = v1; i += rw;
			sm_vals[i] = v2; i += rw;
			sm_vals[i] = v3; i += rw;
			sm_vals[i] = v4; i += rw;
			sm_vals[i] = v5; i += rw;
			sm_vals[i] = v6; i += rw;
			sm_vals[i] = v7; i += rw;
			sm_vals[i] = v8; i += rw;
			sm_vals[i] = v9; i += rw;
			sm_vals[i] = vA; i += rw;
			sm_vals[i] = vB; i += rw;
			sm_vals[i] = vC; i += rw;
			sm_vals[i] = vD; i += rw;
			sm_vals[i] = vE; i += rw;
			sm_vals[i] = vF;
		}
		__syncthreads();
	}

	if(threadIdx.x < 32){
		i = threadIdx.x;
		v0 = sm_vals[i]; i += 32;
		v1 = sm_vals[i]; i += 32;
		v2 = sm_vals[i]; i += 32;
		v3 = sm_vals[i]; i += 32;
		v4 = sm_vals[i]; i += 32;
		v5 = sm_vals[i]; i += 32;
		v6 = sm_vals[i]; i += 32;
		v7 = sm_vals[i]; i += 32;
		v8 = sm_vals[i]; i += 32;
		v9 = sm_vals[i]; i += 32;
		vA = sm_vals[i]; i += 32;
		vB = sm_vals[i]; i += 32;
		vC = sm_vals[i]; i += 32;
		vD = sm_vals[i]; i += 32;
		vE = sm_vals[i]; i += 32;
		vF = sm_vals[i];

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

		//256//
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

		//64//
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

		//32//
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
			v0 = swap(v0,step,dir);
			v8 = swap(v8,step,dir);
		}

		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
		v0 = (threadIdx.x & k) == 0 ? v0 : v8;


		//16//
		for( ; level < 32; level = level << 1){
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
			}
		}
		sm_vals[31 - threadIdx.x] = v0;//TODO:reverse
		//sm_vals[threadIdx.x] = v0;
	}
	__syncthreads();
	if (threadIdx.x < k){
		goffset = k * blockIdx.x + threadIdx.x;
		ovec[goffset] = sm_vals[threadIdx.x];
	}
}

/*
 * At most k=16 rebuild function
 *
 */
template<class T, uint32_t bsize>
__global__ void mrebuild_atm_16(T *ivec, T *ovec, uint64_t k, uint32_t remainder){
	__shared__ T sm_vals[4096];
	uint32_t goffset;
	T v0 = -1.0, v1 = -1.0, v2 = -1.0, v3 = -1.0, v4 = -1.0, v5 = -1.0, v6 = -1.0, v7 = -1.0;
	T v8 = -1.0, v9 = -1.0, vA = -1.0, vB = -1.0, vC = -1.0, vD = -1.0, vE = -1.0, vF = -1.0;

	goffset = (blockIdx.x << 12) + threadIdx.x;
	//check if loaded data are within range//
	if(goffset < remainder) v0 = ivec[goffset];
	if(goffset + 256 < remainder) v1 = ivec[goffset + 256];
	if(goffset + 512 < remainder) v2 = ivec[goffset + 512];
	if(goffset + 768 < remainder) v3 = ivec[goffset + 768];
	if(goffset + 1024 < remainder) v4 = ivec[goffset + 1024];
	if(goffset + 1280 < remainder) v5 = ivec[goffset + 1280];
	if(goffset + 1536 < remainder) v6 = ivec[goffset + 1536];
	if(goffset + 1792 < remainder) v7 = ivec[goffset + 1792];
	if(goffset + 2048 < remainder) v8 = ivec[goffset + 2048];
	if(goffset + 2304 < remainder) v9 = ivec[goffset + 2304];
	if(goffset + 2560 < remainder) vA = ivec[goffset + 2560];
	if(goffset + 2816 < remainder) vB = ivec[goffset + 2816];
	if(goffset + 3072 < remainder) vC = ivec[goffset + 3072];
	if(goffset + 3328 < remainder) vD = ivec[goffset + 3328];
	if(goffset + 3584 < remainder) vE = ivec[goffset + 3584];
	if(goffset + 3840 < remainder) vF = ivec[goffset + 3840];

	uint32_t laneId = threadIdx.x;
	uint32_t level = k >> 1, step, dir;
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

	sm_vals[threadIdx.x] = v0;
	sm_vals[threadIdx.x + 256] = v1;
	sm_vals[threadIdx.x + 512] = v2;
	sm_vals[threadIdx.x + 768] = v3;
	sm_vals[threadIdx.x + 1024] = v4;
	sm_vals[threadIdx.x + 1280] = v5;
	sm_vals[threadIdx.x + 1536] = v6;
	sm_vals[threadIdx.x + 1792] = v7;
	sm_vals[threadIdx.x + 2048] = v8;
	sm_vals[threadIdx.x + 2304] = v9;
	sm_vals[threadIdx.x + 2560] = vA;
	sm_vals[threadIdx.x + 2816] = vB;
	sm_vals[threadIdx.x + 3072] = vC;
	sm_vals[threadIdx.x + 3328] = vD;
	sm_vals[threadIdx.x + 3584] = vE;
	sm_vals[threadIdx.x + 3840] = vF;
	__syncthreads();

	uint32_t i;
	uint32_t rw = 256;
	laneId = threadIdx.x;
	level = k >> 1;
	uint32_t size = (goffset + 4096 < remainder) ? 2048 : remainder - goffset;
	for( ; size > 256; size = size >> 1){//
		if ( threadIdx.x < rw ){
			i = threadIdx.x;
			v0 = sm_vals[i]; i += rw;
			v1 = sm_vals[i]; i += rw;
			v2 = sm_vals[i]; i += rw;
			v3 = sm_vals[i]; i += rw;
			v4 = sm_vals[i]; i += rw;
			v5 = sm_vals[i]; i += rw;
			v6 = sm_vals[i]; i += rw;
			v7 = sm_vals[i]; i += rw;
			v8 = sm_vals[i]; i += rw;
			v9 = sm_vals[i]; i += rw;
			vA = sm_vals[i]; i += rw;
			vB = sm_vals[i]; i += rw;
			vC = sm_vals[i]; i += rw;
			vD = sm_vals[i]; i += rw;
			vE = sm_vals[i]; i += rw;
			vF = sm_vals[i];

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

			i = threadIdx.x;
			sm_vals[i] = v0; i += rw;
			sm_vals[i] = v2; i += rw;
			sm_vals[i] = v4; i += rw;
			sm_vals[i] = v6; i += rw;
			sm_vals[i] = v8; i += rw;
			sm_vals[i] = vA; i += rw;
			sm_vals[i] = vC; i += rw;
			sm_vals[i] = vE;
		}
		__syncthreads();

		rw = rw >> 1;
		if (threadIdx.x < rw){
			i = threadIdx.x;
			v0 = sm_vals[i]; i += rw;
			v1 = sm_vals[i]; i += rw;
			v2 = sm_vals[i]; i += rw;
			v3 = sm_vals[i]; i += rw;
			v4 = sm_vals[i]; i += rw;
			v5 = sm_vals[i]; i += rw;
			v6 = sm_vals[i]; i += rw;
			v7 = sm_vals[i]; i += rw;
			v8 = sm_vals[i]; i += rw;
			v9 = sm_vals[i]; i += rw;
			vA = sm_vals[i]; i += rw;
			vB = sm_vals[i]; i += rw;
			vC = sm_vals[i]; i += rw;
			vD = sm_vals[i]; i += rw;
			vE = sm_vals[i]; i += rw;
			vF = sm_vals[i];

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

			i = threadIdx.x;
			sm_vals[i] = v0; i += rw;
			sm_vals[i] = v1; i += rw;
			sm_vals[i] = v2; i += rw;
			sm_vals[i] = v3; i += rw;
			sm_vals[i] = v4; i += rw;
			sm_vals[i] = v5; i += rw;
			sm_vals[i] = v6; i += rw;
			sm_vals[i] = v7; i += rw;
			sm_vals[i] = v8; i += rw;
			sm_vals[i] = v9; i += rw;
			sm_vals[i] = vA; i += rw;
			sm_vals[i] = vB; i += rw;
			sm_vals[i] = vC; i += rw;
			sm_vals[i] = vD; i += rw;
			sm_vals[i] = vE; i += rw;
			sm_vals[i] = vF;
		}
		__syncthreads();
	}

	//Merge-rebuild one warp//
	if(threadIdx.x < 32){
		i = threadIdx.x;
		v0 = sm_vals[i]; i += 32;
		v1 = sm_vals[i]; i += 32;
		v2 = sm_vals[i]; i += 32;
		v3 = sm_vals[i]; i += 32;
		v4 = sm_vals[i]; i += 32;
		v5 = sm_vals[i]; i += 32;
		v6 = sm_vals[i]; i += 32;
		v7 = sm_vals[i]; i += 32;
		v8 = sm_vals[i]; i += 32;
		v9 = sm_vals[i]; i += 32;
		vA = sm_vals[i]; i += 32;
		vB = sm_vals[i]; i += 32;
		vC = sm_vals[i]; i += 32;
		vD = sm_vals[i]; i += 32;
		vE = sm_vals[i]; i += 32;
		vF = sm_vals[i];

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

		//256//
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

		//64//
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

		//32//
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
			v0 = swap(v0,step,dir);
			v8 = swap(v8,step,dir);
		}

		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
		v0 = (threadIdx.x & k) == 0 ? v0 : v8;


		//16//
		for( ; level < 32; level = level << 1){
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
			}
		}
		sm_vals[31 - threadIdx.x] = v0;
	}

	//Write-back results
	if (threadIdx.x < k){
		goffset = k * blockIdx.x + threadIdx.x;
		ovec[goffset] = sm_vals[threadIdx.x];
	}
}

/*
 * Partial sort top-k function for larger than k=16
 */
template<class T, uint32_t bsize>
__global__ void lsort_geq_32(T *ivec, T *ovec, uint64_t k){
	__shared__ T sm_vals[4096];
	uint32_t goffset;
	T v0, v1, v2, v3, v4, v5, v6, v7;
	T v8, v9, vA, vB, vC, vD, vE, vF;
	T tt;
	//if(threadIdx.x == 0 && blockIdx.x == 0) printf("scores_topk9\n");
	//if(threadIdx.x == 0 && blockIdx.x == 0) printf("%d,%d,%d\n",threadIdx.x,32,__ffs(32));

	//Load in registers
	uint32_t laneId = threadIdx.x;
	uint32_t level, step;
	int dir;
	if ( threadIdx.x < 256 )
	{
		goffset = (blockIdx.x << 12) + threadIdx.x;
		v0 = ivec[goffset];
		v1 = ivec[goffset + 256];
		v2 = ivec[goffset + 512];
		v3 = ivec[goffset + 768];
		v4 = ivec[goffset + 1024];
		v5 = ivec[goffset + 1280];
		v6 = ivec[goffset + 1536];
		v7 = ivec[goffset + 1792];
		v8 = ivec[goffset + 2048];
		v9 = ivec[goffset + 2304];
		vA = ivec[goffset + 2560];
		vB = ivec[goffset + 2816];
		vC = ivec[goffset + 3072];
		vD = ivec[goffset + 3328];
		vE = ivec[goffset + 3584];
		vF = ivec[goffset + 3840];

		//sort in registers
		for(level = 1; level < 32; level = level << 1){
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

		sm_vals[threadIdx.x] = v0;
		sm_vals[threadIdx.x + 256] = v1;
		sm_vals[threadIdx.x + 512] = v2;
		sm_vals[threadIdx.x + 768] = v3;
		sm_vals[threadIdx.x + 1024] = v4;
		sm_vals[threadIdx.x + 1280] = v5;
		sm_vals[threadIdx.x + 1536] = v6;
		sm_vals[threadIdx.x + 1792] = v7;
		sm_vals[threadIdx.x + 2048] = v8;
		sm_vals[threadIdx.x + 2304] = v9;
		sm_vals[threadIdx.x + 2560] = vA;
		sm_vals[threadIdx.x + 2816] = vB;
		sm_vals[threadIdx.x + 3072] = vC;
		sm_vals[threadIdx.x + 3328] = vD;
		sm_vals[threadIdx.x + 3584] = vE;
		sm_vals[threadIdx.x + 3840] = vF;
	}
	__syncthreads();
///////////////////////////////////////
	//sort in shared memory and registers//
	uint32_t low;
	uint32_t i;
	uint32_t loffset;
	//uint32_t tid;
	bool reverse, sp;
	for(level = 32; level < k; level = level << 1){
		dir = level << 1;
		//////////////
		//up to 128//
		for(step = level; step > 16; step = step >> 1){
			low = threadIdx.x & (step - 1);
			i = (threadIdx.x << 1) - low;
			reverse = ((dir & i) == 0);
			for(loffset = 0; loffset < 4096; loffset+=512){
				v0 = sm_vals[loffset+i];
				v1 = sm_vals[loffset+i+step];
				sp = reverse ^ (v0 < v1);
				tt = sp ? v1 : v0; v1 = sp ? v0 : v1; v0 = tt;
				sm_vals[loffset+i] = v0;
				sm_vals[loffset+i+step] = v1;
			}
			__syncthreads();
		}

		if ( threadIdx.x < 256 )
		{
			v0 = sm_vals[threadIdx.x];
			v1 = sm_vals[threadIdx.x + 256];
			v2 = sm_vals[threadIdx.x + 512];
			v3 = sm_vals[threadIdx.x + 768];
			v4 = sm_vals[threadIdx.x + 1024];
			v5 = sm_vals[threadIdx.x + 1280];
			v6 = sm_vals[threadIdx.x + 1536];
			v7 = sm_vals[threadIdx.x + 1792];
			v8 = sm_vals[threadIdx.x + 2048];
			v9 = sm_vals[threadIdx.x + 2304];
			vA = sm_vals[threadIdx.x + 2560];
			vB = sm_vals[threadIdx.x + 2816];
			vC = sm_vals[threadIdx.x + 3072];
			vD = sm_vals[threadIdx.x + 3328];
			vE = sm_vals[threadIdx.x + 3584];
			vF = sm_vals[threadIdx.x + 3840];
			//sort in registers
			for(step = 16; step > 0; step = step >> 1){
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
			//
			sm_vals[threadIdx.x] = v0;
			sm_vals[threadIdx.x + 256] = v1;
			sm_vals[threadIdx.x + 512] = v2;
			sm_vals[threadIdx.x + 768] = v3;
			sm_vals[threadIdx.x + 1024] = v4;
			sm_vals[threadIdx.x + 1280] = v5;
			sm_vals[threadIdx.x + 1536] = v6;
			sm_vals[threadIdx.x + 1792] = v7;
			sm_vals[threadIdx.x + 2048] = v8;
			sm_vals[threadIdx.x + 2304] = v9;
			sm_vals[threadIdx.x + 2560] = vA;
			sm_vals[threadIdx.x + 2816] = vB;
			sm_vals[threadIdx.x + 3072] = vC;
			sm_vals[threadIdx.x + 3328] = vD;
			sm_vals[threadIdx.x + 3584] = vE;
			sm_vals[threadIdx.x + 3840] = vF;
		}
		__syncthreads();
	}

	/////////////////
	//merge-rebuild//
	//uint32_t rw = 256;
	uint32_t sz = 4096;
	level = k >> 1;
	dir = level << 1;
	for(uint32_t size = 2048; size > (k>>1); size = size >> 1){
		//merge
		low = threadIdx.x & (k-1);
		i = (threadIdx.x << 1) - low;
		if( sz == 4096 )
		{
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v1 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v2 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v3 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v4 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v5 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v6 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v7 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}else if ( sz == 2048 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v1 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v2 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v3 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}else if ( sz == 1024 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v1 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}else if ( sz <= 512 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}
		__syncthreads();

		i = threadIdx.x;
		if( sz == 4096 )
		{
			sm_vals[i] = v0; i+=bsize;
			sm_vals[i] = v1; i+=bsize;
			sm_vals[i] = v2; i+=bsize;
			sm_vals[i] = v3; i+=bsize;
			sm_vals[i] = v4; i+=bsize;
			sm_vals[i] = v5; i+=bsize;
			sm_vals[i] = v6; i+=bsize;
			sm_vals[i] = v7;
		}else if ( sz == 2048 ){
			sm_vals[i] = v0; i+=bsize;
			sm_vals[i] = v1; i+=bsize;
			sm_vals[i] = v2; i+=bsize;
			sm_vals[i] = v3;
		}else if ( sz == 1024 ){
			sm_vals[i] = v0; i+=bsize;
			sm_vals[i] = v1;
		}else if ( sz <= 512 ){
			sm_vals[i] = v0;
		}
		__syncthreads();

		sz  = sz >> 1;
		//rebuild
		//Shared memory rebuild
		for(step = level; step > 16; step = step >> 1){
			low = threadIdx.x & (step - 1);
			i = (threadIdx.x << 1) - low;
			reverse = ((dir & i) == 0);

			for(loffset = 0; loffset < sz; loffset+=512){
				v0 = sm_vals[loffset+i];
				v1 = sm_vals[loffset+i+step];
				sp = reverse ^ (v0 < v1);
				tt = sp ? v1 : v0; v1 = sp ? v0 : v1; v0 = tt;
				sm_vals[loffset+i] = v0;
				sm_vals[loffset+i+step] = v1;
			}
			__syncthreads();
		}

		//register rebuild
		i = threadIdx.x;
		laneId = threadIdx.x;
		level = k >> 1;
		if ( sz == 2048 ){
			v0 = sm_vals[i]; i += bsize;
			v1 = sm_vals[i]; i += bsize;
			v2 = sm_vals[i]; i += bsize;
			v3 = sm_vals[i]; i += bsize;
			v4 = sm_vals[i]; i += bsize;
			v5 = sm_vals[i]; i += bsize;
			v6 = sm_vals[i]; i += bsize;
			v7 = sm_vals[i];

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

			i = threadIdx.x;
			sm_vals[i] = v0; i += bsize;
			sm_vals[i] = v1; i += bsize;
			sm_vals[i] = v2; i += bsize;
			sm_vals[i] = v3; i += bsize;
			sm_vals[i] = v4; i += bsize;
			sm_vals[i] = v5; i += bsize;
			sm_vals[i] = v6; i += bsize;
			sm_vals[i] = v7;
		}else if ( sz == 1024 ){
			v0 = sm_vals[i]; i += bsize;
			v1 = sm_vals[i]; i += bsize;
			v2 = sm_vals[i]; i += bsize;
			v3 = sm_vals[i]; i += bsize;

			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
				v1 = swap(v1,step,dir);
				v2 = swap(v2,step,dir);
				v3 = swap(v3,step,dir);
			}

			i = threadIdx.x;
			sm_vals[i] = v0; i += bsize;
			sm_vals[i] = v1; i += bsize;
			sm_vals[i] = v2; i += bsize;
			sm_vals[i] = v3;
		}else if ( sz == 512 ){
			v0 = sm_vals[i]; i += bsize;
			v1 = sm_vals[i];

			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
				v1 = swap(v1,step,dir);
			}

			i = threadIdx.x;
			sm_vals[i] = v0; i += bsize;
			sm_vals[i] = v1;
		}else if ( sz <= 256 ){
			v0 = sm_vals[i];

			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
			}

			i = threadIdx.x;
			sm_vals[i] = v0;
		}
		//if(threadIdx.x == 0 && blockIdx.x == 0) printf("{%d} \n",sz);
		__syncthreads();
	}
	////////////////

	if(threadIdx.x < k)
	{
//		goffset = k * blockIdx.x + threadIdx.x;
//		ovec[goffset] = sm_vals[threadIdx.x];

		if( blockIdx.x & 0x1 == 1 ){
			goffset = k * blockIdx.x + (threadIdx.x^(k-1));
			ovec[goffset] = sm_vals[threadIdx.x];
		}else{
			goffset = k * blockIdx.x + (threadIdx.x);
			ovec[goffset] = sm_vals[threadIdx.x];
		}
	}
//	if(threadIdx.x < k) ovec[goffset] = sm_vals[threadIdx.x];
//	if (threadIdx.x < k){
//		goffset = k * blockIdx.x + threadIdx.x;
//		ovec[goffset] = sm_vals[threadIdx.x];
//	}
//	ovec[goffset] = sm_vals[threadIdx.x];
//	ovec[goffset + 256] = sm_vals[threadIdx.x + 256];
//	ovec[goffset + 512] = sm_vals[threadIdx.x + 512];
//	ovec[goffset + 768] = sm_vals[threadIdx.x + 768];
//	ovec[goffset + 1024] = sm_vals[threadIdx.x + 1024];
//	ovec[goffset + 1280] = sm_vals[threadIdx.x + 1280];
//	ovec[goffset + 1536] = sm_vals[threadIdx.x + 1536];
//	ovec[goffset + 1792] = sm_vals[threadIdx.x + 1792];
//	ovec[goffset + 2048] = sm_vals[threadIdx.x + 2048];
//	ovec[goffset + 2304] = sm_vals[threadIdx.x + 2304];
//	ovec[goffset + 2560] = sm_vals[threadIdx.x + 2560];
//	ovec[goffset + 2816] = sm_vals[threadIdx.x + 2816];
//	ovec[goffset + 3072] = sm_vals[threadIdx.x + 3072];
//	ovec[goffset + 3328] = sm_vals[threadIdx.x + 3328];
//	ovec[goffset + 3584] = sm_vals[threadIdx.x + 3584];
//	ovec[goffset + 3840] = sm_vals[threadIdx.x + 3840];
}

/*
 * aggregate relation and lsort top k <=16
 */
template<class T, uint32_t bsize>
__global__ void agg_lsort_atm_16(T *ivec, T *ovec, uint64_t n, uint64_t d, uint64_t k){
	__shared__ T sm_vals[4096];
	uint32_t goffset;
	T v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0;
	T v8 = 0, v9 = 0, vA = 0, vB = 0, vC = 0, vD = 0, vE = 0, vF = 0;

	goffset = (blockIdx.x << 12) + threadIdx.x;
	for(uint64_t m = 0; m < 1; m++)
	{
		v0 += ivec[goffset + n*m];
		v1 += ivec[goffset + n*m + 256];
		v2 += ivec[goffset + n*m + 512];
		v3 += ivec[goffset + n*m + 768];
		v4 += ivec[goffset + n*m + 1024];
		v5 += ivec[goffset + n*m + 1280];
		v6 += ivec[goffset + n*m + 1536];
		v7 += ivec[goffset + n*m + 1792];
		v8 += ivec[goffset + n*m + 2048];
		v9 += ivec[goffset + n*m + 2304];
		vA += ivec[goffset + n*m + 2560];
		vB += ivec[goffset + n*m + 2816];
		vC += ivec[goffset + n*m + 3072];
		vD += ivec[goffset + n*m + 3328];
		vE += ivec[goffset + n*m + 3584];
		vF += ivec[goffset + n*m + 3840];
	}

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

	sm_vals[threadIdx.x] = v0;
	sm_vals[threadIdx.x + 256] = v1;
	sm_vals[threadIdx.x + 512] = v2;
	sm_vals[threadIdx.x + 768] = v3;
	sm_vals[threadIdx.x + 1024] = v4;
	sm_vals[threadIdx.x + 1280] = v5;
	sm_vals[threadIdx.x + 1536] = v6;
	sm_vals[threadIdx.x + 1792] = v7;
	sm_vals[threadIdx.x + 2048] = v8;
	sm_vals[threadIdx.x + 2304] = v9;
	sm_vals[threadIdx.x + 2560] = vA;
	sm_vals[threadIdx.x + 2816] = vB;
	sm_vals[threadIdx.x + 3072] = vC;
	sm_vals[threadIdx.x + 3328] = vD;
	sm_vals[threadIdx.x + 3584] = vE;
	sm_vals[threadIdx.x + 3840] = vF;
	__syncthreads();

	//merge-rebuild//
	uint32_t i;
	uint32_t rw = 256;
	laneId = threadIdx.x;
	level = k >> 1;
	for(uint32_t size = 2048; size > 256; size = size >> 1){//
		if ( threadIdx.x < rw ){

			i = threadIdx.x;
			v0 = sm_vals[i]; i += rw;
			v1 = sm_vals[i]; i += rw;
			v2 = sm_vals[i]; i += rw;
			v3 = sm_vals[i]; i += rw;
			v4 = sm_vals[i]; i += rw;
			v5 = sm_vals[i]; i += rw;
			v6 = sm_vals[i]; i += rw;
			v7 = sm_vals[i]; i += rw;
			v8 = sm_vals[i]; i += rw;
			v9 = sm_vals[i]; i += rw;
			vA = sm_vals[i]; i += rw;
			vB = sm_vals[i]; i += rw;
			vC = sm_vals[i]; i += rw;
			vD = sm_vals[i]; i += rw;
			vE = sm_vals[i]; i += rw;
			vF = sm_vals[i];

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
			i = threadIdx.x;
			sm_vals[i] = v0; i += rw;
			sm_vals[i] = v2; i += rw;
			sm_vals[i] = v4; i += rw;
			sm_vals[i] = v6; i += rw;
			sm_vals[i] = v8; i += rw;
			sm_vals[i] = vA; i += rw;
			sm_vals[i] = vC; i += rw;
			sm_vals[i] = vE;
		}
		__syncthreads();

		rw = rw >> 1;
		if (threadIdx.x < rw){
			i = threadIdx.x;
			v0 = sm_vals[i]; i += rw;
			v1 = sm_vals[i]; i += rw;
			v2 = sm_vals[i]; i += rw;
			v3 = sm_vals[i]; i += rw;
			v4 = sm_vals[i]; i += rw;
			v5 = sm_vals[i]; i += rw;
			v6 = sm_vals[i]; i += rw;
			v7 = sm_vals[i]; i += rw;
			v8 = sm_vals[i]; i += rw;
			v9 = sm_vals[i]; i += rw;
			vA = sm_vals[i]; i += rw;
			vB = sm_vals[i]; i += rw;
			vC = sm_vals[i]; i += rw;
			vD = sm_vals[i]; i += rw;
			vE = sm_vals[i]; i += rw;
			vF = sm_vals[i];

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

			i = threadIdx.x;
			sm_vals[i] = v0; i += rw;
			sm_vals[i] = v1; i += rw;
			sm_vals[i] = v2; i += rw;
			sm_vals[i] = v3; i += rw;
			sm_vals[i] = v4; i += rw;
			sm_vals[i] = v5; i += rw;
			sm_vals[i] = v6; i += rw;
			sm_vals[i] = v7; i += rw;
			sm_vals[i] = v8; i += rw;
			sm_vals[i] = v9; i += rw;
			sm_vals[i] = vA; i += rw;
			sm_vals[i] = vB; i += rw;
			sm_vals[i] = vC; i += rw;
			sm_vals[i] = vD; i += rw;
			sm_vals[i] = vE; i += rw;
			sm_vals[i] = vF;
		}
		__syncthreads();
	}

	if(threadIdx.x < 32){
		i = threadIdx.x;
		v0 = sm_vals[i]; i += 32;
		v1 = sm_vals[i]; i += 32;
		v2 = sm_vals[i]; i += 32;
		v3 = sm_vals[i]; i += 32;
		v4 = sm_vals[i]; i += 32;
		v5 = sm_vals[i]; i += 32;
		v6 = sm_vals[i]; i += 32;
		v7 = sm_vals[i]; i += 32;
		v8 = sm_vals[i]; i += 32;
		v9 = sm_vals[i]; i += 32;
		vA = sm_vals[i]; i += 32;
		vB = sm_vals[i]; i += 32;
		vC = sm_vals[i]; i += 32;
		vD = sm_vals[i]; i += 32;
		vE = sm_vals[i]; i += 32;
		vF = sm_vals[i];

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

		//256//
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

		//64//
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

		//32//
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
			v0 = swap(v0,step,dir);
			v8 = swap(v8,step,dir);
		}

		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
		v0 = (threadIdx.x & k) == 0 ? v0 : v8;


		//16//
		for( ; level < 32; level = level << 1){
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
			}
		}
		sm_vals[31 - threadIdx.x] = v0;//TODO:reverse
		//sm_vals[threadIdx.x] = v0;
	}
	__syncthreads();
	if (threadIdx.x < k){
		goffset = k * blockIdx.x + threadIdx.x;
		ovec[goffset] = sm_vals[threadIdx.x];
	}
}

/*
 * Rebuild and merge for larger than k=32
 */
template<class T, uint32_t bsize>
__global__ void mrebuild_geq_32(T *ivec, T *ovec, uint64_t k, uint32_t remainder){
	__shared__ T sm_vals[4096];
	uint32_t goffset;
	T v0 = -1.0, v1 = -1.0, v2 = -1.0, v3 = -1.0, v4 = -1.0, v5 = -1.0, v6 = -1.0, v7 = -1.0;
//	T v8 = -1.0, v9 = -1.0, vA = -1.0, vB = -1.0, vC = -1.0, vD = -1.0, vE = -1.0, vF = -1.0;
	T tt;

	goffset = (blockIdx.x << 12) + threadIdx.x;
	//check if loaded data are within range//
	if(goffset < remainder) sm_vals[threadIdx.x] = ivec[goffset];
	if(goffset + 256  < remainder) sm_vals[threadIdx.x + 256] = ivec[goffset + 256];
	if(goffset + 512  < remainder) sm_vals[threadIdx.x + 512] = ivec[goffset + 512];
	if(goffset + 768  < remainder) sm_vals[threadIdx.x + 768] = ivec[goffset + 768];
	if(goffset + 1024 < remainder) sm_vals[threadIdx.x + 1024] = ivec[goffset + 1024];
	if(goffset + 1280 < remainder) sm_vals[threadIdx.x + 1280] = ivec[goffset + 1280];
	if(goffset + 1536 < remainder) sm_vals[threadIdx.x + 1536] = ivec[goffset + 1536];
	if(goffset + 1792 < remainder) sm_vals[threadIdx.x + 1792] = ivec[goffset + 1792];
	if(goffset + 2048 < remainder) sm_vals[threadIdx.x + 2048] = ivec[goffset + 2048];
	if(goffset + 2304 < remainder) sm_vals[threadIdx.x + 2304] = ivec[goffset + 2304];
	if(goffset + 2560 < remainder) sm_vals[threadIdx.x + 2560] = ivec[goffset + 2560];
	if(goffset + 2816 < remainder) sm_vals[threadIdx.x + 2816] = ivec[goffset + 2816];
	if(goffset + 3072 < remainder) sm_vals[threadIdx.x + 3072] = ivec[goffset + 3072];
	if(goffset + 3328 < remainder) sm_vals[threadIdx.x + 3328] = ivec[goffset + 3328];
	if(goffset + 3584 < remainder) sm_vals[threadIdx.x + 3584] = ivec[goffset + 3584];
	if(goffset + 3840 < remainder) sm_vals[threadIdx.x + 3840] = ivec[goffset + 3840];
	__syncthreads();

	uint32_t low;
	uint32_t i;
	uint32_t loffset;
	uint32_t level = k >> 1;
	uint32_t laneId = threadIdx.x;
	uint32_t step;
	int dir = level << 1;
	uint32_t sz = 4096;
	uint32_t size = 2048;
	bool reverse, sp;
	for( size = 2048; size > (k>>1); size = size >> 1 ){
		//merge
		low = threadIdx.x & (k-1);
		i = (threadIdx.x << 1) - low;
		if( sz == 4096 )
		{
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v1 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v2 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v3 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v4 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v5 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v6 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v7 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}else if ( sz == 2048 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v1 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v2 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v3 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}else if ( sz == 1024 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(bsize << 1);
			v1 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}else if ( sz <= 512 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}
		__syncthreads();

		i = threadIdx.x;
		if( sz == 4096 )
		{
			sm_vals[i] = v0; i+=bsize;
			sm_vals[i] = v1; i+=bsize;
			sm_vals[i] = v2; i+=bsize;
			sm_vals[i] = v3; i+=bsize;
			sm_vals[i] = v4; i+=bsize;
			sm_vals[i] = v5; i+=bsize;
			sm_vals[i] = v6; i+=bsize;
			sm_vals[i] = v7;
		}else if ( sz == 2048 ){
			sm_vals[i] = v0; i+=bsize;
			sm_vals[i] = v1; i+=bsize;
			sm_vals[i] = v2; i+=bsize;
			sm_vals[i] = v3;
		}else if ( sz == 1024 ){
			sm_vals[i] = v0; i+=bsize;
			sm_vals[i] = v1;
		}else if ( sz <= 512 ){
			sm_vals[i] = v0;
		}
		__syncthreads();

		sz  = sz >> 1;
		//rebuild
		//Shared memory rebuild
		for(step = level; step > 16; step = step >> 1){
			low = threadIdx.x & (step - 1);
			i = (threadIdx.x << 1) - low;
			reverse = ((dir & i) == 0);

			for(loffset = 0; loffset < sz; loffset+=512){
				v0 = sm_vals[loffset+i];
				v1 = sm_vals[loffset+i+step];
				sp = reverse ^ (v0 < v1);
				tt = sp ? v1 : v0; v1 = sp ? v0 : v1; v0 = tt;
				sm_vals[loffset+i] = v0;
				sm_vals[loffset+i+step] = v1;
			}
			__syncthreads();
		}

		//register rebuild
		i = threadIdx.x;
		laneId = threadIdx.x;
		level = k >> 1;
		if ( sz == 2048 ){
			v0 = sm_vals[i]; i += bsize;
			v1 = sm_vals[i]; i += bsize;
			v2 = sm_vals[i]; i += bsize;
			v3 = sm_vals[i]; i += bsize;
			v4 = sm_vals[i]; i += bsize;
			v5 = sm_vals[i]; i += bsize;
			v6 = sm_vals[i]; i += bsize;
			v7 = sm_vals[i];

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

			i = threadIdx.x;
			sm_vals[i] = v0; i += bsize;
			sm_vals[i] = v1; i += bsize;
			sm_vals[i] = v2; i += bsize;
			sm_vals[i] = v3; i += bsize;
			sm_vals[i] = v4; i += bsize;
			sm_vals[i] = v5; i += bsize;
			sm_vals[i] = v6; i += bsize;
			sm_vals[i] = v7;
		}else if ( sz == 1024 ){
			v0 = sm_vals[i]; i += bsize;
			v1 = sm_vals[i]; i += bsize;
			v2 = sm_vals[i]; i += bsize;
			v3 = sm_vals[i]; i += bsize;

			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
				v1 = swap(v1,step,dir);
				v2 = swap(v2,step,dir);
				v3 = swap(v3,step,dir);
			}

			i = threadIdx.x;
			sm_vals[i] = v0; i += bsize;
			sm_vals[i] = v1; i += bsize;
			sm_vals[i] = v2; i += bsize;
			sm_vals[i] = v3;
		}else if ( sz == 512 ){
			v0 = sm_vals[i]; i += bsize;
			v1 = sm_vals[i];

			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
				v1 = swap(v1,step,dir);
			}

			i = threadIdx.x;
			sm_vals[i] = v0; i += bsize;
			sm_vals[i] = v1;
		}else if ( sz <= 256 ){
			v0 = sm_vals[i];

			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
			}

			i = threadIdx.x;
			sm_vals[i] = v0;
		}
		//if(threadIdx.x == 0 && blockIdx.x == 0) printf("{%d} \n",sz);
		__syncthreads();
	}

	if(threadIdx.x < k)
	{
		if( blockIdx.x & 0x1 == 1 ){
			goffset = k * blockIdx.x + (threadIdx.x^(k-1));
			ovec[goffset] = sm_vals[threadIdx.x];
		}else{
			goffset = k * blockIdx.x + (threadIdx.x);
			ovec[goffset] = sm_vals[threadIdx.x];
		}
	}

//	goffset = (blockIdx.x << 12) + threadIdx.x;
//	ovec[goffset]        = sm_vals[threadIdx.x];
//	ovec[goffset + 256]  = sm_vals[threadIdx.x + 256];
//	ovec[goffset + 512]  = sm_vals[threadIdx.x + 512];
//	ovec[goffset + 768]  = sm_vals[threadIdx.x + 768];
//	ovec[goffset + 1024] = sm_vals[threadIdx.x + 1024];
//	ovec[goffset + 1280] = sm_vals[threadIdx.x + 1280];
//	ovec[goffset + 1536] = sm_vals[threadIdx.x + 1536];
//	ovec[goffset + 1792] = sm_vals[threadIdx.x + 1792];
//	ovec[goffset + 2048] = sm_vals[threadIdx.x + 2048];
//	ovec[goffset + 2304] = sm_vals[threadIdx.x + 2304];
//	ovec[goffset + 2560] = sm_vals[threadIdx.x + 2560];
//	ovec[goffset + 2816] = sm_vals[threadIdx.x + 2816];
//	ovec[goffset + 3072] = sm_vals[threadIdx.x + 3072];
//	ovec[goffset + 3328] = sm_vals[threadIdx.x + 3328];
//	ovec[goffset + 3584] = sm_vals[threadIdx.x + 3584];
//	ovec[goffset + 3840] = sm_vals[threadIdx.x + 3840];
}

__global__ void zcopy (const float* __restrict__ src, float* __restrict__ dst, uint64_t len)
{
    uint64_t stride = gridDim.x * blockDim.x;
    uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (uint64_t i = tid; i < len; i += stride) {
        dst[i] = src[i];
    }
}

template<class T, class Z>
class BTA : public GAA<T,Z>{
	public:
		BTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "BTA";
		};

		~BTA()
		{
			if(this->g_ivec != NULL) cudaFree(this->g_ivec);
			if(this->g_ovec != NULL) cudaFree(this->g_ovec);
			if(this->c_ovec != NULL) cudaFree(this->c_ovec);
		};

		void alloc();
		void init();
		//void copy_query(T *weights, uint32_t *query);
		void findTopK(uint64_t k, uint64_t qq);

	private:
		void evaluate_full_relation(uint64_t k, uint64_t qq);
		void evaluate_single_attribute(uint64_t k, uint64_t qq);
		void bench_kernel(uint64_t k);

		T *g_ivec = NULL;
		T *g_ovec = NULL;
		T *c_ovec = NULL;
};

template<class T, class Z>
void BTA<T,Z>::alloc(){
	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");// Allocate cpu data memory
	cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*this->d,"gdata alloc");//Allocate gpu data memory

	cutil::safeMallocHost<T,uint64_t>(&(this->c_ovec),sizeof(T)*this->n,"c_ovec alloc");//Allocate cpu scores memory
	cutil::safeMalloc<T,uint64_t>(&(this->g_ivec),sizeof(T)*this->n,"g_ivec alloc");//Allocate gpu scores memory
	cutil::safeMalloc<T,uint64_t>(&(this->g_ovec),sizeof(T)*this->n,"g_ovec alloc");//Allocate gpu scores memory
}

template<class T, class Z>
void BTA<T,Z>::init()
{
	normalize_transpose<T>(this->cdata, this->n, this->d);
	cutil::safeCopyToDevice<T,uint64_t>(this->gdata,this->cdata,sizeof(T)*this->n*this->d, " copy from cdata to gdata ");//Copy data from cpu to gpu memory
}

//template<class T, class Z>
//void BTA<T,Z>::copy_query(T *weights, uint32_t *query){
//	this->weights = weights;
//	this->query = query;
//	cutil::cudaCheckErr(cudaMemcpyToSymbol(gpu_weights, weights, sizeof(T)*NUM_DIMS),"copy weights");//Initialize preference vector
//	cutil::cudaCheckErr(cudaMemcpyToSymbol(gpu_query, query, sizeof(uint32_t)*NUM_DIMS),"copy query");//Initialize query vector
//}

template<class T, class Z>
void BTA<T,Z>::evaluate_single_attribute(uint64_t k, uint64_t qq){
	uint32_t block_size = 512;
	dim3 lsort_block(block_size,1,1);
	dim3 lsort_grid((this->n-1)/BTA_TUPLES_PER_BLOCK + 1, 1, 1);

	this->t.start();
	aggregate<T,Z><<<lsort_grid,lsort_block>>>(this->gdata,this->n,qq,this->g_ovec);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing aggregate");
	this->t.lap("calculate and write");
	cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
	//for(uint32_t i = 0; i < this->n; i++){ this->c_ovec[i] = i; }//Linear increasing data
	this->cpu_threshold=find_threshold<T,Z>(this->c_ovec,this->n,k);
	//for(uint32_t i = 0; i < 32; i++){ std::cout << this->c_ovec[i] << std::endl; if((i+1)%k ==0 ){ std::cout << "-----" <<std::endl;}}
	//std::cout << "==================================" << std::endl;
	cutil::safeCopyToDevice<T,uint64_t>(this->g_ivec, this->c_ovec, sizeof(T)*this->n,"copy scores to device");
	cudaFree(this->gdata); this->gdata=NULL;

	//Test global memory bandwidth
	uint32_t iter2 = 10;
	uint32_t ZCOPY_THREADS = 512;
    dim3 dimBlock(ZCOPY_THREADS);
    uint32_t threadBlocks = (this->n-1)/4096 + 1;
    dim3 dimGrid(threadBlocks);

	zcopy<<<dimGrid,dimBlock>>>(this->g_ovec,this->g_ivec,this->n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing zcopy");
    double elapsed = 0;
    for(uint32_t i=0;i<iter2;i++){
    	this->t.start();
    	zcopy<<<dimGrid,dimBlock>>>(this->g_ovec,this->g_ivec,this->n);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing zcopy");
		elapsed += this->t.lap();
    }
	std::cout << "Avg Copy latency:" << elapsed/iter2 << std::endl;
	std::cout << "GB/s:" << (2*this->n*4/((elapsed/iter2)/1000))/(1024*1024*1024) << std::endl;
	/////////////////////////////////////////////////////////////////////////////////
	cutil::safeCopyToDevice<T,uint64_t>(this->g_ivec, this->c_ovec, sizeof(T)*this->n,"copy scores to device");

	//Topk
	uint32_t iter = 10;
	scores_topk<T><<<lsort_grid,lsort_block>>>(this->g_ivec, this->g_ovec, k);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing local_sort");
	this->t.start();
	for(uint32_t i=0;i<iter;i++){
		scores_topk<T><<<lsort_grid,lsort_block>>>(this->g_ivec, this->g_ovec, k);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing local_sort");
	}
	this->tt_processing = this->t.lap()/10;
	std::cout << "local_sort: " << this->tt_processing << std::endl;
	cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
	//find_threshold<T,Z>(this->c_ovec,this->n,k);
	//for(uint32_t i = 0; i < 32; i++){ std::cout << this->c_ovec[i] << std::endl; if((i+1)%k ==0 ){ std::cout << "-----" <<std::endl;}}

	if( k < 32 ){
		dim3 atm_16_block(256,1,1);
		dim3 atm_16_grid((this->n-1)/BTA_TUPLES_PER_BLOCK + 1, 1, 1);
		this->t.start();
		lsort_atm_16<T,256><<<atm_16_grid,atm_16_block>>>(this->g_ivec, this->g_ovec,k);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing lsort_atm_16");
		this->tt_processing = this->t.lap();
		cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
		std::cout << "lsort_atm_16: " << this->tt_processing << std::endl;
		std::cout << "lsort_atm_16 (GB/s):" << ((this->n * 4 + ((this->n/4096)*k))/(this->tt_processing/1000))/(1024*1024*1204) << std::endl;

		uint32_t remainder = ((this->n-1)/BTA_TUPLES_PER_BLOCK + 1) * k;
		find_partial_threshold<T,Z>(this->c_ovec,this->n,k,false,remainder);
		T *pswap;
		while(remainder > k){
			//std::cout << "remainder: " << remainder << std::endl;
			pswap = this->g_ivec;
			this->g_ivec = this->g_ovec;
			this->g_ovec = pswap;
			atm_16_grid.x = ((remainder - 1 ) / BTA_TUPLES_PER_BLOCK) + 1;
			this->t.start();
			mrebuild_atm_16<T,256><<<atm_16_grid,atm_16_block>>>(this->g_ivec, this->g_ovec, k, remainder);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing mrebuild_atm_16");
			this->tt_processing += this->t.lap();
			cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
			this->gpu_threshold = find_partial_threshold<T,Z>(this->c_ovec,this->n,k,false,remainder);
			remainder = ((remainder-1)/BTA_TUPLES_PER_BLOCK + 1) * k;
			std::cout << "remainder: {" << remainder << "}" << std::endl;
		}
		std::cout << "bta_atm_16: " << this->tt_processing << std::endl;
		std::cout << "bta_atm_16 (GB/s):" << ((this->n * 4 + ((this->n/4096)*k))/(this->tt_processing/1000))/(1024*1024*1204) << std::endl;
	}else{
		dim3 geq_32_block(256,1,1);
		dim3 geq_32_grid((this->n-1)/BTA_TUPLES_PER_BLOCK + 1, 1, 1);

		this->t.start();
		lsort_geq_32<T,256><<<geq_32_grid,geq_32_block>>>(this->g_ivec, this->g_ovec, k);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing lsort_leq_32");
		this->tt_processing = this->t.lap();
		cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
		std::cout << "lsort_geq_32: " << this->tt_processing << std::endl;
		std::cout << "lsort_geq_32 (GB/s):" << ((this->n * 4 + ((this->n/4096)*k))/(this->tt_processing/1000))/(1024*1024*1204) << std::endl;
		cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
		//this->gpu_threshold = find_remain_threshold<T,Z>(this->c_ovec,this->n,k,k);

		uint32_t remainder = ((this->n-1)/BTA_TUPLES_PER_BLOCK + 1) * k;
		//this->gpu_threshold = find_partial_threshold<T,Z>(this->c_ovec,this->n,k,false,remainder);
		T *pswap;
		while(remainder > k){
			std::cout << "remainder: {" << remainder << "}" << std::endl;
			pswap = this->g_ivec;
			this->g_ivec = this->g_ovec;
			this->g_ovec = pswap;
			geq_32_grid.x = ((remainder - 1 ) / BTA_TUPLES_PER_BLOCK) + 1;
			this->t.start();
			mrebuild_geq_32<T,256><<<geq_32_grid,geq_32_block>>>(this->g_ivec, this->g_ovec, k, remainder);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing mrebuild_geq_32");
			this->tt_processing += this->t.lap();
			cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
			//this->gpu_threshold = find_remain_threshold<T,Z>(this->c_ovec,remainder,k,k);
			this->gpu_threshold = find_partial_threshold<T,Z>(this->c_ovec,this->n,k,false,remainder);

//			uint32_t blocks = 0;
//			for(uint32_t i = 4096; i < remainder; i+=k*2)
//			{
//				for(uint32_t j = 0; j < k; j++){
//					std::cout << this->c_ovec[i+j] << " --- " << this->c_ovec[i+j+k] << std::endl;
//				}
//				bool s0 = std::is_sorted(&this->c_ovec[i],&this->c_ovec[i+k]);
//				bool s1 = std::is_sorted(&this->c_ovec[i+k],&this->c_ovec[i+k*2],std::greater<T>());
//				std::cout << "is_sorted: " << s0 << "," << s1 << "=" << blocks << std::endl;
//				blocks+=k*2;
//				//do{ std::cout << '\n' << "Press a key to continue..."; } while (std::cin.get() != '\n');
//			}
			remainder = ((remainder-1)/BTA_TUPLES_PER_BLOCK + 1) * k;
		}
		std::cout << "bta_geq_32: " << this->tt_processing << std::endl;
		std::cout << "bta_geq_32 (GB/s):" << ((this->n * 4 + ((this->n/4096)*k))/(this->tt_processing/1000))/(1024*1024*1204) << std::endl;
	}
}

template<class T, class Z>
void BTA<T,Z>::evaluate_full_relation(uint64_t k, uint64_t qq){
	uint32_t block_size = 512;
	dim3 lsort_block(block_size,1,1);
	dim3 lsort_grid((this->n-1)/BTA_TUPLES_PER_BLOCK + 1, 1, 1);
	aggregate<T,Z><<<lsort_grid,lsort_block>>>(this->gdata,this->n,qq,this->g_ivec);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing aggregate");
	cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ivec, sizeof(T)*this->n,"copy scores to host");
	srand(time(NULL));
	//for(uint32_t i = 0; i < this->n; i++){ this->c_ovec[i] = this->n - i; }//Linear increasing data
	//for(uint32_t i = 0; i < this->n; i++){ this->c_ovec[i] = rand()%1000000; }//Linear increasing data
	cutil::safeCopyToDevice<T,uint64_t>(this->g_ivec, this->c_ovec, sizeof(T)*this->n,"copy scores to device");
	this->cpu_threshold=find_threshold<T,Z>(this->c_ovec,this->n,k);

	if( k < 32 )
	{
		dim3 atm_16_block(256,1,1);
		dim3 atm_16_grid((this->n-1)/BTA_TUPLES_PER_BLOCK + 1, 1, 1);

		//agg_lsort_atm_16(T *ivec, T *ovec, uint64_t n, uint64_t d, uint64_t k){
		//std::cout << "qq: " << qq << std::endl;
		this->t.start();
		//agg_lsort_atm_16<T,256><<<atm_16_grid,atm_16_block>>>(this->gdata, this->g_ovec, this->n, qq, k);
		agg_lsort_atm_16<T,256><<<atm_16_grid,atm_16_block>>>(this->g_ivec, this->g_ovec, this->n, qq, k);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing agg_lsort_atm_16");
		this->tt_processing = this->t.lap();
		std::cout << "agg_lsort_atm_16: " << this->tt_processing << std::endl;
		std::cout << "agg_lsort_atm_16 (GB/s):" << ((this->n * qq * 4 + ((this->n/4096)*k))/(this->tt_processing/1000))/(1024*1024*1204) << std::endl;

		T *pswap;
		uint32_t remainder = ((this->n-1)/BTA_TUPLES_PER_BLOCK + 1) * k;
		while(remainder > k){
			//std::cout << "remainder: " << remainder << std::endl;
			pswap = this->g_ivec;
			this->g_ivec = this->g_ovec;
			this->g_ovec = pswap;
			atm_16_grid.x = ((remainder - 1 ) / BTA_TUPLES_PER_BLOCK) + 1;
			this->t.start();
			mrebuild_atm_16<T,256><<<atm_16_grid,atm_16_block>>>(this->g_ivec, this->g_ovec, k, remainder);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing mrebuild_atm_16");
			this->tt_processing += this->t.lap();
			cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
			this->gpu_threshold = find_partial_threshold<T,Z>(this->c_ovec,this->n,k,false,remainder);
			remainder = ((remainder-1)/BTA_TUPLES_PER_BLOCK + 1) * k;
			std::cout << "{" << remainder << "}" << std::endl;
		}
		std::cout << "bta_atm_16: " << this->tt_processing << std::endl;
		std::cout << "bta_atm_16 (GB/s):" << ((this->n * 4 + ((this->n/4096)*k))/(this->tt_processing/1000))/(1024*1024*1204) << std::endl;
	}else{
		dim3 geq_32_block(256,1,1);
		dim3 geq_32_grid((this->n-1)/BTA_TUPLES_PER_BLOCK + 1, 1, 1);

		this->t.start();
		lsort_geq_32<T,256><<<geq_32_grid,geq_32_block>>>(this->g_ivec, this->g_ovec, k);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing lsort_leq_32");
		this->tt_processing = this->t.lap();
		cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
		std::cout << "lsort_geq_32: " << this->tt_processing << std::endl;
		std::cout << "lsort_geq_32 (GB/s):" << ((this->n * 4 + ((this->n/4096)*k))/(this->tt_processing/1000))/(1024*1024*1204) << std::endl;

		T *pswap;
		uint32_t remainder = ((this->n-1)/BTA_TUPLES_PER_BLOCK + 1) * k;
		while(remainder > k){
			std::cout << "remainder: {" << remainder << "}" << std::endl;
			pswap = this->g_ivec;
			this->g_ivec = this->g_ovec;
			this->g_ovec = pswap;
			geq_32_grid.x = ((remainder - 1 ) / BTA_TUPLES_PER_BLOCK) + 1;
			this->t.start();
			mrebuild_geq_32<T,256><<<geq_32_grid,geq_32_block>>>(this->g_ivec, this->g_ovec, k, remainder);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing mrebuild_geq_32");
			this->tt_processing += this->t.lap();
			cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
			//this->gpu_threshold = find_remain_threshold<T,Z>(this->c_ovec,remainder,k,k);
			this->gpu_threshold = find_partial_threshold<T,Z>(this->c_ovec,this->n,k,false,remainder);
			remainder = ((remainder-1)/BTA_TUPLES_PER_BLOCK + 1) * k;
		}
	}
}

template<class T, class Z>
void BTA<T,Z>::findTopK(uint64_t k, uint64_t qq){
	//this->cpu_threshold = compute_threshold<T,Z>(this->cdata, this->n, qq, this->weights, this->query, k);
	//this->evaluate_single_attribute(k,qq);
	this->evaluate_full_relation(k,qq);
}


#endif
