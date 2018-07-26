#ifndef BTA_H
#define BTA_H
/*
 * Bitonic Top-k Aggregation
 */

#include "GAA.h"
#include "../tools/tools.h"
#define BTA_BLOCK_SIZE 512
#define BTA_TUPLES_PER_BLOCK 4096

template<class T, class Z>
__global__ void aggregate(T *gdata, uint64_t n, uint64_t d, uint64_t k, T *gscores){
	__shared__ T tuple_scores[BTA_TUPLES_PER_BLOCK];

	uint32_t tid = threadIdx.x;
	uint64_t goffset = blockIdx.x * BTA_TUPLES_PER_BLOCK + tid;//global starting offset
	for(uint64_t loffset = tid; loffset < BTA_TUPLES_PER_BLOCK; loffset+=blockDim.x){//increment local block offset
		T score = 0;
		for(uint64_t m = 0; m < d; m++){
			uint64_t ai = gpu_query[m];
			score+=gdata[ai*n + goffset] * gpu_weights[ai];//column-store load global relation access
			//score+=gdata[m*n + goffset];
		}
		tuple_scores[loffset] = score;//write scores in shared memory
		gscores[goffset] = tuple_scores[loffset];//write-back scores//optional//for debug purpose
		goffset+= blockDim.x;//increase global offset
	}
}

template<class T, class Z>
__global__ void local_sort2(T *gdata, uint64_t n, uint64_t d, uint64_t k, T *gscores){
	//__shared__ Z tuple_ids[BTA_TUPLES_PER_BLOCK];
	__shared__ T tuple_scores[BTA_TUPLES_PER_BLOCK];
	T *tps;

	//Aggregate scores//
	uint64_t goffset = blockIdx.x * BTA_TUPLES_PER_BLOCK + threadIdx.x;
	for(uint64_t loffset = threadIdx.x; loffset < BTA_TUPLES_PER_BLOCK; loffset+=blockDim.x){
		T score = 0;
		for(uint64_t m = 0; m < d; m++){
			uint64_t ai = gpu_query[m];
			score+=gdata[ai*n + goffset] * gpu_weights[ai];
			//score+=gdata[m*n + goffset];
		}
		tuple_scores[loffset] = score;
		gscores[goffset] = tuple_scores[loffset];//Write-back scores//
		goffset+= blockDim.x;
	}
	__syncthreads();

	//Local sort scores//
	for(uint32_t chunk = 0; chunk < BTA_TUPLES_PER_BLOCK; chunk+=(blockDim.x<<1)){
		tps = &tuple_scores[chunk];
		for(uint32_t level = 1; level < k; level = level << 1){
			for(uint32_t s = level; s > 0; s = s >> 1){
				uint32_t left = (s << 1) * (threadIdx.x/s) + (threadIdx.x&(s-1));
				uint32_t right = (left ^ s);

				bool reverse = ((threadIdx.x & level) == 0);
				T v0 = reverse ? fmaxf(tps[left],tps[right]) : fminf(tps[left],tps[right]);
				T v1 = reverse ? fminf(tps[left],tps[right]) : fmaxf(tps[left],tps[right]);

				tps[left] = v0;
				tps[right] = v1;
				__syncthreads();
			}
		}
	}
	__syncthreads();
	//Write-back//
	goffset = blockIdx.x * BTA_TUPLES_PER_BLOCK + threadIdx.x;
	for(uint64_t loffset = threadIdx.x; loffset < BTA_TUPLES_PER_BLOCK; loffset+=blockDim.x){
		gscores[goffset] = tuple_scores[loffset];//Write-back scores//
		goffset+= blockDim.x;
	}
}

template<class T, class Z>
__global__ void local_sort3(T *gdata, uint64_t n, uint64_t d, uint64_t k, T *gscores){
	//__shared__ Z tuple_ids[BTA_TUPLES_PER_BLOCK];
	__shared__ T tuple_scores[BTA_TUPLES_PER_BLOCK];
	T *tps;

	//Aggregate Scores//
	uint64_t goffset = blockIdx.x * BTA_TUPLES_PER_BLOCK + threadIdx.x;
	for(uint64_t loffset = threadIdx.x; loffset < BTA_TUPLES_PER_BLOCK; loffset+=blockDim.x){
		T score = 0;
		for(uint64_t m = 0; m < d; m++){
			uint64_t ai = gpu_query[m];
			score+=gdata[ai*n + goffset] * gpu_weights[ai];
			//score+=gdata[m*n + goffset];
		}
		tuple_scores[loffset] = score;
		//gscores[goffset] = tuple_scores[loffset];//Write-back scores//
		goffset+= blockDim.x;
	}
	__syncthreads();

	//Local-sort//
	for(uint32_t chunk = 0; chunk < BTA_TUPLES_PER_BLOCK; chunk+=(blockDim.x<<1)){
		tps = &tuple_scores[chunk];
		for(uint32_t level = 1; level < k; level = level << 1){
			for(uint32_t s = level; s > 0; s = s >> 1){
				uint32_t left = (s << 1) * (threadIdx.x/s) + (threadIdx.x&(s-1));
				uint32_t right = (left ^ s);

				bool reverse = ((threadIdx.x & level) == 0);
				T v0 = reverse ? fmaxf(tps[left],tps[right]) : fminf(tps[left],tps[right]);
				T v1 = reverse ? fminf(tps[left],tps[right]) : fmaxf(tps[left],tps[right]);

				tps[left] = v0;
				tps[right] = v1;
				__syncthreads();
			}
		}
	}
	__syncthreads();

	//Merge-sort sequences//
	uint32_t gid = (threadIdx.x / k);//group id
	uint32_t gcount = (blockDim.x / k);//group count
	uint32_t loffset = (threadIdx.x & (k-1));//loffset
	uint32_t stride = ( k << 1 ) * gcount;
	uint32_t low = (k << 1) * gid + loffset;
	//uint32_t count = 0;//debug//
	for(uint32_t m = BTA_TUPLES_PER_BLOCK; m > k; m = m >> 1){
		T max_v = 0;

		for( uint32_t chunk = low; chunk < BTA_TUPLES_PER_BLOCK; chunk+=stride ){
			if (chunk + k < m){
				max_v = fmaxf(tuple_scores[chunk],tuple_scores[chunk + k]);
			}
			__syncthreads();
			if (chunk < m){
				tuple_scores[chunk - (k * gid)] = max_v;
			}
			gid+=gcount;
			__syncthreads();
		}
		gid = (threadIdx.x / k);

		for(uint32_t chunk = 0; chunk < (m >> 1); chunk+=(blockDim.x<<1)){
			tps = &tuple_scores[chunk];
			for(uint32_t level = 1; level < k; level = level << 1){
				for(uint32_t s = level; s > 0; s = s >> 1){
					uint32_t left = (s << 1) * (threadIdx.x/s) + (threadIdx.x&(s-1));
					uint32_t right = (left ^ s);

					bool reverse = ((threadIdx.x & level) == 0);
					T v0 = reverse ? fmaxf(tps[left],tps[right]) : fminf(tps[left],tps[right]);
					T v1 = reverse ? fminf(tps[left],tps[right]) : fmaxf(tps[left],tps[right]);

					tps[left] = v0;
					tps[right] = v1;
					__syncthreads();
				}
			}
		}
		__syncthreads();
		//if(count >= 2) break;
		//break;
	}
	__syncthreads();

	goffset = blockIdx.x * BTA_TUPLES_PER_BLOCK + threadIdx.x;
	//Write k results per block//
	for(uint64_t loffset = threadIdx.x; loffset < k; loffset+=blockDim.x){
		gscores[goffset] = tuple_scores[loffset];//Write-back scores//
		goffset+= blockDim.x;
	}
}

template<class T, class Z>
__global__ void local_sort4(T *gdata, uint64_t n, uint64_t d, uint64_t k, T *gscores){
	//__shared__ Z tuple_ids[BTA_TUPLES_PER_BLOCK];
	__shared__ T tuple_scores[BTA_TUPLES_PER_BLOCK];
	T *tps;

	//Aggregate Scores//
	uint64_t goffset = blockIdx.x * BTA_TUPLES_PER_BLOCK + threadIdx.x;
	for(uint64_t loffset = threadIdx.x; loffset < BTA_TUPLES_PER_BLOCK; loffset+=blockDim.x){
		T score = 0;
		for(uint64_t m = 0; m < d; m++){
			uint64_t ai = gpu_query[m];
			score+=gdata[ai*n + goffset] * gpu_weights[ai];
		}
		tuple_scores[loffset] = score;
		goffset+= blockDim.x;
	}
	__syncthreads();

	//Local-sort//
	for(uint32_t chunk = 0; chunk < BTA_TUPLES_PER_BLOCK; chunk+=(blockDim.x<<1)){
		tps = &tuple_scores[chunk];
		for(uint32_t level = 1; level < k; level = level << 1){
			for(uint32_t s = level; s > 0; s = s >> 1){
				uint32_t left = (s << 1) * (threadIdx.x/s) + (threadIdx.x&(s-1));
				uint32_t right = (left ^ s);

				bool reverse = ((threadIdx.x & level) == 0);
				T v0 = reverse ? fmaxf(tps[left],tps[right]) : fminf(tps[left],tps[right]);
				T v1 = reverse ? fminf(tps[left],tps[right]) : fmaxf(tps[left],tps[right]);

				tps[left] = v0;
				tps[right] = v1;
				__syncthreads();
			}
		}
	}
	__syncthreads();

	//Merge-sort sequences//
	uint32_t gid = (threadIdx.x / k);//group id
	uint32_t gcount = (blockDim.x / k);//group count
	uint32_t loffset = (threadIdx.x & (k-1));//loffset
	uint32_t stride = ( k << 1 ) * gcount;
	uint32_t low = (k << 1) * gid + loffset;
	for(uint32_t m = BTA_TUPLES_PER_BLOCK; m > k; m = m >> 1){
		T max_v = 0;

		for( uint32_t chunk = low; chunk < BTA_TUPLES_PER_BLOCK; chunk+=stride ){
			if (chunk < m){
				max_v = fmaxf(tuple_scores[chunk],tuple_scores[chunk + k]);
			}
			__syncthreads();
			if (chunk < m){
				tuple_scores[chunk - (k * gid)] = max_v;
			}
			gid+=gcount;
			__syncthreads();
		}
		gid = (threadIdx.x / k);

		for(uint32_t chunk = 0; chunk < (m >> 1); chunk+=(blockDim.x<<1)){
			tps = &tuple_scores[chunk];
			for(uint32_t level = 1; level < k; level = level << 1){
				for(uint32_t s = level; s > 0; s = s >> 1){
					uint32_t left = (s << 1) * (threadIdx.x/s) + (threadIdx.x&(s-1));
					uint32_t right = (left ^ s);

					bool reverse = ((threadIdx.x & level) == 0);
					T v0 = reverse ? fmaxf(tps[left],tps[right]) : fminf(tps[left],tps[right]);
					T v1 = reverse ? fminf(tps[left],tps[right]) : fmaxf(tps[left],tps[right]);

					tps[left] = v0;
					tps[right] = v1;
					__syncthreads();
				}
			}
		}
		__syncthreads();
	}
	__syncthreads();

//	goffset = blockIdx.x * k + threadIdx.x;
//	//Write k results per block//
//	for(uint64_t loffset = threadIdx.x; loffset < k; loffset+=blockDim.x){
//		gscores[goffset] = tuple_scores[loffset&(k-1)];//Write-back scores//
//		goffset+= blockDim.x;
//	}
	if ((blockIdx.x & 0x1) == 0){//Write reverse
		goffset = blockIdx.x * k + threadIdx.x;
		for(uint64_t loffset = threadIdx.x; loffset < k; loffset+=blockDim.x){
			gscores[goffset] = tuple_scores[(loffset^(k-1))];//Write-back scores//
			goffset+= blockDim.x;
		}
	}else{//Write regular order
		goffset = blockIdx.x * k + threadIdx.x;
		for(uint64_t loffset = threadIdx.x; loffset < k; loffset+=blockDim.x){
			gscores[goffset] = tuple_scores[loffset];//Write-back scores//
			goffset+= blockDim.x;
		}
	}
}

template<class T, class Z>
__global__ void merge(T *gscores, uint64_t k, uint32_t remainder){
	uint64_t goffset = blockIdx.x * BTA_TUPLES_PER_BLOCK + threadIdx.x;
	//__shared__ Z tuple_ids[BTA_TUPLES_PER_BLOCK];
	__shared__ T tuple_scores[BTA_TUPLES_PER_BLOCK];

	uint32_t size = remainder < BTA_TUPLES_PER_BLOCK ? remainder : BTA_TUPLES_PER_BLOCK;
	//Load scores and ids from
	for(uint64_t loffset = threadIdx.x; loffset < size; loffset+=blockDim.x){
		//tuple_ids[loffset] = gscores[goffset];
		tuple_scores[loffset] = gscores[goffset];
		goffset+= blockDim.x;
	}
	__syncthreads();

	//Merge-sort sequences//
	uint32_t gid = (threadIdx.x / k);//group id
	uint32_t gcount = (blockDim.x / k);//group count
	uint32_t loffset = (threadIdx.x & (k-1));//loffset
	uint32_t stride = ( k << 1 ) * gcount;
	uint32_t low = (k << 1) * gid + loffset;
	T *tps;
	for(uint32_t m = size; m > k; m = m >> 1){
		T max_v = 0;

		for( uint32_t chunk = low; chunk < size; chunk+=stride ){
			if (chunk < m){
				max_v = fmaxf(tuple_scores[chunk],tuple_scores[chunk + k]);
			}
			__syncthreads();
			if (chunk < m){
				tuple_scores[chunk - (k * gid)] = max_v;
			}
			gid+=gcount;
			__syncthreads();
		}
		gid = (threadIdx.x / k);

		for(uint32_t chunk = 0; chunk < (m >> 1); chunk+=(blockDim.x<<1)){
			tps = &tuple_scores[chunk];
			for(uint32_t level = 1; level < k; level = level << 1){
				for(uint32_t s = level; s > 0; s = s >> 1){
					uint32_t left = (s << 1) * (threadIdx.x/s) + (threadIdx.x&(s-1));
					uint32_t right = (left ^ s);

					bool reverse = ((threadIdx.x & level) == 0);
					T v0 = reverse ? fmaxf(tps[left],tps[right]) : fminf(tps[left],tps[right]);
					T v1 = reverse ? fminf(tps[left],tps[right]) : fmaxf(tps[left],tps[right]);

					tps[left] = v0;
					tps[right] = v1;
					__syncthreads();
				}
			}
		}
		__syncthreads();
		//if(count >= 2) break;
		//break;
	}
	__syncthreads();

//	goffset = blockIdx.x * k + threadIdx.x;
//	//Write k results per block//
//	for(uint64_t loffset = threadIdx.x; loffset < k; loffset+=blockDim.x){
//		gscores[goffset] = tuple_scores[loffset];//Write-back scores//
//		goffset+= blockDim.x;
//	}

	if ((blockIdx.x & 0x1) == 0){//Write reverse
		goffset = blockIdx.x * k + threadIdx.x;
		for(uint64_t loffset = threadIdx.x; loffset < k; loffset+=blockDim.x){
			gscores[goffset] = tuple_scores[(loffset^(k-1))];//Write-back scores//
			goffset+= blockDim.x;
		}
	}else{//Write regular order
		goffset = blockIdx.x * k + threadIdx.x;
		for(uint64_t loffset = threadIdx.x; loffset < k; loffset+=blockDim.x){
			gscores[goffset] = tuple_scores[loffset];//Write-back scores//
			goffset+= blockDim.x;
		}
	}
}

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
__global__ void scores_topk2(T *ivec, T *ovec, uint64_t k){
	__shared__ T sm_vals[4096];
	uint64_t goffset, loffset;
	uint32_t level, dir, step;
	int low;
	int i;
	bool reverse, swap;
	T v0,v1;
	T tmp;

	goffset = (blockIdx.x << 12) + threadIdx.x;
	for(loffset = threadIdx.x; loffset < 4096; loffset+=blockDim.x){
		sm_vals[loffset] = ivec[goffset];
		goffset+= blockDim.x;
	}
	__syncthreads();

	for(level = 1; level < k; level = level << 1){
		dir = level << 1;
		for(step = level; step > 0; step = step >> 1){
			low = threadIdx.x & (step - 1);
			i = (threadIdx.x << 1) - low;
			reverse = ((dir & i) == 0);
			for(loffset = 0; loffset < 4096; loffset+=(blockDim.x << 1)){
				v0 = sm_vals[loffset+i];
				v1 = sm_vals[loffset+i+step];
				swap = reverse ^ (v0 < v1);
				tmp = swap ? v0 : v1;
				v1 = swap ? v1 : v0;
				v0 = tmp;
				sm_vals[loffset+i] = v0;
				sm_vals[loffset+i+step] = v1;
			}
			__syncthreads();
		}
	}
	low = threadIdx.x & (k-1);
	i = (threadIdx.x << 1) - low;
	for(loffset = 0; loffset < 4096; loffset+=(blockDim.x << 1)){
		sm_vals[loffset+i] =  fmaxf(sm_vals[loffset+i],sm_vals[loffset+i+k]);
	}
	__syncthreads();

	goffset = (blockIdx.x << 12) + threadIdx.x;
	for(loffset = threadIdx.x; loffset < 4096; loffset+=blockDim.x){
		ovec[goffset] = sm_vals[loffset];//Write-back scores//
		goffset+= blockDim.x;
	}
}

template<class T>
__global__ void scores_topk3(T *ivec, T *ovec, uint64_t k){
	__shared__ T sm_vals[4096];
	uint64_t goffset, loffset, size;
	uint32_t level, dir, step;
	int low;
	int i;
	bool reverse, swap;
	T v0,v1,v2,v3;
	T tmp;

	goffset = (blockIdx.x << 12) + threadIdx.x;
	for(loffset = threadIdx.x; loffset < 4096; loffset+=blockDim.x){
		sm_vals[loffset] = ivec[goffset];
		goffset+= blockDim.x;
	}
	__syncthreads();

	for(level = 1; level < k; level = level << 1){
		dir = level << 1;
		for(step = level; step > 0; step = step >> 1){
			low = threadIdx.x & (step - 1);
			i = (threadIdx.x << 1) - low;
			reverse = ((dir & i) == 0);
			for(loffset = 0; loffset < 4096; loffset+=(blockDim.x << 1)){
				v0 = sm_vals[loffset+i];
				v1 = sm_vals[loffset+i+step];
				swap = reverse ^ (v0 < v1);
				tmp = swap ? v0 : v1;
				v1 = swap ? v1 : v0;
				v0 = tmp;
				sm_vals[loffset+i] = v0;
				sm_vals[loffset+i+step] = v1;
			}
			__syncthreads();
		}
	}

	low = threadIdx.x & (k-1);
	i = (threadIdx.x << 1) - low;
	for(size = 2048; size > k; size = size >> 1){
		//merge//
		if(size == 2048){
			v0 =  fmaxf(sm_vals[i],sm_vals[i+k]);
			v1 =  fmaxf(sm_vals[1024+i],sm_vals[1024+i+k]);
			v2 =  fmaxf(sm_vals[2048+i],sm_vals[2048+i+k]);
			v3 =  fmaxf(sm_vals[3072+i],sm_vals[3072+i+k]);
		}

		//rebuild//
		break;
	}
	__syncthreads();

	goffset = (blockIdx.x << 12) + threadIdx.x;
	for(loffset = threadIdx.x; loffset < 4096; loffset+=blockDim.x){
		ovec[goffset] = sm_vals[loffset];//Write-back scores//
		goffset+= blockDim.x;
	}
}

template<class T, uint32_t bsize>
__global__ void scores_topk4(T *ivec, T *ovec, uint64_t k){
	__shared__ T sm_vals[4096];
	uint64_t goffset,loffset, size;
	uint32_t level, dir, step;
	int low;
	int i;
	bool reverse, swap;
	T v0, v1, v2, v3, v4, v5, v6, v7;
	T tmp;

	//local-sort begins//
	goffset = (blockIdx.x << 12) + threadIdx.x;
	v0 = ivec[goffset];
	v1 = ivec[goffset + 512];
	v2 = ivec[goffset + 1024];
	v3 = ivec[goffset + 1536];
	v4 = ivec[goffset + 2048];
	v5 = ivec[goffset + 2560];
	v6 = ivec[goffset + 3072];
	v7 = ivec[goffset + 3584];

	if( k >= 4 ){
		tmp = fmax(v0,v1); v1 = fmin(v0,v1); v0 = tmp;//1
		tmp = fmax(v2,v3); v3 = fmin(v2,v3); v2 = tmp;
		tmp = fmax(v0,v2); v2 = fmin(v0,v2); v0 = tmp;//2
		tmp = fmax(v1,v3); v3 = fmin(v1,v3); v1 = tmp;
		tmp = fmax(v1,v2); v2 = fmin(v1,v2); v1 = tmp;//3

		tmp = fmin(v4,v5); v5 = fmax(v4,v5); v4 = tmp;//1
		tmp = fmin(v6,v7); v7 = fmax(v6,v7); v6 = tmp;
		tmp = fmin(v4,v6); v6 = fmax(v4,v6); v4 = tmp;//2
		tmp = fmin(v5,v7); v7 = fmax(v5,v7); v5 = tmp;
		tmp = fmin(v5,v6); v6 = fmax(v5,v6); v5 = tmp;//3

		loffset = threadIdx.x << 3;
		sm_vals[loffset] = v0;
		sm_vals[loffset + 1] = v1;
		sm_vals[loffset + 2] = v2;
		sm_vals[loffset + 3] = v3;
		sm_vals[loffset + 4] = v4;
		sm_vals[loffset + 5] = v5;
		sm_vals[loffset + 6] = v6;
		sm_vals[loffset + 7] = v7;
		__syncthreads();
	}
	if( k >= 8 ){
		for(level = 4; level < k; level = level << 1){
			dir = level << 1;
			for(step = level; step > 0; step = step >> 1){
				low = threadIdx.x & (step - 1);
				i = (threadIdx.x << 1) - low;
				reverse = ((dir & i) == 0);
				for(loffset = 0; loffset < 4096; loffset+=(blockDim.x << 1)){
					v0 = sm_vals[loffset+i];
					v1 = sm_vals[loffset+i+step];
					swap = reverse ^ (v0 < v1);
					tmp = swap ? v0 : v1;
					v1 = swap ? v1 : v0;
					v0 = tmp;
					sm_vals[loffset+i] = v0;
					sm_vals[loffset+i+step] = v1;
				}
				__syncthreads();
			}
		}
	}
	//local-sort ends//

	//merge rebuild//
//	for(size = 4096; size > k; size = size >> 1){
//		if(threadIdx.x < rw){
//			low = threadIdx.x & (k-1);
//			i = (threadIdx.x << 1) - low;
//
//			if(rw >= 256){
//				v0 = fmaxf(sm_vals[i],sm_vals[i+k]);
//				v1 = fmaxf(sm_vals[512+i],sm_vals[512+i+k]);
//				v2 = fmaxf(sm_vals[1024+i],sm_vals[1024+i+k]);
//				v3 = fmaxf(sm_vals[1560+i],sm_vals[1560+i+k]);
//
//				v4 = fmaxf(sm_vals[2048+i],sm_vals[2048+i+k]);
//				v5 = fmaxf(sm_vals[2560+i],sm_vals[2560+i+k]);
//				v6 = fmaxf(sm_vals[3072+i],sm_vals[3072+i+k]);
//				v7 = fmaxf(sm_vals[3584+i],sm_vals[3584+i+k]);
//			}
//			__
//			if(rw >=256)
//		}
//	}

	for(size = 4096; size > k; size = size >> 1){
		//break;
		//rebuild-start
		low = threadIdx.x & (k-1);
		i = (threadIdx.x << 1) - low;
		if ( size >= 4096 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]);
			v1 = fmaxf(sm_vals[1024+i],sm_vals[1024+i+k]);
			v2 = fmaxf(sm_vals[2048+i],sm_vals[2048+i+k]);
			v3 = fmaxf(sm_vals[3072+i],sm_vals[3072+i+k]);
		}else if( size >= 2048 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]);
			v1 = fmaxf(sm_vals[1024+i],sm_vals[1024+i+k]);
		}else if( size >= 1024 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}else if( threadIdx.x < (size >> 1)){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}
		__syncthreads();

		if( size >= 4096 ){
			sm_vals[threadIdx.x] = v0;
			sm_vals[threadIdx.x + 512] = v1;
			sm_vals[threadIdx.x + 1024] = v2;
			sm_vals[threadIdx.x + 1536] = v3;
		}else if( size >= 2048 ){
			sm_vals[threadIdx.x] = v0;
			sm_vals[threadIdx.x + 512] = v1;
		}else if( size >= 1024 ){
			sm_vals[threadIdx.x] = v0;
		}else if( threadIdx.x < (size >> 1) ){
			sm_vals[threadIdx.x] = v0;
		}
		//if(threadIdx.x == 0 && blockIdx.x == 0) printf("<%d\n",size);
		__syncthreads();
		//rebuild-end

		level = k >> 1;
		dir = level << 1;
		for(step = level; step > 0; step = step >> 1){
			low = threadIdx.x & (step - 1);
			i = (threadIdx.x << 1) - low;
			reverse = ((dir & i) == 0);

			for(loffset = i; loffset < (size >> 1); loffset+=1024){
				v0 = sm_vals[loffset];
				v1 = sm_vals[loffset+step];
				swap = reverse ^ (v0 < v1);
				tmp = swap ? v0 : v1;
				v1 = swap ? v1 : v0;
				v0 = tmp;
				sm_vals[loffset] = v0;
				sm_vals[loffset+step] = v1;
			}
			__syncthreads();
		}

		//if((size >> 1) == 1024) break;
	}

	//////////////
	//Write-back//
	ovec[goffset] = sm_vals[threadIdx.x];
	//ovec[goffset + 512] = sm_vals[threadIdx.x + 512];
//	ovec[goffset + 1024] = sm_vals[threadIdx.x + 1024];
//	ovec[goffset + 1560] = sm_vals[threadIdx.x + 1560];
//	if( k == 1024 ){
//		ovec[goffset] = sm_vals[threadIdx.x];
//		ovec[goffset + 512] = sm_vals[threadIdx.x + 512];
//	}else if(threadIdx.x <  k){
//		ovec[goffset] = sm_vals[threadIdx.x];
//	}
}

template<class T, uint32_t bsize>
__global__ void scores_topk5(T *ivec, T *ovec, uint64_t k){
	__shared__ T sm_vals[4096];
	uint32_t goffset,loffset, size;
	uint32_t level, dir, step;
	int low;
	int i;
	bool reverse, swap;
	T v0, v1, v2, v3, v4, v5, v6, v7;
	T tmp;
	if(threadIdx.x == 0 && blockIdx.x == 0) printf("scores_topk5\n");
	//local-sort begins//
	goffset = (blockIdx.x << 12) + threadIdx.x;
	v0 = ivec[goffset];
	v1 = ivec[goffset + 512];
	v2 = ivec[goffset + 1024];
	v3 = ivec[goffset + 1536];
	v4 = ivec[goffset + 2048];
	v5 = ivec[goffset + 2560];
	v6 = ivec[goffset + 3072];
	v7 = ivec[goffset + 3584];

	if( k >= 4 ){
		tmp = fmax(v0,v1); v1 = fmin(v0,v1); v0 = tmp;//1
		tmp = fmax(v2,v3); v3 = fmin(v2,v3); v2 = tmp;
		tmp = fmax(v0,v2); v2 = fmin(v0,v2); v0 = tmp;//2
		tmp = fmax(v1,v3); v3 = fmin(v1,v3); v1 = tmp;
		tmp = fmax(v1,v2); v2 = fmin(v1,v2); v1 = tmp;//3

		tmp = fmin(v4,v5); v5 = fmax(v4,v5); v4 = tmp;//1
		tmp = fmin(v6,v7); v7 = fmax(v6,v7); v6 = tmp;
		tmp = fmin(v4,v6); v6 = fmax(v4,v6); v4 = tmp;//2
		tmp = fmin(v5,v7); v7 = fmax(v5,v7); v5 = tmp;
		tmp = fmin(v5,v6); v6 = fmax(v5,v6); v5 = tmp;//3

		loffset = threadIdx.x << 3;
		sm_vals[loffset] = v0;
		sm_vals[loffset + 1] = v1;
		sm_vals[loffset + 2] = v2;
		sm_vals[loffset + 3] = v3;
		sm_vals[loffset + 4] = v4;
		sm_vals[loffset + 5] = v5;
		sm_vals[loffset + 6] = v6;
		sm_vals[loffset + 7] = v7;
		__syncthreads();
	}
	if( k >= 8 ){
		for(level = 4; level < k; level = level << 1){
			dir = level << 1;
			for(step = level; step > 0; step = step >> 1){
				if( level > 4 ){
					low = threadIdx.x & (step - 1);
					i = (threadIdx.x << 1) - low;
					loffset = i + step;
					reverse = ((dir & i) == 0);

					v0 = sm_vals[i];
					v1 = sm_vals[loffset];
					v2 = sm_vals[i+1024];
					v3 = sm_vals[loffset+1024];
					v4 = sm_vals[i+2048];
					v5 = sm_vals[loffset+2048];
					v6 = sm_vals[i+3072];
					v7 = sm_vals[loffset+3072];

					tmp = reverse ? fmaxf(v0,v1) : fminf(v0,v1);
					v0 = reverse ? fminf(v0,v1) : fmaxf(v0,v1);
					v1 = tmp;

//					swap = reverse ^ (v0 < v1);
//					tmp = swap ? v0 : v1;
//					v1 = swap ? v1 : v0;
//					v0 = tmp;

					tmp = reverse ? fmaxf(v2,v3) : fminf(v2,v3);
					v2 = reverse ? fminf(v2,v3) : fmaxf(v2,v3);
					v3 = tmp;

//					swap = reverse ^ (v2 < v3);
//					tmp = swap ? v2 : v3;
//					v2 = swap ? v3 : v2;
//					v3 = tmp;

					tmp = reverse ? fmaxf(v4,v5) : fminf(v4,v5);
					v4 = reverse ? fminf(v4,v5) : fmaxf(v4,v5);
					v5 = tmp;

//					swap = reverse ^ (v4 < v5);
//					tmp = swap ? v4 : v5;
//					v4 = swap ? v5 : v4;
//					v5 = tmp;

					tmp = reverse ? fmaxf(v6,v7) : fminf(v6,v7);
					v6 = reverse ? fminf(v6,v7) : fmaxf(v6,v7);
					v7 = tmp;

//					swap = reverse ^ (v6 < v7);
//					tmp = swap ? v6 : v7;
//					v6 = swap ? v7 : v6;
//					v7 = tmp;

					sm_vals[i] = v0;
					sm_vals[loffset] = v1;
					sm_vals[i+1024] = v2;
					sm_vals[loffset+1024] = v3;
					sm_vals[i+2048] = v4;
					sm_vals[loffset+2048] = v5;
					sm_vals[i+3072] = v6;
					sm_vals[loffset+3072] = v7;
				}else{
					loffset = threadIdx.x << 3;
					reverse = ((dir & loffset) == 0);
					v0 = sm_vals[loffset];
					v1 = sm_vals[loffset + 1];
					v2 = sm_vals[loffset + 2];
					v3 = sm_vals[loffset + 3];
					v4 = sm_vals[loffset + 4];
					v5 = sm_vals[loffset + 5];
					v6 = sm_vals[loffset + 6];
					v7 = sm_vals[loffset + 7];

					swap = reverse ^ (v0 < v4);
					tmp = swap ? v0 : v4; v4 = swap ? v4 : v0; v0 = tmp;
					swap = reverse ^ (v1 < v5);
					tmp = swap ? v1 : v5; v5 = swap ? v5 : v1; v1 = tmp;
					swap = reverse ^ (v2 < v6);
					tmp = swap ? v2 : v6; v6 = swap ? v6 : v2; v2 = tmp;
					swap = reverse ^ (v3 < v7);
					tmp = swap ? v3 : v7; v7 = swap ? v7 : v3; v3 = tmp;

					swap = reverse ^ (v0 < v2);
					tmp = swap ? v0 : v2; v2 = swap ? v2 : v0; v0 = tmp;
					swap = reverse ^ (v1 < v3);
					tmp = swap ? v1 : v3; v3 = swap ? v3 : v1; v1 = tmp;
					swap = reverse ^ (v4 < v6);
					tmp = swap ? v4 : v6; v6 = swap ? v6 : v4; v4 = tmp;
					swap = reverse ^ (v5 < v7);
					tmp = swap ? v5 : v7; v7 = swap ? v7 : v5; v5 = tmp;

					swap = reverse ^ (v0 < v1);
					tmp = swap ? v0 : v1; v1 = swap ? v1 : v0; v0 = tmp;
					swap = reverse ^ (v2 < v3);
					tmp = swap ? v2 : v3; v2 = swap ? v3 : v2; v3 = tmp;
					swap = reverse ^ (v4 < v5);
					tmp = swap ? v4 : v5; v4 = swap ? v5 : v4; v5 = tmp;
					swap = reverse ^ (v6 < v7);
					tmp = swap ? v6 : v7; v6 = swap ? v7 : v6; v7 = tmp;

					sm_vals[loffset] = v0;
					sm_vals[loffset + 1] = v1;
					sm_vals[loffset + 2] = v2;
					sm_vals[loffset + 3] = v3;
					sm_vals[loffset + 4] = v4;
					sm_vals[loffset + 5] = v5;
					sm_vals[loffset + 6] = v6;
					sm_vals[loffset + 7] = v7;
				}
				__syncthreads();
			}

		}
	}
	//local-sort ends//

	//merge rebuild
	level = k >> 1;
	dir = level << 1;
	for(size = 4096; size > k; size = size >> 1){
		//break;
		//rebuild-start
		low = threadIdx.x & (k-1);
		i = (threadIdx.x << 1) - low;
		if ( size >= 4096 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]);
			v1 = fmaxf(sm_vals[1024+i],sm_vals[1024+i+k]);
			v2 = fmaxf(sm_vals[2048+i],sm_vals[2048+i+k]);
			v3 = fmaxf(sm_vals[3072+i],sm_vals[3072+i+k]);
		}else if( size >= 2048 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]);
			v1 = fmaxf(sm_vals[1024+i],sm_vals[1024+i+k]);
		}else if( size >= 1024 ){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}else if( threadIdx.x < (size >> 1)){
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]);
		}
		__syncthreads();
		if( size >= 4096 ){
			sm_vals[threadIdx.x] = v0;
			sm_vals[threadIdx.x + 512] = v1;
			sm_vals[threadIdx.x + 1024] = v2;
			sm_vals[threadIdx.x + 1536] = v3;
		}else if( size >= 2048 ){
			sm_vals[threadIdx.x] = v0;
			sm_vals[threadIdx.x + 512] = v1;
		}else if( size >= 1024 ){
			sm_vals[threadIdx.x] = v0;
		}else if( threadIdx.x < (size >> 1)){
			sm_vals[threadIdx.x] = v0;
		}
		__syncthreads();
		//rebuild-end

		for(step = level; step > 0; step = step >> 1){
			if( threadIdx.x < (size >> 1) ){
				low = threadIdx.x & (step - 1);
				i = (threadIdx.x << 1) - low;
				reverse = ((dir & i) == 0);
				for(loffset = i; loffset < (size >> 1); loffset+=1024){
					v0 = sm_vals[loffset];
					v1 = sm_vals[loffset+step];
					swap = reverse ^ (v0 < v1);
					tmp = swap ? v0 : v1;
					v1 = swap ? v1 : v0;
					v0 = tmp;
					sm_vals[loffset] = v0;
					sm_vals[loffset+step] = v1;
				}
			}
			__syncthreads();
		}
	}

	//////////////
	//Write-back//
	if( k == 1024 ){
		ovec[goffset] = sm_vals[threadIdx.x];
		ovec[goffset + 512] = sm_vals[threadIdx.x + 512];
	}else if(threadIdx.x <  k){
		ovec[goffset] = sm_vals[threadIdx.x];
	}
}

template<class T, uint32_t bsize>
__global__ void scores_topk6(T *ivec, T *ovec, uint64_t k){
//	__shared__ T sm_vals[4096];
//	uint32_t goffset,loffset, size;
//	uint32_t level, dir, step;
	int low;
	int i;
//	bool reverse, swap;
//	T v0, v1, v2, v3, v4, v5, v6, v7;
//	T v8, v9, vA, vB, vC, vD, vE, VF;
//	T tmp;

	__shared__ T sm_vals[4096];
	uint32_t goffset;
	T v0, v1, v2, v3, v4, v5, v6, v7;
	T v8, v9, vA, vB, vC, vD, vE, vF;
	T tt;

	if(threadIdx.x == 0 && blockIdx.x == 0) printf("scores_topk6\n");
	//local-sort begins//
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

	uint32_t level, step;
	uint32_t dir;
	bool swap;
	for(level = 1; level < k; level = level << 1){
		dir = level << 1;
		for(step = level; step > 0; step = step >> 1){
			if(step == 1){
				swap = (dir & 0x0); tt = swap ? fmaxf(v0,v1) : fminf(v0,v1); v1 = swap ? fminf(v0,v1) : fmaxf(v0,v1); v0 = tt;
				swap = (dir & 0x2); tt = swap ? fmaxf(v2,v3) : fminf(v2,v3); v3 = swap ? fminf(v2,v3) : fmaxf(v2,v3); v2 = tt;
				swap = (dir & 0x4); tt = swap ? fmaxf(v4,v5) : fminf(v4,v5); v5 = swap ? fminf(v4,v5) : fmaxf(v4,v5); v4 = tt;
				swap = (dir & 0x6); tt = swap ? fmaxf(v6,v7) : fminf(v6,v7); v7 = swap ? fminf(v6,v7) : fmaxf(v6,v7); v6 = tt;

				swap = (dir & 0x8); tt = swap ? fmaxf(v8,v9) : fminf(v8,v9); v9 = swap ? fminf(v8,v9) : fmaxf(v8,v9); v8 = tt;
				swap = (dir & 0xA); tt = swap ? fmaxf(vA,vB) : fminf(vA,vB); vB = swap ? fminf(vA,vB) : fmaxf(vA,vB); vA = tt;
				swap = (dir & 0xC); tt = swap ? fmaxf(vC,vD) : fminf(vC,vD); vD = swap ? fminf(vC,vD) : fmaxf(vC,vD); vC = tt;
				swap = (dir & 0xE); tt = swap ? fmaxf(vE,vF) : fminf(vE,vF); vF = swap ? fminf(vE,vF) : fmaxf(vE,vF); vE = tt;
			}else if(step == 2){
				swap = (dir & 0x0); tt = swap ? fmaxf(v0,v2) : fminf(v0,v2); v2 = swap ? fminf(v0,v2) : fmaxf(v0,v2); v0 = tt;
				swap = (dir & 0x1); tt = swap ? fmaxf(v1,v3) : fminf(v1,v3); v3 = swap ? fminf(v1,v3) : fmaxf(v1,v3); v1 = tt;
				swap = (dir & 0x4); tt = swap ? fmaxf(v4,v6) : fminf(v4,v6); v6 = swap ? fminf(v4,v6) : fmaxf(v4,v6); v4 = tt;
				swap = (dir & 0x5); tt = swap ? fmaxf(v5,v7) : fminf(v5,v7); v7 = swap ? fminf(v5,v7) : fmaxf(v5,v7); v5 = tt;

				swap = (dir & 0x8); tt = swap ? fmaxf(v8,vA) : fminf(v8,vA); vA = swap ? fminf(v8,vA) : fmaxf(v8,vA); v8 = tt;
				swap = (dir & 0x9); tt = swap ? fmaxf(v9,vB) : fminf(v9,vB); vB = swap ? fminf(v9,vB) : fmaxf(v9,vB); v9 = tt;
				swap = (dir & 0xC); tt = swap ? fmaxf(vC,vE) : fminf(vC,vE); vE = swap ? fminf(vC,vE) : fmaxf(vC,vE); vC = tt;
				swap = (dir & 0xD); tt = swap ? fmaxf(vD,vF) : fminf(vD,vF); vF = swap ? fminf(vD,vF) : fmaxf(vD,vF); vD = tt;
			}else if (step == 4){
				swap = (dir & 0x0); tt = swap ? fmaxf(v0,v4) : fminf(v0,v4); v4 = swap ? fminf(v0,v4) : fmaxf(v0,v4); v0 = tt;
				swap = (dir & 0x1); tt = swap ? fmaxf(v1,v5) : fminf(v1,v5); v5 = swap ? fminf(v1,v5) : fmaxf(v1,v5); v1 = tt;
				swap = (dir & 0x2); tt = swap ? fmaxf(v2,v6) : fminf(v2,v6); v6 = swap ? fminf(v2,v6) : fmaxf(v2,v6); v2 = tt;
				swap = (dir & 0x3); tt = swap ? fmaxf(v3,v7) : fminf(v3,v7); v7 = swap ? fminf(v3,v7) : fmaxf(v3,v7); v3 = tt;

				swap = (dir & 0x8); tt = swap ? fmaxf(v8,vC) : fminf(v8,vC); vC = swap ? fminf(v8,vC) : fmaxf(v8,vC); v8 = tt;
				swap = (dir & 0x9); tt = swap ? fmaxf(v9,vD) : fminf(v9,vD); vD = swap ? fminf(v9,vD) : fmaxf(v9,vD); v9 = tt;
				swap = (dir & 0xA); tt = swap ? fmaxf(vA,vE) : fminf(vA,vE); vE = swap ? fminf(vA,vE) : fmaxf(vA,vE); vA = tt;
				swap = (dir & 0xB); tt = swap ? fmaxf(vB,vF) : fminf(vB,vF); vF = swap ? fminf(vB,vF) : fmaxf(vB,vF); vB = tt;
			}
		}
	}

	uint32_t loffset = threadIdx.x << 4;
	sm_vals[loffset] = v0;
	sm_vals[loffset + 1] = v1;
	sm_vals[loffset + 2] = v2;
	sm_vals[loffset + 3] = v3;
	sm_vals[loffset + 4] = v4;
	sm_vals[loffset + 5] = v5;
	sm_vals[loffset + 6] = v6;
	sm_vals[loffset + 7] = v7;
	sm_vals[loffset + 8] = v8;
	sm_vals[loffset + 9] = v9;
	sm_vals[loffset + 10] = vA;
	sm_vals[loffset + 11] = vB;
	sm_vals[loffset + 12] = vC;
	sm_vals[loffset + 13] = vD;
	sm_vals[loffset + 14] = vE;
	sm_vals[loffset + 15] = vF;
	__syncthreads();
//
	uint32_t rw = 128;
	for(uint32_t size = 2048; size > 32; size = size >> 1){//
		if(threadIdx.x < rw){
			if(threadIdx.x == 0 && blockIdx.x == 0) printf("Hello(%d)<%d>\n",rw,size);
			low = threadIdx.x & (k-1);
			i = (threadIdx.x << 1) - low;
			v0 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			v1 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			v2 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			v3 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			v4 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			v5 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			v6 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			v7 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);

			v8 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			v9 = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			vA = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			vB = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			vC = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			vD = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			vE = fmaxf(sm_vals[i],sm_vals[i+k]); i+=(rw << 1);
			vF = fmaxf(sm_vals[i],sm_vals[i+k]);

			level = k >> 1;
			dir = level << 1;
			for(step = level; step > 0; step = step >> 1){
				if(step == 1){
					swap = (dir & 0x0); tt = swap ? fmaxf(v0,v1) : fminf(v0,v1); v1 = swap ? fminf(v0,v1) : fmaxf(v0,v1); v0 = tt;
					swap = (dir & 0x2); tt = swap ? fmaxf(v2,v3) : fminf(v2,v3); v3 = swap ? fminf(v2,v3) : fmaxf(v2,v3); v2 = tt;
					swap = (dir & 0x4); tt = swap ? fmaxf(v4,v5) : fminf(v4,v5); v5 = swap ? fminf(v4,v5) : fmaxf(v4,v5); v4 = tt;
					swap = (dir & 0x6); tt = swap ? fmaxf(v6,v7) : fminf(v6,v7); v7 = swap ? fminf(v6,v7) : fmaxf(v6,v7); v6 = tt;

					swap = (dir & 0x8); tt = swap ? fmaxf(v8,v9) : fminf(v8,v9); v9 = swap ? fminf(v8,v9) : fmaxf(v8,v9); v8 = tt;
					swap = (dir & 0xA); tt = swap ? fmaxf(vA,vB) : fminf(vA,vB); vB = swap ? fminf(vA,vB) : fmaxf(vA,vB); vA = tt;
					swap = (dir & 0xC); tt = swap ? fmaxf(vC,vD) : fminf(vC,vD); vD = swap ? fminf(vC,vD) : fmaxf(vC,vD); vC = tt;
					swap = (dir & 0xE); tt = swap ? fmaxf(vE,vF) : fminf(vE,vF); vF = swap ? fminf(vE,vF) : fmaxf(vE,vF); vE = tt;
				}else if(step == 2){
					swap = (dir & 0x0); tt = swap ? fmaxf(v0,v2) : fminf(v0,v2); v2 = swap ? fminf(v0,v2) : fmaxf(v0,v2); v0 = tt;
					swap = (dir & 0x1); tt = swap ? fmaxf(v1,v3) : fminf(v1,v3); v3 = swap ? fminf(v1,v3) : fmaxf(v1,v3); v1 = tt;
					swap = (dir & 0x4); tt = swap ? fmaxf(v4,v6) : fminf(v4,v6); v6 = swap ? fminf(v4,v6) : fmaxf(v4,v6); v4 = tt;
					swap = (dir & 0x5); tt = swap ? fmaxf(v5,v7) : fminf(v5,v7); v7 = swap ? fminf(v5,v7) : fmaxf(v5,v7); v5 = tt;

					swap = (dir & 0x8); tt = swap ? fmaxf(v8,vA) : fminf(v8,vA); vA = swap ? fminf(v8,vA) : fmaxf(v8,vA); v8 = tt;
					swap = (dir & 0x9); tt = swap ? fmaxf(v9,vB) : fminf(v9,vB); vB = swap ? fminf(v9,vB) : fmaxf(v9,vB); v9 = tt;
					swap = (dir & 0xC); tt = swap ? fmaxf(vC,vE) : fminf(vC,vE); vE = swap ? fminf(vC,vE) : fmaxf(vC,vE); vC = tt;
					swap = (dir & 0xD); tt = swap ? fmaxf(vD,vF) : fminf(vD,vF); vF = swap ? fminf(vD,vF) : fmaxf(vD,vF); vD = tt;
				}else if (step == 4){
					swap = (dir & 0x0); tt = swap ? fmaxf(v0,v4) : fminf(v0,v4); v4 = swap ? fminf(v0,v4) : fmaxf(v0,v4); v0 = tt;
					swap = (dir & 0x1); tt = swap ? fmaxf(v1,v5) : fminf(v1,v5); v5 = swap ? fminf(v1,v5) : fmaxf(v1,v5); v1 = tt;
					swap = (dir & 0x2); tt = swap ? fmaxf(v2,v6) : fminf(v2,v6); v6 = swap ? fminf(v2,v6) : fmaxf(v2,v6); v2 = tt;
					swap = (dir & 0x3); tt = swap ? fmaxf(v3,v7) : fminf(v3,v7); v7 = swap ? fminf(v3,v7) : fmaxf(v3,v7); v3 = tt;

					swap = (dir & 0x8); tt = swap ? fmaxf(v8,vC) : fminf(v8,vC); vC = swap ? fminf(v8,vC) : fmaxf(v8,vC); v8 = tt;
					swap = (dir & 0x9); tt = swap ? fmaxf(v9,vD) : fminf(v9,vD); vD = swap ? fminf(v9,vD) : fmaxf(v9,vD); v9 = tt;
					swap = (dir & 0xA); tt = swap ? fmaxf(vA,vE) : fminf(vA,vE); vE = swap ? fminf(vA,vE) : fmaxf(vA,vE); vA = tt;
					swap = (dir & 0xB); tt = swap ? fmaxf(vB,vF) : fminf(vB,vF); vF = swap ? fminf(vB,vF) : fmaxf(vB,vF); vB = tt;
				}
			}

			i = threadIdx.x << 4;
			sm_vals[i] = v0;
			sm_vals[i+1] = v1;
			sm_vals[i+2] = v2;
			sm_vals[i+3] = v3;
			sm_vals[i+4] = v4;
			sm_vals[i+5] = v5;
			sm_vals[i+6] = v6;
			sm_vals[i+7] = v7;

			sm_vals[i+8] = v8;
			sm_vals[i+9] = v9;
			sm_vals[i+10] = vA;
			sm_vals[i+11] = vB;
			sm_vals[i+12] = vC;
			sm_vals[i+13] = vD;
			sm_vals[i+14] = vE;
			sm_vals[i+15] = vF;
		}
		__syncthreads();
		rw = rw >> 1;//128,64,32
	}
//
	if( threadIdx.x < 32 ){
		low = threadIdx.x & (k-1);
		i = (threadIdx.x << 1) - low;

		if( k < 32 ){
			//Reduce k
			v0 = sm_vals[threadIdx.x];
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			//compact
			v0 = __shfl_sync(0xFFFFFFFF,v0,i);
			sm_vals[threadIdx.x] = v0;
		}

		if( k < 16 ){

		}
	}

	//////////////
	//Write-back//
//	if(threadIdx.x <  32) ovec[goffset] = sm_vals[threadIdx.x];
	ovec[goffset] = sm_vals[threadIdx.x];
//	ovec[goffset + 256] = sm_vals[threadIdx.x + 256];
//	ovec[goffset + 512] = sm_vals[threadIdx.x + 512];
//	ovec[goffset + 768] = sm_vals[threadIdx.x + 768];
//	ovec[goffset + 1024] = sm_vals[threadIdx.x + 1024];
//	ovec[goffset + 1280] = sm_vals[threadIdx.x + 1280];
//	ovec[goffset + 1536] = sm_vals[threadIdx.x + 1536];
//	ovec[goffset + 1792] = sm_vals[threadIdx.x + 1792];

//	if( k == 1024 ){
//		ovec[goffset] = sm_vals[threadIdx.x];
//		ovec[goffset + 512] = sm_vals[threadIdx.x + 512];
//	}else if(threadIdx.x <  k){
//		ovec[goffset] = sm_vals[threadIdx.x];
//	}
}

template<class T>
__device__ T swap(T a, uint32_t stride, int dir){
	T b = __shfl_xor_sync(0xFFFFFFFF,a,stride);
	return ((a < b) == dir) ? b : a;
}

__device__ uint32_t bfe(uint32_t a, uint32_t i){
	return (a >> i) & 0x1;
}

template<class T, uint32_t bsize>
__global__ void scores_topk7(T *ivec, T *ovec, uint64_t k){
//	__shared__ T sm_vals[4096];
//	uint32_t goffset,loffset, size;
//	uint32_t level, dir, step;
//	int low;
//	int i;
//	bool reverse, swap;
//	T v0, v1, v2, v3, v4, v5, v6, v7;
//	T v8, v9, vA, vB, vC, vD, vE, VF;
//	T tmp;

	__shared__ T sm_vals[4096];
	uint32_t goffset;
	T v0, v1, v2, v3, v4, v5, v6, v7;
	T v8, v9, vA, vB, vC, vD, vE, vF;
	if(threadIdx.x == 0 && blockIdx.x == 0) printf("scores_topk7\n");

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

	uint32_t laneId = (threadIdx.x & 31);
	if( k>= 2 ){//2
		v0 = swap(v0,0x01,bfe(laneId,1)^bfe(laneId,0));
		v1 = swap(v1,0x01,bfe(laneId,1)^bfe(laneId,0));
		v2 = swap(v2,0x01,bfe(laneId,1)^bfe(laneId,0));
		v3 = swap(v3,0x01,bfe(laneId,1)^bfe(laneId,0));
		v4 = swap(v4,0x01,bfe(laneId,1)^bfe(laneId,0));
		v5 = swap(v5,0x01,bfe(laneId,1)^bfe(laneId,0));
		v6 = swap(v6,0x01,bfe(laneId,1)^bfe(laneId,0));
		v7 = swap(v7,0x01,bfe(laneId,1)^bfe(laneId,0));
		v8 = swap(v8,0x01,bfe(laneId,1)^bfe(laneId,0));
		v9 = swap(v9,0x01,bfe(laneId,1)^bfe(laneId,0));
		vA = swap(vA,0x01,bfe(laneId,1)^bfe(laneId,0));
		vB = swap(vB,0x01,bfe(laneId,1)^bfe(laneId,0));
		vC = swap(vC,0x01,bfe(laneId,1)^bfe(laneId,0));
		vD = swap(vE,0x01,bfe(laneId,1)^bfe(laneId,0));
		vE = swap(vD,0x01,bfe(laneId,1)^bfe(laneId,0));
		vF = swap(vF,0x01,bfe(laneId,1)^bfe(laneId,0));
	}

	if( k >= 4 ){//4
		v0 = swap(v0,0x02,bfe(laneId,2)^bfe(laneId,1));
		v0 = swap(v0,0x01,bfe(laneId,2)^bfe(laneId,0));
		v1 = swap(v1,0x02,bfe(laneId,2)^bfe(laneId,1));
		v1 = swap(v1,0x01,bfe(laneId,2)^bfe(laneId,0));
		v2 = swap(v2,0x02,bfe(laneId,2)^bfe(laneId,1));
		v2 = swap(v2,0x01,bfe(laneId,2)^bfe(laneId,0));
		v3 = swap(v3,0x02,bfe(laneId,2)^bfe(laneId,1));
		v3 = swap(v3,0x01,bfe(laneId,2)^bfe(laneId,0));
		v4 = swap(v4,0x02,bfe(laneId,2)^bfe(laneId,1));
		v4 = swap(v4,0x01,bfe(laneId,2)^bfe(laneId,0));
		v5 = swap(v5,0x02,bfe(laneId,2)^bfe(laneId,1));
		v5 = swap(v5,0x01,bfe(laneId,2)^bfe(laneId,0));
		v6 = swap(v6,0x02,bfe(laneId,2)^bfe(laneId,1));
		v6 = swap(v6,0x01,bfe(laneId,2)^bfe(laneId,0));
		v7 = swap(v7,0x02,bfe(laneId,2)^bfe(laneId,1));
		v7 = swap(v7,0x01,bfe(laneId,2)^bfe(laneId,0));
		v8 = swap(v8,0x02,bfe(laneId,2)^bfe(laneId,1));
		v8 = swap(v8,0x01,bfe(laneId,2)^bfe(laneId,0));
		v9 = swap(v9,0x02,bfe(laneId,2)^bfe(laneId,1));
		v9 = swap(v9,0x01,bfe(laneId,2)^bfe(laneId,0));
		vA = swap(vA,0x02,bfe(laneId,2)^bfe(laneId,1));
		vA = swap(vA,0x01,bfe(laneId,2)^bfe(laneId,0));
		vB = swap(vB,0x02,bfe(laneId,2)^bfe(laneId,1));
		vB = swap(vB,0x01,bfe(laneId,2)^bfe(laneId,0));
		vC = swap(vC,0x02,bfe(laneId,2)^bfe(laneId,1));
		vC = swap(vC,0x01,bfe(laneId,2)^bfe(laneId,0));
		vD = swap(vD,0x02,bfe(laneId,2)^bfe(laneId,1));
		vD = swap(vD,0x01,bfe(laneId,2)^bfe(laneId,0));
		vF = swap(vF,0x02,bfe(laneId,2)^bfe(laneId,1));
		vF = swap(vF,0x01,bfe(laneId,2)^bfe(laneId,0));
	}

	if( k >= 8 ){//8
		v0 = swap(v0,0x04,bfe(laneId,3)^bfe(laneId,2));
		v0 = swap(v0,0x02,bfe(laneId,3)^bfe(laneId,1));
		v0 = swap(v0,0x01,bfe(laneId,3)^bfe(laneId,0));
		v1 = swap(v1,0x04,bfe(laneId,3)^bfe(laneId,2));
		v1 = swap(v1,0x02,bfe(laneId,3)^bfe(laneId,1));
		v1 = swap(v1,0x01,bfe(laneId,3)^bfe(laneId,0));
		v2 = swap(v2,0x04,bfe(laneId,3)^bfe(laneId,2));
		v2 = swap(v2,0x02,bfe(laneId,3)^bfe(laneId,1));
		v2 = swap(v2,0x01,bfe(laneId,3)^bfe(laneId,0));
		v3 = swap(v3,0x04,bfe(laneId,3)^bfe(laneId,2));
		v3 = swap(v3,0x02,bfe(laneId,3)^bfe(laneId,1));
		v3 = swap(v3,0x01,bfe(laneId,3)^bfe(laneId,0));
		v4 = swap(v4,0x04,bfe(laneId,3)^bfe(laneId,2));
		v4 = swap(v4,0x02,bfe(laneId,3)^bfe(laneId,1));
		v4 = swap(v4,0x01,bfe(laneId,3)^bfe(laneId,0));
		v5 = swap(v5,0x04,bfe(laneId,3)^bfe(laneId,2));
		v5 = swap(v5,0x02,bfe(laneId,3)^bfe(laneId,1));
		v5 = swap(v5,0x01,bfe(laneId,3)^bfe(laneId,0));
		v6 = swap(v6,0x04,bfe(laneId,3)^bfe(laneId,2));
		v6 = swap(v6,0x02,bfe(laneId,3)^bfe(laneId,1));
		v6 = swap(v6,0x01,bfe(laneId,3)^bfe(laneId,0));
		v7 = swap(v7,0x04,bfe(laneId,3)^bfe(laneId,2));
		v7 = swap(v7,0x02,bfe(laneId,3)^bfe(laneId,1));
		v7 = swap(v7,0x01,bfe(laneId,3)^bfe(laneId,0));
		v8 = swap(v8,0x04,bfe(laneId,3)^bfe(laneId,2));
		v8 = swap(v8,0x02,bfe(laneId,3)^bfe(laneId,1));
		v8 = swap(v8,0x01,bfe(laneId,3)^bfe(laneId,0));
		v9 = swap(v9,0x04,bfe(laneId,3)^bfe(laneId,2));
		v9 = swap(v9,0x02,bfe(laneId,3)^bfe(laneId,1));
		v9 = swap(v9,0x01,bfe(laneId,3)^bfe(laneId,0));
		vA = swap(vA,0x04,bfe(laneId,3)^bfe(laneId,2));
		vA = swap(vA,0x02,bfe(laneId,3)^bfe(laneId,1));
		vA = swap(vA,0x01,bfe(laneId,3)^bfe(laneId,0));
		vB = swap(vB,0x04,bfe(laneId,3)^bfe(laneId,2));
		vB = swap(vB,0x02,bfe(laneId,3)^bfe(laneId,1));
		vB = swap(vB,0x01,bfe(laneId,3)^bfe(laneId,0));
		vC = swap(vC,0x04,bfe(laneId,3)^bfe(laneId,2));
		vC = swap(vC,0x02,bfe(laneId,3)^bfe(laneId,1));
		vC = swap(vC,0x01,bfe(laneId,3)^bfe(laneId,0));
		vD = swap(vD,0x04,bfe(laneId,3)^bfe(laneId,2));
		vD = swap(vD,0x02,bfe(laneId,3)^bfe(laneId,1));
		vD = swap(vD,0x01,bfe(laneId,3)^bfe(laneId,0));
		vE = swap(vE,0x04,bfe(laneId,3)^bfe(laneId,2));
		vE = swap(vE,0x02,bfe(laneId,3)^bfe(laneId,1));
		vE = swap(vE,0x01,bfe(laneId,3)^bfe(laneId,0));
		vF = swap(vF,0x04,bfe(laneId,3)^bfe(laneId,2));
		vF = swap(vF,0x02,bfe(laneId,3)^bfe(laneId,1));
		vF = swap(vF,0x01,bfe(laneId,3)^bfe(laneId,0));
	}

	if( k >= 16 ){
		v0 = swap(v0,0x08,bfe(laneId,4)^bfe(laneId,3));
		v0 = swap(v0,0x04,bfe(laneId,4)^bfe(laneId,2));
		v0 = swap(v0,0x02,bfe(laneId,4)^bfe(laneId,1));
		v0 = swap(v0,0x01,bfe(laneId,4)^bfe(laneId,0));
		v1 = swap(v1,0x08,bfe(laneId,4)^bfe(laneId,3));
		v1 = swap(v1,0x04,bfe(laneId,4)^bfe(laneId,2));
		v1 = swap(v1,0x02,bfe(laneId,4)^bfe(laneId,1));
		v1 = swap(v1,0x01,bfe(laneId,4)^bfe(laneId,0));
		v2 = swap(v2,0x08,bfe(laneId,4)^bfe(laneId,3));
		v2 = swap(v2,0x04,bfe(laneId,4)^bfe(laneId,2));
		v2 = swap(v2,0x02,bfe(laneId,4)^bfe(laneId,1));
		v2 = swap(v2,0x01,bfe(laneId,4)^bfe(laneId,0));
		v3 = swap(v3,0x08,bfe(laneId,4)^bfe(laneId,3));
		v3 = swap(v3,0x04,bfe(laneId,4)^bfe(laneId,2));
		v3 = swap(v3,0x02,bfe(laneId,4)^bfe(laneId,1));
		v3 = swap(v3,0x01,bfe(laneId,4)^bfe(laneId,0));
		v4 = swap(v4,0x08,bfe(laneId,4)^bfe(laneId,3));
		v4 = swap(v4,0x04,bfe(laneId,4)^bfe(laneId,2));
		v4 = swap(v4,0x02,bfe(laneId,4)^bfe(laneId,1));
		v4 = swap(v4,0x01,bfe(laneId,4)^bfe(laneId,0));
		v5 = swap(v5,0x08,bfe(laneId,4)^bfe(laneId,3));
		v5 = swap(v5,0x04,bfe(laneId,4)^bfe(laneId,2));
		v5 = swap(v5,0x02,bfe(laneId,4)^bfe(laneId,1));
		v5 = swap(v5,0x01,bfe(laneId,4)^bfe(laneId,0));
		v6 = swap(v6,0x08,bfe(laneId,4)^bfe(laneId,3));
		v6 = swap(v6,0x04,bfe(laneId,4)^bfe(laneId,2));
		v6 = swap(v6,0x02,bfe(laneId,4)^bfe(laneId,1));
		v6 = swap(v6,0x01,bfe(laneId,4)^bfe(laneId,0));
		v7 = swap(v7,0x08,bfe(laneId,4)^bfe(laneId,3));
		v7 = swap(v7,0x04,bfe(laneId,4)^bfe(laneId,2));
		v7 = swap(v7,0x02,bfe(laneId,4)^bfe(laneId,1));
		v7 = swap(v7,0x01,bfe(laneId,4)^bfe(laneId,0));
		v8 = swap(v8,0x08,bfe(laneId,4)^bfe(laneId,3));
		v8 = swap(v8,0x04,bfe(laneId,4)^bfe(laneId,2));
		v8 = swap(v8,0x02,bfe(laneId,4)^bfe(laneId,1));
		v8 = swap(v8,0x01,bfe(laneId,4)^bfe(laneId,0));
		v9 = swap(v9,0x08,bfe(laneId,4)^bfe(laneId,3));
		v9 = swap(v9,0x04,bfe(laneId,4)^bfe(laneId,2));
		v9 = swap(v9,0x02,bfe(laneId,4)^bfe(laneId,1));
		v9 = swap(v9,0x01,bfe(laneId,4)^bfe(laneId,0));
		vA = swap(vA,0x08,bfe(laneId,4)^bfe(laneId,3));
		vA = swap(vA,0x04,bfe(laneId,4)^bfe(laneId,2));
		vA = swap(vA,0x02,bfe(laneId,4)^bfe(laneId,1));
		vA = swap(vA,0x01,bfe(laneId,4)^bfe(laneId,0));
		vB = swap(vB,0x08,bfe(laneId,4)^bfe(laneId,3));
		vB = swap(vB,0x04,bfe(laneId,4)^bfe(laneId,2));
		vB = swap(vB,0x02,bfe(laneId,4)^bfe(laneId,1));
		vB = swap(vB,0x01,bfe(laneId,4)^bfe(laneId,0));
		vC = swap(vC,0x08,bfe(laneId,4)^bfe(laneId,3));
		vC = swap(vC,0x04,bfe(laneId,4)^bfe(laneId,2));
		vC = swap(vC,0x02,bfe(laneId,4)^bfe(laneId,1));
		vC = swap(vC,0x01,bfe(laneId,4)^bfe(laneId,0));
		vD = swap(vD,0x08,bfe(laneId,4)^bfe(laneId,3));
		vD = swap(vD,0x04,bfe(laneId,4)^bfe(laneId,2));
		vD = swap(vD,0x02,bfe(laneId,4)^bfe(laneId,1));
		vD = swap(vD,0x01,bfe(laneId,4)^bfe(laneId,0));
		vE = swap(vE,0x08,bfe(laneId,4)^bfe(laneId,3));
		vE = swap(vE,0x04,bfe(laneId,4)^bfe(laneId,2));
		vE = swap(vE,0x02,bfe(laneId,4)^bfe(laneId,1));
		vE = swap(vE,0x01,bfe(laneId,4)^bfe(laneId,0));
		vF = swap(vF,0x08,bfe(laneId,4)^bfe(laneId,3));
		vF = swap(vF,0x04,bfe(laneId,4)^bfe(laneId,2));
		vF = swap(vF,0x02,bfe(laneId,4)^bfe(laneId,1));
		vF = swap(vF,0x01,bfe(laneId,4)^bfe(laneId,0));
	}
	laneId = threadIdx.x;

	if( k >= 32 ){
		v0 = swap(v0,0x10,bfe(laneId,5)^bfe(laneId,4));
		v0 = swap(v0,0x08,bfe(laneId,5)^bfe(laneId,3));
		v0 = swap(v0,0x04,bfe(laneId,5)^bfe(laneId,2));
		v0 = swap(v0,0x02,bfe(laneId,5)^bfe(laneId,1));
		v0 = swap(v0,0x01,bfe(laneId,5)^bfe(laneId,0));
		v1 = swap(v1,0x10,bfe(laneId,5)^bfe(laneId,4));
		v1 = swap(v1,0x08,bfe(laneId,5)^bfe(laneId,3));
		v1 = swap(v1,0x04,bfe(laneId,5)^bfe(laneId,2));
		v1 = swap(v1,0x02,bfe(laneId,5)^bfe(laneId,1));
		v1 = swap(v1,0x01,bfe(laneId,5)^bfe(laneId,0));
		v2 = swap(v2,0x10,bfe(laneId,5)^bfe(laneId,4));
		v2 = swap(v2,0x08,bfe(laneId,5)^bfe(laneId,3));
		v2 = swap(v2,0x04,bfe(laneId,5)^bfe(laneId,2));
		v2 = swap(v2,0x02,bfe(laneId,5)^bfe(laneId,1));
		v2 = swap(v2,0x01,bfe(laneId,5)^bfe(laneId,0));
		v3 = swap(v3,0x10,bfe(laneId,5)^bfe(laneId,4));
		v3 = swap(v3,0x08,bfe(laneId,5)^bfe(laneId,3));
		v3 = swap(v3,0x04,bfe(laneId,5)^bfe(laneId,2));
		v3 = swap(v3,0x02,bfe(laneId,5)^bfe(laneId,1));
		v3 = swap(v3,0x01,bfe(laneId,5)^bfe(laneId,0));
		v4 = swap(v4,0x10,bfe(laneId,5)^bfe(laneId,4));
		v4 = swap(v4,0x08,bfe(laneId,5)^bfe(laneId,3));
		v4 = swap(v4,0x04,bfe(laneId,5)^bfe(laneId,2));
		v4 = swap(v4,0x02,bfe(laneId,5)^bfe(laneId,1));
		v4 = swap(v4,0x01,bfe(laneId,5)^bfe(laneId,0));
		v5 = swap(v5,0x10,bfe(laneId,5)^bfe(laneId,4));
		v5 = swap(v5,0x08,bfe(laneId,5)^bfe(laneId,3));
		v5 = swap(v5,0x04,bfe(laneId,5)^bfe(laneId,2));
		v5 = swap(v5,0x02,bfe(laneId,5)^bfe(laneId,1));
		v5 = swap(v5,0x01,bfe(laneId,5)^bfe(laneId,0));
		v6 = swap(v6,0x10,bfe(laneId,5)^bfe(laneId,4));
		v6 = swap(v6,0x08,bfe(laneId,5)^bfe(laneId,3));
		v6 = swap(v6,0x04,bfe(laneId,5)^bfe(laneId,2));
		v6 = swap(v6,0x02,bfe(laneId,5)^bfe(laneId,1));
		v6 = swap(v6,0x01,bfe(laneId,5)^bfe(laneId,0));
		v7 = swap(v7,0x10,bfe(laneId,5)^bfe(laneId,4));
		v7 = swap(v7,0x08,bfe(laneId,5)^bfe(laneId,3));
		v7 = swap(v7,0x04,bfe(laneId,5)^bfe(laneId,2));
		v7 = swap(v7,0x02,bfe(laneId,5)^bfe(laneId,1));
		v7 = swap(v7,0x01,bfe(laneId,5)^bfe(laneId,0));
		v8 = swap(v8,0x10,bfe(laneId,5)^bfe(laneId,4));
		v8 = swap(v8,0x08,bfe(laneId,5)^bfe(laneId,3));
		v8 = swap(v8,0x04,bfe(laneId,5)^bfe(laneId,2));
		v8 = swap(v8,0x02,bfe(laneId,5)^bfe(laneId,1));
		v8 = swap(v8,0x01,bfe(laneId,5)^bfe(laneId,0));
		v9 = swap(v9,0x10,bfe(laneId,5)^bfe(laneId,4));
		v9 = swap(v9,0x08,bfe(laneId,5)^bfe(laneId,3));
		v9 = swap(v9,0x04,bfe(laneId,5)^bfe(laneId,2));
		v9 = swap(v9,0x02,bfe(laneId,5)^bfe(laneId,1));
		v9 = swap(v9,0x01,bfe(laneId,5)^bfe(laneId,0));
		vA = swap(vA,0x10,bfe(laneId,5)^bfe(laneId,4));
		vA = swap(vA,0x08,bfe(laneId,5)^bfe(laneId,3));
		vA = swap(vA,0x04,bfe(laneId,5)^bfe(laneId,2));
		vA = swap(vA,0x02,bfe(laneId,5)^bfe(laneId,1));
		vA = swap(vA,0x01,bfe(laneId,5)^bfe(laneId,0));
		vB = swap(vB,0x10,bfe(laneId,5)^bfe(laneId,4));
		vB = swap(vB,0x08,bfe(laneId,5)^bfe(laneId,3));
		vB = swap(vB,0x04,bfe(laneId,5)^bfe(laneId,2));
		vB = swap(vB,0x02,bfe(laneId,5)^bfe(laneId,1));
		vB = swap(vB,0x01,bfe(laneId,5)^bfe(laneId,0));
		vC = swap(vC,0x10,bfe(laneId,5)^bfe(laneId,4));
		vC = swap(vC,0x08,bfe(laneId,5)^bfe(laneId,3));
		vC = swap(vC,0x04,bfe(laneId,5)^bfe(laneId,2));
		vC = swap(vC,0x02,bfe(laneId,5)^bfe(laneId,1));
		vC = swap(vC,0x01,bfe(laneId,5)^bfe(laneId,0));
		vD = swap(vD,0x10,bfe(laneId,5)^bfe(laneId,4));
		vD = swap(vD,0x08,bfe(laneId,5)^bfe(laneId,3));
		vD = swap(vD,0x04,bfe(laneId,5)^bfe(laneId,2));
		vD = swap(vD,0x02,bfe(laneId,5)^bfe(laneId,1));
		vD = swap(vD,0x01,bfe(laneId,5)^bfe(laneId,0));
		vE = swap(vE,0x10,bfe(laneId,5)^bfe(laneId,4));
		vE = swap(vE,0x08,bfe(laneId,5)^bfe(laneId,3));
		vE = swap(vE,0x04,bfe(laneId,5)^bfe(laneId,2));
		vE = swap(vE,0x02,bfe(laneId,5)^bfe(laneId,1));
		vE = swap(vE,0x01,bfe(laneId,5)^bfe(laneId,0));
		vF = swap(vF,0x10,bfe(laneId,5)^bfe(laneId,4));
		vF = swap(vF,0x08,bfe(laneId,5)^bfe(laneId,3));
		vF = swap(vF,0x04,bfe(laneId,5)^bfe(laneId,2));
		vF = swap(vF,0x02,bfe(laneId,5)^bfe(laneId,1));
		vF = swap(vF,0x01,bfe(laneId,5)^bfe(laneId,0));
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



	ovec[goffset] = sm_vals[threadIdx.x];
	ovec[goffset + 256] = sm_vals[threadIdx.x + 256];
	ovec[goffset + 512] = sm_vals[threadIdx.x + 512];
	ovec[goffset + 768] = sm_vals[threadIdx.x + 768];
	ovec[goffset + 1024] = sm_vals[threadIdx.x + 1024];
	ovec[goffset + 1280] = sm_vals[threadIdx.x + 1280];
	ovec[goffset + 1536] = sm_vals[threadIdx.x + 1536];
	ovec[goffset + 1792] = sm_vals[threadIdx.x + 1792];
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

		void alloc();
		void init(T *weights, uint32_t *query);
		void findTopK(uint64_t k, uint64_t qq);

	private:
		void full_relation_test(uint64_t k, uint64_t qq);
		void single_attribute_test(uint64_t k, uint64_t qq);
		void bench_kernel(uint64_t k);

		T *g_ivec;
		T *g_ovec;
		T *c_ovec;
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
void BTA<T,Z>::init(T *weights, uint32_t *query){
	normalize_transpose<T>(this->cdata, this->n, this->d);
	cutil::safeCopyToDevice<T,uint64_t>(this->gdata,this->cdata,sizeof(T)*this->n*this->d, " copy from cdata to gdata ");//Copy data from cpu to gpu memory

	cutil::cudaCheckErr(cudaMemcpyToSymbol(gpu_weights, weights, sizeof(T)*NUM_DIMS),"copy weights");//Initialize preference vector
	cutil::cudaCheckErr(cudaMemcpyToSymbol(gpu_query, query, sizeof(uint32_t)*NUM_DIMS),"copy query");//Initialize query vector
}

template<class T, class Z>
void BTA<T,Z>::full_relation_test(uint64_t k, uint64_t qq){
	uint32_t block_size = std::max((uint64_t)BTA_BLOCK_SIZE,k);
	dim3 lsort_block(block_size,1,1);
	//dim3 _block(BTA_BLOCK_SIZE,1,1);
	dim3 lsort_grid((this->n-1)/BTA_TUPLES_PER_BLOCK + 1, 1, 1);

//	for(uint32_t level = 1; level < 8; level= level <<1){
//		std::cout << "<>";
//		for(uint32_t s = level; s > 0; s = s >> 1){
//			std::cout <<  " [" << std::setfill('0') << std::setw(3) << s << "] ";
//		}
//		std::cout<<std::endl;
//		for(uint32_t tid = 0; tid < 8; tid++){
//			std::cout << std::setfill('0') << std::setw(2) << tid;
//			for(uint32_t s = level; s > 0; s = s >> 1){
//				uint32_t left = (s << 1) * (tid/s) + (tid&(s-1));
//				std::cout <<  " <" << std::setfill('0') << std::setw(3) << left << "," << (left ^ s) << "> ";
//			}
//			std::cout << std::endl;
//		}
//		std::cout <<"<===========================================>\n";
//	}

	std::cout << std::fixed << std::setprecision(4);
	//First step debug//
	this->t.start();
	local_sort2<T,Z><<<lsort_grid,lsort_block>>>(this->gdata,this->n,qq,k,this->gscores);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing local_sort");
	this->tt_processing = this->t.lap();
	cutil::safeCopyToHost<T,uint64_t>(this->cscores, this->gscores, sizeof(T)*this->n,"copy scores to host");
	this->cpu_threshold = find_threshold<T,Z>(this->cscores,this->n,k);
	//for(uint32_t i = 0; i < 32; i++){ std::cout << this->cscores[i] << std::endl; if((i+1)%k ==0 ){ std::cout << "-----" <<std::endl;}}
	this->cpu_threshold = find_threshold<T,Z>(this->cscores,this->n,k);

	this->t.start();
	local_sort3<T,Z><<<lsort_grid,lsort_block>>>(this->gdata,this->n,qq,k,this->gscores);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing local_sort2");
	this->tt_processing = this->t.lap("Sort only");
	cutil::safeCopyToHost<T,uint64_t>(this->cscores, this->gscores, sizeof(T)*this->n,"copy scores to host");
	//for(uint32_t i = 0; i < 32; i++){ std::cout << this->cscores[i] << std::endl; if((i+1)%k ==0 ){ std::cout << "-----" <<std::endl;}}
	this->cpu_threshold = find_partial_threshold<T,Z>(this->cscores,this->n,k,true,0);

	//Last development phase//
	this->t.start();
	local_sort4<T,Z><<<lsort_grid,lsort_block>>>(this->gdata,this->n,qq,k,this->gscores);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing local_sort2");
	uint32_t remainder = ((this->n-1)/BTA_TUPLES_PER_BLOCK + 1)*k;
	while(remainder > k){
		std::cout << "remainder:" << remainder << std::endl;
		//break;
		dim3 merge_block(block_size,1,1);
		dim3 merge_grid((remainder-1)/BTA_TUPLES_PER_BLOCK + 1, 1, 1);
		merge<T,Z><<<merge_grid,merge_block>>>(this->gscores,k,remainder);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing local_sort2");
		remainder = ((remainder-1)/BTA_TUPLES_PER_BLOCK + 1)*k;
		//break;
	}
	this->tt_processing = this->t.lap("Full BTA");
	std::cout << "remainder:" << remainder << std::endl;
	cutil::safeCopyToHost<T,uint64_t>(this->cscores, this->gscores, sizeof(T)*this->n,"copy scores to host");
	//for(uint32_t i = 0; i < 32; i++){ std::cout << this->cscores[i] << std::endl; if((i+1)%k ==0 ){ std::cout << "-----" <<std::endl;}}
	this->cpu_threshold = find_partial_threshold<T,Z>(this->cscores,this->n,k,false,remainder);
}

template<class T, class Z>
void BTA<T,Z>::single_attribute_test(uint64_t k, uint64_t qq){
	uint32_t block_size = 512;
	dim3 lsort_block(block_size,1,1);
	dim3 lsort_grid((this->n-1)/BTA_TUPLES_PER_BLOCK + 1, 1, 1);

	this->t.start();
	aggregate<T,Z><<<lsort_grid,lsort_block>>>(this->gdata,this->n,qq,k,this->g_ovec);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing aggregate");
	this->t.lap("calculate and write");
	cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
	for(uint32_t i = 0; i < this->n; i++){ this->c_ovec[i] = i;}
	find_threshold<T,Z>(this->c_ovec,this->n,k);
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
	find_threshold<T,Z>(this->c_ovec,this->n,k);
	//for(uint32_t i = 0; i < 32; i++){ std::cout << this->c_ovec[i] << std::endl; if((i+1)%k ==0 ){ std::cout << "-----" <<std::endl;}}

	//Topk2
	this->t.start();
	//scores_topk4<T,512><<<lsort_grid,lsort_block>>>(this->g_ivec, this->g_ovec,k);
	scores_topk5<T,512><<<lsort_grid,lsort_block>>>(this->g_ivec, this->g_ovec,k);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing local_sort");
	this->tt_processing = this->t.lap();
	cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
	find_remain_threshold<T,Z>(this->c_ovec,this->n,k,k);
	std::cout << "local_sort5: " << this->tt_processing << std::endl;
	std::cout << "local_sort5 (GB/s):" << ((this->n * 4 + ((this->n/4096)*k)*4)/(this->tt_processing/1000))/(1024*1024*1204) << std::endl;
	//for(uint32_t i = 0; i < 32; i++){ std::cout << this->c_ovec[i] << std::endl; if((i+1)%k ==0 ){ std::cout << "-----" <<std::endl;}}

	uint32_t block_size2 = 256;
	dim3 lsort_block2(block_size2,1,1);
	dim3 lsort_grid2((this->n-1)/BTA_TUPLES_PER_BLOCK + 1, 1, 1);
	this->t.start();
	scores_topk7<T,256><<<lsort_grid2,lsort_block2>>>(this->g_ivec, this->g_ovec,k);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing local_sort7");
	this->tt_processing = this->t.lap();
	cutil::safeCopyToHost<T,uint64_t>(this->c_ovec, this->g_ovec, sizeof(T)*this->n,"copy scores to host");
	for(uint32_t i = 0; i < 64; i++){ std::cout << this->c_ovec[i] << std::endl; if((i+1)%k ==0 ){ std::cout << "-----" <<std::endl;}}
	find_remain_threshold<T,Z>(this->c_ovec,this->n,k,16);
	std::cout << "local_sort6: " << this->tt_processing << std::endl;
	std::cout << "local_sort6 (GB/s):" << ((this->n * 4 + ((this->n/4096)*k))/(this->tt_processing/1000))/(1024*1024*1204) << std::endl;
}

template<class T, class Z>
void BTA<T,Z>::findTopK(uint64_t k, uint64_t qq){
	//this->full_relation_test(k,qq);
	this->single_attribute_test(k,qq);
}


#endif
