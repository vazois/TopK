#ifndef BTA_H
#define BTA_H
/*
 * Bitonic Top-k Aggregation
 */

#include "GAA.h"
#define BTA_BLOCK_SIZE 256
#define BTA_TUPLES_PER_BLOCK 4096

template<class T, class Z>
__global__ void local_sort(T *gdata, uint64_t n, uint64_t d, uint64_t k, T *gscores){
	//__shared__ Z tuple_ids[BTA_TUPLES_PER_BLOCK];
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

template<class T, class Z>
class BTA : public GAA<T,Z>{
	public:
		BTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "BTA";
		};

		void alloc();
		void init(T *weights, uint32_t *query);
		void findTopK(uint64_t k, uint64_t qq);
};

template<class T, class Z>
void BTA<T,Z>::alloc(){
	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");// Allocate cpu data memory
	cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*this->d,"gdata alloc");//Allocate gpu data memory

	cutil::safeMallocHost<T,uint64_t>(&(this->cscores),sizeof(T)*this->n,"cscores alloc");//Allocate cpu scores memory
	cutil::safeMalloc<T,uint64_t>(&(this->gscores),sizeof(T)*this->n,"gscores alloc");//Allocate gpu scores memory
}

template<class T, class Z>
void BTA<T,Z>::init(T *weights, uint32_t *query){
	cutil::safeCopyToDevice<T,uint64_t>(this->gdata,this->cdata,sizeof(T)*this->n*this->d, " copy from cdata to gdata ");//Copy data from cpu to gpu memory

	cutil::cudaCheckErr(cudaMemcpyToSymbol(gpu_weights, weights, sizeof(T)*NUM_DIMS),"copy weights");//Initialize preference vector
	cutil::cudaCheckErr(cudaMemcpyToSymbol(gpu_query, query, sizeof(uint32_t)*NUM_DIMS),"copy query");//Initialize query vector
}

template<class T, class Z>
void BTA<T,Z>::findTopK(uint64_t k, uint64_t qq){
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
	for(uint32_t i = 0; i < 32; i++){ std::cout << this->cscores[i] << std::endl; if((i+1)%k ==0 ){ std::cout << "-----" <<std::endl;}}
	this->cpu_threshold = find_threshold<T,Z>(this->cscores,this->n,k);

	this->t.start();
	local_sort3<T,Z><<<lsort_grid,lsort_block>>>(this->gdata,this->n,qq,k,this->gscores);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing local_sort2");
	this->tt_processing = this->t.lap();
	cutil::safeCopyToHost<T,uint64_t>(this->cscores, this->gscores, sizeof(T)*this->n,"copy scores to host");
	for(uint32_t i = 0; i < 32; i++){ std::cout << this->cscores[i] << std::endl; if((i+1)%k ==0 ){ std::cout << "-----" <<std::endl;}}
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
	this->tt_processing = this->t.lap();
	std::cout << "remainder:" << remainder << std::endl;
	cutil::safeCopyToHost<T,uint64_t>(this->cscores, this->gscores, sizeof(T)*this->n,"copy scores to host");
	for(uint32_t i = 0; i < 32; i++){ std::cout << this->cscores[i] << std::endl; if((i+1)%k ==0 ){ std::cout << "-----" <<std::endl;}}
	this->cpu_threshold = find_partial_threshold<T,Z>(this->cscores,this->n,k,false,remainder);
}


#endif
