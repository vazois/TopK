#ifndef BTA_H
#define BTA_H

#include "GAA.h"

#define BTA_TUPLES_PER_BLOCK 4096

template<class T, class Z>
class BTA : public GAA<T,Z>{
	public:
		BTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "BTA";
		};

		~BTA(){
			if(this->csvector!=NULL) cudaFreeHost(this->csvector);
			#if USE_DEVICE_MEM
				if(this->gsvector!=NULL) cudaFree(this->gsvector);
			#endif
		};

		void alloc();
		void init();
		void findTopK(uint64_t k, uint64_t qq);

	private:
		void clear(T *vec, uint64_t size);

		T *csvector = NULL;
		T *gsvector = NULL;
};

template<class T, class Z>
void BTA<T,Z>::clear(T * vec, uint64_t size)
{
	for(uint64_t i = 0; i < size; i++) vec[i] = 0;
}

template<class T>
__global__ void aggregate(T *gdata, uint64_t n, uint64_t qq, T *gscores)
{
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if( i < n )
	{
		T score = 0;
		for(uint64_t m = 0; m <qq; m++)
		{
			uint64_t ai = gpu_query[m];
			score+=gdata[i + ai * n] * gpu_weights[ai];
		}
		gscores[i] = score;
	}
}

template<class T>
__global__ void agg_lsort_atm_16(T *gdata, uint64_t n, uint64_t qq, uint64_t k, T *gscores){
	uint32_t i;
	__shared__ T buffer[256];
	T v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0;
	T v8 = 0, v9 = 0, vA = 0, vB = 0, vC = 0, vD = 0, vE = 0, vF = 0;

	/*
	 * Aggregate
	 */
	i = (blockIdx.x << 12) + threadIdx.x;
	for(uint64_t m = 0; m < qq; m++)
	{
		uint64_t ai = gpu_query[m];
		uint64_t index = n * ai + i;
		v0 += gdata[index] * gpu_weights[ai];
		v1 += gdata[index + 256] * gpu_weights[ai];
		v2 += gdata[index + 512] * gpu_weights[ai];
		v3 += gdata[index + 768] * gpu_weights[ai];
		v4 += gdata[index + 1024] * gpu_weights[ai];
		v5 += gdata[index + 1280] * gpu_weights[ai];
		v6 += gdata[index + 1536] * gpu_weights[ai];
		v7 += gdata[index + 1792] * gpu_weights[ai];
		v8 += gdata[index + 2048] * gpu_weights[ai];
		v9 += gdata[index + 2304] * gpu_weights[ai];
		vA += gdata[index + 2560] * gpu_weights[ai];
		vB += gdata[index + 2816] * gpu_weights[ai];
		vC += gdata[index + 3072] * gpu_weights[ai];
		vD += gdata[index + 3328] * gpu_weights[ai];
		vE += gdata[index + 3584] * gpu_weights[ai];
		vF += gdata[index + 3840] * gpu_weights[ai];
	}

//	gscores[i] = v0;
//	gscores[i+256] = v1;
//	gscores[i+512] = v2;
//	gscores[i+768] = v3;
//	gscores[i+1024] = v4;
//	gscores[i+1280] = v5;
//	gscores[i+1536] = v6;
//	gscores[i+1792] = v7;
//	gscores[i+2048] = v8;
//	gscores[i+2304] = v9;
//	gscores[i+2560] = vA;
//	gscores[i+2816] = vB;
//	gscores[i+3072] = vC;
//	gscores[i+3328] = vD;
//	gscores[i+3584] = vE;
//	gscores[i+3840] = vF;

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

//	gscores[i] = v0;
//	gscores[i+256] = v2;
//	gscores[i+512] = v4;
//	gscores[i+768] = v6;
//	gscores[i+1024] = v8;
//	gscores[i+1280] = vA;
//	gscores[i+1536] = vC;
//	gscores[i+1792] = vE;


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

//	gscores[i] = v0;
//	gscores[i+256] = v4;
//	gscores[i+512] = v8;
//	gscores[i+768] = vC;

	/*
	 * Rebuild - Reduce 1024 -> 512
	 */
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

//	gscores[i] = v0;
//	gscores[i+256] = v8;

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

	//gscores[i] = v0;
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
		 * 256->128
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

//		gscores[i] = v0;
//		gscores[i+32] = v2;
//		gscores[i+64] = v4;
//		gscores[i+96] = v6;

		/*
		 * 128->64
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

//		gscores[i] = v0;
//		gscores[i+32] = v4;

		/*
		 * 64->32
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
		 * 32->16
		 */
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
			v0 = swap(v0,step,dir);
		}
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v0 = (threadIdx.x & k) == 0 ? v0 : 0;

//		/*
//		 * Sort 16
//		 */
		for(level = k; level < 32; level = level << 1){
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = rswap(v0,step,dir);
			}
		}
		//gscores[i] = v0;

		if(threadIdx.x < k)
		{
			i = blockIdx.x * k;
			if((blockIdx.x & 0x1) == 0) gscores[i + threadIdx.x] = v0; else gscores[i + (threadIdx.x ^ (k-1))] = v0;
		}
	}
}


template<class T>
__global__ void agg_lsort_geq_32(T *gdata, uint64_t n, uint64_t qq, uint64_t k, T *gscores){
	uint32_t i;
	//__shared__ T buffer[256];
	T v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0;
	T v8 = 0, v9 = 0, vA = 0, vB = 0, vC = 0, vD = 0, vE = 0, vF = 0;

	/*
	 * Aggregate
	 */
	i = (blockIdx.x << 12) + threadIdx.x;
	for(uint64_t m = 0; m < qq; m++)
	{
		uint64_t ai = gpu_query[m];
		uint64_t index = n * ai + i;
		v0 += gdata[index] * gpu_weights[ai];
		v1 += gdata[index + 256] * gpu_weights[ai];
		v2 += gdata[index + 512] * gpu_weights[ai];
		v3 += gdata[index + 768] * gpu_weights[ai];
		v4 += gdata[index + 1024] * gpu_weights[ai];
		v5 += gdata[index + 1280] * gpu_weights[ai];
		v6 += gdata[index + 1536] * gpu_weights[ai];
		v7 += gdata[index + 1792] * gpu_weights[ai];
		v8 += gdata[index + 2048] * gpu_weights[ai];
		v9 += gdata[index + 2304] * gpu_weights[ai];
		vA += gdata[index + 2560] * gpu_weights[ai];
		vB += gdata[index + 2816] * gpu_weights[ai];
		vC += gdata[index + 3072] * gpu_weights[ai];
		vD += gdata[index + 3328] * gpu_weights[ai];
		vE += gdata[index + 3584] * gpu_weights[ai];
		vF += gdata[index + 3840] * gpu_weights[ai];
	}

	gscores[i] = v0;
	gscores[i+256] = v1;
	gscores[i+512] = v2;
	gscores[i+768] = v3;
	gscores[i+1024] = v4;
	gscores[i+1280] = v5;
	gscores[i+1536] = v6;
	gscores[i+1792] = v7;
	gscores[i+2048] = v8;
	gscores[i+2304] = v9;
	gscores[i+2560] = vA;
	gscores[i+2816] = vB;
	gscores[i+3072] = vC;
	gscores[i+3328] = vD;
	gscores[i+3584] = vE;
	gscores[i+3840] = vF;
}

template<class T>
__global__ void gclear(T *vec, uint64_t size)
{
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size) vec[i] = 0;
}

template<class T, class Z>
void BTA<T,Z>::alloc(){
	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");// Allocate cpu data memory
	cutil::safeMallocHost<T,uint64_t>(&(this->csvector),sizeof(T)*this->n,"csvector alloc");//Allocate cpu scores memory
	this->clear(this->csvector,this->n);
}

template<class T, class Z>
void BTA<T,Z>::init()
{
	normalize_transpose<T>(this->cdata, this->n, this->d);
	#if USE_DEVICE_MEM
		cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*this->d,"gdata alloc");//Allocate gpu data memory
		cutil::safeCopyToDevice<T,uint64_t>(this->gdata,this->cdata,sizeof(T)*this->n*this->d, " copy from cdata to gdata ");//Copy data from cpu to gpu memory
		cutil::safeMalloc<T,uint64_t>(&(this->gsvector),sizeof(T)*this->n,"gsvector alloc");//Allocate gpu scores memory
	#else
		this->gdata = this->cdata;
		this->gsvector = this->csvector;
	#endif
}

template<class T, class Z>
void BTA<T,Z>::findTopK(uint64_t k, uint64_t qq){
	double tt_processing = 0;
	T threshold = 0;

	VAGG<T,Z> vagg(this->cdata,this->n,this->d);
	this->t.start();
	this->cpu_threshold = vagg.findTopKtpac(k, qq,this->weights,this->query);
	this->t.lap("vagg");

	//TODO: DEBUG
	//T threshold = 0;
	gclear<T><<<((this->n-1) / 256) + 1, 256>>>(this->gsvector,this->n);
	this->t.start();
	aggregate<T><<<((this->n-1) / 256) + 1, 256>>>(this->gdata,this->n,qq,this->gsvector);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing gclear");
	tt_processing = this->t.lap();
	std::cout << "aggregate: " << tt_processing << std::endl;
	std::cout << "aggregate (GB/s): " << ((this->n * qq * 4) / (tt_processing/1000))/(1024*1024*1024) << std::endl;
	#if USE_DEVICE_MEM
		cutil::safeCopyToHost<T,uint64_t>(this->csvector,this->gsvector,sizeof(T)*this->n, "copy from gsvector to csvector ");
	#endif
	std::sort(this->csvector,this->csvector + this->n,std::greater<T>());
	threshold = this->csvector[k-1];

	dim3 agg_lsort_block(256,1,1);
	dim3 agg_lsort_grid((this->n-1)/4096,1,1);
	gclear<T><<<((this->n-1) / 256) + 1, 256>>>(this->gsvector,this->n);
	if( k < 32 ){
		this->t.start();
		agg_lsort_atm_16<T><<<agg_lsort_grid,agg_lsort_block>>>(this->gdata, this->n, qq, k, this->gsvector);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing agg_lsort_atm_16");
		tt_processing = this->t.lap();
		std::cout << "agg_lsort_atm_16: " << tt_processing << std::endl;
		std::cout << "agg_lsort_atm_16 (GB/s): " << ((this->n * qq * 4) / (tt_processing/1000))/(1024*1024*1024) << std::endl;
		#if USE_DEVICE_MEM
			cutil::safeCopyToHost<T,uint64_t>(this->csvector,this->gsvector,sizeof(T)*this->n, "copy from gsvector to csvector ");
		#endif
	}else{
		this->t.start();
		agg_lsort_geq_32<T><<<agg_lsort_grid,agg_lsort_block>>>(this->gdata, this->n, qq, k, this->gsvector);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing agg_lsort_geq_32");
		tt_processing = this->t.lap();
		std::cout << "agg_lsort_geq_32: " << tt_processing << std::endl;
		std::cout << "agg_lsort_geq_32 (GB/s): " << ((this->n * qq * 4) / (tt_processing/1000))/(1024*1024*1024) << std::endl;
		#if USE_DEVICE_MEM
			cutil::safeCopyToHost<T,uint64_t>(this->csvector,this->gsvector,sizeof(T)*this->n, "copy from gsvector to csvector ");
		#endif
	}
	//DEBUG
//	for(uint32_t i = 0; i < 128; i+=k)
//	{
//		for(uint32_t j = i; j < i + k; j++) std::cout << this->csvector[j] << " ";
//		std::cout << "[" << std::is_sorted(&this->csvector[i],(&this->csvector[i+k])) << "]" << std::endl;
//	}
	//std::sort(this->csvector,this->csvector + this->n,std::greater<T>());
	std::sort(this->csvector,this->csvector + (agg_lsort_grid.x * k),std::greater<T>());
	T atm_lsort_16_threshold = this->csvector[k-1];
	if(abs((double)atm_lsort_16_threshold - (double)threshold) > (double)0.00000000000001)
	//if(abs((double)this->cpu_threshold - (double)threshold) > (double)0.00000000000001)
	{
		std::cout << std::fixed << std::setprecision(16);
		std::cout << "{ERROR}: " << atm_lsort_16_threshold << "," << threshold << "," << this->cpu_threshold << std::endl;
		exit(1);
	}

	std::cout << "threshold=[" << atm_lsort_16_threshold << "," << threshold << "," << this->cpu_threshold << "]"<< std::endl;
}

#endif
