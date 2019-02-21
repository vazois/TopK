#ifndef BTA_H
#define BTA_H

#include "GAA.h"

#define BTA_TUPLES_PER_BLOCK 4096
#define BTA_USE_DEV_MEM_PROCESSING true
#define BTA_USE_DEV_MEM_FOR_SCORES true
#define BTA_USE_PREFETCH false
#define BTA_MEM_MANAGED false

template<class T>
__global__ void aggregate(T *gdata, uint64_t n, uint64_t qq, T *gscores);

template<class T>
__global__ void agg_lsort_atm_16(T *gdata, uint64_t n, uint64_t qq, uint64_t k, T *gscores);

template<class T>
__global__ void agg_lsort_geq_32(T *gdata, uint64_t n, uint64_t qq, uint64_t k, T *gscores);

template<class T>
__global__ void gclear(T *vec, uint64_t size);

template<class T, class Z>
class BTA : public GAA<T,Z>{
	public:
		BTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "BTA";

			this->using_dev_mem = BTA_USE_DEV_MEM_PROCESSING;
			this->parts = ((this->n-1)/BTA_TUPLES_PER_BLOCK) + 1;
			this->block_size = BTA_TUPLES_PER_BLOCK;
		};

		~BTA(){
			if(this->csvector) cutil::safeCudaFreeHost<T>(this->csvector,"free csvector");
			if(this->csvector_out) cutil::safeCudaFreeHost<T>(this->csvector_out,"free csvector_out");
			#if BTA_MEM_MANAGED
				if(this->cdata) cutil::safeCudaFree<T>(this->cdata,"free BTA cdata");
			#endif
			#if BTA_USE_DEV_MEM_PROCESSING
				if(this->gsvector) cutil::safeCudaFree<T>(this->gsvector,"free gsvector");
				if(this->gsvector_out) cutil::safeCudaFree<T>(this->gsvector_out,"free gsvector_out");
			#else
				this->gdata = NULL;
				#if BTA_USE_DEV_MEM_FOR_SCORES
					if(this->gsvector) cutil::safeCudaFree<T>(this->gsvector,"free gsvector");
					if(this->gsvector_out) cutil::safeCudaFree<T>(this->gsvector_out,"free gsvector_out");
				#endif
			#endif
		};

		void alloc();
		void init();
		void findTopK(uint64_t k, uint64_t qq);

	private:
		void clear(T *vec, uint64_t size);
		void gclear_driver(T *vec, uint64_t size);
		T cpuTopK(uint64_t k, uint64_t qq);
		void validate(uint64_t k, uint64_t qq);

		void atm_16_driver(uint64_t k, uint64_t qq);
		void geq_32_driver(uint64_t k, uint64_t qq);

		T *csvector = NULL;
		T *csvector_out = NULL;
		T *gsvector = NULL;
		T *gsvector_out = NULL;
		uint64_t svec_size;
};

template<class T, class Z>
void BTA<T,Z>::clear(T * vec, uint64_t size)
{
	for(uint64_t i = 0; i < size; i++) vec[i] = 0;
}

template<class T, class Z>
void BTA<T,Z>::gclear_driver(T *vec, uint64_t size){
	gclear<T><<<((this->n-1) / 256) + 1, 256>>>(vec,this->n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing gclear");
}

template<class T, class Z>
void BTA<T,Z>::alloc(){
	#if BTA_MEM_MANAGED
		cutil::safeMallocManaged<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");// Allocate cpu data memory
	#else
		cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");// Allocate cpu data memory
	#endif
	#if USE_DEVICE_MEM
		cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*this->d,"gdata alloc");//Allocate gpu data memory
	#endif
}

template<class T, class Z>
void BTA<T,Z>::init()
{
	svec_size = ((this->n-1)/BTA_TUPLES_PER_BLOCK) * KKE;
	normalize_transpose<T>(this->cdata, this->n, this->d);
	cutil::safeMallocHost<T,uint64_t>(&(this->csvector),sizeof(T) * svec_size,"csvector alloc");//Allocate cpu scores memory
	cutil::safeMallocHost<T,uint64_t>(&(this->csvector_out),sizeof(T) * svec_size,"csvector alloc");//Allocate cpu scores memory
//	this->clear(this->csvector, BLOCKS * KKE);
//	this->clear(this->csvector_out, BLOCKS * KKE);

	#if BTA_USE_DEV_MEM_PROCESSING
		cutil::safeCopyToDevice<T,uint64_t>(this->gdata,this->cdata,sizeof(T)*this->n*this->d, " copy from cdata to gdata ");//Copy data from cpu to gpu memory
		cutil::safeMalloc<T,uint64_t>(&(this->gsvector),sizeof(T) * svec_size,"gsvector alloc");//Allocate gpu scores memory
		cutil::safeMalloc<T,uint64_t>(&(this->gsvector_out),sizeof(T) * svec_size,"gsvector_out alloc");//Allocate gpu scores memory
	#else
		this->gdata = this->cdata;
		#if BTA_USE_DEV_MEM_FOR_SCORES
			cutil::safeMalloc<T,uint64_t>(&(this->gsvector),sizeof(T) * svec_size,"gsvector alloc");//Allocate gpu scores memory
			cutil::safeMalloc<T,uint64_t>(&(this->gsvector_out),sizeof(T) * svec_size,"gsvector_out alloc");//Allocate gpu scores memory
		#else
			this->gsvector = this->csvector;
			this->gsvector_out = this->csvector_out;
		#endif
	#endif
}

template<class T, class Z>
T BTA<T,Z>::cpuTopK(uint64_t k, uint64_t qq){
	std::priority_queue<T, std::vector<ranked_tuple<T,Z>>, MaxFirst<T,Z>> q;

	T threshold = 0;
	for(uint64_t i = 0; i < this->n; i++)
	{
		T score = 0;
		for(uint64_t m = 0; m < qq; m++)
		{
			uint32_t ai = this->query[m];
			score += this->cdata[ai * this->n + i] * this->weights[ai];
		}

		if(q.size() < k)
		{
			q.push(ranked_tuple<T,Z>(i,score));
		}else if( q.top().score < score ){
			q.pop();
			q.push(ranked_tuple<T,Z>(i,score));
		}
	}
	threshold = q.top().score;
	return threshold;
}

template<class T, class Z>
void BTA<T,Z>::validate(uint64_t k, uint64_t qq){
	#if VALIDATE
		this->cpu_threshold = this->cpuTopK(k,qq);
		if( abs((double)this->gpu_threshold - (double)this->cpu_threshold) > (double)0.000001 ) {
			std::cout << std::fixed << std::setprecision(16);
			std::cout << "ERROR: {" << this->gpu_threshold << "," << this->cpu_threshold << "}" << std::endl; exit(1);
		}
	#endif
}

template<class T, class Z>
void BTA<T,Z>::atm_16_driver(uint64_t k, uint64_t qq){
	T *csvector = this->csvector;
	T *gsvector;
	T *gsvector_out;

	#if BTA_USE_PREFETCH
		uint32_t ai = this->query[0];
		std::cout << "CPU Device: " << cudaCpuDeviceId << std::endl;
		cudaMemAdvise(&this->cdata[ai * this->n], sizeof(T) * qq, cudaMemAdviseSetReadMostly, 0);
		cudaMemPrefetchAsync(&this->cdata[ai * this->n], sizeof(T) * qq, 0, cudaStreamLegacy);
		std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	#endif

	#if BTA_USE_DEV_MEM_PROCESSING
		gsvector = this->gsvector;
		gsvector_out = this->gsvector_out;
	#else
		#if BTA_USE_DEV_MEM_FOR_SCORES
			gsvector = this->gsvector;
			gsvector_out = this->gsvector_out;
		#else
			gsvector = csvector;
			gsvector_out = this->csvector_out;
		#endif
	#endif

	dim3 agg_lsort_block(256,1,1);
	dim3 agg_lsort_grid(((this->n-1)/BTA_TUPLES_PER_BLOCK) + 1,1,1);
//	gclear_driver(gsvector,BLOCKS * KKE);
//	gclear_driver(gsvector_out,BLOCKS * KKE);

	//1:Local sort
	this->t.start();
	agg_lsort_atm_16<T><<<agg_lsort_grid,agg_lsort_block>>>(this->gdata, this->n, qq, k, gsvector);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing agg_lsort_atm_16");
	this->tt_processing += this->t.lap();

	//2:Local reduce
	uint64_t remainder = (agg_lsort_grid.x * k);
	while(remainder > k)
	{
		agg_lsort_grid.x = ((remainder - 1) /4096) + 1;
		this->t.start();
		reduce_rebuild_atm_16<T><<<agg_lsort_grid,agg_lsort_block>>>(gsvector,remainder,k,gsvector_out);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing reduce_rebuild_atm_16");
		this->tt_processing += this->t.lap();

		remainder = (agg_lsort_grid.x * k);
		std::swap(gsvector,gsvector_out);
	}

	#if BTA_USE_DEV_MEM_PROCESSING
		cutil::safeCopyToHost<T,uint64_t>(csvector,gsvector,sizeof(T) * k,"copy from csvector to gsvector");
	#else
		#if BTA_USE_DEV_MEM_FOR_SCORES
			cutil::safeCopyToHost<T,uint64_t>(csvector,gsvector,sizeof(T) * k,"copy from csvector to gsvector");
		#else
			csvector = gsvector;
		#endif
	#endif
	this->gpu_threshold = csvector[k-1];
	this->validate(k,qq);
}

template<class T, class Z>
void BTA<T,Z>::geq_32_driver(uint64_t k, uint64_t qq){
	T *csvector = this->csvector;
	T *gsvector;
	T *gsvector_out;

	#if BTA_USE_DEV_MEM_PROCESSING
		gsvector = this->gsvector;
		gsvector_out = this->gsvector_out;
	#else
		#if BTA_USE_DEV_MEM_FOR_SCORES
			gsvector = this->gsvector;
			gsvector_out = this->gsvector_out;
		#else
			gsvector = csvector;
			gsvector_out = this->csvector_out;
		#endif
	#endif

	dim3 agg_lsort_block(256,1,1);
	dim3 agg_lsort_grid(((this->n-1)/BTA_TUPLES_PER_BLOCK) + 1,1,1);
//	gclear_driver(gsvector,BLOCKS * KKE);
//	gclear_driver(gsvector_out,BLOCKS * KKE);

	//1: local sort
	this->t.start();
	agg_lsort_geq_32<<<agg_lsort_grid,agg_lsort_block>>>(this->gdata, this->n, qq, k, gsvector);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing agg_lsort_geq_32");
	this->tt_processing += this->t.lap();

	//2:Local reduce
	uint64_t remainder = (agg_lsort_grid.x * k);
	while(remainder > k)
	{
		//std::cout << "remainder: " << remainder << std::endl;
		agg_lsort_grid.x = ((remainder - 1) /4096) + 1;
		this->t.start();
		reduce_rebuild_qeq_32<T><<<agg_lsort_grid,agg_lsort_block>>>(gsvector,remainder,k,gsvector_out);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing reduce_rebuild_atm_16");
		this->tt_processing += this->t.lap();

		remainder = (agg_lsort_grid.x * k);
		std::swap(gsvector,gsvector_out);
	}

	#if BTA_USE_DEV_MEM_PROCESSING
		cutil::safeCopyToHost<T,uint64_t>(csvector,gsvector,sizeof(T) * k,"copy from csvector to gsvector");
	#else
		#if BTA_USE_DEV_MEM_FOR_SCORES
			cutil::safeCopyToHost<T,uint64_t>(csvector,gsvector,sizeof(T) * k,"copy from csvector to gsvector");
		#else
			csvector = gsvector;
		#endif
	#endif
	this->gpu_threshold = csvector[k-1];
	this->validate(k,qq);
}

template<class T, class Z>
void BTA<T,Z>::findTopK(uint64_t k, uint64_t qq){
	if( k <= 16 ){
		this->atm_16_driver(k,qq);
	}else if ( k <= 256 ){
		this->geq_32_driver(k,qq);
	}else{
		std::cout << "GPU Maximum K = 256" << std::endl;
		exit(1);
	}
}

template<class T>
__global__ void gclear(T *vec, uint64_t size)
{
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size) vec[i] = 0;
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
	__shared__ T buffer[256];
	#if BTA_TUPLES_PER_BLOCK >= 1024
		T v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0;
	#endif
	#if BTA_TUPLES_PER_BLOCK >= 4096
		T v8 = 0, v9 = 0, vA = 0, vB = 0, vC = 0, vD = 0, vE = 0, vF = 0;
	#endif

	/*
	 * Aggregate
	 */
	for(uint32_t m = 0; m < qq; m++)
	{
		uint32_t ai = gpu_query[m];
		//uint32_t index = n * ai + (blockIdx.x << 12) + threadIdx.x;
		uint32_t index = n * ai + blockIdx.x * BTA_TUPLES_PER_BLOCK + threadIdx.x;
		T w = gpu_weights[ai];
		#if BTA_TUPLES_PER_BLOCK >= 1024
			v0 += gdata[index       ] * w;
			v1 += gdata[index +  256] * w;
			v2 += gdata[index +  512] * w;
			v3 += gdata[index +  768] * w;
		#endif
		#if BTA_TUPLES_PER_BLOCK >= 2048
			v4 += gdata[index + 1024] * w;
			v5 += gdata[index + 1280] * w;
			v6 += gdata[index + 1536] * w;
			v7 += gdata[index + 1792] * w;
		#endif
		#if BTA_TUPLES_PER_BLOCK >= 4096
			v8 += gdata[index + 2048] * w;
			v9 += gdata[index + 2304] * w;
			vA += gdata[index + 2560] * w;
			vB += gdata[index + 2816] * w;
			vC += gdata[index + 3072] * w;
			vD += gdata[index + 3328] * w;
			vE += gdata[index + 3584] * w;
			vF += gdata[index + 3840] * w;
		#endif
	}

	//Sort in registers//
	uint32_t level, step, dir;
	for(level = 1; level < k; level = level << 1){
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
			#if BTA_TUPLES_PER_BLOCK >= 1024
				v0 = swap(v0,step,dir);
				v1 = swap(v1,step,dir);
				v2 = swap(v2,step,dir);
				v3 = swap(v3,step,dir);
			#endif
			#if BTA_TUPLES_PER_BLOCK >= 2048
				v4 = swap(v4,step,dir);
				v5 = swap(v5,step,dir);
				v6 = swap(v6,step,dir);
				v7 = swap(v7,step,dir);
			#endif
			#if BTA_TUPLES_PER_BLOCK >= 4096
				v8 = swap(v8,step,dir);
				v9 = swap(v9,step,dir);
				vA = swap(vA,step,dir);
				vB = swap(vB,step,dir);
				vC = swap(vC,step,dir);
				vD = swap(vD,step,dir);
				vE = swap(vE,step,dir);
				vF = swap(vF,step,dir);
			#endif
		}
	}

	/*
	 * Reduce - Rebuild 4096 -> 2048
	 */
	#if BTA_TUPLES_PER_BLOCK >= 4096
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
		v1 = (threadIdx.x & k) == 0 ? v2 : v3;
		v2 = (threadIdx.x & k) == 0 ? v4 : v5;
		v3 = (threadIdx.x & k) == 0 ? v6 : v7;
		v4 = (threadIdx.x & k) == 0 ? v8 : v9;
		v5 = (threadIdx.x & k) == 0 ? vA : vB;
		v6 = (threadIdx.x & k) == 0 ? vC : vD;
		v7 = (threadIdx.x & k) == 0 ? vE : vF;
		level = k >> 1;
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
			v0 = swap(v0,step,dir);
			v1 = swap(v1,step,dir);
			v2 = swap(v2,step,dir);
			v3 = swap(v3,step,dir);
			v4 = swap(v4,step,dir);
			v5 = swap(v5,step,dir);
			v6 = swap(v6,step,dir);
			v7 = swap(v7,step,dir);
		}
	#endif

	/*
	 * Reduce - Rebuild 2048 -> 1024
	 */
	#if BTA_TUPLES_PER_BLOCK >= 2048
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
		v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
		v3 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v3, k),v3);
		v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
		v5 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v5, k),v5);
		v6 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v6, k),v6);
		v7 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v7, k),v7);
		v0 = (threadIdx.x & k) == 0 ? v0 : v1;
		v1 = (threadIdx.x & k) == 0 ? v2 : v3;
		v2 = (threadIdx.x & k) == 0 ? v4 : v5;
		v3 = (threadIdx.x & k) == 0 ? v6 : v7;
		level = k >> 1;
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
			v0 = swap(v0,step,dir);
			v1 = swap(v1,step,dir);
			v2 = swap(v2,step,dir);
			v3 = swap(v3,step,dir);
		}
	#endif

	/*
	 * Reduce - Rebuild 1024 -> 512
	 */
	#if BTA_TUPLES_PER_BLOCK >= 1024
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
		v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
		v3 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v3, k),v3);
		v0 = (threadIdx.x & k) == 0 ? v0 : v1;
		v1 = (threadIdx.x & k) == 0 ? v2 : v3;
		level = k >> 1;
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
			v0 = swap(v0,step,dir);
			v1 = swap(v1,step,dir);
		}
	#endif

	/*
	 * Reduce - Rebuild 512 -> 256
	 */
	#if BTA_TUPLES_PER_BLOCK >= 512
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
		v0 = (threadIdx.x & k) == 0 ? v0 : v1;
		level = k >> 1;
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
			v0 = swap(v0,step,dir);
		}
	#endif

	//Merge//
	buffer[threadIdx.x] = v0;
	__syncthreads();
	//k=1, 	0x55555555
	//k=2, 	0x33333333
	//k=4, 	0x0F0F0F0F
	//k=8, 	0x00FF00FF
	//k=16, 0xFFFFFFFF
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
//		for(step = level; step > 0; step = step >> 1){
//			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
//			v0 = swap(v0,step,dir);
//			v1 = swap(v1,step,dir);
//			v2 = swap(v2,step,dir);
//			v3 = swap(v3,step,dir);
//			v4 = swap(v4,step,dir);
//			v5 = swap(v5,step,dir);
//			v6 = swap(v6,step,dir);
//			v7 = swap(v7,step,dir);
//		}
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
		v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
		v3 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v3, k),v3);
		v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
		v5 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v5, k),v5);
		v6 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v6, k),v6);
		v7 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v7, k),v7);
		v0 = (threadIdx.x & k) == 0 ? v0 : v1;
		v1 = (threadIdx.x & k) == 0 ? v2 : v3;
		v2 = (threadIdx.x & k) == 0 ? v4 : v5;
		v3 = (threadIdx.x & k) == 0 ? v6 : v7;

		/*
		 * 128->64
		 */
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
			v0 = swap(v0,step,dir);
			v1 = swap(v1,step,dir);
			v2 = swap(v2,step,dir);
			v3 = swap(v3,step,dir);
		}
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
		v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
		v3 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v3, k),v3);
		v0 = (threadIdx.x & k) == 0 ? v0 : v1;
		v1 = (threadIdx.x & k) == 0 ? v2 : v3;

		/*
		 * 64->32
		 */
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
			v0 = swap(v0,step,dir);
			v1 = swap(v1,step,dir);
		}
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
		v0 = (threadIdx.x & k) == 0 ? v0 : v1;

		/*
		 * 32->16
		 */
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
			v0 = swap(v0,step,dir);
		}
		v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
		v0 = __shfl_sync(0x0000FFFF,v0,(threadIdx.x << 1) - (threadIdx.x & (k - 1)));

		/*
		 * 16
		 */
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
			v0 = swap(v0,step,dir);
		}

		//8
		if(k <= 8)
		{
			v0 = fmaxf(__shfl_xor_sync(0x0000FFFF, v0, k),v0);
			v0 = __shfl_sync(0x000000FF,v0,(threadIdx.x << 1) - (threadIdx.x & (k - 1)));

			for(step = level; step > 0; step = step >> 1){
				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
				v0 = swap(v0,step,dir);
			}
		}

		//4
		if(k <= 4)
		{
			v0 = fmaxf(__shfl_xor_sync(0x000000FF, v0, k),v0);
			v0 = __shfl_sync(0x0000000F,v0,(threadIdx.x << 1) - (threadIdx.x & (k - 1)));
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
				v0 = swap(v0,step,dir);
			}
		}

		//2
		if(k <= 2)
		{
			v0 = fmaxf(__shfl_xor_sync(0x0000000F, v0, k),v0);
			v0 = __shfl_sync(0x00000003,v0,(threadIdx.x << 1) - (threadIdx.x & (k - 1)));
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
				v0 = swap(v0,step,dir);
			}
		}

		//1
		if(k <= 1)
		{
			v0 = fmaxf(__shfl_xor_sync(0x00000003, v0, k),v0);
			v0 = __shfl_sync(0x00000001,v0,(threadIdx.x << 1) - (threadIdx.x & (k - 1)));
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
				v0 = swap(v0,step,dir);
			}
		}


//		/*
//		 * Sort 16
//		 */
//		v0 = (threadIdx.x & k) == 0 ? v0 : 0;
//		for(level = k; level < 32; level = level << 1){
//			for(step = level; step > 0; step = step >> 1){
//				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
//				v0 = rswap(v0,step,dir);
//			}
//		}

		/*
		 * Write-back heaps of each partition
		 */
		if(threadIdx.x < k)
		{
			if((blockIdx.x & 0x1) == 0) gscores[blockIdx.x * k + threadIdx.x] = v0;
			else gscores[blockIdx.x * k + (threadIdx.x ^ (k-1))] = v0;
		}
	}
}

template<class T>
__global__ void agg_lsort_geq_32(T *gdata, uint64_t n, uint64_t qq, uint64_t k, T *gscores){
	__shared__ T buffer[BTA_TUPLES_PER_BLOCK];
	#if BTA_TUPLES_PER_BLOCK >= 1024
		T v0 = 0, v1 = 0, v2 = 0, v3 = 0;
	#endif
	#if BTA_TUPLES_PER_BLOCK >= 2048
		T v4 = 0, v5 = 0, v6 = 0, v7 = 0;
	#endif
	#if BTA_TUPLES_PER_BLOCK >= 4096
		T v8 = 0, v9 = 0, vA = 0, vB = 0, vC = 0, vD = 0, vE = 0, vF = 0;
	#endif

	/*
	 * Aggregate
	 */
	for(uint32_t m = 0; m < qq; m++)
	{
		uint32_t ai = gpu_query[m];
		uint32_t index = n * ai + blockIdx.x * BTA_TUPLES_PER_BLOCK + threadIdx.x;
		T w = gpu_weights[ai];
		#if BTA_TUPLES_PER_BLOCK >= 1024
			v0 += gdata[index       ] * w;
			v1 += gdata[index +  256] * w;
			v2 += gdata[index +  512] * w;
			v3 += gdata[index +  768] * w;
		#endif
		#if BTA_TUPLES_PER_BLOCK >= 2048
			v4 += gdata[index + 1024] * w;
			v5 += gdata[index + 1280] * w;
			v6 += gdata[index + 1536] * w;
			v7 += gdata[index + 1792] * w;
		#endif
		#if BTA_TUPLES_PER_BLOCK >= 4096
			v8 += gdata[index + 2048] * w;
			v9 += gdata[index + 2304] * w;
			vA += gdata[index + 2560] * w;
			vB += gdata[index + 2816] * w;
			vC += gdata[index + 3072] * w;
			vD += gdata[index + 3328] * w;
			vE += gdata[index + 3584] * w;
			vF += gdata[index + 3840] * w;
		#endif
	}

	/*
	 * Sort using registers
	 */
	uint32_t level, step, dir, i;
	for(level = 1; level < 32; level = level << 1){
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
			#if BTA_TUPLES_PER_BLOCK >= 1024
				v0 = swap(v0,step,dir);
				v1 = swap(v1,step,dir);
				v2 = swap(v2,step,dir);
				v3 = swap(v3,step,dir);
			#endif
			#if BTA_TUPLES_PER_BLOCK >= 2048
				v4 = swap(v4,step,dir);
				v5 = swap(v5,step,dir);
				v6 = swap(v6,step,dir);
				v7 = swap(v7,step,dir);
			#endif
			#if BTA_TUPLES_PER_BLOCK >= 4096
				v8 = swap(v8,step,dir);
				v9 = swap(v9,step,dir);
				vA = swap(vA,step,dir);
				vB = swap(vB,step,dir);
				vC = swap(vC,step,dir);
				vD = swap(vD,step,dir);
				vE = swap(vE,step,dir);
				vF = swap(vF,step,dir);
			#endif
		}
	}

	#if BTA_TUPLES_PER_BLOCK >= 1024
		buffer[threadIdx.x       ] = v0;
		buffer[threadIdx.x +  256] = v1;
		buffer[threadIdx.x +  512] = v2;
		buffer[threadIdx.x +  768] = v3;
	#endif
	#if BTA_TUPLES_PER_BLOCK >= 2048
		buffer[threadIdx.x + 1024] = v4;
		buffer[threadIdx.x + 1280] = v5;
		buffer[threadIdx.x + 1536] = v6;
		buffer[threadIdx.x + 1792] = v7;
	#endif
	#if BTA_TUPLES_PER_BLOCK >= 4096
		buffer[threadIdx.x + 2048] = v8;
		buffer[threadIdx.x + 2304] = v9;
		buffer[threadIdx.x + 2560] = vA;
		buffer[threadIdx.x + 2816] = vB;
		buffer[threadIdx.x + 3072] = vC;
		buffer[threadIdx.x + 3328] = vD;
		buffer[threadIdx.x + 3584] = vE;
		buffer[threadIdx.x + 3840] = vF;
	#endif
	__syncthreads();

	//Sort in shared memory//
	for(level = 32; level < k; level = level << 1){
		dir = level << 1;
		for(step = level; step > 0; step = step >> 1){
			i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
			bool r = ((dir & i) == 0);
			#if BTA_TUPLES_PER_BLOCK >= 1024
				swap_shared<T>(buffer[i       ], buffer[i +        step], r);
				swap_shared<T>(buffer[i +  512], buffer[i +  512 + step], r);
			#endif
			#if BTA_TUPLES_PER_BLOCK >= 2048
				swap_shared<T>(buffer[i + 1024], buffer[i + 1024 + step], r);
				swap_shared<T>(buffer[i + 1536], buffer[i + 1536 + step], r);
			#endif
			#if BTA_TUPLES_PER_BLOCK >= 4096
				swap_shared<T>(buffer[i + 2048], buffer[i + 2048 + step], r);
				swap_shared<T>(buffer[i + 2560], buffer[i + 2560 + step], r);
				swap_shared<T>(buffer[i + 3072], buffer[i + 3072 + step], r);
				swap_shared<T>(buffer[i + 3584], buffer[i + 3584 + step], r);
			#endif
			__syncthreads();
		}
	}

	//////////////////////////////////////////////
	//////////Reduce-Rebuild 4096 - 2048//////////
	#if BTA_TUPLES_PER_BLOCK >= 4096
		i = (threadIdx.x << 1) - (threadIdx.x & (k - 1));
		v0 = fmaxf(buffer[i       ], buffer[i +        k]);
		v1 = fmaxf(buffer[i +  512], buffer[i +  512 + k]);
		v2 = fmaxf(buffer[i + 1024], buffer[i + 1024 + k]);
		v3 = fmaxf(buffer[i + 1536], buffer[i + 1536 + k]);
		v4 = fmaxf(buffer[i + 2048], buffer[i + 2048 + k]);
		v5 = fmaxf(buffer[i + 2560], buffer[i + 2560 + k]);
		v6 = fmaxf(buffer[i + 3072], buffer[i + 3072 + k]);
		v7 = fmaxf(buffer[i + 3584], buffer[i + 3584 + k]);
		__syncthreads();
		buffer[threadIdx.x       ] = v0;
		buffer[threadIdx.x +  256] = v1;
		buffer[threadIdx.x +  512] = v2;
		buffer[threadIdx.x +  768] = v3;
		buffer[threadIdx.x + 1024] = v4;
		buffer[threadIdx.x + 1280] = v5;
		buffer[threadIdx.x + 1536] = v6;
		buffer[threadIdx.x + 1792] = v7;
		__syncthreads();
		level = k >> 1;
		dir = level << 1;
		for(step = level; step > 0; step = step >> 1){
			i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
			bool r = ((dir & i) == 0);
			swap_shared<T>(buffer[i       ], buffer[i +        step], r);
			swap_shared<T>(buffer[i +  512], buffer[i +  512 + step], r);
			swap_shared<T>(buffer[i + 1024], buffer[i + 1024 + step], r);
			swap_shared<T>(buffer[i + 1536], buffer[i + 1536 + step], r);
			__syncthreads();
		}
	#endif

	//////////////////////////////////////////////
	//////////Reduce-Rebuild 2048 - 1024//////////
	#if BTA_TUPLES_PER_BLOCK >= 2048
		i = (threadIdx.x << 1) - (threadIdx.x & (k - 1));
		v0 = fmaxf(buffer[i       ], buffer[i +        k]);
		v1 = fmaxf(buffer[i +  512], buffer[i +  512 + k]);
		v2 = fmaxf(buffer[i + 1024], buffer[i + 1024 + k]);
		v3 = fmaxf(buffer[i + 1536], buffer[i + 1536 + k]);
		__syncthreads();
		buffer[threadIdx.x       ] = v0;
		buffer[threadIdx.x +  256] = v1;
		buffer[threadIdx.x +  512] = v2;
		buffer[threadIdx.x +  768] = v3;
		__syncthreads();
		level = k >> 1;
		dir = level << 1;
		for(step = level; step > 0; step = step >> 1){
			i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
			bool r = ((dir & i) == 0);
			swap_shared<T>(buffer[i       ], buffer[i +        step], r);
			swap_shared<T>(buffer[i +  512], buffer[i +  512 + step], r);
			__syncthreads();
		}
	#endif

	//////////////////////////////////////////////
	//////////Reduce-Rebuild 1024 - 512//////////
	#if BTA_TUPLES_PER_BLOCK >= 1024
		i = (threadIdx.x << 1) - (threadIdx.x & (k - 1));
		v0 = fmaxf(buffer[i       ], buffer[i +        k]);
		v1 = fmaxf(buffer[i +  512], buffer[i +  512 + k]);
		__syncthreads();
		buffer[threadIdx.x       ] = v0;
		buffer[threadIdx.x +  256] = v1;
		__syncthreads();
		level = k >> 1;
		dir = level << 1;
		for(step = level; step > 0; step = step >> 1){
			i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
			bool r = ((dir & i) == 0);
			swap_shared<T>(buffer[i       ], buffer[i +        step], r);
			__syncthreads();
		}
	#endif

	////////////////////////////////////////////
	//////////Reduce-Rebuild 512 - 256//////////
	#if BTA_TUPLES_PER_BLOCK >= 1024
		i = (threadIdx.x << 1) - (threadIdx.x & (k - 1));
		v0 = fmaxf(buffer[i       ], buffer[i +        k]);
		__syncthreads();
		buffer[threadIdx.x       ] = v0;
		__syncthreads();
		level = k >> 1;
		dir = level << 1;
		for(step = level; step > 0; step = step >> 1){
			if(threadIdx.x < 128){
				i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
				bool r = ((dir & i) == 0);
				swap_shared<T>(buffer[i       ], buffer[i +        step], r);
			}
			__syncthreads();
		}
	#endif

	if(k == 256) {
		if((blockIdx.x & 0x1) == 0) gscores[(blockIdx.x << 8) + threadIdx.x] = buffer[threadIdx.x];
		else gscores[(blockIdx.x << 8) + threadIdx.x] = buffer[(k - 1) ^ threadIdx.x];
		return ;
	}

	////////////////////////////////////////////
	//////////Reduce-Rebuild 256 - 128//////////
	if(threadIdx.x < 128){
		i = (threadIdx.x << 1) - (threadIdx.x & (k - 1));
		v0 = fmaxf(buffer[i       ], buffer[i +        k]);
	}
	__syncthreads();
	if(threadIdx.x < 128) buffer[threadIdx.x] = v0;
	__syncthreads();
	level = k >> 1;
	dir = level << 1;
	for(step = level; step > 0; step = step >> 1){
		if(threadIdx.x < 64){
			i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
			bool r = ((dir & i) == 0);
			swap_shared<T>(buffer[i       ], buffer[i +        step], r);
		}
		__syncthreads();
	}
	if(k == 128 && threadIdx.x < 128) {
		if((blockIdx.x & 0x1) == 0) gscores[(blockIdx.x << 7) + threadIdx.x] = buffer[threadIdx.x];
		else gscores[(blockIdx.x << 7) + threadIdx.x] = buffer[(k - 1) ^ threadIdx.x];
		return ;
	}

	////////////////////////////////////////////
	//////////Reduce-Rebuild 128 - 64//////////
	if(threadIdx.x < 64){
		i = (threadIdx.x << 1) - (threadIdx.x & (k - 1));
		v0 = fmaxf(buffer[i       ], buffer[i +        k]);
	}
	__syncthreads();
	if(threadIdx.x < 64) buffer[threadIdx.x] = v0;
	__syncthreads();
	level = k >> 1;
	dir = level << 1;
	for(step = level; step > 0; step = step >> 1){
		if(threadIdx.x < 32){
			i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
			bool r = ((dir & i) == 0);
			swap_shared<T>(buffer[i       ], buffer[i +        step], r);
		}
		__syncthreads();
	}
	if(k == 64 && threadIdx.x < 64) {
		if((blockIdx.x & 0x1) == 0) gscores[(blockIdx.x << 6) + threadIdx.x] = buffer[threadIdx.x];
		else gscores[(blockIdx.x << 6) + threadIdx.x] = buffer[(k - 1) ^ threadIdx.x];
		return ;
	}

	////////////////////////////////////////////
	//////////Reduce-Rebuild 64 - 32//////////
	if(threadIdx.x < 32){
		i = (threadIdx.x << 1) - (threadIdx.x & (k - 1));
		v0 = fmaxf(buffer[i       ], buffer[i +        k]);
	}
	__syncthreads();
	if(threadIdx.x < 32) buffer[threadIdx.x] = v0;
	__syncthreads();
	level = k >> 1;
	dir = level << 1;
	for(step = level; step > 0; step = step >> 1){
		if(threadIdx.x < 16){
			i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
			bool r = ((dir & i) == 0);
			swap_shared<T>(buffer[i       ], buffer[i +        step], r);
		}
		__syncthreads();
	}
	if(k == 32 && threadIdx.x < 32) {
		if((blockIdx.x & 0x1) == 0) gscores[(blockIdx.x << 5) + threadIdx.x] = buffer[threadIdx.x];
		else gscores[(blockIdx.x << 5) + threadIdx.x] = buffer[(k - 1) ^ threadIdx.x];
		return;
	}

}

#endif

