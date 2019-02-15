#ifndef GVTA_H
#define GVTA_H

#include "GAA.h"

//PTA DEVICE MEMORY USAGE//
#define GVTA_USE_DEV_MEM_REORDER true
#define GVTA_USE_DEV_MEM_PROCESSING true

#define GVTA_PARTITIONS 256 //at least 8 partitions//
#define GVTA_BLOCK_SIZE 4096

template<class T,class Z>
struct gvta_block
{
	gvta_block(){};
	T data[GVTA_BLOCK_SIZE * GVTA_PARTITIONS * NUM_DIMS];
	T tvector[GVTA_PARTITIONS * NUM_DIMS];
	Z num_tuples;
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
__global__ void gvta_atm_16(gvta_block<T,Z> *blocks, uint64_t nb, uint64_t qq, uint64_t k, T *out);

template<class T, class Z>
class GVTA : public GAA<T,Z>{
	public:
		GVTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "GVTA";
			cudaStreamCreate(&s0);
			cudaStreamCreate(&s1);

			this->cblocks = NULL;
			this->gblocks = NULL;

			this->cout = NULL;
			this->cout2 = NULL;
			this->gout = NULL;
			this->gout2 = NULL;

			this->using_dev_mem = GVTA_USE_DEV_MEM_PROCESSING;
			this->parts = GVTA_PARTITIONS;
			this->block_size = GVTA_BLOCK_SIZE;
		};

		~GVTA()
		{
			cudaStreamDestroy(s0);
			cudaStreamDestroy(s1);
			if(this->cblocks) cutil::safeCudaFreeHost(this->cblocks, "free cblocks"); //cudaFreeHost(this->blocks);
			if(this->cout) cutil::safeCudaFreeHost(this->cout,"free cout"); //cudaFreeHost(cout);
			if(this->cout2) cutil::safeCudaFreeHost(this->cout2,"free cout2"); //cudaFreeHost(cout);

			#if GVTA_USE_DEV_MEM_PROCESSING
				if(this->gblocks) cutil::safeCudaFree(this->gblocks,"free gblocks"); //cudaFree(this->gblocks);
				if(this->gout) cutil::safeCudaFree(this->gout,"free gout"); //cudaFree(gout);
				if(this->gout2) cutil::safeCudaFree(this->gout2,"free gout2"); //cudaFree(gout2);
			#endif
		};

		void alloc();
		void init();
		void findTopK(uint64_t k, uint64_t qq);

	private:
		uint64_t tuples_per_part;
		uint64_t num_blocks;
		gvta_block<T,Z> *cblocks = NULL;
		gvta_block<T,Z> *gblocks = NULL;
		T *cout = NULL;
		T *cout2 = NULL;
		T *gout = NULL;
		T *gout2 = NULL;

		void cclear();
		void reorder();
		T cpuTopK(uint64_t k, uint64_t qq);

		void atm_16(uint64_t k, uint64_t qq);
		void atm_16_driver(uint64_t k, uint64_t qq);

		cudaStream_t s0;
		cudaStream_t s1;
};

template<class T, class Z>
void GVTA<T,Z>::alloc(){
	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");// Allocate cpu data memory
}

template<class T, class Z>
void GVTA<T,Z>::reorder()
{
	uint64_t mem = 0;
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
	cutil::safeMallocHost<gvta_block<T,Z>,uint64_t>(&(this->cblocks),sizeof(gvta_block<T,Z>)*this->num_blocks,"alloc host gvta_blocks");
	//std::cout << this->n << " = p,psz(" << GVTA_PARTITIONS <<"," << this->tuples_per_part << ") - " << "b,bsz(" << this->num_blocks << "," << GVTA_BLOCK_SIZE << ")" << std::endl;

	/////////////////////////////////////////////
	//2: Allocate temporary space for reordering
	//hkeys = (Z*)malloc(sizeof(Z)*this->tuples_per_part);
	cutil::safeMallocHost<Z,uint64_t>(&hkeys,sizeof(Z)*this->tuples_per_part,"hkeys alloc"); mem += sizeof(Z)*this->tuples_per_part;
	cutil::safeMallocHost<Z,uint64_t>(&(grvector),sizeof(Z)*this->tuples_per_part,"alloc rvector"); mem += sizeof(Z)*this->tuples_per_part;
	cutil::safeMallocHost<Z,uint64_t>(&(grvector_out),sizeof(Z)*this->tuples_per_part,"alloc rvector"); mem += sizeof(Z)*this->tuples_per_part;
	cutil::safeMallocHost<Z,uint64_t>(&(dkeys_in),sizeof(Z)*this->tuples_per_part,"alloc dkeys_in"); mem += sizeof(Z)*this->tuples_per_part;
	cutil::safeMallocHost<Z,uint64_t>(&(dkeys_out),sizeof(Z)*this->tuples_per_part,"alloc dkeys_out"); mem += sizeof(Z)*this->tuples_per_part;
	cutil::safeMallocHost<T,uint64_t>(&(dvalues_in),sizeof(T)*this->tuples_per_part,"alloc dvalues_in"); mem += sizeof(T)*this->tuples_per_part;
	cutil::safeMallocHost<T,uint64_t>(&(dvalues_out),sizeof(T)*this->tuples_per_part,"alloc dvalues_out"); mem += sizeof(T)*this->tuples_per_part;
	cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, dkeys_in, dkeys_out, dvalues_in, dvalues_out, this->tuples_per_part);
	cutil::cudaCheckErr(cudaMalloc(&d_temp_storage, temp_storage_bytes),"alloc d_temp_storage");
	std::cout << "BUILD PARTITION MEMORY OVERHEAD: " << ((double)mem)/(1 << 20) << " MB" << std::endl;

	uint64_t offset = 0;
	//a: for each partition

	for(uint64_t i = 0; i < GVTA_PARTITIONS; i++){
		//b:Find partition size
		uint64_t psize = offset + this->tuples_per_part < this->n ? this->tuples_per_part : (this->n - offset);
		grid.x = (this->tuples_per_part-1)/block.x + 1;

		this->t.start();
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
		this->tt_init += this->t.lap();

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
						this->cblocks[bi].data[GVTA_BLOCK_SIZE * GVTA_PARTITIONS * m + (i * GVTA_BLOCK_SIZE + jj)] = v;
						tvector[m] = std::max(tvector[m],v);
					}
				}else{//initialize to negative if block is partially full
					for(uint64_t m = 0; m < this->d; m++)
					{
						this->cblocks[bi].data[GVTA_BLOCK_SIZE * GVTA_PARTITIONS * m + (i *GVTA_BLOCK_SIZE + jj)] = -1.0;
					}
				}
			}
			if(bi > 0){
				T *tvec = this->cblocks[bi-1].tvector;
				for(uint64_t m = 0; m < this->d; m++){
					//tvec[ GVTA_PARTITIONS * m  + i] = tvector[m];
					tvec[ GVTA_PARTITIONS * i  + m] = tvector[m];
				}
			}
			bi++;
		}
		//
		offset += this->tuples_per_part;
	}

//	for(uint64_t i = 0; i < GVTA_PARTITIONS; i++){
//		T mx[NUM_DIMS];
//		for(uint64_t m = 0; m < this->d; m++) mx[m] = 0;
//		for(uint64_t b = this->num_blocks - 1; b > 0; b--){
//			for(uint64_t j = i * GVTA_BLOCK_SIZE; j < (i+1) * GVTA_BLOCK_SIZE; j++){
//				for(uint64_t m = 0; m < this->d; m++){
//					mx[m] = std::max(mx[m], this->cblocks[b].data[GVTA_BLOCK_SIZE * GVTA_PARTITIONS * m + j]);
//				}
//				T *tvec = &this->cblocks[b-1].tvector[i*this->d];
//				std::memcpy(tvec,mx,sizeof(T)*this->d);
//			}
//		}
//	}

	#if VALIDATE
		for(uint64_t i = 0; i < GVTA_PARTITIONS; i++){
			for(uint64_t b = 1; b < this->num_blocks; b++){
				for(uint64_t m = 0; m < this->d; m++){
					if( this->cblocks[b].tvector[ GVTA_PARTITIONS * m  + i] > this->cblocks[b-1].tvector[ GVTA_PARTITIONS * m  + i] )
					{
						std::cout << std::fixed << std::setprecision(4);
						std::cout << "ERROR" << std::endl;
						std::cout << "[" << std::setfill('0') << std::setw(3) << i << "," << b <<  "] ";
						for(uint64_t mm = 0; mm < this->d; mm++){ std::cout << this->cblocks[b-1].tvector[ GVTA_PARTITIONS * mm  + i] << " "; }
						std::cout << std::endl << "+++++++++++++++++++++" << std::endl;
						std::cout << "[" << std::setfill('0') << std::setw(3) << i << "," << b <<  "] ";
						for(uint64_t mm = 0; mm < this->d; mm++){ std::cout << this->cblocks[b].tvector[ GVTA_PARTITIONS * mm  + i] << " "; }
						std::cout << std::endl;
						exit(1);
					}
				}
			}
		}
	#endif

	/////////////////////////
	//Free not needed space//
	cutil::safeCudaFree(d_temp_storage,"free d_temp_storage"); //cudaFree(d_temp_storage);
	cutil::safeCudaFreeHost(grvector,"free grvector"); //cudaFreeHost(grvector);
	cutil::safeCudaFreeHost(grvector_out,"free grvector_out"); //cudaFreeHost(grvector_out);
	cutil::safeCudaFreeHost(dkeys_in,"free dkeys_in"); //cudaFreeHost(dkeys_in);
	cutil::safeCudaFreeHost(dkeys_out,"free dkeys_out"); //cudaFreeHost(dkeys_out);
	cutil::safeCudaFreeHost(dvalues_in,"free dvalues_in");//cudaFreeHost(dvalues_in);
	cutil::safeCudaFreeHost(dvalues_out,"free dvalues_out"); //cudaFreeHost(dvalues_out);
	cutil::safeCudaFreeHost(hkeys,"free hkeys"); //cudaFreeHost(hkeys);
	#if !VALIDATE
		cutil::safeCudaFreeHost(this->cdata,"free cdata"); //cudaFreeHost(this->cdata);
	#endif
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
	this->reorder();

	#if GVTA_USE_DEV_MEM_PROCESSING
		cutil::safeMalloc<gvta_block<T,Z>,uint64_t>(&(this->gblocks),sizeof(gvta_block<T,Z>)*this->num_blocks,"alloc gpu gvta_blocks");
		cutil::safeCopyToDevice<gvta_block<T,Z>,uint64_t>(this->gblocks,this->cblocks,sizeof(gvta_block<T,Z>)*this->num_blocks,"error copying to gpu gvta_blocks");
	#else
		this->gblocks = this->cblocks;
	#endif

	cutil::safeMallocHost<T,uint64_t>(&cout,sizeof(T) * GVTA_PARTITIONS * KKE,"cout alloc");
	#if GVTA_USE_DEV_MEM_PROCESSING
		cutil::safeMalloc<T,uint64_t>(&gout,sizeof(T) * GVTA_PARTITIONS * KKE,"gout alloc");
		cutil::safeMalloc<T,uint64_t>(&gout2,sizeof(T) * GVTA_PARTITIONS * KKE,"gout2 alloc");
	#else
		cutil::safeMallocHost<T,uint64_t>(&cout2,sizeof(T) * GVTA_PARTITIONS * KKE,"cout alloc");
		gout = cout;
		gout2 = cout2;
	#endif
}

template<class T, class Z>
T GVTA<T,Z>::cpuTopK(uint64_t k, uint64_t qq){
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

template<class T,class Z>
void GVTA<T,Z>::atm_16_driver(uint64_t k, uint64_t qq){
	dim3 atm_16_block(256,1,1);
	dim3 atm_16_grid(GVTA_PARTITIONS, 1, 1);

	this->t.start();
	gvta_atm_16<T,Z><<<atm_16_grid,atm_16_block,256*sizeof(T)>>>(this->gblocks,this->num_blocks,qq,k,gout);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing gvta_atm_16");
	this->tt_processing += this->t.lap("");

	//First step check
	#if GVTA_USE_DEV_MEM_PROCESSING
		cutil::safeCopyToHost<T,uint64_t>(cout,gout,sizeof(T) * GVTA_PARTITIONS * k, "error copying from gout to out");
	#endif
	std::sort(cout, cout + GVTA_PARTITIONS * k,std::greater<T>());
	this->gpu_threshold = cout[k-1];

	uint64_t remainder = (GVTA_PARTITIONS * k);
	this->t.start();
	while(remainder > k){
		//std::cout << "remainder: " << remainder << std::endl;
		atm_16_grid.x = ((remainder - 1) / 4096) + 1;
		reduce_rebuild_atm_16<T><<<atm_16_grid,atm_16_block>>>(gout,remainder,k,gout2);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"executing gpta_rr_atm_16");
		remainder = (atm_16_grid.x * k);
		std::swap(gout,gout2);
	}
	this->tt_processing += this->t.lap();

	//Second step check
	#if GVTA_USE_DEV_MEM_PROCESSING
		cutil::safeCopyToHost<T,uint64_t>(cout, gout, sizeof(T) * k, "error copying (k) from gout to out");
	#endif
	std::sort(cout, cout + k, std::greater<T>());
	this->gpu_threshold = cout[k-1];

	#if VALIDATE
		this->cpu_threshold = this->cpuTopK(k,qq);
		if( abs((double)this->gpu_threshold - (double)this->cpu_threshold) > (double)0.000001 ) {
			std::cout << std::fixed << std::setprecision(16);
			std::cout << "ERROR: {" << this->gpu_threshold << "," << this->cpu_threshold << "}" << std::endl; exit(1);
		}
	#endif
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
	gvta_atm_16<T,Z><<<atm_16_grid,atm_16_block,256*sizeof(T),s0>>>(this->gblocks,this->num_blocks,qq,k,gout);
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
//#if VALIDATE
//	std::sort(cout, cout + GVTA_PARTITIONS * k,std::greater<T>());
//	std::cout << "threshold=[" << cout[k-1] << "," << this->cpu_threshold << "," << cpu_gvagg << "]"<< std::endl;
//	if(abs((double)cout[k-1] - (double)cpu_gvagg) > (double)0.00000000000001
//			||
//			abs((double)cout[k-1] - (double)this->cpu_threshold) > (double)0.00000000000001
//			) {
//		std::cout << std::fixed << std::setprecision(16);
//		std::cout << "ERROR: {" << cout[k-1] << "," << this->cpu_threshold << "," << cpu_gvagg << "}" << std::endl; exit(1);
//	}
//#endif
}

template<class T, class Z>
void GVTA<T,Z>::findTopK(uint64_t k, uint64_t qq){
	this->cclear();
	this->atm_16_driver(k,qq);
}

template<class T, class Z>
__global__ void gvta_atm_16(gvta_block<T,Z> *blocks, uint64_t nb, uint64_t qq, uint64_t k, T *out)
{
	uint64_t b = 0;//data block level
	uint64_t offset = blockIdx.x * nb * GVTA_BLOCK_SIZE;

	__shared__ T threshold[NUM_DIMS+1];
	__shared__ T heap[32];
	__shared__ T buffer[256];
	if(threadIdx.x < 32) heap[threadIdx.x] = 0;
	while(b < nb)
	{
		#if GVTA_BLOCK_SIZE >= 1024
			T v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0;
		#endif
		#if GVTA_BLOCK_SIZE >= 4096
			T v8 = 0, v9 = 0, vA = 0, vB = 0, vC = 0, vD = 0, vE = 0, vF = 0;
		#endif
		T *data = &blocks[b].data[blockIdx.x * GVTA_BLOCK_SIZE];

		if(threadIdx.x < NUM_DIMS)
		{
			threshold[threadIdx.x] = blocks[b].tvector[GVTA_PARTITIONS * blockIdx.x + threadIdx.x];
			if(threadIdx.x == 0) threshold[NUM_DIMS] = 0;
		}

		for(uint32_t m = 0; m < qq; m++)
		{
			uint32_t ai = gpu_query[m];
			uint32_t start = GVTA_BLOCK_SIZE * GVTA_PARTITIONS * ai + threadIdx.x;
			T w = gpu_weights[ai];
			if(threadIdx.x == 0) threshold[NUM_DIMS] += threshold[ai] * w;

			#if GVTA_BLOCK_SIZE >= 1024
				v0 += data[start       ] * w;
				v1 += data[start + 256 ] * w;
				v2 += data[start + 512 ] * w;
				v3 += data[start + 768 ] * w;
			#endif
			#if GVTA_BLOCK_SIZE >= 2048
				v4 += data[start + 1024] * w;
				v5 += data[start + 1280] * w;
				v6 += data[start + 1536] * w;
				v7 += data[start + 1792] * w;
			#endif
			#if GVTA_BLOCK_SIZE >= 4096
				v8 += data[start + 2048] * w;
				v9 += data[start + 2304] * w;
				vA += data[start + 2560] * w;
				vB += data[start + 2816] * w;
				vC += data[start + 3072] * w;
				vD += data[start + 3328] * w;
				vE += data[start + 3584] * w;
				vF += data[start + 3840] * w;
			#endif
		}

		/*
		 * {4096,2048,1024} -> {2048,1024,512}
		 */
		uint32_t laneId = threadIdx.x;
		uint32_t level, step, dir;
		for(level = 1; level < k; level = level << 1){
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				#if GVTA_BLOCK_SIZE >= 1024
					v0 = swap(v0,step,dir);
					v1 = swap(v1,step,dir);
					v2 = swap(v2,step,dir);
					v3 = swap(v3,step,dir);
				#endif
				#if GVTA_BLOCK_SIZE >= 2048
					v4 = swap(v4,step,dir);
					v5 = swap(v5,step,dir);
					v6 = swap(v6,step,dir);
					v7 = swap(v7,step,dir);
				#endif
				#if GVTA_BLOCK_SIZE >= 4096
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
		#if GVTA_BLOCK_SIZE >= 1024
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
			v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
			v3 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v3, k),v3);
			v0 = (threadIdx.x & k) == 0 ? v0 : v1;
			v2 = (threadIdx.x & k) == 0 ? v2 : v3;
		#endif
		#if GVTA_BLOCK_SIZE >= 2048
			v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
			v5 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v5, k),v5);
			v6 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v6, k),v6);
			v7 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v7, k),v7);
			v4 = (threadIdx.x & k) == 0 ? v4 : v5;
			v6 = (threadIdx.x & k) == 0 ? v6 : v7;
		#endif
		#if GVTA_BLOCK_SIZE >= 4096
			v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
			v9 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v9, k),v9);
			vA = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vA, k),vA);
			vB = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vB, k),vB);
			vC = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vC, k),vC);
			vD = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vD, k),vD);
			vE = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vE, k),vE);
			vF = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vF, k),vF);
			v8 = (threadIdx.x & k) == 0 ? v8 : v9;
			vA = (threadIdx.x & k) == 0 ? vA : vB;
			vC = (threadIdx.x & k) == 0 ? vC : vD;
			vE = (threadIdx.x & k) == 0 ? vE : vF;
		#endif

		/*
		 * {2048,1024,512} -> {1024,512,256}
		 */
		level = k >> 1;
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
			#if GVTA_BLOCK_SIZE >= 1024
				v0 = swap(v0,step,dir);
				v2 = swap(v2,step,dir);
			#endif
			#if GVTA_BLOCK_SIZE >= 2048
				v4 = swap(v4,step,dir);
				v6 = swap(v6,step,dir);
			#endif
			#if GVTA_BLOCK_SIZE >= 4096
				v8 = swap(v8,step,dir);
				vA = swap(vA,step,dir);
				vC = swap(vC,step,dir);
				vE = swap(vE,step,dir);
			#endif
		}
		#if GVTA_BLOCK_SIZE >= 1024
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
			v0 = (threadIdx.x & k) == 0 ? v0 : v2;
		#endif
		#if GVTA_BLOCK_SIZE >= 2048
			v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
			v6 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v6, k),v6);
			v4 = (threadIdx.x & k) == 0 ? v4 : v6;
		#endif
		#if GVTA_BLOCK_SIZE >= 4096
			v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
			vA = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vA, k),vA);
			vC = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vC, k),vC);
			vE = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vE, k),vE);
			v8 = (threadIdx.x & k) == 0 ? v8 : vA;
			vC = (threadIdx.x & k) == 0 ? vC : vE;
		#endif

		/*
		 * {1024,512} -> {512,256}
		 */
		#if GVTA_BLOCK_SIZE >= 2048
			level = k >> 1;
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
				v0 = swap(v0,step,dir);
				v4 = swap(v4,step,dir);
				#if GVTA_BLOCK_SIZE >= 4096
					v8 = swap(v8,step,dir);
					vC = swap(vC,step,dir);
				#endif
			}

			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
			v0 = (threadIdx.x & k) == 0 ? v0 : v4;
			#if GVTA_BLOCK_SIZE >= 4096
				v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
				vC = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vC, k),vC);
				v8 = (threadIdx.x & k) == 0 ? v8 : vC;
			#endif
		#endif

		/*
		 * {512} -> {256}
		 */
		#if GVTA_BLOCK_SIZE >= 4096
				level = k >> 1;
				for(step = level; step > 0; step = step >> 1){
					dir = bfe(laneId,__ffs(level))^bfe(laneId,__ffs(step>>1));
					v0 = swap(v0,step,dir);
					v8 = swap(v8,step,dir);
				}
				v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
				v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
				v0 = (threadIdx.x & k) == 0 ? v0 : v8;
		#endif

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
			if(b == 0)
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
		if(heap[k-1] >= threshold[NUM_DIMS]) break;
		__syncthreads();
		b++;
	}

	/*
	 * Write-back heaps of each partition
	 */
	if(threadIdx.x < k){
		offset = blockIdx.x * k;
		if((blockIdx.x & 0x1) == 0) out[offset + (k-1) - threadIdx.x] = heap[threadIdx.x];
		else out[offset + threadIdx.x] = heap[threadIdx.x];
	}
}

#endif
