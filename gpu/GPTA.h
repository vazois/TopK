#ifndef GPTA_H
#define GPTA_H
/*
 * GPU Parallel Threshold Algorithm
 */

#include "GAA.h"
#include "../tools/tools.h"
#include <map>

#define ALPHA 1.0
#define GPTA_PI 3.1415926535
#define GPTA_PI_2 (180.0f/PI)

#define GPTA_SPLITS 2
#define GPTA_PARTS (((uint64_t)pow(GPTA_SPLITS,NUM_DIMS-1)))
#define GPTA_PART_BITS ((uint64_t)log2f(GPTA_SPLITS))

#define GPTA_BLOCK_SIZE 1024

template<class T, class Z>
struct gpta_block{
	T data[GPTA_BLOCK_SIZE * NUM_DIMS];
	T tvector[NUM_DIMS];
	Z offset = 0;
};

template<class T, class Z>
struct gpta_part{
	gpta_block<T,Z> *blocks = NULL;
	uint32_t bnum = 0;
	uint32_t size = 0;
};

template<class Z>
__global__ void update_minimum_pos(Z *pos, Z* tid_vec, uint64_t n);
template<class Z>
__global__ void init_pos(Z *pos, uint64_t n);
template<class Z>
__global__ void init_tid_vec(Z *tid_vec, uint64_t n);
__global__ void assign(uint32_t *keys, uint32_t *part, uint64_t n, uint32_t mul);
__global__ void init_part(uint32_t *part, uint64_t n);
__global__ void init_keys(uint32_t *keys, uint64_t n);

template<class T>
__global__ void init_num_vec(T *data, uint64_t n, uint64_t m, T *num_vec);

template<class T>
__global__ void next_angle(T *data, uint64_t n, int m, T *num_vec, T *angle_vec);

template<class T, class Z>
class GPTA : public GAA<T,Z>{
	public:
		GPTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "GPTA";

			this->cparts = NULL;
			this->part_size = NULL;
		};

		~GPTA(){
			if(this->part_size) cudaFreeHost(this->part_size);
			if(this->cparts)
			{
				for(uint64_t i = 0; i < GPTA_PARTS; i++) if(this->cparts[i].blocks) cudaFreeHost(this->cparts[i].blocks);
				cudaFreeHost(this->cparts);
			}
		};

		void alloc();
		void init();
		void findTopK(uint64_t k, uint64_t qq);

	private:
		gpta_part<T,Z> *cparts;
		gpta_part<T,Z> *gparts;
		Z *part_tid;
		uint32_t *part_size;
		uint32_t max_part_size;

		void polar_partition();
		void reorder_partition();

		T cpuTopK(uint64_t k, uint64_t qq);
};

template<class T, class Z>
void GPTA<T,Z>::alloc(){
	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");// Allocate cpu data memory
	cutil::safeMallocHost<Z,uint64_t>(&(this->part_size),sizeof(Z)*GPTA_PARTS,"part size alloc");// Allocate cpu data memory
}

template<class T, class Z>
void GPTA<T,Z>::init(){
	normalize_transpose<T>(this->cdata, this->n, this->d);
	this->tt_init = 0;
	//this->t.start();
	this->polar_partition();
	this->reorder_partition();
	//this->tt_init=this->t.lap();
}

template<class T, class Z>
void GPTA<T,Z>::polar_partition()
{
	uint64_t mem = 0;
	uint32_t *keys_in;
	uint32_t *keys_out;
	T *values_out;

	T *num_vec; // Numerator vector
	T *angle_vec; // Angle vector
	Z *part; // Partition assignment vector
	dim3 polar_block(256,1,1);
	dim3 polar_grid(((this->n - 1)/256) + 1, 1, 1);

	//Allocate buffers for partitioning
	#if USE_POLAR_DEV_MEM//ONLY DEVICE MEMORY
		cutil::safeMalloc<Z,uint64_t>(&keys_in,sizeof(Z)*this->n,"keys_in alloc"); mem+=this->n * sizeof(Z);
		cutil::safeMalloc<Z,uint64_t>(&keys_out,sizeof(Z)*this->n,"keys_out alloc"); mem+=this->n * sizeof(Z);
		cutil::safeMalloc<T,uint64_t>(&values_out,sizeof(T)*this->n,"values_out alloc"); mem+=this->n * sizeof(T);

		cutil::safeMalloc<Z,uint64_t>(&part,sizeof(Z)*this->n,"part alloc"); mem+=this->n * sizeof(Z);
		cutil::safeMalloc<T,uint64_t>(&num_vec,sizeof(T)*this->n,"numerator alloc"); mem+=this->n * sizeof(T);
		cutil::safeMalloc<T,uint64_t>(&angle_vec,sizeof(T)*this->n,"numerator alloc"); mem+=this->n * sizeof(T);
	#else//ONLY HOST MEMORY
		cutil::safeMallocHost<Z,uint64_t>(&keys_in,sizeof(Z)*this->n,"keys_in alloc"); mem+=this->n * sizeof(Z);
		cutil::safeMallocHost<Z,uint64_t>(&keys_out,sizeof(Z)*this->n,"keys_out alloc"); mem+=this->n * sizeof(Z);
		cutil::safeMallocHost<T,uint64_t>(&values_out,sizeof(T)*this->n,"values_out alloc"); mem+=this->n * sizeof(T);

		cutil::safeMallocHost<Z,uint64_t>(&part,sizeof(Z)*this->n,"part alloc"); mem+=this->n * sizeof(Z);
		cutil::safeMallocHost<T,uint64_t>(&num_vec,sizeof(T)*this->n,"numerator alloc"); mem+=this->n * sizeof(T);
		cutil::safeMallocHost<T,uint64_t>(&angle_vec,sizeof(T)*this->n,"numerator alloc"); mem+=this->n * sizeof(T);
	#endif

	//initialize tmp buffer for sorting//
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs(d_temp_storage,temp_storage_bytes, angle_vec, values_out, keys_in, keys_out, this->n);
	cutil::cudaCheckErr(cudaMalloc(&d_temp_storage, temp_storage_bytes),"alloc d_temp_storage"); mem+=temp_storage_bytes;
	std::cout << "temp_storage_bytes: " << temp_storage_bytes << "," << this->n << std::endl;
	std::cout << "PARTITION ASSIGNMENT MEMORY OVERHEAD: " << ((double)mem)/(1 << 20) << " MB" << std::endl;

	//assign tuples to partitions//
	this->t.start();
	init_part<<<polar_grid,polar_block>>>(part,this->n);//initialize part vector
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_part");
	init_num_vec<T><<<polar_grid, polar_block>>>(this->cdata,this->n,this->d-1,num_vec);//initialize numerator vector
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_num_vec");
	this->tt_init += this->t.lap();
	uint32_t mul = 1;
	for(int m = this->d-1; m > 0; m--)
	{
		this->t.start();
		//a: calculate next angle
		next_angle<T><<<polar_grid,polar_block>>>(this->cdata,this->n,m-1,num_vec,angle_vec);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_num_vec");

		//b: initialize keys for sorting
		init_keys<<<polar_grid,polar_block>>>(keys_in,this->n);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_keys");

		//c: sort according to angle
		cub::DeviceRadixSort::SortPairs(d_temp_storage,temp_storage_bytes, angle_vec, values_out, keys_in, keys_out, this->n);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing SortPairs");

		//d: assign to partition by adding offset value
		assign<<<polar_grid,polar_block>>>(keys_out,part,this->n,mul);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing assign");
		this->tt_init += this->t.lap();

		mul *= GPTA_SPLITS;
	}

	//Gather partition size information//
	Z *cpart;
	#if USE_POLAR_DEV_MEM
		cutil::safeMallocHost<Z,uint64_t>(&cpart,sizeof(Z)*this->n,"keys_in alloc");
		cutil::safeCopyToHost<Z,uint64_t>(cpart,part,sizeof(Z)*this->n,"copy part to cpart");
	#else
		cpart = part;
	#endif

	this->t.start();
	max_part_size = 0;
	for(uint64_t i = 0; i < GPTA_PARTS; i++) part_size[i] = 0;
	for(uint64_t i = 0; i<this->n; i++){
		part_size[cpart[i]]++;
		max_part_size = std::max(max_part_size,part_size[cpart[i]]);
		if(cpart[i]>=GPTA_PARTS){ std::cout << "ERROR (polar): " << i << "," << cpart[i] << std::endl;  exit(1); }
	}
	this->tt_init += this->t.lap();
	//for(uint64_t i = 0; i < GPTA_PARTS; i++){ std::cout << i << " = " << part_size[i] << std::endl; }
	//std::cout << "max_part_size: " << this->max_part_size << std::endl;

	////////////////
	//Free buffers//
	#if USE_POLAR_DEV_MEM
		cudaFree(keys_in);
		cudaFree(keys_out);
		cudaFree(values_out);
		cudaFree(num_vec);
		cudaFree(angle_vec);
		cudaFreeHost(cpart);
	#else
		cudaFreeHost(keys_in);
		cudaFreeHost(keys_out);
		cudaFreeHost(values_out);
		cudaFreeHost(num_vec);
		cudaFreeHost(angle_vec);
	#endif

	////////////////////////////////////
	//Group tids in the same partition//
	Z *tid_in;
	Z *tid_out;
	Z *part_out;
	mem = temp_storage_bytes;
	#if USE_PART_REORDER_DEV_MEM
		cutil::safeMalloc<Z,uint64_t>(&tid_in,sizeof(Z)*this->n,"alloc tid_in"); mem+=this->n * sizeof(Z);
		cutil::safeMalloc<Z,uint64_t>(&tid_out,sizeof(Z)*this->n,"alloc tid_out"); mem+=this->n * sizeof(Z);
		cutil::safeMalloc<Z,uint64_t>(&part_out,sizeof(Z)*this->n,"alloc part_out"); mem+=this->n * sizeof(Z);
	#else
		cutil::safeMallocHost<Z,uint64_t>(&tid_in,sizeof(Z)*this->n,"alloc tid_in"); mem+=this->n * sizeof(Z);
		cutil::safeMallocHost<Z,uint64_t>(&tid_out,sizeof(Z)*this->n,"alloc tid_out"); mem+=this->n * sizeof(Z);
		cutil::safeMallocHost<Z,uint64_t>(&part_out,sizeof(Z)*this->n,"alloc part_out"); mem+=this->n * sizeof(Z);
	#endif
	std::cout << "PART REORDER MEMORY OVERHEAD: " << ((double)mem)/(1 << 20) << " MB" << std::endl;

	//e:Sort tuples according to partition assignment
	this->t.start();
	init_keys<<<polar_grid,polar_block>>>(tid_in,this->n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_keys for sorting according to partition");
	cub::DeviceRadixSort::SortPairs(d_temp_storage,temp_storage_bytes, part, part_out, tid_in, tid_out, this->n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing SortPairs for sorting according to partition");
	this->tt_init += this->t.lap();

	//Copy ordered tuple ids ordered by partition assignment
	cutil::safeMallocHost<Z,uint64_t>(&this->part_tid,sizeof(Z)*this->n,"alloc part_tid");
	cutil::safeCopyToHost<Z,uint64_t>(this->part_tid,tid_out,sizeof(Z)*this->n,"copy tid_out to part_tid");

	#if USE_PART_REORDER_DEV_MEM
		cudaFree(tid_in);
		cudaFree(tid_out);
		cudaFree(part_out);
	#else
		cudaFreeHost(tid_in);
		cudaFreeHost(tid_out);
		cudaFreeHost(part_out);
	#endif
	cudaFree(d_temp_storage);
}

template<class T, class Z>
void GPTA<T,Z>::reorder_partition(){
	uint64_t mem = 0;
	T *cattr_vec_in;
	Z *gtid_vec_in, *gtid_vec_out, *ctid_vec_out;
	T *gattr_vec_in, *gattr_vec_out;
	Z *cpos, *gpos, *gpos_out;

	dim3 reorder_block(256,1,1);
	dim3 reorder_grid(((this->max_part_size - 1)/256) + 1, 1, 1);

	cutil::safeMallocHost<T,uint64_t>(&cattr_vec_in,sizeof(T)*this->max_part_size,"cattr_vec_in alloc");
	cutil::safeMallocHost<Z,uint64_t>(&ctid_vec_out,sizeof(Z)*this->max_part_size,"ctid_vec_out alloc");
	cutil::safeMallocHost<Z,uint64_t>(&cpos,sizeof(Z)*this->max_part_size,"cpos alloc");

	//Device memory
	cutil::safeMalloc<T,uint64_t>(&gattr_vec_in,sizeof(T)*this->max_part_size,"gattr_vec_in alloc"); mem += sizeof(T)*this->max_part_size;
	cutil::safeMalloc<T,uint64_t>(&gattr_vec_out,sizeof(T)*this->max_part_size,"gattr_vec_out alloc"); mem += sizeof(T)*this->max_part_size;
	cutil::safeMalloc<Z,uint64_t>(&gtid_vec_in,sizeof(Z)*this->max_part_size,"gtid_vec_in alloc"); mem += sizeof(Z)*this->max_part_size;
	cutil::safeMalloc<Z,uint64_t>(&gtid_vec_out,sizeof(Z)*this->max_part_size,"gtid_vec_out alloc"); mem += sizeof(Z)*this->max_part_size;
	cutil::safeMalloc<Z,uint64_t>(&gpos,sizeof(Z)*this->max_part_size,"gpos alloc"); mem += sizeof(Z)*this->max_part_size;
	cutil::safeMalloc<Z,uint64_t>(&gpos_out,sizeof(Z)*this->max_part_size,"gpos_out alloc"); mem += sizeof(Z)*this->max_part_size;

	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortPairsDescending(d_temp_storage,temp_storage_bytes, gattr_vec_in, gattr_vec_out, gtid_vec_in, gtid_vec_out, this->max_part_size);
	cutil::cudaCheckErr(cudaMalloc(&d_temp_storage, temp_storage_bytes),"alloc d_temp_storage"); mem+=temp_storage_bytes;
	std::cout << "BUILD PARTITION MEMORY OVERHEAD: " << ((double)mem)/(1 << 20) << " MB" << std::endl;

	uint64_t part_offset = 0;
	cutil::safeMallocHost<gpta_part<T,Z>,uint64_t>(&this->cparts,sizeof(gpta_part<T,Z>)*GPTA_PARTS,"cparts alloc");
	for(uint64_t i = 0; i < GPTA_PARTS; i++)
	{
		this->cparts[i].size = this->part_size[i];
		this->cparts[i].bnum = ((this->part_size[i] - 1)/GPTA_BLOCK_SIZE) + 1;
		cutil::safeMallocHost<gpta_block<T,Z>,uint64_t>(&this->cparts[i].blocks,sizeof(gpta_block<T,Z>)*this->cparts[i].bnum,"gpta_block alloc");

		//initialize minimum positions//
		this->t.start();
		init_pos<Z><<<reorder_grid,reorder_block>>>(gpos,this->cparts[i].size);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_pos");
		this->tt_init += this->t.lap();

		//Find local order for tuples//
		this->t.start();
		for(uint64_t m = 0; m < this->d; m++)
		{
			//a: create vector with m-th attribute
			for(uint64_t j = 0; j < this->cparts[i].size; j++)
			{
				Z tid = this->part_tid[part_offset + j];
				cattr_vec_in[j] = this->cdata[m*this->n + tid];
			}

			//b: copy vector with attribute values
			cutil::safeCopyToDevice<T,uint64_t>(gattr_vec_in,cattr_vec_in,sizeof(T)*this->cparts[i].size,"copy cattr_vec_in to gattr_vec_in");

			//c: initialize local key vector
			init_tid_vec<Z><<<reorder_grid,reorder_block>>>(gtid_vec_in,this->cparts[i].size);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_tid_vec");

			//d: sort local tids in ascending order//
			cub::DeviceRadixSort::SortPairsDescending(d_temp_storage,temp_storage_bytes, gattr_vec_in, gattr_vec_out, gtid_vec_in, gtid_vec_out, this->cparts[i].size);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing Sort Pairs Descending");

			//e: update minimum position
			update_minimum_pos<Z><<<reorder_grid,reorder_block>>>(gpos,gtid_vec_out,this->cparts[i].size);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing update minimum pos");
		}
		this->tt_init += this->t.lap();

		this->t.start();
		//f:reorder based on first seen position
		init_tid_vec<Z><<<reorder_grid,reorder_block>>>(gtid_vec_in,this->cparts[i].size);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_tid_vec");
		cub::DeviceRadixSort::SortPairs(d_temp_storage,temp_storage_bytes, gpos, gpos_out, gtid_vec_in, gtid_vec_out, this->cparts[i].size);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing Sort Pairs Descending");
		this->tt_init += this->t.lap();

		//Build partition//
		cutil::safeCopyToHost<Z,uint64_t>(ctid_vec_out,gtid_vec_out,sizeof(Z)*this->cparts[i].size,"copy gtid_vec_out to ctid_vec_out");
		uint64_t b = 0;
		this->t.start();
		for(Z j = 0; j < this->cparts[i].size; j+=GPTA_BLOCK_SIZE)
		{
			Z size = (j + GPTA_BLOCK_SIZE) < this->cparts[i].size ? GPTA_BLOCK_SIZE : this->cparts[i].size - j;
			T *data = this->cparts[i].blocks[b].data;
			this->cparts[i].blocks[b].offset = j;
			//std::cout << b << "," << this->cparts[i].bnum << std::endl;
			for(Z jj = 0; jj < GPTA_BLOCK_SIZE; jj++)
			{
				if( jj < size ){
					Z tid = this->part_tid[part_offset + ctid_vec_out[j + jj]];
					for(uint64_t m = 0; m < this->d; m++) data[m * GPTA_BLOCK_SIZE + jj] = this->cdata[m * this->n + tid];
				}else{
					for(uint64_t m = 0; m < this->d; m++) data[m * GPTA_BLOCK_SIZE + jj] = 0;
				}
			}
			b++;
		}
		this->tt_init += this->t.lap();
		part_offset += this->cparts[i].size;
		//if(i == 0) break;
	}

	//Calculate thresholds//
	T mx[NUM_DIMS];
	this->t.start();
	for(uint64_t m = 0; m < this->d; m++) mx[m] = 0;
	for(uint64_t i = 0; i < GPTA_PARTS; i++){
		for(uint64_t b = this->cparts[i].bnum - 1; b > 0; b--){
			T *tvector = this->cparts[i].blocks[b-1].tvector;
			T *data = this->cparts[i].blocks[b].data;
			for(uint64_t j = 0; j < GPTA_BLOCK_SIZE; j++){
				for(uint64_t m = 0; m < this->d; m++) mx[m] = std::max(mx[m],data[m * GPTA_BLOCK_SIZE + j]);
			}
			std::memcpy(tvector,mx,sizeof(T)*NUM_DIMS);
		}
	}
	this->tt_init += this->t.lap();

	//Validate thresholds//
//	for(uint64_t i = 0; i < 1; i++){
//		for(uint64_t b = 0; b < this->cparts[i].bnum; b++){
//			std::cout << std::fixed << std::setprecision(4);
//			std::cout << "[" << std::setfill('0') << std::setw(3) << b <<  "] ";
//			for(uint64_t m = 0; m < this->d; m++){
//				std::cout << this->cparts[i].blocks[b].tvector[ m ] << " ";
//			}
//			std::cout << std::endl;
//		}
//	}

	for(uint64_t i = 0; i < GPTA_PARTS; i++){
		for(uint64_t b = 1; b < this->cparts[i].bnum; b++){
			for(uint64_t m = 0; m < this->d; m++){
				if(this->cparts[i].blocks[b].tvector[ m ] > this->cparts[i].blocks[b-1].tvector[ m ]){
					std::cout << "{ERROR}" << std::endl;
					std::cout << std::fixed << std::setprecision(4);
					std::cout << "[" << std::setfill('0') << std::setw(3) << b <<  "] ";
					for(uint64_t mm = 0; mm < this->d; mm++){ std::cout << this->cparts[i].blocks[b].tvector[ mm ] << " "; }
					std::cout << std::endl;
					std::cout << "[" << std::setfill('0') << std::setw(3) << b <<  "] ";
					for(uint64_t mm = 0; mm < this->d; mm++){ std::cout << this->cparts[i].blocks[b].tvector[ mm ] << " "; }
					std::cout << std::endl;
					exit(1);
				}
			}
		}
	}

	cudaFreeHost(ctid_vec_out);
	cudaFreeHost(cattr_vec_in);
	cudaFreeHost(cpos);

	//Device Memory
	cudaFree(gattr_vec_in);
	cudaFree(gattr_vec_out);
	cudaFree(gtid_vec_in);
	cudaFree(gtid_vec_out);
	cudaFree(gpos);
	cudaFree(gpos_out);

	cudaFree(d_temp_storage);
}

template<class T, class Z>
T GPTA<T,Z>::cpuTopK(uint64_t k, uint64_t qq){
	std::priority_queue<T, std::vector<ranked_tuple<T,Z>>, MaxFirst<T,Z>> q;

//	T threshold = 0;
//	for(uint64_t i = 0; i < this->n; i++)
//	{
//		T score = 0;
//		for(uint64_t m = 0; m < qq; m++)
//		{
//			uint32_t ai = this->query[m];
//			score += this->cdata[ai * this->n + i] * this->weights[ai];
//		}
//
//		if(q.size() < k)
//		{
//			q.push(ranked_tuple<T,Z>(i,score));
//		}else if( q.top().score < score ){
//			q.pop();
//			q.push(ranked_tuple<T,Z>(i,score));
//		}
//	}
//	threshold = q.top().score;

	q = std::priority_queue<T, std::vector<ranked_tuple<T,Z>>, MaxFirst<T,Z>>();
	T threshold2 = 0;
	for(uint64_t i = 0; i < GPTA_PARTS; i++)
	{
		gpta_block<T,Z> *blocks = this->cparts[i].blocks;
		for(uint64_t b = 0; b < this->cparts[i].bnum; b++)
		{
			T *data = blocks[b].data;
			T *tvector = blocks[b].tvector;
			for(uint64_t j = 0; j < GPTA_BLOCK_SIZE; j++)
			{
				T score = 0;
				for(uint64_t m = 0; m < qq; m++)
				{
					Z ai = this->query[m];
					score += data[ai*GPTA_BLOCK_SIZE + j] * this->weights[ai];
				}
				if(q.size() < k)
				{
					q.push(ranked_tuple<T,Z>(i,score));
				}else if( q.top().score < score ){
					q.pop();
					q.push(ranked_tuple<T,Z>(i,score));
				}
				this->tuple_count+=1;
			}
			T t = 0;
			for(uint64_t m = 0; m < qq; m++){
				Z ai = this->query[m];
				t += tvector[ai] * this->weights[ai];
			}
			if(q.size() >= k && q.top().score >= t){
				//std::cout << "{BREAK}: " << i << "," << b << " < " << this->cparts[i].bnum << std::endl;
				break;
			}
		}
	}
	threshold2 = q.top().score;
	//std::cout << "[" << threshold << "," << threshold2 << "]" << std::endl;
	return threshold2;
}

template<class T, class Z>
void GPTA<T,Z>::findTopK(uint64_t k, uint64_t qq){
	this->tuple_count=0;
	this->cpu_threshold = this->cpuTopK(k,qq);
}

template<class Z>
__global__ void update_minimum_pos(Z *pos, Z* tid_vec, uint64_t n)
{
	Z i = blockIdx.x * blockDim.x + threadIdx.x;

	if( i < n )
	{
		Z ltid = tid_vec[i];
		pos[ltid] = min(pos[ltid],i);
	}
}

template<class Z>
__global__ void init_pos(Z *pos, uint64_t n)
{
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if( i < n ) pos[i] = n;
}

template<class Z>
__global__ void init_tid_vec(Z *tid_vec, uint64_t n)
{
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if( i < n ) tid_vec[i] = i;
}

__global__ void assign(uint32_t *keys_out, uint32_t *part, uint64_t n, uint32_t mul)
{
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if( i < n )
	{
		uint32_t tid = keys_out[i];
		part[tid] = part[tid] + (i/(((n-1)/GPTA_SPLITS)+1)) * mul;
	}
}

__global__ void init_part(uint32_t *part, uint64_t n)
{
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n) part[i] = 0;
}

__global__ void init_keys(uint32_t *keys, uint64_t n)
{
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n) keys[i] = i;
}

template<class T>
__global__ void init_num_vec(T *data, uint64_t n, uint64_t m, T *num_vec)
{
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < n)
	{
		T num = 0;
		num = (ALPHA - data[m*n + i]);
		num = num * num;
		num_vec[i] = num;
	}
}

template<class T>
__global__ void next_angle(T *data, uint64_t n, int m, T *num_vec, T *angle_vec)
{
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < n)
	{
		T num = 0;
		T dnm = 0;
		T angle = 0;

		num = num_vec[i];//Load previous numerator
		dnm = (ALPHA - data[m*n + i]);//Load current denominator

		angle = atan(sqrtf(num)/dnm);//calculate tan(Fj)

		num_vec[i] += dnm * dnm;
		angle_vec[i] = angle;
	}
}

#endif
