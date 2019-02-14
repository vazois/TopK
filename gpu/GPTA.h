#ifndef GPTA_H
#define GPTA_H
/*
 * GPU Parallel Threshold Algorithm
 */

#include "GAA.h"
#include "../tools/tools.h"
#include <map>

//PTA DEVICE MEMORY USAGE//
#define USE_POLAR_DEV_MEM false // USE DEVICE MEMORY TO COMPUTE POLAR COORDINATES
#define USE_PART_REORDER_DEV_MEM true
#define USE_BUILD_PART_DEV_MEM true
#define USE_PTA_DEVICE_MEM true
//

#define ALPHA 1.1
#define GPTA_PI 3.1415926535
#define GPTA_PI_2 (180.0f/PI)

#define GPTA_SPLITS 2
#define GPTA_PARTS (((uint64_t)pow(GPTA_SPLITS,NUM_DIMS-1)))
#define GPTA_PART_BITS ((uint64_t)log2f(GPTA_SPLITS))

#define GPTA_BLOCK_SIZE 4096

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

template<class T, class Z>
__global__ void gpta_geq_32(gpta_part<T,Z> *gparts, uint64_t qq, uint64_t k, T *out);
template<class T, class Z>
__global__ void gpta_atm_16(gpta_part<T,Z> *gparts, uint64_t qq, uint64_t k, T *out);
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
			this->cprts = NULL;
			this->gparts = NULL;
			this->part_size = NULL;
			this->cout = NULL;
			this->gout = NULL;
			this->cout2 = NULL;
			this->gout2 = NULL;

			this->using_dev_mem = USE_PTA_DEVICE_MEM;

			this->parts = GPTA_PARTS;
			this->block_size = GPTA_BLOCK_SIZE;
		};

		~GPTA(){
			if(this->part_size) cutil::safeCudaFreeHost(this->part_size,"free part_size"); //cudaFreeHost(this->part_size);
			if(this->cparts)
			{
				for(uint64_t i = 0; i < GPTA_PARTS; i++) if(this->cparts[i].blocks) cutil::safeCudaFreeHost(this->cparts[i].blocks,"free cparts blocks"); //cudaFreeHost(this->cparts[i].blocks);
				cutil::safeCudaFreeHost(this->cparts,"free cparts"); //cudaFreeHost(this->cparts);
			}

			if(this->cprts)
			{
				for(uint64_t i = 0; i < GPTA_PARTS; i++) if(this->cparts[i].blocks) cutil::safeCudaFree(this->cprts[i].blocks,"free cprts blocks"); //cudaFree(this->cprts[i].blocks);
				cutil::safeCudaFreeHost(this->cprts,"free cprts"); //cudaFreeHost(this->cprts);
			}

			if(this->gparts)
			{
				cutil::safeCudaFree(this->gparts,"free gparts"); //cudaFree(this->gparts);
			}

			if(this->cout) cutil::safeCudaFreeHost(this->cout,"free cout");
			if(this->cout2) cutil::safeCudaFreeHost(this->cout2,"free cout2");
			if(this->gout) cutil::safeCudaFree(this->gout,"free gout");
			if(this->gout2) cutil::safeCudaFree(this->gout2,"free gout2");
		};

		void alloc();
		void init();
		void findTopK(uint64_t k, uint64_t qq);

	private:
		gpta_part<T,Z> *cparts;
		gpta_part<T,Z> *cprts;
		gpta_part<T,Z> *gparts;
		Z *part_tid;
		uint32_t *part_size;
		uint32_t max_part_size;
		T *cout = NULL;
		T *gout = NULL;
		T *cout2 = NULL;
		T *gout2 = NULL;

		void polar_partition();
		void reorder_partition();

		void atm_16_driver(uint64_t k, uint64_t qq);
		void geq_32_driver(uint64_t k, uint64_t qq);

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

	cutil::safeMallocHost<T,uint64_t>(&cout,sizeof(T) * GPTA_PARTS * KKE,"cout alloc");
	cutil::safeMallocHost<T,uint64_t>(&cout2,sizeof(T) * GPTA_PARTS * KKE,"cout2 alloc");
	#if USE_PTA_DEVICE_MEM
		std::cout << "ALLOCATING DEVICE MEMORY" << std::endl;
		cutil::safeMalloc<T,uint64_t>(&gout, sizeof(T) * GPTA_PARTS * KKE,"gout alloc");
		cutil::safeMalloc<T,uint64_t>(&gout2, sizeof(T) * GPTA_PARTS * KKE,"gout2 alloc");

		cutil::safeMalloc<gpta_part<T,Z>,uint64_t>(&this->gparts,sizeof(gpta_part<T,Z>)*GPTA_PARTS,"gparts alloc");
		cutil::safeMallocHost<gpta_part<T,Z>,uint64_t>(&this->cprts,sizeof(gpta_part<T,Z>)*GPTA_PARTS,"cprts alloc");
		for(uint64_t i = 0; i < GPTA_PARTS; i++)
		{
			this->cprts[i].size = this->cparts[i].size;
			this->cprts[i].bnum = this->cparts[i].bnum;

			cutil::safeMalloc<gpta_block<T,Z>,uint64_t>(&(this->cprts[i].blocks),sizeof(gpta_block<T,Z>)*this->cprts[i].bnum,"cprts gpta_block alloc");
			cutil::safeCopyToDevice<gpta_block<T,Z>,uint64_t>(this->cprts[i].blocks,this->cparts[i].blocks,sizeof(gpta_block<T,Z>)*this->cprts[i].bnum, "copy cparts to cprts");
		}
		cutil::safeCopyToDevice<gpta_part<T,Z>,uint64_t>(this->gparts,this->cprts,sizeof(gpta_part<T,Z>)*GPTA_PARTS,"copy cprts to gparts");
		cutil::cudaCheckErr(cudaPeekAtLastError(),"copy cprts to gparts");
	#else
		this->gparts = this->cparts;
		this->gout = cout;
		this->gout = cout2;
	#endif
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

//		if( m == 1 ){
//			for(uint32_t i = 0; i < 32; i++)
//			{
//				std::cout << std::fixed << std::setprecision(4);
//				std::cout << "[" << std::setfill('0') << std::setw(8) << i << "] : {" << angle_vec[i] << "} = ";
//				for(uint64_t mm = 0; mm < this->d; mm++){ std::cout << this->cdata[mm*this->n + i] << " "; } std::cout << std::endl;
//			}
//		}

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
//	for(uint64_t i = 0; i<64; i++){
//		std::cout << "[" << std::setfill('0') << std::setw(8) << i << "] : {" << std::setfill('0') << std::setw(3) << cpart[i] << "} = ";
//		std::cout << std::fixed << std::setprecision(4);
//		for(uint64_t m = 0; m < this->d; m++){ std::cout << this->cdata[m*this->n + i] << " "; } std::cout << std::endl;
//	}

	max_part_size = 0;
	for(uint64_t i = 0; i < GPTA_PARTS; i++) part_size[i] = 0;
	for(uint64_t i = 0; i<this->n; i++){
		part_size[cpart[i]]++;
		max_part_size = std::max(max_part_size,part_size[cpart[i]]);
		if(cpart[i]>=GPTA_PARTS){ std::cout << "ERROR (polar): " << i << "," << cpart[i] << std::endl;  exit(1); }
	}
	this->tt_init += this->t.lap();
//	for(uint64_t i = 0; i < GPTA_PARTS; i++){
//		std::cout << "g(" << i << "):" << std::setfill('0') << std::setw(8) << part_size[i] << std::endl;
//	}
//	std::cout << "max_part_size: " << this->max_part_size << std::endl;

	////////////////
	//Free buffers//
	#if USE_POLAR_DEV_MEM
		cutil::safeCudaFree(keys_in,"free keys_in"); //cudaFree(keys_in);
		cutil::safeCudaFree(keys_out,"free keys_out"); //cudaFree(keys_out);
		cutil::safeCudaFree(values_out,"free values_out"); //cudaFree(values_out);
		cutil::safeCudaFree(num_vec,"free num_vec"); //cudaFree(num_vec);
		cutil::safeCudaFree(angle_vec,"free angle_vec"); //cudaFree(angle_vec);
		cutil::safeCudaFreeHost(cpart,"free cpart"); //cudaFreeHost(cpart);
	#else
		cutil::safeCudaFreeHost(keys_in,"free keys_in"); //cudaFreeHost(keys_in);
		cutil::safeCudaFreeHost(keys_out,"free keys_out"); //cudaFreeHost(keys_out);
		cutil::safeCudaFreeHost(values_out,"free values_out"); //cudaFreeHost(values_out);
		cutil::safeCudaFreeHost(num_vec,"free num_vec"); //cudaFreeHost(num_vec);
		cutil::safeCudaFreeHost(angle_vec,"free angle_vec"); //cudaFreeHost(angle_vec);
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
		cutil::safeCudaFree(tid_in,"free tid_in"); //cudaFree(tid_in);
		cutil::safeCudaFree(tid_out,"free tid_out"); //cudaFree(tid_out);
		cutil::safeCudaFree(part_out,"free part_out"); //cudaFree(part_out);
	#else
		cutil::safeCudaFreeHost(tid_in,"free tid_in"); //cudaFreeHost(tid_in);
		cutil::safeCudaFreeHost(tid_out,"free tid_out"); //cudaFreeHost(tid_out);
		cutil::safeCudaFreeHost(part_out,"free part_out"); //cudaFreeHost(part_out);
	#endif
	cutil::safeCudaFree(d_temp_storage,"free d_temp_storage"); //cudaFree(d_temp_storage);
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
		cutil::safeMallocHost<gpta_block<T,Z>,uint64_t>(&this->cparts[i].blocks,sizeof(gpta_block<T,Z>)*this->cparts[i].bnum,"cparts gpta_block alloc");

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

			T mx[NUM_DIMS];
			for(uint64_t m = 0; m < this->d; m++) mx[m] = 0;
			for(Z jj = 0; jj < GPTA_BLOCK_SIZE; jj++)
			{
				if( jj < size ){
					Z tid = this->part_tid[part_offset + ctid_vec_out[j + jj]];
					for(uint64_t m = 0; m < this->d; m++){
						T v = this->cdata[m * this->n + tid];
						data[m * GPTA_BLOCK_SIZE + jj] = v;
						mx[m] = std::max(mx[m],v);
					}
				}else{
					for(uint64_t m = 0; m < this->d; m++) data[m * GPTA_BLOCK_SIZE + jj] = 0;
				}
			}
			b++;
		}
		this->tt_init += this->t.lap();
		part_offset += this->cparts[i].size;
	}

	//Calculate thresholds//
	this->t.start();
	for(uint64_t i = 0; i < GPTA_PARTS; i++){
		T mx[NUM_DIMS];
		for(uint64_t m = 0; m < this->d; m++) mx[m] = 0;
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

	cutil::safeCudaFreeHost(ctid_vec_out,"free ctid_vec_out"); //cudaFreeHost(ctid_vec_out);
	cutil::safeCudaFreeHost(cattr_vec_in,"free cattr_vec_in"); //cudaFreeHost(cattr_vec_in);
	cutil::safeCudaFreeHost(cpos,"free cpos"); //cudaFreeHost(cpos);

	//Device Memory
	cutil::safeCudaFree(gattr_vec_in,"free gattr_vec_in"); //cudaFree(gattr_vec_in);
	cutil::safeCudaFree(gattr_vec_out,"free gattr_vec_out"); //cudaFree(gattr_vec_out);
	cutil::safeCudaFree(gtid_vec_in,"free gtid_vec_in"); //cudaFree(gtid_vec_in);
	cutil::safeCudaFree(gtid_vec_out,"free gtid_vec_out"); //cudaFree(gtid_vec_out);
	cutil::safeCudaFree(gpos,"free gpos"); //cudaFree(gpos);
	cutil::safeCudaFree(gpos_out,"free gpos_out"); //cudaFree(gpos_out);

	cutil::safeCudaFree(d_temp_storage,"free d_temp_storage"); //cudaFree(d_temp_storage);
}

template<class T, class Z>
T GPTA<T,Z>::cpuTopK(uint64_t k, uint64_t qq){
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

	q = std::priority_queue<T, std::vector<ranked_tuple<T,Z>>, MaxFirst<T,Z>>();
	T threshold2 = 0;
	for(uint64_t i = 0; i < GPTA_PARTS; i++)
	{
		gpta_block<T,Z> *blocks = this->cparts[i].blocks;
		for(uint64_t b = 0; b < this->cparts[i].bnum; b++)
		{
			T *data = blocks[b].data;
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
			T *tvector = blocks[b].tvector;
			for(uint64_t m = 0; m < qq; m++){
				Z ai = this->query[m];
				t += tvector[ai] * this->weights[ai];
			}
			if(q.size() >= k && q.top().score >= t){ break; }
		}
	}
	threshold2 = q.top().score;
	if( abs((double)threshold - (double)threshold2) > (double)0.000001) {
		std::cout << "ERROR (cpu):[" << threshold << "," << threshold2 << "]" << std::endl;
	}
	return threshold2;
}

template<class T, class Z>
void GPTA<T,Z>::findTopK(uint64_t k, uint64_t qq){
	this->tuple_count=0;
	if( k <= 16 ){
		this->atm_16_driver(k,qq);
	}else if ( k <= 256 ){
		this->geq_32_driver(k,qq);
	}else{
		std::cout << "GPU Maximum K = 256" << std::endl;
		exit(1);
	}
}

template<class T, class Z>
void GPTA<T,Z>::atm_16_driver(uint64_t k, uint64_t qq)
{
	dim3 atm_16_block(256,1,1);
	dim3 atm_16_grid(GPTA_PARTS, 1, 1);

	this->t.start();
	gpta_atm_16<T,Z><<<atm_16_grid,atm_16_block>>>(gparts, qq, k, gout);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"executing gpta_atm_16");
	this->tt_processing += this->t.lap();
	//First step check
//	#if USE_PTA_DEVICE_MEM
//		cutil::safeCopyToHost<T,uint64_t>(cout, gout, sizeof(T) * GPTA_PARTS * k, "error copying from gout to out");
//	#endif
//	std::sort(cout, cout + GPTA_PARTS * k, std::greater<T>());
//	this->gpu_threshold = cout[k-1];

	uint64_t remainder = (GPTA_PARTS * k);
	this->t.start();
	while(remainder > k){
		std::cout << "remainder: " << remainder << std::endl;
		atm_16_grid.x = ((remainder - 1) / 4096) + 1;
		reduce_rebuild_atm_16<T><<<atm_16_grid,atm_16_block>>>(gout,remainder,k,gout2);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"executing gpta_rr_atm_16");
		remainder = (atm_16_grid.x * k);
		std::swap(gout,gout2);
	}
	this->tt_processing += this->t.lap();

	//Second step check
	#if USE_PTA_DEVICE_MEM
		cutil::safeCopyToHost<T,uint64_t>(cout, gout, sizeof(T) * k, "error copying (k) from gout to out");
	#endif
	this->gpu_threshold = cout[k-1];

	#if VALIDATE
		this->cpu_threshold = this->cpuTopK(k,qq);
		if( abs((double)this->gpu_threshold - (double)this->cpu_threshold) > (double)0.000001 ) {
			std::cout << std::fixed << std::setprecision(16);
			std::cout << "ERROR: {" << cout[k-1] << "," << this->cpu_threshold << "}" << std::endl; exit(1);
		}
	#endif
}

template<class T, class Z>
void GPTA<T,Z>::geq_32_driver(uint64_t k, uint64_t qq)
{
	dim3 geq_32_block(256,1,1);
	dim3 geq_32_grid(GPTA_PARTS, 1, 1);

	this->t.start();
	gpta_geq_32<T,Z><<<geq_32_grid,geq_32_block>>>(gparts, qq, k, gout);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"executing gpta_geq_32");
	this->tt_processing += this->t.lap();

	#if USE_PTA_DEVICE_MEM
		cutil::safeCopyToHost<T,uint64_t>(cout, gout, sizeof(T) * GPTA_PARTS * k, "error copying from gout to out");
	#endif
	std::sort(cout, cout + GPTA_PARTS * k, std::greater<T>());
	this->gpu_threshold = cout[k-1];

	#if VALIDATE
		this->cpu_threshold = this->cpuTopK(k,qq);
		if( abs((double)this->gpu_threshold - (double)this->cpu_threshold) > (double)0.000001 ) {
			std::cout << std::fixed << std::setprecision(16);
			std::cout << "ERROR: {" << cout[k-1] << "," << this->cpu_threshold << "}" << std::endl; exit(1);
		}
	#endif
}

template<class T, class Z>
__global__ void gpta_geq_32(gpta_part<T,Z> *gparts, uint64_t qq, uint64_t k, T *out)
{
	__shared__ T threshold[NUM_DIMS+1];
	__shared__ T heap[256];
	__shared__ T buffer[GPTA_BLOCK_SIZE];

	uint32_t b = 0;
	uint32_t nb = gparts[blockIdx.x].bnum;
	heap[threadIdx.x] = 0;
	while(b < nb)
	{
		#if GPTA_BLOCK_SIZE >= 1024
			T v0 = 0, v1 = 0, v2 = 0, v3 = 0;
		#endif
		#if GPTA_BLOCK_SIZE >= 1024
			T v4 = 0, v5 = 0, v6 = 0, v7 = 0;
		#endif
		#if GPTA_BLOCK_SIZE >= 4096
			T v8 = 0, v9 = 0, vA = 0, vB = 0;
			T vC = 0, vD = 0, vE = 0, vF = 0;
		#endif
		T *data = gparts[blockIdx.x].blocks[b].data;

		if(threadIdx.x < NUM_DIMS)
		{
			threshold[threadIdx.x] = gparts[blockIdx.x].blocks[b].tvector[threadIdx.x];
			if(threadIdx.x == 0) threshold[NUM_DIMS] = 0;
		}

		for(uint32_t m = 0; m < qq; m++)
		{
			Z ai = gpu_query[m];
			Z start = ai * GPTA_BLOCK_SIZE + threadIdx.x;
			T w = gpu_weights[ai];

			if(threadIdx.x == 0) threshold[NUM_DIMS] += threshold[ai] * w;
			#if GPTA_BLOCK_SIZE >= 1024
				v0 += data[start       ] * w;
				v1 += data[start +  256] * w;
				v2 += data[start +  512] * w;
				v3 += data[start +  768] * w;
			#endif
			#if GPTA_BLOCK_SIZE >= 2048
				v4 += data[start + 1024] * w;
				v5 += data[start + 1280] * w;
				v6 += data[start + 1536] * w;
				v7 += data[start + 1792] * w;
			#endif
			#if GPTA_BLOCK_SIZE >= 4096
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

		uint32_t level, step, dir, i;
		for(level = 1; level < k; level = level << 1){
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
				#if GPTA_BLOCK_SIZE >= 1024
					v0 = swap(v0,step,dir);
					v1 = swap(v1,step,dir);
					v2 = swap(v2,step,dir);
					v3 = swap(v3,step,dir);
				#endif
				#if GPTA_BLOCK_SIZE >= 2048
					v4 = swap(v4,step,dir);
					v5 = swap(v5,step,dir);
					v6 = swap(v6,step,dir);
					v7 = swap(v7,step,dir);
				#endif
				#if GPTA_BLOCK_SIZE >= 4096
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
		buffer[threadIdx.x       ] = v0;
		buffer[threadIdx.x +  256] = v1;
		buffer[threadIdx.x +  512] = v2;
		buffer[threadIdx.x +  768] = v3;
		buffer[threadIdx.x + 1024] = v4;
		buffer[threadIdx.x + 1280] = v5;
		buffer[threadIdx.x + 1536] = v6;
		buffer[threadIdx.x + 1792] = v7;
		buffer[threadIdx.x + 2048] = v8;
		buffer[threadIdx.x + 2304] = v9;
		buffer[threadIdx.x + 2560] = vA;
		buffer[threadIdx.x + 2816] = vB;
		buffer[threadIdx.x + 3072] = vC;
		buffer[threadIdx.x + 3328] = vD;
		buffer[threadIdx.x + 3584] = vE;
		buffer[threadIdx.x + 3840] = vF;
		__syncthreads();


		//sort in shared memory
		for(level = 32; level < k; level = level << 1){
			dir = level << 1;
			for(step = level; step > 0; step = step >> 1){
				i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
				bool r = ((dir & i) == 0);
				swap_shared<T>(buffer[i       ], buffer[i +        step], r);
				swap_shared<T>(buffer[i +  512], buffer[i +  512 + step], r);
				swap_shared<T>(buffer[i + 1024], buffer[i + 1024 + step], r);
				swap_shared<T>(buffer[i + 1536], buffer[i + 1536 + step], r);
				swap_shared<T>(buffer[i + 2048], buffer[i + 2048 + step], r);
				swap_shared<T>(buffer[i + 2560], buffer[i + 2560 + step], r);
				swap_shared<T>(buffer[i + 3072], buffer[i + 3072 + step], r);
				swap_shared<T>(buffer[i + 3584], buffer[i + 3584 + step], r);
				__syncthreads();
			}
		}

		//////////////////////////////////////////////
		//////////Reduce-Rebuild 4096 - 2048//////////
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

		//////////////////////////////////////////////
		//////////Reduce-Rebuild 2048 - 1024//////////
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

		//////////////////////////////////////////////
		//////////Reduce-Rebuild 1024 - 512//////////
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

		////////////////////////////////////////////
		//////////Reduce-Rebuild 512 - 256//////////
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

		////////////////////////////////////////////
		//////////Reduce-Rebuild 256 - 128//////////
//		if( k < 128 ){
//			if(threadIdx.x < 128){
//				i = (threadIdx.x << 1) - (threadIdx.x & (k - 1));
//				v0 = fmaxf(buffer[i       ], buffer[i +        k]);
//			}
//			__syncthreads();
//			if(threadIdx.x < 128) buffer[threadIdx.x] = v0;
//			__syncthreads();
//			level = k >> 1;
//			dir = level << 1;
//			for(step = level; step > 0; step = step >> 1){
//				if(threadIdx.x < 128){
//					i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
//					bool r = ((dir & i) == 0);
//					swap_shared<T>(buffer[i       ], buffer[i +        step], r);
//				}
//				__syncthreads();
//			}
//		}

//		////////////////////////////////////////////
//		//////////Reduce-Rebuild 128 - 64//////////
//		if( k < 64 ){
//			if(threadIdx.x < 64){
//				i = (threadIdx.x << 1) - (threadIdx.x & (k - 1));
//				v0 = fmaxf(buffer[i       ], buffer[i +        k]);
//			}
//			__syncthreads();
//			if(threadIdx.x < 64) buffer[threadIdx.x] = v0;
//			__syncthreads();
//			level = k >> 1;
//			dir = level << 1;
//			for(step = level; step > 0; step = step >> 1){
//				if(threadIdx.x < 64){
//					i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
//					bool r = ((dir & i) == 0);
//					swap_shared<T>(buffer[i       ], buffer[i +        step], r);
//				}
//				__syncthreads();
//			}
//		}
//
//		////////////////////////////////////////////
//		//////////Reduce-Rebuild 64 - 32//////////
//		if( k < 32 ){
//			if(threadIdx.x < 32){
//				i = (threadIdx.x << 1) - (threadIdx.x & (k - 1));
//				v0 = fmaxf(buffer[i       ], buffer[i +        k]);
//			}
//			__syncthreads();
//			if(threadIdx.x < 32) buffer[threadIdx.x] = v0;
//			__syncthreads();
//			level = k >> 1;
//			dir = level << 1;
//			for(step = level; step > 0; step = step >> 1){
//				if(threadIdx.x < 32){
//					i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
//					bool r = ((dir & i) == 0);
//					swap_shared<T>(buffer[i       ], buffer[i +        step], r);
//				}
//				__syncthreads();
//			}
//		}

		///////////////
		//Merge heaps//
		if(b == 0)
		{
			if(threadIdx.x < k)  heap[(k - 1) - threadIdx.x] = buffer[threadIdx.x];
		}else{
			if(threadIdx.x < k) buffer[k + threadIdx.x] = heap[threadIdx.x];

			//2k -> k
			if(threadIdx.x < k){
				i = (threadIdx.x << 1) - (threadIdx.x & (k - 1));
				v0 = fmaxf(buffer[i       ], buffer[i +        k]);
			}
			__syncthreads();
			if(threadIdx.x < k) buffer[threadIdx.x       ] = v0;
			__syncthreads();

			level = k >> 1;
			dir = level << 1;
			for(step = level; step > 0; step = step >> 1){
				if(threadIdx.x < k){
					i = (threadIdx.x << 1) - (threadIdx.x & (step - 1));
					bool r = ((dir & i) == 0);
					swap_shared<T>(buffer[i       ], buffer[i +        step], r);
				}
				__syncthreads();
			}

			if(threadIdx.x < k) heap[(k - 1) - threadIdx.x] = buffer[threadIdx.x];
		}
		__syncthreads();

		if(heap[k-1] >= threshold[NUM_DIMS]){ break; }
		__syncthreads();
		b++;
	}

	if(threadIdx.x < k){
		uint64_t offset = blockIdx.x * k;
		if((blockIdx.x & 0x1) == 0) out[offset + (k-1) - threadIdx.x] = heap[threadIdx.x];
		else out[offset + threadIdx.x] = heap[threadIdx.x];
	}
}

template<class T, class Z>
__global__ void gpta_atm_16(gpta_part<T,Z> *gparts, uint64_t qq, uint64_t k, T *out)
{
	__shared__ T threshold[NUM_DIMS+1];
	__shared__ T heap[32];
	__shared__ T buffer[256];

	uint32_t b = 0;
	uint32_t nb = gparts[blockIdx.x].bnum;
	if(threadIdx.x < 32) heap[threadIdx.x] = 0;
	while(b < nb)
	{
		#if GPTA_BLOCK_SIZE >= 1024
			T v0 = 0, v1 = 0, v2 = 0, v3 = 0;
		#endif
		#if GPTA_BLOCK_SIZE >= 1024
			T v4 = 0, v5 = 0, v6 = 0, v7 = 0;
		#endif
		#if GPTA_BLOCK_SIZE >= 4096
			T v8 = 0, v9 = 0, vA = 0, vB = 0;
			T vC = 0, vD = 0, vE = 0, vF = 0;
		#endif
		T *data = gparts[blockIdx.x].blocks[b].data;

		if(threadIdx.x < NUM_DIMS)
		{
			threshold[threadIdx.x] = gparts[blockIdx.x].blocks[b].tvector[threadIdx.x];
			if(threadIdx.x == 0) threshold[NUM_DIMS] = 0;
		}

		for(uint32_t m = 0; m < qq; m++)
		{
			Z ai = gpu_query[m];
			Z start = ai * GPTA_BLOCK_SIZE + threadIdx.x;
			T w = gpu_weights[ai];

			if(threadIdx.x == 0) threshold[NUM_DIMS] += threshold[ai] * w;
			#if GPTA_BLOCK_SIZE >= 1024
				v0 += data[start       ] * w;
				v1 += data[start +  256] * w;
				v2 += data[start +  512] * w;
				v3 += data[start +  768] * w;
			#endif
			#if GPTA_BLOCK_SIZE >= 2048
				v4 += data[start + 1024] * w;
				v5 += data[start + 1280] * w;
				v6 += data[start + 1536] * w;
				v7 += data[start + 1792] * w;
			#endif
			#if GPTA_BLOCK_SIZE >= 4096
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

		//{4096,2048,1024} -> {2048,1024,512}
		//uint32_t laneId = threadIdx.x;
		uint32_t level, step, dir;
		for(level = 1; level < k; level = level << 1){
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
				#if GPTA_BLOCK_SIZE >= 1024
					v0 = swap(v0,step,dir);
					v1 = swap(v1,step,dir);
					v2 = swap(v2,step,dir);
					v3 = swap(v3,step,dir);
				#endif
				#if GPTA_BLOCK_SIZE >= 2048
					v4 = swap(v4,step,dir);
					v5 = swap(v5,step,dir);
					v6 = swap(v6,step,dir);
					v7 = swap(v7,step,dir);
				#endif
				#if GPTA_BLOCK_SIZE >= 4096
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
		#if GPTA_BLOCK_SIZE >= 1024
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
			v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
			v3 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v3, k),v3);
			v0 = (threadIdx.x & k) == 0 ? v0 : v1;
			v2 = (threadIdx.x & k) == 0 ? v2 : v3;
		#endif
		#if GPTA_BLOCK_SIZE >= 2048
			v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
			v5 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v5, k),v5);
			v6 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v6, k),v6);
			v7 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v7, k),v7);
			v4 = (threadIdx.x & k) == 0 ? v4 : v5;
			v6 = (threadIdx.x & k) == 0 ? v6 : v7;
		#endif
		#if GPTA_BLOCK_SIZE >= 4096
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

		//{2048,1024,512} -> {1024,512,256}
		level = k >> 1;
		for(step = level; step > 0; step = step >> 1){
			dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
			#if GPTA_BLOCK_SIZE >= 1024
				v0 = swap(v0,step,dir);
				v2 = swap(v2,step,dir);
			#endif
			#if GPTA_BLOCK_SIZE >= 2048
				v4 = swap(v4,step,dir);
				v6 = swap(v6,step,dir);
			#endif
			#if GPTA_BLOCK_SIZE >= 4096
				v8 = swap(v8,step,dir);
				vA = swap(vA,step,dir);
				vC = swap(vC,step,dir);
				vE = swap(vE,step,dir);
			#endif
		}
		#if GPTA_BLOCK_SIZE >= 1024
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v2 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v2, k),v2);
			v0 = (threadIdx.x & k) == 0 ? v0 : v2;
		#endif
		#if GPTA_BLOCK_SIZE >= 2048
			v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
			v6 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v6, k),v6);
			v4 = (threadIdx.x & k) == 0 ? v4 : v6;
		#endif
		#if GPTA_BLOCK_SIZE >= 4096
			v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
			vA = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vA, k),vA);
			vC = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vC, k),vC);
			vE = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vE, k),vE);
			v8 = (threadIdx.x & k) == 0 ? v8 : vA;
			vC = (threadIdx.x & k) == 0 ? vC : vE;
		#endif

		//{1024,512} -> {512,256}
		#if GPTA_BLOCK_SIZE >= 2048
			level = k >> 1;
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
				v0 = swap(v0,step,dir);
				v4 = swap(v4,step,dir);
				#if GPTA_BLOCK_SIZE >= 4096
					v8 = swap(v8,step,dir);
					vC = swap(vC,step,dir);
				#endif
			}
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
			v0 = (threadIdx.x & k) == 0 ? v0 : v4;
			#if GPTA_BLOCK_SIZE >= 4096
				v8 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v8, k),v8);
				vC = fmaxf(__shfl_xor_sync(0xFFFFFFFF, vC, k),vC);
				v8 = (threadIdx.x & k) == 0 ? v8 : vC;
			#endif
		#endif

		//{512} -> {256}
		#if GPTA_BLOCK_SIZE >= 4096
			level = k >> 1;
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
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

			//256 -> 128
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

			//128 -> 64
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
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

			//64 -> 32
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
				v0 = swap(v0,step,dir);
				v4 = swap(v4,step,dir);
			}
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v4 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v4, k),v4);
			v0 = (threadIdx.x & k) == 0 ? v0 : v4;

			//32 -> 16
			for(step = level; step > 0; step = step >> 1){
				dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
				v0 = swap(v0,step,dir);
			}
			v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
			v0 = (threadIdx.x & k) == 0 ? v0 : 0;

			//sort
			for(level = k; level < 32; level = level << 1){
				for(step = level; step > 0; step = step >> 1){
					dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
					v0 = swap(v0,step,dir);
				}
			}

			//Merge heaps//
			if(b == 0)
			{
				heap[31 - threadIdx.x] = v0;
			}else{
				v1 = heap[threadIdx.x];
				v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
				v1 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v1, k),v1);
				v0 = (threadIdx.x & k) == 0 ? v0 : v1;

				for(step = level; step > 0; step = step >> 1){
					dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
					v0 = swap(v0,step,dir);
				}
				v0 = fmaxf(__shfl_xor_sync(0xFFFFFFFF, v0, k),v0);
				v0 = (threadIdx.x & k) == 0 ? v0 : 0;

				for(level = k; level < 32; level = level << 1){
					for(step = level; step > 0; step = step >> 1){
						dir = bfe(threadIdx.x,__ffs(level))^bfe(threadIdx.x,__ffs(step>>1));
						v0 = swap(v0,step,dir);
					}
				}
				heap[31 - threadIdx.x] = v0;
			}
		}
		__syncthreads();

		if(heap[k-1] >= threshold[NUM_DIMS]){
			break;
		}
		__syncthreads();
		b++;
	}

	if(threadIdx.x < k){
		uint64_t offset = blockIdx.x * k;
		if((blockIdx.x & 0x1) == 0) out[offset + (k-1) - threadIdx.x] = heap[threadIdx.x];
		else out[offset + threadIdx.x] = heap[threadIdx.x];
	}
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

		angle = fabsf(atan(sqrtf(num)/dnm) * GPTA_PI_2);//calculate tan(Fj)

		num_vec[i] += dnm * dnm;
		angle_vec[i] = angle;
	}
}

#endif
