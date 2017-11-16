#ifndef GPA_H
#define GPA_H

#include <cub/cub.cuh>

#include "../cpu/AA.h"
#include "../input/GInput.h"
#include "CudaHelper.h"
#include "reorder_attr.h"
#include "init_tupples.h"

#include <list>

#define BLOCK_SIZE 512

template<class T>
struct cpred{
	cpred(){ tid = 0; curr_attr = 0; total = 0; }
	cpred(uint64_t t, T a){ tid = t; curr_attr = a; total = a;}
	uint64_t tid;
	T curr_attr;
	T total;
};

template<class T>
static bool cmp_max_cpred(const cpred<T> &a, const cpred<T> &b){ return a.total > b.total; };

template<class T>
class GPA : public AA<T>{
	public:
		GPA(GInput<T>* ginput) : AA<T>(ginput){ this->algo = "GPA"; };
		~GPA(){
			if(this->gdata!=NULL){ cudaFree(this->gdata); }
			if(this->gtupples!=NULL){ cudaFree(this->gtupples); }
			if(this->ctupples!=NULL){ cudaFree(this->ctupples); }
		};

		void init();
		void findTopK(uint64_t k);
		void findTopK_cpu(uint64_t k);


	protected:
		uint64_t *ctupples = NULL;//cpu tupples id
		uint64_t *gtupples = NULL;//gpu tupples id
		T *gdata = NULL;//column major tupple data

	private:
		void check_order();
};

/*
 * Allocate GPU memory, copy data and order attributes
 */
template<class T>
void GPA<T>::init(){
	std::cout << "gdata: (" << this->n << "," << this->d << ")" << std::endl;
	cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*this->d,"gdata alloc");
	cutil::safeMalloc<uint64_t,uint64_t>(&(this->gtupples),sizeof(uint64_t)*this->n,"gtupples alloc");
	cutil::safeMallocHost<uint64_t,uint64_t>(&(this->ctupples),sizeof(uint64_t)*this->n,"ctupples alloc");

	cutil::safeCopyToDevice<T,uint64_t>(this->gdata,this->cdata,sizeof(T)*this->n*this->d, " copy from cdata to gdata ");
	dim3 grid(this->n/BLOCK_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	switch(this->d){
		case 4:
			init_tupples_4<BLOCK_SIZE><<<grid,block>>>(this->gtupples,this->n);
			reorder_max_4_full<T,BLOCK_SIZE><<<grid,block>>>(this->gdata,this->n,this->d);
			break;
		default:
			break;
	}
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing reorder");

	this->check_order();//TODO: Comment
}

template<class T>
void GPA<T>::check_order(){
	cutil::safeCopyToHost<T,uint64_t>(this->cdata,this->gdata,sizeof(T)*this->n*this->d, " copy from gdata to cdata ");

	std::string passed = "(PASSED)";
	for(uint64_t i = 0; i < this->n; i++){
		bool ordered = true;
		for(uint64_t j = 0; j < (this->d - 1); j++){ ordered &=(this->cdata[j * this->n + i] >= this->cdata[(j+1) * this->n + i]); }

		if(!ordered){
			passed = "(FAILED)";
			std::cout << "i: ";
			for(uint64_t j = 0; j < this->d; j++) std::cout << this->cdata[j * this->n + i] << " ";
			std::cout << std::endl;
		}
	}
	std::cout << "check_order: " << passed << std::endl;
}

template<class T>
__global__ void debug(T *gvalues, uint64_t *gtupples){
	//uint64_t offset = block * blockIdx.x + threadIdx.x;
	for(uint64_t i = 0; i <10; i++){ printf("< %f,%d >\n",gvalues[i],gtupples[i]); }
}

template<class T>
void GPA<T>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ..." << std::endl;

	uint64_t *gtupples_out = NULL;
	T *gvalues_out = NULL;
	void *d_temp_storage = NULL;
	uint64_t temp_storage_bytes = 0;
	cutil::safeMalloc<uint64_t,uint64_t>(&(gtupples_out),sizeof(uint64_t)*this->n,"gkeys_out alloc");
	cutil::safeMalloc<T,uint64_t>(&(gvalues_out),sizeof(T)*this->n,"gvalues_out alloc");

	//cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, this->gtupples, gtupples_out, this->gdata, gvalues_out, this->n);
	cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, this->gdata, gvalues_out, this->gtupples, gtupples_out, this->n);
	std::cout << "temp_storage bytes: " << temp_storage_bytes << std::endl;

	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	//cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, this->gtupples, gtupples_out, this->gdata, gvalues_out, this->n);
	cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, this->gdata, gvalues_out, this->gtupples, gtupples_out, this->n);

	debug<T><<<1,1>>>(gvalues_out,gtupples_out);

	cudaFree(d_temp_storage);
	cudaFree(gtupples_out);
	cudaFree(gvalues_out);
}




#endif
