#ifndef GPA_H
#define GPA_H

#include <cub/cub.cuh>

#include "../cpu/AA.h"
#include "CudaHelper.h"
#include "reorder_attr.h"

#define BLOCK_SIZE 256

template<class T>
class GPA : public AA<T>{
	public:
		GPA(GInput<T>* ginput) : AA<T>(ginput){ this->algo = "GPA"; this->cdata= ginput->get_dt(); };
		~GPA(){
			if(this->gdata!=NULL){ cudaFree(this->gdata); }
			if(this->gkeys!=NULL){ cudaFree(this->gkeys); }
			if(this->ckeys!=NULL){ cudaFree(this->ckeys); }
		};

		void init();
		void findTopK(uint64_t k);

	protected:
		uint64_t *ckeys = NULL;
		uint64_t *gkeys = NULL;
		T *gdata = NULL;

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
	cutil::safeMalloc<uint64_t,uint64_t>(&(this->gkeys),sizeof(uint64_t)*this->n,"gkeys alloc");
	cutil::safeMallocHost<uint64_t,uint64_t>(&(this->ckeys),sizeof(uint64_t)*this->n,"ckeys alloc");

	cutil::safeCopyToDevice<T,uint64_t>(this->gdata,this->cdata,sizeof(T)*this->n*this->d, " copy from cdata to gdata ");
	dim3 grid(this->n/BLOCK_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	switch(this->d){
		case 4:
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
	cutil::safeCopyToDevice<T,uint64_t>(this->cdata,this->gdata,sizeof(T)*this->n*this->d, " copy from gdata to cdata ");

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
void GPA<T>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ..." << std::endl;

	uint64_t *gkeys_out = NULL;
	T *gvalues_out = NULL;
	void *d_temp_storage = NULL;
	uint64_t temp_storage_bytes = 0;
	cutil::safeMalloc<uint64_t,uint64_t>(&(gkeys_out),sizeof(uint64_t)*this->n,"gkeys_out alloc");
	cutil::safeMalloc<T,uint64_t>(&(gvalues_out),sizeof(T)*this->n,"gvalues_out alloc");

	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, this->gkeys, gkeys_out, this->gdata, gvalues_out, this->n);
	std::cout << "temp_storage bytes: " << temp_storage_bytes << std::endl;

	cudaFree(gkeys_out);
	cudaFree(gvalues_out);
}

#endif
