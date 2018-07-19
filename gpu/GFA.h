#ifndef GFA_H
#define GFA_H

#include "../tools/CudaHelper.h"
#include <inttypes.h>
#include <algorithm>
#include <vector>

#define BLOCK_SIZE 512

template<class T>
class GFA{
	public:
		GFA() { this->algo = "GFA"; };

		~GFA(){
			if(this->cdata!=NULL){ cudaFreeHost(this->cdata); }
			if(this->gdata!=NULL){ cudaFree(this->gdata); }
		};

		void init();
		void findTopK(uint64_t k);

		void alloc(uint64_t items, uint64_t rows);
		void set_cdata(T *cdata){ this->cdata = cdata; }
		void set_gdata(T *gdata){ this->gdata = gdata; }
		T*& get_cdata(){ return this->cdata; }
		T*& get_gdata(){ return this->gdata; }

	private:

		T *gdata = NULL;//column major tupple data
		T *cdata;
		uint64_t n,d;
		std::string algo;
};

template<class T>
void GFA<T>::alloc(uint64_t items, uint64_t rows){
	this->d = items; this->n=rows;

	cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*this->d,"gdata alloc");
	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");
}

template<class T>
void GFA<T>::findTopK(uint64_t k){
	dim3 grid((this->n-1)/BLOCK_SIZE+1,1,1);
	dim3 block(BLOCK_SIZE,1,1);
}


#endif
