#ifndef GPAm_H
#define GPAm_H

#include "CudaHelper.h"
#include "reorder_attr.h"
#include "init_tupples.h"
#include "radix_select.h"
#include "prune_tupples.h"

#include "GPA.h"

template<class T>
class GPAm : public GPA<T>{
	public:
		GPAm() { this->algo = "GPA"; this->gdata = NULL; this->cdata = NULL;};
		~GPAm(){
			if(this->cdata!=NULL){ cudaFreeHost(this->cdata); cudaFree(this->gdata);}
		};

		void alloc(uint64_t items, uint64_t rows);
		void init();
		void findTopK(uint64_t k);

	private:
		T *cdata;
		T *gdata;
};


template<class T>
void GPAm<T>::alloc(uint64_t items, uint64_t rows){
	this->d = items; this->n=rows;
	if(this->gdata==NULL){ cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*this->d,"gdata alloc"); }
	if(this->cdata==NULL){ cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc"); }
}

template<class T>
void GPAm<T>::findTopK(uint64_t k){
	dim3 grid((this->n-1)/BLOCK_SIZE+1,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	cutil::safeCopyToHost<T,uint64_t>(this->cdata,this->gdata,sizeof(T)*this->n*this->d, " copy from gdata to cdata ");
	cudaFree(this->gdata);//Free Array After Ordering Attributes

	cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n,"gdata alloc");
	cutil::safeCopyToDevice<T,uint64_t>(this->gdata,this->cdata,sizeof(T)*this->n," copy from cdata to gdata ");

	//Allocate candidate tuples//
	Tuple<T> tuples;
	alloc_tuples<T>(tuples,this->n);
	init_tuples<T,BLOCK_SIZE><<<grid,block>>>(tuples.tuple_ids,tuples.scores,this->gdata,this->n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_tupples");
	//---//


	free_tuples<T>(tuples);
}



#endif
