#ifndef GPAm_H
#define GPAm_H

#include "CudaHelper.h"
#include "reorder_attr.h"
#include "init_tuples.h"
#include "radix_select.h"
#include "prune_tuples.h"

#include "GPA.h"

template<class T>
class GPAm : public GPA<T>{
	public:
		GPAm() { this->algo = "GPAm"; this->gdata = NULL; this->cdata = NULL;};
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

	//Copy back ordered attributes
	cutil::safeCopyToHost<T,uint64_t>(this->cdata,this->gdata,sizeof(T)*this->n*this->d, " copy from gdata to cdata ");
	cudaFree(this->gdata);//Free Array After Ordering Attributes

	//Iteratively send attributes to device
	cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*2,"gdata alloc");
	cutil::safeCopyToDevice<T,uint64_t>(this->gdata,this->cdata,sizeof(T)*this->n*2," copy from cdata to gdata ");

	//Allocate candidate tuples//
	Tuple<T> tuples;
	alloc_tuples<T>(tuples,this->n);
	init_tuples<T,BLOCK_SIZE><<<grid,block>>>(tuples.tuple_ids,tuples.scores,this->gdata,this->n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_tupples");
	//---//

	//Calculate TopK
	uint32_t *gcol_ui;
	cutil::safeMalloc<uint32_t,uint64_t>(&(gcol_ui),sizeof(uint32_t)*this->n,"gcol_ui alloc");
	CompactConfig<T> cconfig;
	uint64_t suffix_len = this->d - 1;
	uint64_t tmpN = this->n;
	bool even = true;
	this->t.start();
	for(uint64_t i = 0;i <this->d;i++){
		dim3 ggrid((tmpN-1)/BLOCK_SIZE + 1,1,1);
		dim3 gblock(BLOCK_SIZE,1,1);
		extract_bin<T,BLOCK_SIZE><<<ggrid,gblock>>>(gcol_ui,tuples.scores,tmpN);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing extract_bin");

		uint32_t gpu_prefix = radix_select_gpu_findK(gcol_ui,tmpN,tmpN - k);
		float threshold = *(float*)&gpu_prefix;
		printf("kgpu: 0x%08x, %f\n",gpu_prefix,threshold);//Debug

		if(even){
			cconfig.prune_tuples(tuples.tuple_ids,tuples.scores,&this->gdata[0],&this->gdata[this->n],tmpN,threshold,suffix_len);
			if(i+2 < this->d) cutil::safeCopyToDevice<T,uint64_t>(&this->gdata[0],&this->cdata[this->n*(i+2)],sizeof(T)*this->n," copy from cdata to gdata ");
			even=false;
		}else{
			cconfig.prune_tuples(tuples.tuple_ids,tuples.scores,&this->gdata[this->n],&this->gdata[0],tmpN,threshold,suffix_len);
			if(i+2 < this->d) cutil::safeCopyToDevice<T,uint64_t>(&this->gdata[this->n],&this->cdata[this->n*(i+2)],sizeof(T)*this->n," copy from cdata to gdata ");
			even=false;
		}
		suffix_len--;
	}
	this->tt_processing = this->t.lap();
	cudaFree(gcol_ui);
	free_tuples<T>(tuples);
}



#endif
