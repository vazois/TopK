#ifndef GPA_H
#define GPA_H

#include "CudaHelper.h"
#include "reorder_attr.h"
#include "init_tupples.h"
#include "radix_select.h"
#include "prune_tupples.h"

#include <inttypes.h>
#include <algorithm>
#include <vector>


#define BLOCK_SIZE 512
#define DEBUG false

template<class T>
class GPA{
	public:
		GPA() { this->algo = "GPA"; };
		~GPA(){
			if(this->cdata!=NULL){ cudaFreeHost(this->cdata); }
			if(this->gdata!=NULL){ cudaFree(this->gdata); }
			if(this->gtupples!=NULL){ cudaFree(this->gtupples); }
			if(this->ctupples!=NULL){ cudaFree(this->ctupples); }
		};

		void init();
		void findTopK(uint64_t k);

		void alloc(uint64_t items, uint64_t rows);
		T*& get_cdata(){ return this->cdata; }

	protected:
		uint64_t *ctupples = NULL;//cpu tupples id
		uint64_t *gtupples = NULL;//gpu tupples id
		T *gdata = NULL;//column major tupple data
		T *cdata;

	private:
		uint32_t radix_select_findK(uint32_t *col_ui,uint64_t n, uint64_t k);
		uint32_t radix_select_count(uint32_t *col_ui,uint64_t n,uint64_t &k,uint32_t prefix, uint32_t prefix_mask,uint32_t digit_mask, uint32_t digit_shf);
		void check_order();

		uint64_t n,d;
		std::string algo;
};

template<class T>
void GPA<T>::alloc(uint64_t items, uint64_t rows){
	this->d = items; this->n=rows;

	cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*this->d,"gdata alloc");
	cutil::safeMalloc<uint64_t,uint64_t>(&(this->gtupples),sizeof(uint64_t)*this->n,"gtupples alloc");
	cutil::safeMallocHost<uint64_t,uint64_t>(&(this->ctupples),sizeof(uint64_t)*this->n,"ctupples alloc");

	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");
}

template<class T>
void GPA<T>::init(){
	cutil::safeCopyToDevice<T,uint64_t>(this->gdata,this->cdata,sizeof(T)*this->n*this->d, " copy from cdata to gdata ");

	dim3 grid(this->n/BLOCK_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	switch(this->d){
		case 4:
			//init_tupples_4<BLOCK_SIZE><<<grid,block>>>(this->gtupples,this->n);//
			reorder_max_4_full<T,BLOCK_SIZE><<<grid,block>>>(this->gdata,this->n,this->d);
			break;
		case 6:
			reorder_max_6_full<T,BLOCK_SIZE><<<grid,block>>>(this->gdata,this->n,this->d);
			break;
		case 8:
			reorder_max_8_full<T,BLOCK_SIZE><<<grid,block>>>(this->gdata,this->n,this->d);
			break;
		case 10:
			reorder_max_10_full<T,BLOCK_SIZE><<<grid,block>>>(this->gdata,this->n,this->d);
			break;
		case 12:
			reorder_max_12_full<T,BLOCK_SIZE><<<grid,block>>>(this->gdata,this->n,this->d);
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
			std::cout << "i: <" << i << "> ";
			for(uint64_t j = 0; j < this->d; j++) std::cout << this->cdata[j * this->n + i] << " ";
			std::cout << std::endl;
			std::cout << "check_order: " << passed << std::endl;
			exit(1);
		}
		//if(i < 10){ for(uint64_t j = 0; j < this->d; j++){ std::cout << this->cdata[j * this->n + i] << " "; } std::cout << std::endl;}
	}
}


template<class T>
uint32_t GPA<T>::radix_select_count(uint32_t *col_ui,uint64_t n,uint64_t &k,uint32_t prefix, uint32_t prefix_mask,uint32_t digit_mask, uint32_t digit_shf){
	uint32_t bins[16];
	for(int i = 0;i < 16;i++) bins[i]=0;

	for(uint64_t i = 0;i<n;i++){
		uint32_t vi = col_ui[i];
		uint8_t digit = (vi & digit_mask) >> digit_shf;
		bins[digit]+= ((vi & prefix_mask) == prefix);
	}

	if (bins[0] > k) return 0x0;
	for(int i = 1;i < 16;i++){
		bins[i]+=bins[i-1];
		if( bins[i] > k ){
			k = k-bins[i-1];
			return i;
		}
	}
	return 0xF;

//	for(int i = 1;i < 16;i++) bins[i]+=bins[i-1];
//	for(int i = 0;i < 16;i++) printf("%d | ",bins[i]);
//	printf("\n");
//	for(int i = 0;i < 16;i++){
//		if( bins[i] > k ){// >= or >
//			k = k-bins[i-1];
//			return i;
//		}
//	}
//	return 0xF;
}

template<class T>
uint32_t GPA<T>::radix_select_findK(uint32_t *col_ui,uint64_t n, uint64_t k){
	uint32_t prefix=0x00000000;
	uint32_t prefix_mask=0x00000000;
	uint32_t digit_mask=0xF0000000;
	uint32_t digit_shf=28;

	uint64_t tmpK = k;
	for(int i = 0;i <8;i++){
		//printf("0x%08x,0x%08x,0x%08x,%02d, %"PRIu64"\n",prefix,prefix_mask,digit_mask,digit_shf,tmpK);

		uint32_t digit = this->radix_select_count(col_ui,n,tmpK,prefix,prefix_mask,digit_mask,digit_shf);

		prefix = prefix | (digit << digit_shf);
		prefix_mask=prefix_mask | digit_mask;
		digit_mask>>=4;
		digit_shf-=4;
	}
//	printf("k largest: 0x%08x\n",prefix);
	return prefix;
}

template<class T>
void GPA<T>::findTopK(uint64_t k){
	dim3 grid((this->n-1)/BLOCK_SIZE+1,1,1);
	dim3 block(BLOCK_SIZE,1,1);
	uint32_t *col_ui, *gcol_ui;
	Tupple<T> tupples;

	alloc_tupples<T>(tupples,this->n);
	init_tupples<T,BLOCK_SIZE><<<grid,block>>>(tupples.tupple_ids,tupples.scores,this->gdata,this->n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_tupples");

	/////////////////////
	//Debug Code
	if(DEBUG){
		cutil::safeMallocHost<uint32_t,uint64_t>(&(col_ui),sizeof(uint32_t)*this->n,"col_ui alloc");
		cutil::safeMalloc<uint32_t,uint64_t>(&(gcol_ui),sizeof(uint32_t)*this->n,"gcol_ui alloc");
		for(int i = 0;i<d;i++){
			printf("test: %d column\n",i);
			extract_bin<T,BLOCK_SIZE><<<grid,block>>>(gcol_ui,&this->gdata[i*n],this->n);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing extract_bin");
			{
				cutil::safeCopyToHost<uint32_t,uint64_t>(col_ui,gcol_ui,sizeof(uint32_t)*this->n, " copy from gcol_ui to col_ui ");
				uint32_t prefix_cpu=this->radix_select_findK(col_ui,this->n,this->n - k);
				printf("kcpu: 0x%08x\n",prefix_cpu);

				std::vector<uint32_t> myvector (col_ui, col_ui+this->n);
				std::sort (myvector.begin(), myvector.begin()+this->n);
				uint32_t sK = myvector[this->n - k];

				printf("scpu: 0x%08x\n",sK);

				uint32_t gpu_prefix = radix_select_gpu_findK(gcol_ui,n,this->n - k);
				float gpu_prefixf = *(float*)&gpu_prefix;
				printf("kgpu: 0x%08x, %f\n",gpu_prefix,gpu_prefixf);
			}
			cudaFreeHost(col_ui);
			cudaFree(gcol_ui);
		}
	/////////////
	}else{
		cutil::safeMallocHost<uint32_t,uint64_t>(&(col_ui),sizeof(uint32_t)*this->n,"col_ui alloc");
		cutil::safeMalloc<uint32_t,uint64_t>(&(gcol_ui),sizeof(uint32_t)*this->n,"gcol_ui alloc");

		CompactConfig<T> cconfig;
		uint64_t suffix_len = this->d - 1;
		uint64_t tmpN = n;
		for(uint64_t i = 0; i < d;i++){
			dim3 ggrid((tmpN-1)/BLOCK_SIZE + 1,1,1);
			dim3 gblock(BLOCK_SIZE,1,1);
			extract_bin<T,BLOCK_SIZE><<<ggrid,gblock>>>(gcol_ui,tupples.scores,tmpN);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing extract_bin");

			uint32_t gpu_prefix = radix_select_gpu_findK(gcol_ui,tmpN,tmpN - k);
			float threshold = *(float*)&gpu_prefix;
			printf("kgpu: 0x%08x, %f\n",gpu_prefix,threshold);

			cconfig.prune_tupples(tupples.tupple_ids,tupples.scores,&this->gdata[i*n],&this->gdata[(i+1)*n],tmpN,threshold,suffix_len);

			suffix_len--;
		}
		cudaFreeHost(col_ui);
		cudaFree(gcol_ui);
	}


	free_tupples<T>(tupples);
}

#endif
