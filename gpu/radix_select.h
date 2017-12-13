#ifndef RADIX_SELECT_H
#define RADIX_SELECT_H

#include <inttypes.h>
#include "CudaHelper.h"

#define VALUE_PER_THREAD 32
#define BLOCK_SIZE 512

class RdxSelect{
	public:
		RdxSelect(){
			cutil::safeMalloc<uint32_t,uint64_t>(&(this->gpu_bins),sizeof(uint32_t)*16,"gpu_bins alloc");
			cutil::safeMallocHost<uint32_t,uint64_t>(&(this->cpu_bins),sizeof(uint32_t)*16,"cpu_bins alloc");
		}

		~RdxSelect(){
			if(this->gpu_bins !=NULL ) cudaFree(this->gpu_bins);
			if(this->cpu_bins !=NULL ) cudaFreeHost(this->cpu_bins);
		}

		uint32_t* get_cbins(){ return this->cpu_bins; }
		uint32_t* get_gbins(){ return this->gpu_bins; }

		void clear(){
			for(int i = 0;i<16;i++) cpu_bins[i]=0;
			cutil::safeCopyToDevice<uint32_t,uint64_t>(this->get_gbins(),this->get_cbins(),sizeof(uint32_t)*16, " copy from cpu_bins to gpu_bins");
		}

	private:
		uint32_t *gpu_bins;
		uint32_t *cpu_bins;

};

static RdxSelect rdx;

template<class T,uint32_t block>
__global__ void extract_bin(uint32_t *gcol_ui, T *gcol_f,uint64_t n){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	if(offset < n){
		T vf = gcol_f[offset];
		uint32_t vi = *(uint32_t*)&vf;
		gcol_ui[offset] = vi;
		//if(offset == 1){ printf("<<< %f , 0x%x\n",vf,vi); }
	}
}

template<uint32_t block, uint32_t radix>
__global__ void radix_select_count(uint32_t *gcol_ui,uint64_t n,uint64_t k,uint32_t prefix, uint32_t prefix_mask,uint32_t digit_mask, uint32_t digit_shf,uint32_t *gpu_bins){
	__shared__ uint32_t bbins[block][16];//Counting bins
	uint64_t offset = block * blockIdx.x + threadIdx.x;//Vector offset

	for(int i = 0;i<16;i++) bbins[threadIdx.x][i] = 0;
	while (offset < n){
		uint32_t vi = gcol_ui[offset];
		uint8_t digit = (vi & digit_mask) >> digit_shf;
		bbins[threadIdx.x][digit]+= ((vi & prefix_mask) == prefix);
		offset+=gridDim.x*block;
	}
	__syncthreads();

	if(threadIdx.x < 16){
		for(int i = 1; i < block;i++){
			bbins[0][threadIdx.x]+= bbins[i][threadIdx.x];
		}
		//gpu_bins[threadIdx.x] = tbins[threadIdx.x];
		atomicAdd(&gpu_bins[threadIdx.x],bbins[0][threadIdx.x]);
	}
}

__host__ uint32_t radix_select_gpu_findK(uint32_t *gcol_ui,uint64_t n, uint64_t k){
//	dim3 grid(GRID_SIZE,1,1);
//	dim3 block(BLOCK_SIZE,1,1);

	uint32_t GRID_SIZE = (n-1)/(VALUE_PER_THREAD * BLOCK_SIZE) + 1;
	dim3 grid(GRID_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	uint32_t prefix=0x00000000;
	uint32_t prefix_mask=0x00000000;
	uint32_t digit_mask=0xF0000000;
	uint32_t digit_shf=28;
	uint8_t digit =0x0;

	uint64_t tmpK = k;
	for(int i = 0;i <8;i++){
		//printf("0x%08x,0x%08x,0x%08x,%02d, %"PRIu64"\n",prefix,prefix_mask,digit_mask,digit_shf,tmpK);

		radix_select_count<BLOCK_SIZE,4><<<grid,block>>>(gcol_ui,n,tmpK,prefix,prefix_mask,digit_mask, digit_shf,rdx.get_gbins());
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing radix_select_count");
		cutil::safeCopyToHost<uint32_t,uint64_t>(rdx.get_cbins(),rdx.get_gbins(),sizeof(uint32_t)*16, " copy from gpu_bins to cpu_bins");

		//for(int i = 0;i < 16;i++) printf("%d | ",rdx.get_cbins()[i]);
		//printf("\n");

		if (rdx.get_cbins()[0] > tmpK){
			digit = 0x0;
		}else{
			for(int i = 1;i < 16;i++){
				rdx.get_cbins()[i]+=rdx.get_cbins()[i-1];
				if( rdx.get_cbins()[i] > tmpK ){
					tmpK = tmpK-rdx.get_cbins()[i-1];
					digit=i;
					break;
				}
			}
		}

		//printf("digit 0x%x\n",digit);
		rdx.clear();
		prefix = prefix | (digit << digit_shf);
		prefix_mask=prefix_mask | digit_mask;
		digit_mask>>=4;
		digit_shf-=4;
	}
	//printf("k largest(gpu): 0x%08x\n",prefix);
	return prefix;
}


#endif

