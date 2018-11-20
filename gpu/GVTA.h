#ifndef GVTA_H
#define GVTA_H

#define GVTA_PARTITIONS 128

template<class T>
struct gvta_block
{
	T *data;
	T *tvector;
};

template<class T, class Z>
class GVTA : public GAA<T,Z>{
	public:
		GVTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "GVTA";
		};

		~GVTA()
		{

		};

		void alloc();
		void init();

	private:
		uint64_t tuples_per_part;
		gvta_block<T> *blocks = NULL;
		void layer_data();

};

template<class T, class Z>
void GVTA<T,Z>::alloc(){
	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");// Allocate cpu data memory
	//cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*this->d,"gdata alloc");//Allocate gpu data memory
}

template<class T, class Z>
void GVTA<T,Z>::layer_data()
{
	std::cout << "layer_data" << std::endl;
	dim3 block(256,1,1);
	dim3 grid(1,1,1);
	Z *rvector = NULL;
	Z *dkeys_in = NULL;
	Z *dkeys_out = NULL;
	T *dvalues_in = NULL;
	T *dvalues_out = NULL;

	Z *hkeys = NULL;
	T *hvalues = NULL;
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	this->tuples_per_part = ((this->n - 1)/GVTA_PARTITIONS) + 1;

	hkeys = (Z*)malloc(sizeof(Z)*this->tuples_per_part);
	hvalues = (T*)malloc(sizeof(T)*this->tuples_per_part);
	cutil::safeMallocHost<Z,uint64_t>(&(rvector),sizeof(Z)*this->n,"alloc rvector");
	cutil::safeMallocHost<Z,uint64_t>(&(dkeys_in),sizeof(Z)*this->tuples_per_part,"alloc dkeys_in");
	cutil::safeMallocHost<Z,uint64_t>(&(dkeys_out),sizeof(Z)*this->tuples_per_part,"alloc dkeys_out");
	cutil::safeMallocHost<T,uint64_t>(&(dvalues_in),sizeof(T)*this->tuples_per_part,"alloc dvalues_in");
	cutil::safeMallocHost<T,uint64_t>(&(dvalues_out),sizeof(T)*this->tuples_per_part,"alloc dvalues_out");
	cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, dkeys_in, dkeys_out, dvalues_in, dvalues_out, this->tuples_per_part);
	cutil::cudaCheckErr(cudaMalloc(&d_temp_storage, temp_storage_bytes),"alloc d_temp_storage");

	//initialize global rearrange vector//
	grid.x = (this->tuples_per_part-1)/block.x + 1;
	init_rvglobal<Z><<<grid,block>>>(rvector,this->n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_rvglobal");

	uint64_t offset = 0;
	for(uint64_t i = 0; i < GVTA_PARTITIONS; i++){
		//std::cout << "PART (" << i << ")" << std::endl;
		grid.x = (this->tuples_per_part-1)/block.x + 1;
		init_rvlocal<Z><<<grid,block>>>(dkeys_in,this->tuples_per_part);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init_rvlocal");

		for(uint64_t m = 0; m < this->d; m++){
			//Copy attribute and sort them in descending order
			cutil::safeCopyToDevice<T,uint64_t>(dvalues_in,&this->cdata[m*this->n + offset],sizeof(T)*this->tuples_per_part, " copy from cdata to dvalues_in");
			cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, dvalues_in, dvalues_out, dkeys_in, dkeys_out, this->tuples_per_part);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing SortPairsDescending");

			//Update global rvector with local rvector
			max_rvglobal<Z><<<grid,block>>>(&rvector[offset], dkeys_out,this->tuples_per_part);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing max_rvglobal");

			if(i == 0){
				std::cout << std::fixed << std::setprecision(4);
				cutil::safeCopyToHost<Z,uint64_t>(hkeys,dkeys_out,sizeof(Z)*this->tuples_per_part, " copy from cdata to hkeys");
				cutil::safeCopyToHost<T,uint64_t>(hvalues,dvalues_out,sizeof(T)*this->tuples_per_part, " copy from cdata to hvalues");
				for(int j = 0; j < 20; j++){ std::cout << hkeys[j] << " "; } std::cout << std::endl;
				for(int j = 0; j < 20; j++){ std::cout << hvalues[j] << " "; } std::cout << std::endl;
				std::cout << " --------- " << std::endl;
			}
		}
		offset += this->tuples_per_part;
	}

	cudaFree(d_temp_storage);
	cudaFree(rvector);
	cudaFree(dkeys_in);
	cudaFree(dkeys_out);
	cudaFree(dvalues_in);
	cudaFree(dvalues_out);

	free(hkeys);
	free(hvalues);

}

template<class T, class Z>
void GVTA<T,Z>::init()
{
	this->layer_data();
}

#endif
