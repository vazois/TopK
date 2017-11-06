#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

/*
 * @author Vasileios Zois
 * @email vzois@usc.edu
 *
 * Definition of CUDA utility functions.
 */


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <iostream>
#include <string>

#include "../tools/Utils.h"

#define RAND_BLOCKS 8
#define RAND_THREADS 512
#define RAND_BLOCK_THREADS 256
#define RAND_STATES 256 * 512

namespace cutil{
	static __device__ curandState devStates[RAND_BLOCKS*RAND_THREADS];
	static __device__ curandState randDevStates[RAND_STATES];
	static cudaError_t error = cudaSuccess;

	/*
	 * Utility functions
	 */
	__host__ cudaError_t printDeviceSpecs(bool);
	__host__ void cudaCheckErr(cudaError_t error, std::string comment);
	__host__ void cublasCheckErr(cublasStatus_t status, std::string comment);

	/*
	 * Grid and Block dimension computation.
	 */
	dim3 grid_1D(unsigned int N, unsigned int data_per_block);
	dim3 grid_1D(unsigned int N, unsigned int data_per_block, unsigned int amplification);
	dim3 block_1D(unsigned int data_per_block);
	void print_grid(dim3 blocks, dim3 threads);


	/*
	 * Memory handling functions.
	 */
	template<typename DATA_T, typename SIZE_T>
	__host__ void safeMalloc(DATA_T **addr, SIZE_T size, std::string msg);
	template<typename DATA_T, typename SIZE_T>
	__host__ void safeMallocHost(DATA_T **addr, SIZE_T size, std::string msg);
	template<typename DATA_T, typename SIZE_T>
	__host__ void safeCopyToDevice(DATA_T *to, DATA_T *from, SIZE_T size, std::string msg);
	template<typename DATA_T, typename SIZE_T>
	__host__ void safeCopyToHost(DATA_T *to, DATA_T *from, SIZE_T size, std::string msg);

	/*
	 * Random number generation tools
	 * 		cudaUniRand: call inside kernel with threaIdx.x or blockIdx.x to get random float
	 * 		cudaSetupRandStatesKernel: call directly or through cudaInitRandStates to initialize states for random number.
	 * 		cudaInitRandStates: call from host to initialize random states.
	 */
	__device__ float cudaUniRand(unsigned int tid){
		return curand_uniform(&randDevStates[tid % RAND_STATES]);
	}


	__global__ void cudaInitRandStatesKernel(unsigned int seed){
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		curand_init(seed, blockIdx.x, 0, &randDevStates[i]);
	}

	__host__ void cudaInitRandStates(){
		dim3 grid = grid_1D(RAND_STATES,RAND_BLOCK_THREADS);
		dim3 block = block_1D(RAND_BLOCK_THREADS);

		Utils<unsigned int> u;
		cudaInitRandStatesKernel<<<grid,block>>>(u.uni(UINT_MAX));
		cudaCheckErr(cudaDeviceSynchronize(),"Error initializing random states");
	}

	template<typename DATA_T, typename SIZE_T>
	__global__ void cudaRandInitKernel(DATA_T *arr, SIZE_T size){
		SIZE_T i = blockIdx.x * blockDim.x + threadIdx.x;

		while(i < size){
			arr[i] = cudaUniRand(i);
			//arr[i] = 1;
			i+=gridDim.x * blockDim.x;
		}
	}

	template<typename DATA_T,typename SIZE_T>
	__host__ void cudaRandInit(DATA_T *arr, SIZE_T size){
		cudaInitRandStates();
		dim3 grid = grid_1D(size, RAND_BLOCK_THREADS);
		dim3 block = block_1D(RAND_BLOCK_THREADS);

		cudaRandInitKernel<DATA_T,SIZE_T><<<grid,block>>>(arr,size);
		cudaCheckErr(cudaDeviceSynchronize(),"Error calling cudaRandInitKernel");
	}

	/*
	 * Allocating device memory
	 * addr: Address on GPU
	 * size: Data size in bytes
	 * msg: Error message to be displayed
	 */
	template<typename DATA_T, typename SIZE_T>
	void safeMalloc(DATA_T **addr, SIZE_T size, std::string msg){
		error = cudaMalloc(&(*addr), size);
		cudaCheckErr(error, "Error Allocating device " + msg);
	}

	/*
	 * Allocating Pinned Memory
	 *
	 * addr: Address on GPU
	 * size: Data size in bytes
	 * msg: error message to be displayed
	 */
	template<typename DATA_T, typename SIZE_T>
	__host__ void safeMallocHost(DATA_T **addr, SIZE_T size, std::string msg){
		error = cudaMallocHost(&(*addr), size);
		cudaCheckErr(error, "Error Allocating host "+msg);
	}

	/*
	 * Copy Arrays to Device
	 * to: GPU address
	 * from: DRAM address
	 * size: data size in bytes
	 * msg: error message to be displayed
	 */
	template<typename DATA_T, typename SIZE_T>
	__host__ void safeCopyToDevice(DATA_T *to, DATA_T *from, SIZE_T size, std::string msg){
		error = cudaMemcpy(to,from,size,cudaMemcpyHostToDevice);
		cudaCheckErr(error, "Error Copying to device "+ msg);
	}

	template<typename DATA_T, typename SIZE_T>
	__host__ void safeCopyToHost(DATA_T *to, DATA_T *from, SIZE_T size, std::string msg){
		error = cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost);
		cudaCheckErr(error, "Error Copying to device " + msg);
	}

	/*
	 * Use to compute grid dimension and block dimension based on flat array space.
	 */
	dim3 grid_1D(unsigned int N, unsigned int data_per_block){
		return dim3((N - 1) / data_per_block + 1, 1, 1);
	}

	//AMPLIFY = # ELEMENTS PER THREAD
	dim3 grid_1D(unsigned int N, unsigned int data_per_block, unsigned int amplification){
		return dim3((N - 1) / (data_per_block*amplification) + 1, 1, 1);
	}

	dim3 block_1D(unsigned int data_per_block){
		return dim3(data_per_block, 1, 1);
	}

	void print_grid(dim3 grid, dim3 block,std::string msg){
		std::cout<< msg <<" = ";
		std::cout<<"grid("<<grid.x <<","<<grid.y << "," << grid.z <<") -- ";
		std::cout<<"block("<<block.x <<","<<block.y << "," << block.z <<")"<<std::endl;
	}

	/*
	 * Device error handling
	 *
	 */
	__host__ void cudaCheckErr(cudaError_t error, std::string comment){
		if (error != cudaSuccess){ std::cout << "Cuda Error: " << comment << "," << cudaGetErrorString(error) << std::endl; exit(1); }
	}

	__host__ void cublasCheckErr(cublasStatus_t status, std::string comment){
		if ( status != CUBLAS_STATUS_SUCCESS ){ std::cout<<"Cublas Error: "<< comment << std::endl; }
	}

	__host__ void setActiveDevice(int devID){
		int devi = devID;
		cudaDeviceProp devp;
		cudaCheckErr(cudaGetDeviceProperties(&devp, devi),"Retrieving Device Properties");
		printf("Using GPU Device %d: \"%s\", compute capability %d.%d\n", devi, devp.name, devp.major, devp.minor);
		cudaSetDevice(devi);
	}

	__host__ void cudaFinalize(){
		cudaDeviceReset();
	}

	/*
	 * Print all device specifications.
	 */
	__host__ cudaError_t printDeviceSpecs(bool print){
		cudaDeviceProp prop;
		cudaError_t error = cudaSuccess;
		int devs = 0;

		error = cudaGetDeviceCount(&devs);
		if (!print) return error;
		if (error != cudaSuccess){ cudaCheckErr(error, "Error Getting Number of Devices");  return error; }
		std::cout << std::endl;
		std::cout << "Number of Devices: (" << devs << ")" << std::endl;

		for (int i = 0; i < devs; i++){
			error = cudaGetDeviceProperties(&prop, i);
			if (error != cudaSuccess){ cudaCheckErr(error, "Error Reading Device Properties");  return error; }
			std::cout << "<<<<<< Device " << i << " >>>>>>" << std::endl;

			std::cout << "Device Name: " << prop.name << std::endl;

			std::cout << "Device Compute Mode: " << prop.computeMode <<std::endl;
			std::cout << "Device Major Compute Capability: " << prop.major << std::endl;
			std::cout << "Device Minor Compute Capability: " << prop.minor << std::endl;

			std::cout << "Number of AsyncEngineCount: " << prop.asyncEngineCount << std::endl;
			std::cout << "Global Memory Size: " << prop.totalGlobalMem << std::endl;
			std::cout << "Constant Memory Size: " << prop.totalConstMem << std::endl;

			std::cout << "Number of Multiprocessors: " << prop.multiProcessorCount << std::endl;
			std::cout << "Shared Memory Per Multiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
			std::cout << "Shared Memory Per Block: " << ((float)prop.sharedMemPerMultiprocessor) << std::endl;

			/*int x = 0;
			error = cudaDeviceGetAttribute(&x, cudaDevAttrMaxBlockDimX, 0);
			std::cout << "Device Block Number X:" << x << endl;
			error = cudaDeviceGetAttribute(&x, cudaDevAttrMaxBlockDimY, 0);
			std::cout << "Device Block Number Y:" << x << endl;
			error = cudaDeviceGetAttribute(&x, cudaDevAttrMaxBlockDimZ, 0);
			std::cout << "Device Block Number Z:" << x << endl;*/

			std::cout << "Maximum Grid Size (X,Y,Z): (" << prop.maxGridSize[0] << "),("
					<< prop.maxGridSize[1] << "),(" << prop.maxGridSize[2] << ")" << std::endl;

			std::cout << "Maximum Threads Per Block: " << prop.maxThreadsPerBlock<< std::endl;
			std::cout << "Maximum Number of Blocks (X,Y,Z): (" << prop.maxThreadsDim[0] << "),("
					<< prop.maxThreadsDim[1] << "),(" << prop.maxThreadsDim[2] << ")" << std::endl;

		}
		std::cout << std::endl;

		return cudaSuccess;
	}

}

static __device__ curandState devStates[RAND_BLOCKS*RAND_THREADS];

#define RAND_BLOCK_THREADS 256
#define RAND_STATES 256 * 512
static __device__ curandState randDevStates[RAND_STATES];

static cudaError_t error = cudaSuccess;

template<class V>
__host__ void printDevData(V *devAddr,unsigned int row, unsigned col);

/*
 * Random number generators.
 */
__host__ void cudaInitRandStates();
__global__ void cudaSetupRandStatesKernel(unsigned int seed);
__device__ float cudaUniRand(unsigned int tid);

/*
 * Utility functions
 */
__host__ cudaError_t printDeviceSpecs(bool);
__host__ void cudaCheckErr(cudaError_t error, std::string comment);
__host__ void setActiveDevice(int devi);
__host__ void cudaFinalize();

/*
 * Memory handling wrappers.
 */
template<class V> __host__ void safeMalloc(V **addr, unsigned int size, std::string msg);
template<class V> __host__ void safeMallocHost(V **addr, unsigned int size, std::string msg);
template<class V> __host__ void safeCpyToSymbol(V *symbol, V *data, std::string msg);
template<class V> __host__ void safeCopyToDevice(V *to, V *from, unsigned int size, std::string msg);
template<class V> __host__ void safeCopyToHost(V *to, V *from, unsigned int size, std::string msg);

/*
 * Grid and Block dimension computation.
 */
dim3 grid_1D(unsigned int N, unsigned int data_per_block);
dim3 grid_1D(unsigned int N, unsigned int data_per_block, unsigned int amplification);
dim3 block_1D(unsigned int data_per_block);
void print_grid(dim3 blocks, dim3 threads);

#endif
