CC=g++
NVCC=/usr/local/cuda-8.0/bin/nvcc

#CPU CONFIGURATION
CC_MAIN=cpu/main.cpp
CC_FLAGS=-std=c++11
CC_EXE=cpu_run
CC_OPT_FLAGS= -O3 -ffast-math -funroll-loops -msse -msse3 -mmmx -fomit-frame-pointer -m64

#GPU CONFIGURATION
GC_MAIN=gpu/main.cu
GC_EXE=gpu_run
#NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35 -rdc=true
ARCH = -gencode arch=compute_61,code=sm_61
#ARCH = -gencode arch=compute_35,code=sm_35

all: cpu_cc

cpu_cc: 
	$(CC) $(CC_FLAGS) $(CC_OPT_FLAGS) $(CC_MAIN) -o $(CC_EXE)

gpu_cc:
	$(NVCC) -std=c++11 $(ARCH) $(GC_MAIN) -o $(GC_EXE)
	
clean:
	rm -rf $(CC_EXE)
	rm -rf $(GC_EXE)  
