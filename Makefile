CC=g++
NVCC=/usr/local/cuda-8.0/bin/nvcc

#REORDER APP
CC_REORDER=input/main.cpp
CC_EXE_RE=reorder_run

DIMS=${IDIM}
V=VERBOSE
DT=0
PROFILER=0

#CPU CONFIGURATION
CC_MAIN=cpu/main.cpp skyline/hybrid/hybrid.cpp
CC_FLAGS=-std=c++11
CC_EXE=cpu_run
CC_OPT_FLAGS= -O3 -march=native -DNUM_DIMS=$(DIMS) -D$(V) -DCOUNT_DT=$(DT) -DPROFILER=$(PROFILER) -ffast-math -funroll-loops -msse -msse2 -msse3 -msse4.1 -mbmi2 -mmmx -mavx -mavx2 -fomit-frame-pointer -m64 -fopenmp

#GPU CONFIGURATION
GC_MAIN=gpu/main.cu
GC_EXE=gpu_run
#NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35 -rdc=true
ARCH = -gencode arch=compute_61,code=sm_61
#ARCH = -gencode arch=compute_35,code=sm_35

all: cpu_cc

cpu_cc: 
	$(CC) $(CC_FLAGS) $(CC_OPT_FLAGS) $(CC_MAIN) -o $(CC_EXE)

reorder_cc:
	$(CC) $(CC_FLAGS) $(CC_OPT_FLAGS) $(CC_REORDER) -o $(CC_EXE_RE)

gpu_cc:
	$(NVCC) -std=c++11 $(ARCH) $(GC_MAIN) -o $(GC_EXE) -I cub-1.7.4/
	
clean:
	rm -rf $(CC_EXE)
	rm -rf $(GC_EXE) 
	rm -rf $(CC_EXE_RE) 
