CC=g++
#CC=icpc
NVCC=/usr/local/cuda-8.0/bin/nvcc

#REORDER APP
CC_REORDER=input/main.cpp
CC_EXE_RE=reorder_run

#########
#Skyline#
DIMS=4
V=VERBOSE
DT=0
PROFILER=0

######
#TopK#
#QM 0:Multiple dimension queries, 1:Single dimension Queries
QM=0
#QD Dimension interval for testing
QD=1
#IMP 0:Scalar, 1:SIMD, 2:Threads
IMP=2
#ITER Testing iterations
ITER=1
#LD 0:load from file, 1: generate in memory
LD=0
#DISTR c:correlated i:independent a:anticorrelated
DISTR=1

#TA Benchmark
TA_B=0
#TPAc Benchmark
TPAc_B=0
#TPAr Benchmark
TPAr_B=0
#VTA Benchmark
VTA_B=0
#PTA Benchmark
PTA_B=0

#Top-K
K=16

BENCH= -DTA_B=$(TA_B) -DTPAc_B=$(TPAc_B) -DTPAr_B=$(TPAr_B) -DVTA_B=$(VTA_B) -DPTA_B=$(PTA_B)

#CPU CONFIGURATION
CC_MAIN=cpu/main.cpp skyline/hybrid/hybrid.cpp input/randdataset-1.1.0/src/randdataset.c
CC_FLAGS=-std=c++11
CC_EXE=cpu_run
CC_OPT_FLAGS_GNU= -O3 -march=native $(BENCH) -DK=$(K) -DGNU=0 -DQM=$(QM) -DQD=$(QD) -DIMP=$(IMP) -DITER=$(ITER) -DLD=$(LD) -DDISTR=$(DISTR) -DNUM_DIMS=$(DIMS) -D$(V) -DCOUNT_DT=$(DT) -DPROFILER=$(PROFILER) -ffast-math -funroll-loops -msse -msse2 -msse3 -msse4.1 -mbmi2 -mmmx -mavx -mavx2 -fomit-frame-pointer -m64 -fopenmp
CC_OPT_FLAGS_INTEL= -O3 -DNUM_DIMS=$(DIMS) -D$(V) -DCOUNT_DT=$(DT) -DPROFILER=$(PROFILER) -ffast-math -funroll-loops -fomit-frame-pointer -mavx -fopenmp

#GPU CONFIGURATION
GC_MAIN=gpu/main.cu
GC_EXE=gpu_run
#NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35 -rdc=true
ARCH = -gencode arch=compute_61,code=sm_61
#ARCH = -gencode arch=compute_35,code=sm_35

all: cpu_cc

cpu_cc:
ifeq ($(CC),g++)
	$(CC) $(CC_FLAGS) $(CC_OPT_FLAGS_GNU) $(CC_MAIN) -o $(CC_EXE)
else
	$(CC) $(CC_FLAGS) $(CC_OPT_FLAGS_INTEL) $(CC_MAIN) -o $(CC_EXE)
endif

reorder_cc:
	$(CC) $(CC_FLAGS) $(CC_OPT_FLAGS) $(CC_REORDER) -o $(CC_EXE_RE)

gpu_cc:
	$(NVCC) -std=c++11 $(ARCH) $(GC_MAIN) -o $(GC_EXE) -I cub-1.7.4/
	
clean:
	rm -rf $(CC_EXE)
	rm -rf $(GC_EXE) 
	rm -rf $(CC_EXE_RE) 
