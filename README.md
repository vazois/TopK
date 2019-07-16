# TopK Algorithms Benchmark #
###### Note: switch to the appropriate branch to run the algorithms (cpu\_ks for CPUs, gpu\_topk for GPUs)

This repository contains the source code for a collection of efficient parallel in-memory Top-k selection algorithms.
CPU-only versions are verified and tested for correctness. GPU-only implementations are still under development.

The following is a short description of each CPU algorithm:

* cpu/TA.h
	<br /> - Scalar implementation of the Threshold Algorithm using standard C++ library.

* cpu/TPAc.h
	<br /> - Scalar, SIMD, SIMD + multi-threaded, and multi-query implementation of Full Table Evaluation (FTE) using column major order.
	
* cpu/TPAr.h
	<br /> - Scalar, SIMD, and SIMD + multi-threaded implementation of Full Table Evaluation (FTE) using row major order.
	
* cpu/VTA.h
	<br /> - Scalar, SIMD, SIMD + multi-threaded, and multi-query implementation of Vectorized Threshold Algorithm (VTA).
	
* cpu/SLA.h	
	<br /> - Scalar, SIMD, and SIMD + multi-threaded implementation of Skyline Layered Algorithm (SLA).

* cpu/PTA.h	
	<br /> - Scalar, SIMD, SIMD + multi-threaded, and multi-query implementation of Partitioned Threshold Algorithm (PTA).
	
# Behnchmark Instructions #

To evaluate the above implementations, you need to configure ``debug.sh``. Following we provide a short description
of those parameters and a few example how to configure them.

* START\_N, END\_N
	<br /> - For synthetic data, choose number of objects to test.

* DIMS
	<br /> - For synthetic data, choose number of attributes for each object. Specified also queries to be tested (i.e. DIMS=8, queries on 2,3,4,5,6,7,8 attributes).
	
* KKS, KKE
	<br /> - Select range of k to test incrementing by a factor of 2 (i.e. KKS=16 KKE=128, 16,32,64,128).
	
* LD
	<br /> - Specified synthetic or real data. LD=0 synthetic data on disk, LD=1 synthetic data generated in-memory, LD=3 real data.
	
* distr
	<br /> - Choose data distribution (i.e. distr=c(orrelated), distr=i(ndependent), distr=a(nticorrelated) to generate on disk or in-memory.
	
* script
	<br /> - script name used to generate synthetic data on disk.
	
* REAL\_DATA\_PATH
	<br /> - Path to real data file. It needs to follow a specific naming convention <name>\_<objects>\_<dims>
	
* device
	<br /> - Choose between cpu or gpu evaluation. device=0, cpu-only for now.
	
* QM=0
	<br /> - Query attributes starting from the last or first attribute, 0 or 1 respectively
* QD
	<br /> - Query interval testing (i.e. QD=2 then 2,4,6,8).

* IMP
	<br /> - Choose what implementation to test (i.e. Scalar=0, SIMD=1, Threads=2, Multi-query random dimension=3, Multi-query fixed dimension=4)
	
* ITER
	<br /> - Choose number of iterations to benchmark a query.

* MQTHREADS
	<br /> - Multi-query number of threads.
	
* STATS\_EFF
 	<br /> - Gather statistics associated with number of objects evaluated.
 	
 * WORKLOAD
 	<br /> - Number of queries generated for multi-query evaluation.

* TA\_B
	<br /> - Enable TA benchmark.

* TPAc\_B
	<br /> - Enable FTE column major benchmark.

* TPAr\_B
	<br /> - Enable FTE row major benchmark.

* VTA\_B
	<br /> - Enable VTA benchmark.

* PTA\_B
	<br /> - Enable PTA benchmark.

* SLA\_B
	<br /> - Enable SLA benchmark.






