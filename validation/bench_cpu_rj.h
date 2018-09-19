#ifndef BENCH_CPU_RJ
#define BENCH_CPU_RJ

//#include "../cpu_rj/ARJ.h"
#include "../cpu_rj/HJR.h"

void test_bench(){
	uint32_t n0 = 1024, d0 = 2;
	uint32_t n1 = 1024*1024, d1 = 2;
	RankJoinInstance<uint32_t,float> rj_inst(n0,d0,n1,d1);
	GenData<uint32_t,float> gen_data(&rj_inst);

	rj_inst.sample();
}


#endif

