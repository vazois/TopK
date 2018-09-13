#ifndef BENCH_CPU_RJ
#define BENCH_CPU_RJ

#include "../cpu_rj/ARJ.h"

void test_bench(){
	uint32_t n0 = 128, d0 = 2;
	uint32_t n1 = 4096, d1 = 2;
	GenData<uint32_t,float> gen_data;
	RankJoinInstance<uint32_t,float> rj_inst(n0,d0,n1,d1);

	gen_data.populate(rj_inst.getR(),rj_inst.getS());
	rj_inst.sample();
}


#endif

