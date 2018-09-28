#include "../validation/bench_cpu_rj.h"

int main(int argc, char **argv){
	std::cout << "RankJoin Main" << std::endl;

	uint32_t n0 = RSIZE, d0 = DIMS;
	uint32_t n1 = SSIZE, d1 = DIMS;
	uint32_t k = 10;

	if(n0 > n1){
		perror("RSIZE should be always greater than SSIZE\n");
		exit(1);
	}

	RankJoinInstance<uint32_t,float> rj_inst(n0,d0,n1,d1,k);
	GenData<uint32_t,float> gen_data(&rj_inst);

	bench_hjr(&rj_inst);
	bench_pbrj(&rj_inst);
	
	return 0;
}
