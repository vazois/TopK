#include "../validation/bench_cpu_rj.h"

int main(int argc, char **argv){
	std::cout << "RankJoin Main" << std::endl;
	//test_bench();
	bench_no_part_hjr();
	
	return 0;
}
