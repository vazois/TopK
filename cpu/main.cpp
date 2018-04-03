#include<limits>
#include <stdio.h>
#include <cstdint>
#include <stdio.h>
#include <tmmintrin.h>
#include <immintrin.h>

#include "validation/bench_cpu.h"
#include "test/Tests.h"

#define RUN_PAR false
#define K 100

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	if(!ap.exists("-f")){
		std::cout << "Missing file input!!!" << std::endl;
		exit(1);
	}

	if(!ap.exists("-d")){
		std::cout << "Missing d!!!" << std::endl;
		exit(1);
	}

	if(!ap.exists("-n")){
		std::cout << "Missing n!!!" << std::endl;
		exit(1);
	}

	uint64_t n = ap.getInt("-n");
	uint64_t d = ap.getInt("-d");
	uint64_t nu;
	if(!ap.exists("-nu")){
		nu = n;
	}else{
		nu = ap.getInt("-nu");
	}

	uint64_t nl;
	if(!ap.exists("-nl")){
		nl = n;
	}else{
		nl = ap.getInt("-nl");
	}

//	float v[4] = {0.3,0.2,0.7,0.5};
//	std::cout << "00: "<<v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;
//	simd4sort(v);
//	std::cout << "ee: " <<v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;
//	test_dt();

	bench_ta(ap.getString("-f"),n,d,K);
	bench_tpac(ap.getString("-f"),n,d,K);
	//bench_tpar(ap.getString("-f"),n,d,K);
	//bench_pta(ap.getString("-f"),n,d,K);
	//bench_sla(ap.getString("-f"),n,d,K);
	//bench_ptap(ap.getString("-f"),n,d,K);
	//test_dt();
	bench_vta(ap.getString("-f"),n,d,K);

	return 0;
}
