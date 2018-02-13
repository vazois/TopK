#include<limits>

#include "validation/bench_cpu.h"
#include "validation/debug_cpu.h"

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

	//debug(ap.getString("-f"),n,d,K);
	bench_fa(ap.getString("-f"),n,d,K);
	//bench_ta(ap.getString("-f"),n,d,K);
	//bench_cba(ap.getString("-f"),n,d,K);
	//bench_fac(ap.getString("-f"),n,d,K);


	return 0;
}
