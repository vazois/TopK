#include "../validation/bench_gpu.h"
#include "../tools/ArgParser.h"

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

	//bench_gpa(ap.getString("-f"),n,d,KKS);
	//bench_gpam(ap.getString("-f"),n,d,K);
	bench_bta(ap.getString("-f"),n,d,KKS);

	return 0;
}
