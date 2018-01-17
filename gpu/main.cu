#include "GPA.h"
#include "GFA.h"
#include "input/File.h"
#include "tools/ArgParser.h"

#define K 100

void test(std::string fname, uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,true);
	GPA<float> gpa;

	gpa.alloc(f.items(),f.rows());
	f.set_transpose(true);
	std::cout << "Loading data..." << std::endl;
	f.load(gpa.get_cdata());
	//f.sample();
	gpa.init();
	gpa.findTopK(K);
	gpa.benchmark();
}

void simple_benchmark(std::string fname, uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,true);
	GPA<float> gpa;
	GFA<float> gfa;

	gpa.alloc(f.items(),f.rows());
	gfa.set_gdata(gpa.get_gdata());
	f.set_transpose(true);
	std::cout << "Loading data..." << std::endl;
	f.load(gpa.get_cdata());
	std::cout << "Finished Loading data..." << std::endl;

	gpa.init();
	gpa.findTopK(K);
	gpa.benchmark();
}



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
	test(ap.getString("-f"),n,d, K);


	return 0;
}
