#include "GPA.h"
#include "GFA.h"
#include "GPAm.h"
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
	File<float> f(fname,true,n,d);
	GPA<float> gpa;
	GPAm<float> gpam;

	gpa.alloc(f.items(),f.rows());

	f.set_transpose(true);
	std::cout << "Loading data..." << std::endl;
	f.load(gpa.get_cdata());
	std::cout << "Finished Loading data..." << std::endl;


	gpa.init();
	gpa.findTopK(K);
	gpa.benchmark();


	gpam.set_dim(f.items(),f.rows());
	gpam.set_cdata(gpa.get_cdata());
	gpam.alloc(f.items(),f.rows());
	gpam.findTopK(K);
}

void bench_gpa(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,true,n,d);
	GPA<float> gpa;

	gpa.alloc(f.items(),f.rows());

	f.set_transpose(true);
	std::cout << "Loading data..." << std::endl;
	f.load(gpa.get_cdata());
	std::cout << "Finished Loading data..." << std::endl;

	gpa.init();
	gpa.findTopK(K);
	gpa.benchmark();
}

void bench_gpam(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,true,n,d);
	GPA<float> gpam;

	gpam.alloc(f.items(),f.rows());

	f.set_transpose(true);
	std::cout << "Loading data..." << std::endl;
	f.load(gpam.get_cdata());
	std::cout << "Finished Loading data..." << std::endl;

	gpam.init();
	gpam.findTopK(K);
	gpam.benchmark();
}

void mem_gpam(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,true,n,d);
	GPA<float> gpam;

	gpam.alloc(f.items(),f.rows());

	//f.set_transpose(true);
	std::cout << "Loading data..." << std::endl;
	//f.load(gpam.get_cdata());
	std::cout << "Finished Loading data..." << std::endl;

	gpam.init();
	//gpam.findTopK(K);
	//gpam.benchmark();
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

//	uint64_t nu;
//	if(!ap.exists("-nu")){
//		nu = n;
//	}else{
//		nu = ap.getInt("-nu");
//	}
//
//	uint64_t nl;
//	if(!ap.exists("-nl")){
//		nl = n;
//	}else{
//		nl = ap.getInt("-nl");
//	}

	//test(ap.getString("-f"),n,d, K);
	//simple_benchmark(ap.getString("-f"),n,d,K);
	bench_gpa(ap.getString("-f"),n,d,K);
	//bench_gpam(ap.getString("-f"),n,d,K);
	//mem_gpam(ap.getString("-f"),n,d,K);

	return 0;
}
