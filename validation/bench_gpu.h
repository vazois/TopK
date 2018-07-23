#ifndef BENCH_GPU_H
#define BENCHGPU_H

#include "../input/File.h"

//#include "../gpu/GPA.h"
//#include "../gpu/GFA.h"
//#include "../gpu/GPAm.h"
#include "../gpu/BTA.h"
#include "../gpu/GPTA.h"
#include "../gpu/GTA.h"

float bench_weights[8] = { 1,1,1,1,1,1,1,1 };//Q0
//float bench_weights[NUM_DIMS] = { 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 };//Q1
//float bench_weights[NUM_DIMS] = { 0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1 };//Q2
//float bench_weights[NUM_DIMS] = { 0.1,0.2,0.3,0.4,0.4,0.3,0.2,0.1 };//Q3
//float bench_weights[NUM_DIMS] = { 0.4,0.3,0.2,0.1,0.1,0.2,0.3,0.4 };//Q4

uint32_t bench_query[8] = {0,1,2,3,4,5,6,7};
const std::string distributions[3] ={"correlated","independent","anticorrelated"};

//void bench_gpam(std::string fname,uint64_t n, uint64_t d, uint64_t k){
//	File<float> f(fname,true,n,d);
//	GPA<float> gpam;
//
//	gpam.alloc(f.items(),f.rows());
//
//	f.set_transpose(true);
//	std::cout << "Loading data..." << std::endl;
//	f.load(gpam.get_cdata());
//	std::cout << "Finished Loading data..." << std::endl;
//
//	gpam.init();
//	gpam.findTopK(k);
//	gpam.benchmark();
//}
//
//void bench_gpa(std::string fname,uint64_t n, uint64_t d, uint64_t k){
//	File<float> f(fname,true,n,d);
//	GPA<float> gpa;
//
//	gpa.alloc(f.items(),f.rows());
//	f.set_transpose(true);
//	std::cout << "Loading data..." << std::endl;
//	f.load(gpa.get_cdata());
//	std::cout << "Finished Loading data..." << std::endl;
//
//	gpa.init();
//	gpa.findTopK(k);
//	gpa.benchmark();
//}

void bench_bta(std::string fname, uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,true,n,d);
	BTA<float,uint64_t> bta(n,d);

	std::cout << "Allocating buffers ..." << std::endl;
	bta.alloc();
	std::cout << "Loading data ... " << std::endl;
	f.set_transpose(true);

	if (LD != 1){
		std::cout << "Loading data from file !!!" <<std::endl;
		f.load(bta.get_cdata());
	}else{
		std::cout << "Generating ( "<< distributions[DISTR] <<" ) data in memory !!!" <<std::endl;
		f.gen(bta.get_cdata(),DISTR);
	}

	std::cout << "Calculating top-k ... " << std::endl;
	bta.init(bench_weights,bench_query);
	for(uint64_t m = NUM_DIMS ; m <= NUM_DIMS; m++){
		bta.findTopK(k,m);
		bta.benchmark();
	}
}

void bench_gta(std::string fname, uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,true,n,d);
	GTA<float,uint64_t> gta(n,d);

	std::cout << "Allocating buffers ..." << std::endl;
	gta.alloc();
	std::cout << "Loading data ... " << std::endl;
	f.set_transpose(true);

	if (LD != 1){
		std::cout << "Loading data from file !!!" <<std::endl;
		f.load(gta.get_cdata());
	}else{
		std::cout << "Generating ( "<< distributions[DISTR] <<" ) data in memory !!!" <<std::endl;
		f.gen(gta.get_cdata(),DISTR);
	}

	//gta.init(bench_weights,bench_query);
	//gta.benchmark();
}

void bench_gpta(std::string fname, uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,true,n,d);
	GPTA<float,uint64_t> gpta(n,d);

	std::cout << "Allocating buffers ..." << std::endl;
	gpta.alloc();
	std::cout << "Loading data ... " << std::endl;
	f.set_transpose(true);

	if (LD != 1){
		std::cout << "Loading data from file !!!" <<std::endl;
		f.load(gpta.get_cdata());
	}else{
		std::cout << "Generating ( "<< distributions[DISTR] <<" ) data in memory !!!" <<std::endl;
		f.gen(gpta.get_cdata(),DISTR);
	}

	gpta.init(bench_weights,bench_query);
	gpta.benchmark();
}



#endif
