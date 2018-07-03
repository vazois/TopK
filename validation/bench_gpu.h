#ifndef BENCH_GPU_H
#define BENCHGPU_H

#include "../input/File.h"

#include "../gpu/GPA.h"
#include "../gpu/GFA.h"
#include "../gpu/GPAm.h"
#include "../gpu/BTA.h"

float weights[MAX_ATTRIBUTES] = { 1,1,1,1,1,1,1,1 };//Q0
//float weights[MAX_ATTRIBUTES] = { 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 };//Q1
//float weights[MAX_ATTRIBUTES] = { 0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1 };//Q2
//float weights[MAX_ATTRIBUTES] = { 0.1,0.2,0.3,0.4,0.4,0.3,0.2,0.1 };//Q3
//float weights[MAX_ATTRIBUTES] = { 0.4,0.3,0.2,0.1,0.1,0.2,0.3,0.4 };//Q4

uint32_t query[MAX_ATTRIBUTES] = {0,1,2,3,4,5,6,7};

void bench_gpam(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,true,n,d);
	GPA<float> gpam;

	gpam.alloc(f.items(),f.rows());

	f.set_transpose(true);
	std::cout << "Loading data..." << std::endl;
	f.load(gpam.get_cdata());
	std::cout << "Finished Loading data..." << std::endl;

	gpam.init();
	gpam.findTopK(k);
	gpam.benchmark();
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
	gpa.findTopK(k);
	gpa.benchmark();
}

void bench_bta(std::string fname, uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,true,n,d);
	BTA<float,uint64_t> bta(n,d);

	std::cout << "Allocating buffers ..." << std::endl;
	bta.alloc();
	std::cout << "Loading data ... " << std::endl;
	f.set_transpose(true);
	f.load(bta.get_cdata());

	std::cout << "Calculating top-k ... " << std::endl;
	bta.init(weights,query);
	bta.findTopK(k);
}


#endif
