#ifndef BENCH_GPU_H
#define BENCHGPU_H

#include "../input/File.h"

#include "../gpu/GPA.h"
#include "../gpu/GFA.h"
#include "../gpu/GPAm.h"
#include "../gpu/BTA.h"

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

	bta.findTopK(k);
}


#endif
