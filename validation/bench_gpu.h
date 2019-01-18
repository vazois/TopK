#ifndef BENCH_GPU_H
#define BENCHGPU_H

#include "../input/File.h"
#include "bench_common.h"

//#include "../gpu/GPA.h"
//#include "../gpu/GFA.h"
//#include "../gpu/GPAm.h"
#include "../gpu/BTA.h"
#include "../gpu/GVTA.h"
#include "../gpu/GPTA.h"
#include "../gpu/GTA.h"

uint32_t bench_query[8] = {0,1,2,3,4,5,6,7};
const std::string distributions[3] ={"correlated","independent","anticorrelated"};

void bench_bta(std::string fname, uint64_t n, uint64_t d, uint64_t ks, uint64_t ke){
	File<float> f(fname,true,n,d);
	BTA<float,uint32_t> bta(n,d);

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
	bta.init();
	uint64_t q = 2;
	if(IMP < 3){
		for(uint64_t k = ks; k <= ke; k*=2){
			for(uint64_t i = q; i <= f.items();i+=QD){
				std::cout << "Benchmark <<<-------------" << f.rows() << "," << (int)i << "," << k << "------------->>> " << std::endl;
				bta.copy_query(weights,attr[i-q]);
				bta.findTopK(k,i);
				bta.benchmark();
			}
		}
	}
}

void bench_gvta(std::string fname, uint64_t n, uint64_t d, uint64_t ks, uint64_t ke){
	File<float> f(fname,true,n,d);
	GVTA<float,uint32_t> gvta(n,d);

	std::cout << "Allocating buffers ..." << std::endl;
	gvta.alloc();
	std::cout << "Loading data ... " << std::endl;
	f.set_transpose(true);

	if (LD != 1){
		std::cout << "Loading data from file !!!" <<std::endl;
		f.load(gvta.get_cdata());
	}else{
		std::cout << "Generating ( "<< distributions[DISTR] <<" ) data in memory !!!" <<std::endl;
		f.gen(gvta.get_cdata(),DISTR);
	}

	gvta.init();
	gvta.set_iter(ITER);
	uint64_t q = 2;
	//uint64_t q = NUM_DIMS;
	if(IMP < 3){
		for(uint64_t k = ks; k <= ke; k*=2){
			for(uint64_t i = q; i <= f.items();i+=QD){
				std::cout << "Benchmark <<<-------------" << f.rows() << "," << (int)i << "," << k << "------------->>> " << std::endl;
				gvta.copy_query(weights,attr[i-q]);
				for(uint8_t m = 0; m < ITER;m++){
					gvta.findTopK(k,i);
				}
				gvta.benchmark();
			}
		}
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


}

#endif
