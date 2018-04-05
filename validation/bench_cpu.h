#ifndef BENCH_CPU_H
#define BENCH_CPU_H

#include "../time/Time.h"
#include "../tools/ArgParser.h"
#include "../input/File.h"

#include "../cpu/AA.h"
#include "../cpu/FA.h"
#include "../cpu/TA.h"

#include "../cpu_opt/MSA.h"
#include "../cpu_opt/LSA.h"
#include "../cpu_opt/TPAc.h"
#include "../cpu_opt/TPAr.h"
#include "../cpu_opt/PTA.h"
#include "../cpu_opt/SLA.h"
#include "../cpu_opt/VTA.h"
#include "../cpu_opt/PVTA.h"

#define ITER 10
#define IMP 1//0:Scalar 1:SIMD 2:Threads + SIMD

//[0,1,2,3][4,5,6,7][8,9,10,11][12,13,14,15]
uint8_t qq[72] =
	{
		1,15,//2
		1,5,9,13,//4
		1,3,5,7,9,13,//6
		1,3,5,7,9,11,13,15,//8
		0,1,3,4,5,7,8,11,12,15,//10
		0,1,3,4,5,7,9,10,11,12,13,14,//12
		0,1,2,3,4,5,7,8,9,10,11,12,13,14,//14
		0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15//16
	};

void bench_ta(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	TA<float,uint32_t> ta(f.rows(),f.items());

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(ta.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	ta.init();
	ta.set_iter(ITER);
	for(uint8_t i = 2; i <= f.items();i+=2){
		//Warm up
		if (IMP == 0){
			ta.findTopK(k,i);
		}else if(IMP == 1){
			ta.findTopK(k,i);
		}else if(IMP == 2){
			ta.findTopKthreads(k,i);
		}
		ta.reset_clocks();
		//Benchmark
		for(uint8_t m = 0; m < ITER;m++){
			if (IMP == 0){
				ta.findTopK(k,i);
			}else if(IMP == 1){
				ta.findTopK(k,i);
			}else if(IMP == 2){
				ta.findTopKthreads(k,i);
			}
		}
		ta.benchmark();
	}
}

void bench_tpar(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	TPAr<float,uint32_t> tpar(f.rows(),f.items());

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(tpar.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	tpar.init();
	tpar.set_iter(ITER);
	for(uint8_t i = 2; i <= f.items();i+=2){
		//Warm up
		if (IMP == 0){
			tpar.findTopKscalar(k,i);
		}else if(IMP == 1){
			tpar.findTopKsimd(k,i);
		}else if(IMP == 2){
			tpar.findTopKthreads(k,i);
		}
		tpar.reset_clocks();
		//Benchmark
		for(uint8_t m = 0; m < ITER;m++){
			if (IMP == 0){
				tpar.findTopKscalar(k,i);
			}else if(IMP == 1){
				tpar.findTopKsimd(k,i);
			}else if(IMP == 2){
				tpar.findTopKthreads(k,i);
			}
		}
		tpar.benchmark();
	}
}

void bench_tpac(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	TPAc<float,uint32_t> tpac(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(tpac.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	tpac.init();
	tpac.set_iter(ITER);
	for(uint8_t i = 2; i <= f.items();i+=2){
		//Warm up
		if (IMP == 0){
			tpac.findTopKscalar(k,i);
		}else if(IMP == 1){
			tpac.findTopKsimd(k,i);
		}else if(IMP == 2){
			tpac.findTopKthreads(k,i);
		}
		tpac.reset_clocks();
		//Benchmark
		for(uint8_t m = 0; m < ITER;m++){
			if (IMP == 0){
				tpac.findTopKscalar(k,i);
			}else if(IMP == 1){
				tpac.findTopKsimd(k,i);
			}else if(IMP == 2){
				tpac.findTopKthreads(k,i);
			}
		}
		tpac.benchmark();
	}
}

void bench_pta(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	PTA<float,uint32_t> pta(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(pta.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	pta.init();
	pta.set_iter(ITER);
	for(uint8_t i = 2; i <= f.items();i+=2){
		//Warm up
		if (IMP == 0){
			pta.findTopKscalar(k,i);
		}else if(IMP == 1){
			pta.findTopKsimd(k,i);
		}
		pta.reset_clocks();
		//Benchmark
		for(uint8_t m = 0; m < ITER;m++){
			if (IMP == 0){
				pta.findTopKscalar(k,i);
			}else if(IMP == 1){
				pta.findTopKsimd(k,i);
			}
		}
		pta.benchmark();
	}
}

void bench_vta(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	VTA<float,uint32_t> vta(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(vta.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	//bta.init();
	//bta.findTopKthreads(k,2);
	vta.init();
	//return;
	vta.set_iter(ITER);
	for(uint8_t i = f.items(); i <= f.items();i+=2){
		//Warm up
		if (IMP == 0){
			vta.findTopKscalar(k,i);
		}else if(IMP == 1){
			vta.findTopKsimd(k,i);
		}else if(IMP == 2){
			vta.findTopKthreads(k,i);
		}
		vta.reset_clocks();
		//Benchmark
		for(uint8_t m = 0; m < ITER;m++){
			if (IMP == 0){
				vta.findTopKscalar(k,i);
			}else if(IMP == 1){
				vta.findTopKsimd(k,i);
			}else if(IMP == 2){
				vta.findTopKthreads(k,i);
			}
		}
		vta.benchmark();
	}
}

void bench_fa(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	FA<float,uint32_t> fa(f.rows(),f.items());

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(fa.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	fa.init(); fa.findTopK(k);
	fa.benchmark();
}

void bench_msa(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	MSA<float,uint32_t> msa(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(msa.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	msa.init();
	for(uint8_t m = 0; m < ITER;m++) msa.findTopK(k);
	msa.benchmark();
}

void bench_lsa(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	LSA<float,uint32_t> lsa(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(lsa.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	lsa.init();
	//for(uint8_t m = 0; m < ITER;m++) lsa.findTopK(k);
	for(uint8_t m = 0; m < ITER;m++) lsa.findTopKscalar(k);
	lsa.benchmark();
}

void bench_sla(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	SLA<float,uint32_t> sla(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(sla.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	sla.init();
	sla.findTopK(k,d);
	sla.benchmark();
}

#endif
