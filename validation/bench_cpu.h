#ifndef BENCH_CPU_H
#define BENCH_CPU_H

#include "../time/Time.h"
#include "../tools/ArgParser.h"
#include "../input/File.h"

#include "../cpu/AA.h"
#include "../cpu/TA.h"

#include "../cpu_opt/MSA.h"
#include "../cpu_opt/LSA.h"
#include "../cpu_opt/TPAc.h"
#include "../cpu_opt/TPAr.h"
#include "../cpu_opt/VTA.h"
#include "../cpu_opt/PTA.h"
#include "../cpu_opt/SLA.h"

//#define ITER 1
//#define IMP 1//0:Scalar 1:SIMD 2:Threads + SIMD
//#define QM 1//0:Multiple Queries 1: Single Queries
//#define QD 1

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

//float weights[8] = { 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 };
float weights[8] = { 1,1,1,1,1,1,1,1 };

#if QM == 0
	#if NUM_DIMS == 2
		uint8_t attr[1][2] = { {0,1} };
	#elif NUM_DIMS == 3
		uint8_t attr[2][3] = { {1,2,0}, {0,1,2} };
	#elif NUM_DIMS == 4
		uint8_t attr[3][4] = { {2,3,0,0}, {1,2,3,0}, {0,1,2,3} };
	#elif NUM_DIMS == 6
		uint8_t attr[5][6] = { {4,5,0,0,0,0}, {3,4,5,0,0,0}, {2,3,4,5,0,0}, {1,2,3,4,5,0}, {0,1,2,3,4,5} };
	#else
		uint8_t attr[7][8] = {
				{6,7,0,0,0,0,0,0}, {5,6,7,0,0,0,0,0}, {4,5,6,7,0,0,0,0},
				{3,4,5,6,7,0,0,0}, {2,3,4,5,6,7,0,0}, {1,2,3,4,5,6,7,0}, {0,1,2,3,4,5,6,7}
		};
	#endif
#else
	#if NUM_DIMS == 2
		uint8_t attr[1][2] = { {0,1} };
	#elif NUM_DIMS == 3
		uint8_t attr[2][3] = { {0,1,0}, {0,1,2} };
	#elif NUM_DIMS == 4
		uint8_t attr[3][4] = { {0,1,0,0}, {0,1,2,0}, {0,1,2,3} };
	#elif NUM_DIMS == 6
		uint8_t attr[5][6] = {{0,1,0,0,0,0}, {0,1,2,0,0,0},  {0,1,2,3,0,0}, {0,1,2,3,4,0}, {0,1,2,3,4,5} };
	#else
		uint8_t attr[7][8] = {
				{0,1,0,0,0,0,0,0}, {0,1,2,0,0,0,0,0}, {0,1,2,3,0,0,0,0},
				{0,1,2,3,4,0,0,0}, {0,1,2,3,4,5,0,0}, {0,1,2,3,4,5,6,0}, {0,1,2,3,4,5,6,7}
		};
	#endif
#endif

const std::string distributions[3] ={"correlated","independent","anticorrelated"};

void bench_ta(std::string fname,uint64_t n, uint64_t d, uint64_t ks, uint64_t ke){
	File<float> f(fname,false,n,d);
	TA<float,uint64_t> ta(f.rows(),f.items());

	if (LD != 0){
		std::cout << "Loading data from file !!!" <<std::endl;
		f.load(ta.get_cdata());
	}else{
		std::cout << "Generating ( "<< distributions[DISTR] <<" ) data in memory !!!" <<std::endl;
		f.gen(ta.get_cdata(),DISTR);
	}

	ta.init();
	ta.set_iter(ITER);
	uint8_t q = 2;
	for(uint64_t k = ks; k <= ke; k*=2){
		std::cout << "Benchmark <<<-------------" << f.rows() << "," << f.items() << "," << k << "------------->>> " << std::endl;
		for(uint8_t i = q; i <= f.items();i+=QD){
			//Warm up
			ta.findTopK(k,i,weights,attr[i-q]);
			ta.reset_clocks();
			//Benchmark
			for(uint8_t m = 0; m < ITER;m++){
				ta.findTopK(k,i,weights,attr[i-q]);
			}
			ta.benchmark();
		}
	}
}

void bench_tpar(std::string fname,uint64_t n, uint64_t d, uint64_t ks, uint64_t ke){
	File<float> f(fname,false,n,d);
	TPAr<float,uint64_t> tpar(f.rows(),f.items());

	if (LD != 1){
		std::cout << "Loading data from file !!!" <<std::endl;
		f.load(tpar.get_cdata());
	}else{
		std::cout << "Generating ( "<< distributions[DISTR] <<" ) data in memory !!!" <<std::endl;
		f.gen(tpar.get_cdata(),DISTR);
	}

	tpar.init();
	tpar.set_iter(ITER);
	uint8_t q = 2;
	for(uint64_t k = ks; k <= ke; k*=2){
		std::cout << "Benchmark <<<-------------" << f.rows() << "," << f.items() << "," << k << "------------->>> " << std::endl;
		for(uint8_t i = q; i <= f.items();i+=QD){
			//Warm up
			if (IMP == 0){
				tpar.findTopKscalar(k,i,weights,attr[i-q]);
			}else if(IMP == 1){
				tpar.findTopKsimd(k,i,weights,attr[i-q]);
			}else if(IMP == 2){
				tpar.findTopKthreads(k,i,weights,attr[i-q]);
			}
			tpar.reset_clocks();
			//Benchmark
			for(uint8_t m = 0; m < ITER;m++){
				if (IMP == 0){
					tpar.findTopKscalar(k,i,weights,attr[i-q]);
				}else if(IMP == 1){
					tpar.findTopKsimd(k,i,weights,attr[i-q]);
				}else if(IMP == 2){
					tpar.findTopKthreads(k,i,weights,attr[i-q]);
				}
			}
			tpar.benchmark();
		}
	}
}

void bench_tpac(std::string fname,uint64_t n, uint64_t d, uint64_t ks, uint64_t ke){
	File<float> f(fname,false,n,d);
	TPAc<float,uint64_t> tpac(f.rows(),f.items());
	f.set_transpose(true);

	if (LD != 1){
		std::cout << "Loading data from file !!!" <<std::endl;
		f.load(tpac.get_cdata());
	}else{
		std::cout << "Generating ( "<< distributions[DISTR] <<" ) data in memory !!!" <<std::endl;
		f.gen(tpac.get_cdata(),DISTR);
	}

	tpac.init();
	tpac.set_iter(ITER);
	uint8_t q = 2;
	for(uint64_t k = ks; k <= ke; k*=2){
		std::cout << "Benchmark <<<-------------" << f.rows() << "," << f.items() << "," << k << "------------->>> " << std::endl;
		for(uint8_t i = q; i <= f.items();i+=QD){
			//Warm up
			if (IMP == 0){
				tpac.findTopKscalar(k,i,weights,attr[i-q]);
			}else if(IMP == 1){
				tpac.findTopKsimd(k,i,weights,attr[i-q]);
			}else if(IMP == 2){
				tpac.findTopKthreads(k,i,weights,attr[i-q]);
			}
			tpac.reset_clocks();
			//Benchmark
			for(uint8_t m = 0; m < ITER;m++){
				if (IMP == 0){
					tpac.findTopKscalar(k,i,weights,attr[i-q]);
				}else if(IMP == 1){
					tpac.findTopKsimd(k,i,weights,attr[i-q]);
				}else if(IMP == 2){
					tpac.findTopKthreads(k,i,weights,attr[i-q]);
				}
			}
			tpac.benchmark();
		}
	}
}

void bench_vta(std::string fname,uint64_t n, uint64_t d, uint64_t ks, uint64_t ke){
	File<float> f(fname,false,n,d);
	VTA<float,uint64_t> vta(f.rows(),f.items());
	f.set_transpose(true);

	if (LD != 1){
		std::cout << "Loading data from file !!!" <<std::endl;
		f.load(vta.get_cdata());
	}else{
		std::cout << "Generating ( "<< distributions[DISTR] <<" ) data in memory !!!" <<std::endl;
		f.gen(vta.get_cdata(),DISTR);
	}

	vta.init();
	vta.set_iter(ITER);
	uint8_t q = 2;
	for(uint64_t k = ks; k <= ke; k*=2){
		std::cout << "Benchmark <<<-------------" << f.rows() << "," << f.items() << "," << k << "------------->>> " << std::endl;
		for(uint8_t i = q; i <= f.items();i+=QD){
			//Warm up
			if (IMP == 0){
				vta.findTopKscalar(k,i,weights,attr[i-q]);
			}else if(IMP == 1){
				vta.findTopKsimd(k,i,weights, attr[i-q]);
			}else if(IMP == 2){
				vta.findTopKthreads(k,i,weights,attr[i-q]);
			}
			vta.reset_clocks();
			//Benchmark
			for(uint8_t m = 0; m < ITER;m++){
				if (IMP == 0){
					vta.findTopKscalar(k,i,weights,attr[i-q]);
				}else if(IMP == 1){
					vta.findTopKsimd(k,i,weights,attr[i-q]);
				}else if(IMP == 2){
					vta.findTopKthreads(k,i,weights,attr[i-q]);
				}
			}
			vta.benchmark();
		}
	}
}

void bench_pta(std::string fname,uint64_t n, uint64_t d, uint64_t ks, uint64_t ke){
	File<float> f(fname,false,n,d);
	PTA<float,uint64_t> pta(f.rows(),f.items());
	f.set_transpose(true);

	if (LD != 1){
		std::cout << "Loading data from file !!!" <<std::endl;
		f.load(pta.get_cdata());
	}else{
		std::cout << "Generating ( "<< distributions[DISTR] <<" ) data in memory !!!" <<std::endl;
		f.gen(pta.get_cdata(),DISTR);
	}

	pta.init();
	pta.set_iter(ITER);
	uint8_t q = 2;
	for(uint64_t k = ks; k <= ke; k*=2){
		std::cout << "Benchmark <<<-------------" << f.rows() << "," << f.items() << "," << k << "------------->>> " << std::endl;
		for(uint8_t i = q; i <= f.items();i+=QD){
			//Warm up
			if (IMP == 0){
				pta.findTopKscalar(k,i,weights,attr[i-q]);
			}else if(IMP == 1){
				pta.findTopKsimd(k,i,weights,attr[i-q]);
			}else if(IMP == 2){
				pta.findTopKthreads2(k,i,weights,attr[i-q]);
			}
			pta.reset_clocks();
			//Benchmark
			for(uint8_t m = 0; m < ITER;m++){
				if (IMP == 0){
					pta.findTopKscalar(k,i,weights,attr[i-q]);
				}else if(IMP == 1){
					pta.findTopKsimd(k,i,weights,attr[i-q]);
				}else if(IMP == 2){
					pta.findTopKthreads2(k,i,weights,attr[i-q]);
				}
			}
			pta.benchmark();
		}
	}
}

void bench_sla(std::string fname,uint64_t n, uint64_t d, uint64_t ks,uint64_t ke){
	File<float> f(fname,false,n,d);
	SLA<float,uint64_t> sla(f.rows(),f.items());
	f.set_transpose(true);

	if (LD != 1){
		std::cout << "Loading data from file !!!" <<std::endl;
		f.load(sla.get_cdata());
	}else{
		std::cout << "Generating ( "<< distributions[DISTR] <<" ) data in memory !!!" <<std::endl;
		f.gen(sla.get_cdata(),DISTR);
	}

	sla.init();
	sla.set_iter(ITER);
	uint8_t q = 2;
	for(uint64_t k = ks; k <= ke; k*=2){
		std::cout << "Benchmark <<<-------------" << f.rows() << "," << f.items() << "," << k << "------------->>> " << std::endl;
		for(uint8_t i = q; i <= f.items();i+=QD){
			//Warm up
			if (IMP == 0){
				sla.findTopKscalar(k,i,weights,attr[i-q]);
			}else if(IMP == 1){
				sla.findTopKsimd(k,i,weights,attr[i-q]);
			}else if(IMP == 2){
				sla.findTopKthreads(k,i,weights,attr[i-q]);
			}
			sla.reset_clocks();
			//Benchmark
			for(uint8_t m = 0; m < ITER;m++){
				if (IMP == 0){
					sla.findTopKscalar(k,i,weights,attr[i-q]);
				}else if(IMP == 1){
					sla.findTopKsimd(k,i,weights,attr[i-q]);
				}else if(IMP == 2){
					sla.findTopKthreads(k,i,weights,attr[i-q]);
				}
			}
			sla.benchmark();
		}
	}
}

void bench_msa(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	MSA<float,uint64_t> msa(f.rows(),f.items());
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
	LSA<float,uint64_t> lsa(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(lsa.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	lsa.init();
	//for(uint8_t m = 0; m < ITER;m++) lsa.findTopK(k);
	for(uint8_t m = 0; m < ITER;m++) lsa.findTopKscalar(k);
	lsa.benchmark();
}


#endif
