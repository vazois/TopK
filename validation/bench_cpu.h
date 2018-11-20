#ifndef BENCH_CPU_H
#define BENCH_CPU_H

#include "../time/Time.h"
#include "../tools/ArgParser.h"
#include "../input/File.h"
#include "bench_common.h"

#include "../cpu/AA.h"
#include "../cpu/TA.h"

#include "../cpu/MSA.h"
#include "../cpu/LSA.h"
#include "../cpu/TPAc.h"
#include "../cpu/TPAr.h"
#include "../cpu/VTA.h"
#include "../cpu/PTA.h"
#include "../cpu/SLA.h"

const std::string distributions[3] ={"correlated","independent","anticorrelated"};

uint8_t work_array[WORKLOAD];

void random_workload(){
	srand(time(NULL));
	for(uint64_t i = 0; i < WORKLOAD;i++){
		work_array[i]=rand() % ((NUM_DIMS) - 2 + 1) + 2;
		//std::cout << "work_array: " << (int)work_array[i] << std::endl;
	}
}

void bench_ta(std::string fname,uint64_t n, uint64_t d, uint64_t ks, uint64_t ke){
	File<float> f(fname,false,n,d);
	TA<float,uint64_t> ta(f.rows(),f.items());

	if (LD != 1){
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
		for(uint8_t i = q; i <= f.items();i+=QD){
			std::cout << "Benchmark <<<-------------" << f.rows() << "," << (int)i << "," << k << "------------->>> " << std::endl;
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
		for(uint8_t i = q; i <= f.items();i+=QD){
			std::cout << "Benchmark <<<-------------" << f.rows() << "," << (int)i << "," << k << "------------->>> " << std::endl;
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
	if(IMP!=3){
		for(uint64_t k = ks; k <= ke; k*=2){
			for(uint8_t i = q; i <= f.items();i+=QD){
				std::cout << "Benchmark <<<-------------" << f.rows() << "," << (int)i << "," << k << "------------->>> " << std::endl;
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
	}else{
		if(IMP==3){
			random_workload();
			omp_set_num_threads(MQTHREADS);
			std::cout << "<<<Random Attribute Multiple Queries PTA - ("<<MQTHREADS<< ") threads >>>" << std::endl;
			for(uint64_t k = ks; k <= ke; k*=2){
				std::cout << "Benchmark <<<-------------" << f.rows() << "," << f.items() << "," << k << "------------->>> " << std::endl;
				#pragma omp parallel
				{
					uint32_t tid = omp_get_thread_num();
					for(uint64_t j = tid; j < WORKLOAD; j+=MQTHREADS){
						uint8_t i = work_array[j];
						for(uint8_t m = 0; m < ITER;m++){
							tpac.findTopKsimdMQ(k,i,weights,attr[i-2],tid);
						}
					}
				}
				tpac.benchmark();
			}
		}else{
			omp_set_num_threads(MQTHREADS);
			std::cout << "<<<Same Attribute Multiple Queries PTA - ("<<MQTHREADS<< ") threads >>>" << std::endl;
			for(uint64_t k = ks; k <= ke; k*=2){
				for(uint8_t i = q; i <= f.items();i+=QD){
					std::cout << "Benchmark <<<-------------" << f.rows() << "," << (int)i << "," << k << "------------->>> " << std::endl;
					#pragma omp parallel
					{
						uint32_t tid = omp_get_thread_num();
						for(uint64_t j = tid; j < WORKLOAD; j+=MQTHREADS){
							for(uint8_t m = 0; m < ITER;m++){
								tpac.findTopKsimdMQ(k,i,weights,attr[i-2],tid);
							}
						}
					}
					tpac.benchmark();
					tpac.reset_clocks();
				}
			}
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
	if(IMP < 3){
		for(uint64_t k = ks; k <= ke; k*=2){
			for(uint8_t i = q; i <= f.items();i+=QD){
				std::cout << "Benchmark <<<-------------" << f.rows() << "," << (int)i << "," << k << "------------->>> " << std::endl;
				//Warm up
				if (IMP == 0){
					vta.findTopKscalar(k,i,weights,attr[i-q]);
				}else if(IMP == 1){
					vta.findTopKsimd(k,i,weights, attr[i-q]);
				}else if(IMP == 2){
					vta.findTopKthreads2(k,i,weights,attr[i-q]);
				}
				vta.reset_clocks();
				//Benchmark
				for(uint8_t m = 0; m < ITER;m++){
					if (IMP == 0){
						vta.findTopKscalar(k,i,weights,attr[i-q]);
					}else if(IMP == 1){
						vta.findTopKsimd(k,i,weights,attr[i-q]);
					}else if(IMP == 2){
						vta.findTopKthreads2(k,i,weights,attr[i-q]);
					}
				}
				vta.benchmark();
			}
		}
	}else{
		if(IMP==3){
			random_workload();
			omp_set_num_threads(MQTHREADS);
			std::cout << "<<<Random Attribute Multiple Queries PTA - ("<<MQTHREADS<< ") threads >>>" << std::endl;
			for(uint64_t k = ks; k <= ke; k*=2){
				std::cout << "Benchmark <<<-------------" << f.rows() << "," << f.items() << "," << k << "------------->>> " << std::endl;
				#pragma omp parallel
				{
					uint32_t tid = omp_get_thread_num();
					for(uint64_t j = tid; j < WORKLOAD; j+=MQTHREADS){
						uint8_t i = work_array[j];
						for(uint8_t m = 0; m < ITER;m++){
							vta.findTopKsimdMQ(k,i,weights,attr[i-2],tid);
						}
					}
				}
				vta.benchmark();
			}
		}else{
			omp_set_num_threads(MQTHREADS);
			std::cout << "<<<Same Attribute Multiple Queries PTA - ("<<MQTHREADS<< ") threads >>>" << std::endl;
			for(uint64_t k = ks; k <= ke; k*=2){
				for(uint8_t i = q; i <= f.items();i+=QD){
					std::cout << "Benchmark <<<-------------" << f.rows() << "," << (int)i << "," << k << "------------->>> " << std::endl;
					#pragma omp parallel
					{
						uint32_t tid = omp_get_thread_num();
						for(uint64_t j = tid; j < WORKLOAD; j+=MQTHREADS){
							for(uint8_t m = 0; m < ITER;m++){
								vta.findTopKsimdMQ(k,i,weights,attr[i-2],tid);
							}
						}
					}
					vta.benchmark();
					vta.reset_clocks();
				}
			}
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
	if(IMP<3){
		for(uint64_t k = ks; k <= ke; k*=2){
			for(uint8_t i = q; i <= f.items();i+=QD){
				std::cout << "Benchmark <<<-------------" << f.rows() << "," << (int)i << "," << k << "------------->>> " << std::endl;
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
	}else{
		if(IMP==3){
			random_workload();
			omp_set_num_threads(MQTHREADS);
			std::cout << "<<<Random Attribute Multiple Queries PTA - ("<<MQTHREADS<< ") threads >>>" << std::endl;
			for(uint64_t k = ks; k <= ke; k*=2){
				std::cout << "Benchmark <<<-------------" << f.rows() << "," << f.items() << "," << k << "------------->>> " << std::endl;
				#pragma omp parallel
				{
					uint32_t tid = omp_get_thread_num();
					for(uint64_t j = tid; j < WORKLOAD; j+=MQTHREADS){
						uint8_t i = work_array[j];
						for(uint8_t m = 0; m < ITER;m++){ pta.findTopKsimdMQ(k,i,weights,attr[i-2],tid); }
					}
				}
				pta.benchmark();
			}
		}else{

			omp_set_num_threads(MQTHREADS);
			std::cout << "<<<Same Attribute Multiple Queries PTA - ("<<(int)MQTHREADS<< ") threads >>>" << std::endl;
			for(uint64_t k = ks; k <= ke; k*=2){
				for(uint8_t i = q; i <= f.items();i+=QD){
					std::cout << "Benchmark <<<-------------" << f.rows() << "," << (int)i << "," << k << "------------->>> " << std::endl;
					#pragma omp parallel
					{
						uint32_t tid = omp_get_thread_num();
						for(uint64_t j = tid; j < WORKLOAD; j+=MQTHREADS){
							for(uint8_t m = 0; m < ITER;m++){ pta.findTopKsimdMQ(k,i,weights,attr[i-2],tid); }
						}
					}
					pta.benchmark();
					pta.reset_clocks();
				}
			}
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
		for(uint8_t i = q; i <= f.items();i+=QD){
			std::cout << "Benchmark <<<-------------" << f.rows() << "," << (int)i << "," << k << "------------->>> " << std::endl;
			//Warm up
			if (IMP == 0){
				sla.findTopKscalar(k,i,weights,attr[i-q]);
			}else if(IMP == 1){
				sla.findTopKsimd(k,i,weights,attr[i-q]);
			}else if(IMP == 2){
				sla.findTopKthreads2(k,i,weights,attr[i-q]);
			}
			sla.reset_clocks();
			//Benchmark
			for(uint8_t m = 0; m < ITER;m++){
				if (IMP == 0){
					sla.findTopKscalar(k,i,weights,attr[i-q]);
				}else if(IMP == 1){
					sla.findTopKsimd(k,i,weights,attr[i-q]);
				}else if(IMP == 2){
					sla.findTopKthreads2(k,i,weights,attr[i-q]);
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
