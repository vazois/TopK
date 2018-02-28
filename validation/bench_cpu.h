#ifndef BENCH_CPU_H
#define BENCH_CPU_H

#include "time/Time.h"
#include "tools/ArgParser.h"
#include "input/File.h"

#include "cpu/AA.h"
#include "cpu/FA.h"
#include "cpu/TA.h"

#include "cpu_opt/MSA.h"
#include "cpu_opt/LSA.h"
#include "cpu_opt/GSA.h"
#include "cpu_opt/TPAc.h"
#include "cpu_opt/TPAr.h"
#include "cpu_opt/TAcsimd.h"
#include "cpu_opt/QLA.h"

#define ITER 5

void bench_fa(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	FA<float,uint32_t> fa(f.rows(),f.items());

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(fa.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	fa.init(); fa.findTopK(k);
	fa.benchmark();
}

void bench_ta(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	TA<float,uint32_t> ta(f.rows(),f.items());

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(ta.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	ta.init();
	for(uint8_t m = 0; m < ITER;m++) ta.findTopK(k);
	ta.benchmark();
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

void bench_gsa(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	GSA<float,uint32_t> gsa(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(gsa.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	gsa.init();
	gsa.findTopK(k);
	gsa.benchmark();
}

void bench_tpac(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	TPAc<float,uint32_t> tpac(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(tpac.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	tpac.init();
	for(uint8_t m = 0; m < ITER;m++) tpac.findTopKsimd(k);
	tpac.benchmark();
}

void bench_tpar(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	TPAr<float,uint32_t> tpar(f.rows(),f.items());

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(tpar.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	tpar.init();
	tpar.findTopK(k);
	tpar.benchmark();
}

void bench_ta_simd(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	TAcsimd<float,uint32_t> ta_simd(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(ta_simd.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	ta_simd.init2();
	//for(uint8_t m = 0; m < 10;m++) ta_simd.findTopK(k);
	for(uint8_t m = 0; m < ITER;m++) ta_simd.findTopKsimd(k);
	ta_simd.benchmark();
}

void bench_qla(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	QLA<float,uint32_t> qla(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(qla.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	qla.init();
	//for(uint8_t m = 0; m < 10;m++) ta_simd.findTopK(k);
	for(uint8_t m = 0; m < ITER;m++) qla.findTopK(k);
	qla.benchmark();
}
#endif
