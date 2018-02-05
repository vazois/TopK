#ifndef BENCH_CPU_H
#define BENCH_CPU_H

#include "time/Time.h"
#include "tools/ArgParser.h"
#include "input/File.h"

#include "cpu/AA.h"
#include "cpu/NA.h"
#include "cpu/FA.h"
#include "cpu/TA.h"
#include "cpu/CBA.h"
#include "cpu/BPA.h"
#include "cpu/T2S.h"
#include "cpu/FAc.h"
#include "cpu/PFAc.h"

void bench_ta(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	TA<float,uint32_t> ta(f.rows(),f.items());

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(ta.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	ta.init(); ta.findTopK(k);
	ta.benchmark();
}

void bench_cba(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	CBA<float,uint32_t> cba(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(cba.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	cba.init(); cba.findTopK(k);
	cba.benchmark();
}

void debug_fac(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	FAc<float,uint32_t> fac(f.rows(),f.items());

	std::cout << "Loading data ..." << std::endl;
	f.load(fac.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	fac.init2();fac.findTopK2(k);
	fac.benchmark();
}


#endif
