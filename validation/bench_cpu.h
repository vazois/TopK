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
#include "cpu/cFA.h"

#include "cpu/MaxScore.h"
#include "cpu_opt/CBAopt.h"
#include "cpu_opt/cFAopt.h"
#include "cpu_opt/MSA.h"
#include "cpu_opt/GSA.h"

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

void bench_cfa(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	cFA<float,uint32_t> cfa(f.rows(),f.items());

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(cfa.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	cfa.init(); cfa.findTopK(k);
	cfa.benchmark();
}

void bench_maxscore(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	MaxScore<float,uint32_t> ms(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(ms.get_cdata());

	ms.init();
	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	ms.findTopK2(k);
	ms.benchmark();
}

void bench_cba_opt(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	CBAopt<float,uint32_t> cba(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(cba.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	cba.init(); cba.findTopK2(k);
	cba.benchmark();
}

void bench_cfa_opt(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	cFAopt<float,uint32_t> cfa_opt(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(cfa_opt.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	cfa_opt.init(); cfa_opt.findTopK(k);
	cfa_opt.benchmark();
}

void bench_msa(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	MSA<float,uint32_t> msa(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" << std::endl;
	f.load(msa.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	msa.init();
	msa.findTopK(k);
	msa.benchmark();
}

void bench_gsa(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	GSA<float,uint32_t> gsa(f.rows(),f.items());
	f.set_transpose(true);

	std::cout << "Loading data from file !!!" <<std::endl;
	f.load(gsa.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	gsa.init();

	gsa.benchmark();
}

#endif
