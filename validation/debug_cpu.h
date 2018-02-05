#ifndef DEBUG_CPU_H
#define DEBUG_CPU_H

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

void debug(std::string fname, uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	File<float> f2(fname,false,n,d);
	f2.set_transpose(true);

	NA<float,uint32_t> na(f.rows(),f.items());
	TA<float,uint32_t> ta(f.rows(),f.items());
	CBA<float,uint32_t> cba(f.rows(),f.items());
	FA<float,uint32_t> fa(f.rows(),f.items());
	FAc<float,uint32_t> fac(f.rows(),f.items());


	std::cout << "Loading data ..." << std::endl;
	f.load(na.get_cdata());
	//f.sample(25);
	f2.load(cba.get_cdata());
	ta.set_cdata(na.get_cdata());
	fa.set_cdata(na.get_cdata());
	fac.set_cdata(na.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;

	na.init(); na.findTopK(k);
	ta.init(); ta.findTopK(k);
	fa.init(); fa.findTopK(k);
	cba.init(); cba.findTopK(k);
	fac.init2(); fac.findTopK2(k);

	ta.compare(na);
	cba.compare(na);
	fa.compare(na);
	fac.compare(na);

	na.benchmark();
	ta.benchmark();
	fa.benchmark();
	cba.benchmark();
	fac.benchmark();
}



#endif
