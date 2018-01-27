#include<limits>

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

#define RUN_PAR false
#define K 100

void debug(std::string fname, uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	File<float> f2(fname,false,n,d);
	f2.set_transpose(true);

	NA<float,uint64_t> na(f.rows(),f.items());
	TA<float,uint64_t> ta(f.rows(),f.items());
	CBA<float,uint64_t> cba(f.rows(),f.items());

	std::cout << "Loading data ..." << std::endl;
	f.load(na.get_cdata());
	f2.load(cba.get_cdata());
	ta.set_cdata(na.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;

	na.init(); na.findTopK(k);
	ta.init2(); ta.findTopK2(k);
	cba.init(); cba.findTopK(k);

	ta.compare(na);
	cba.compare(na);

	na.benchmark();
	ta.benchmark();
	cba.benchmark();
}

void debug_cba(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	f.set_transpose(true);

	CBA<float,uint32_t> cba_seq(f.rows(),f.items());
	cba_seq.set_topk_exec(false);

	std::cout << "Loading data ..." << std::endl;
	f.load(cba_seq.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	cba_seq.init(); cba_seq.findTopK(k);

	cba_seq.benchmark();
}

void debug_ta(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	TA<float,uint32_t> ta_seq(f.rows(),f.items());

	std::cout << "Loading data ..." << std::endl;
	f.load(ta_seq.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	ta_seq.init(); ta_seq.findTopK(k);

	//ta_par.compare(ta_seq);
	ta_seq.benchmark();
}

void debug_t2s(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	T2S<float,uint32_t> t2s(f.rows(),f.items());

	std::cout << "Loading data ..." << std::endl;
	f.load(t2s.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	t2s.init();
	t2s.benchmark();
}

void bench_ta(std::string fname,uint64_t n, uint64_t d, uint64_t k, uint64_t nl, uint64_t nu){
	File<float> f(fname,false,n,d);
	f.set_transpose(true);
	float *data=NULL;
	std::cout << "Loading data ..." << std::endl;
	f.load(data);

	for(uint64_t i = nl; i <= nu; i*=2){
		TA<float,uint32_t> ta(i,f.items());

		ta.set_cdata(data);
		std::cout << "Benchmark <<<" << i << "," << d << "," << k << ">>> " << std::endl;
		ta.init2();
		ta.findTopK2(k);
		ta.benchmark();
	}
//	if(data != NULL ) free(data);
}

void bench_cba(std::string fname,uint64_t n, uint64_t d, uint64_t k, uint64_t nl, uint64_t nu){
	File<float> f(fname,false,n,d);
	float *data=NULL;
	std::cout << "Loading data ..." << std::endl;
	f.load(data);

	for(uint64_t i = nl; i <= nu; i*=2){
		CBA<float,uint32_t> cba(i,f.items());

		cba.set_cdata(data);
		std::cout << "Benchmark <<<" << i << "," << d << "," << k << ">>> " << std::endl;
		cba.init();
		cba.findTopK(k);
		cba.benchmark();
	}
	//if(data != NULL ) free(data);
}

void mem_bench(uint32_t bsize, uint32_t dim, uint32_t n, uint32_t p){
	float *base = (float*)malloc(sizeof(float) * dim * n);
	float *buffer = (float*)malloc(sizeof(float) * dim * bsize);
	float **bucket = (float**) malloc (sizeof(float**) * p);

	for(uint64_t i = 0 ; i < p; i++) bucket[i] = (float*)malloc(sizeof(float)*dim*bsize);
	uint64_t count = 0;
	Time<msecs> t;
	double tt = 0;
	std::cout << "Running..." << std::endl;
	for(uint64_t i = 0; i < n ; i+=bsize)
	{
		t.start();
		memcpy(buffer,&base[i * dim],sizeof(float) * dim * bsize);
		for(uint64_t j = 0 ; j < p; j++) memcpy(&bucket[j][0],buffer, sizeof(float) * dim * bsize);
		tt+=t.lap();
		count++;
	}

	std::cout << "cc: " << count << std::endl;
	std::cout << "total: " << tt << std::endl;
	std::cout << "per iteration: " << tt/count << std::endl;

	for(uint64_t i = 0 ; i < p; i++) free(bucket[i]);
	free(bucket);
	free(base);
	free(buffer);
}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	if(!ap.exists("-f")){
		std::cout << "Missing file input!!!" << std::endl;
		exit(1);
	}

	if(!ap.exists("-d")){
		std::cout << "Missing d!!!" << std::endl;
		exit(1);
	}

	if(!ap.exists("-n")){
		std::cout << "Missing n!!!" << std::endl;
		exit(1);
	}

	uint64_t n = ap.getInt("-n");
	uint64_t d = ap.getInt("-d");
	uint64_t nu;
	if(!ap.exists("-nu")){
		nu = n;
	}else{
		nu = ap.getInt("-nu");
	}

	uint64_t nl;
	if(!ap.exists("-nl")){
		nl = n;
	}else{
		nl = ap.getInt("-nl");
	}


	//debug(ap.getString("-f"),n,d,K);
	//debug_ta(ap.getString("-f"),n,d,K);
	debug_cba(ap.getString("-f"),n,d,K);
	//debug_t2s(ap.getString("-f"),n,d,K);
	//bench_ta(ap.getString("-f"),n,d,K,nl,nu);
	//bench_cba(ap.getString("-f"),n,d,K,nl,nu);

	//mem_bench(256*1024,4, 1*1024*1024, 1);

	return 0;
}
