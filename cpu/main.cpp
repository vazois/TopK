#include "time/Time.h"
#include "tools/ArgParser.h"
#include "input/File.h"

#include "cpu/AA.h"
#include "cpu/NA.h"
#include "cpu/FA.h"
#include "cpu/TA.h"
#include "cpu/CBA.h"
#include "cpu/BPA.h"

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
	ta.init(); ta.findTopK(k);
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

	CBA<float,uint64_t> cba_seq(f.rows(),f.items());
	CBA<float,uint64_t> cba_par(f.rows(),f.items());
	cba_seq.set_topk_exec(false);
	cba_par.set_topk_exec(true);


	std::cout << "Loading data ..." << std::endl;
	f.load(cba_seq.get_cdata());
	cba_par.set_cdata(cba_seq.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	cba_seq.init(); cba_seq.findTopK(k);

	cba_seq.benchmark();
}

void debug_ta(std::string fname,uint64_t n, uint64_t d, uint64_t k){
	File<float> f(fname,false,n,d);
	TA<float,uint64_t> ta_seq(f.rows(),f.items());

	std::cout << "Loading data ..." << std::endl;
	f.load(ta_seq.get_cdata());

	std::cout << "Benchmark <<<" << f.rows() << "," << f.items() << "," << k << ">>> " << std::endl;
	ta_seq.init(); ta_seq.findTopK(k);

	//ta_par.compare(ta_seq);
	ta_seq.benchmark();
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

	if(!ap.exists("-md")){
		//std::cout << "Default mode: <debug>" <<std::endl;
		ap.addArg("-md","debug");
	}

	uint64_t n = ap.getInt("-n");
	uint64_t d = ap.getInt("-d");
	if(ap.getString("-md") == "debug"){
		debug(ap.getString("-f"),n,d,K);
		//debug_cba(ap.getString("-f"),n,d,K);
		//debug_ta(ap.getString("-f"),n,d,K);
	}

	return 0;
}
