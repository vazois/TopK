#include "time/Time.h"
#include "tools/ArgParser.h"
#include "input/File.h"

#include "cpu/AA.h"
#include "cpu/NA.h"
#include "cpu/FA.h"
#include "cpu/TA.h"
#include "cpu/CBA.h"
#include "cpu/BPA.h"

#define RUN_PAR true

void debug(std::string fname, uint64_t k){
	File<float> f(fname,false);
	File<float> f2(fname,false,f.rows(),f.items());
	f2.set_transpose(true);

	NA<float> na(f.rows(),f.items());
	FA<float> fa(f.rows(),f.items());
	TA<float> ta(f.rows(),f.items());
	BPA<float> bpa(f.rows(),f.items());
	CBA<float> cba(f.rows(),f.items());

	f.load(na.get_cdata());
	f2.load(cba.get_cdata());
	fa.set_cdata(na.get_cdata());
	bpa.set_cdata(na.get_cdata());
	ta.set_cdata(na.get_cdata());

//	f.sample();
//	cba.set_cdata(na.get_cdata());
//	cba.transpose();

	fa.set_init_exec(RUN_PAR); //fa.set_topk_exec(RUN_PAR);
	ta.set_init_exec(RUN_PAR);
	bpa.set_init_exec(RUN_PAR);

	na.init(); na.findTopK(k);
	fa.init(); fa.findTopK(k);
	ta.init(); ta.findTopK(k);
	bpa.init(); bpa.findTopK(k);
	cba.init(); cba.findTopK(k);

	fa.compare(na);
	ta.compare(na);
	bpa.compare(na);
	cba.compare(na);

	na.benchmark();
	fa.benchmark();
	ta.benchmark();
	bpa.benchmark();
	cba.benchmark();
}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	if(!ap.exists("-f")){
		std::cout << "Missing file input!!!" << std::endl;
		exit(1);
	}

	if(!ap.exists("-md")){
		//std::cout << "Default mode: <debug>" <<std::endl;
		ap.addArg("-md","debug");
	}

	if(ap.getString("-md") == "debug"){
		debug(ap.getString("-f"),100);
	}

	return 0;
}
