#include "time/Time.h"
#include "input/Input.h"
#include "tools/ArgParser.h"

#include "cpu/AA.h"
#include "cpu/NA.h"
#include "cpu/FA.h"
#include "cpu/CBA.h"

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	if(!ap.exists("-f")){
		std::cout << "Missing file input!!!" << std::endl;
		return -1;
	}

	//Input<float> input("data/d_16777216_4_i");
	Input<float> input(ap.getString("-f"));
	input.init();
	input.sample();

	//Input<float> ginput("data/d_16777216_4_i");
	Input<float> ginput(ap.getString("-f"));
	ginput.transpose(true);
	ginput.init();
	ginput.sample();


	//Algorithms
	NA<float> na(&input);
	na.init();
	na.findTopK(100);
	na.benchmark();

	FA<float> fa(&input);
	fa.init();
	fa.findTopK(100);
	fa.benchmark();
	fa.compare(na);

	CBA<float> cba(&ginput);
	cba.init();
	cba.findTopK(100);
	cba.benchmark();
	cba.compare(fa);


	return 0;
}
