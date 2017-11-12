#include "time/Time.h"
#include "input/Input.h"

#include "cpu/AA.h"
#include "cpu/NA.h"
#include "cpu/FA.h"



int main(int argc, char **argv){
	Time<msecs> t;
	//Input<float> input("data/d_16777216_4_i");
	Input<float> input("data/d_1048576_4_i");

	input.init();
//	input.sample();

	NA<float> na(&input);
	na.init();
	na.findTopK(1000);
	na.benchmark();

	FA<float> fa(&input);
	fa.init();
	fa.findTopK(1000);
	fa.benchmark();

	fa.compare(na);

	return 0;
}
