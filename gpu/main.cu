#include "CudaHelper.h"
#include "time/Time.h"
#include "input/GInput.h"


int main(int argc, char **argv){
	Time<msecs> t;
	GInput<float> input("data/d_16777216_4_i");

	input.init();

	return 0;
}
