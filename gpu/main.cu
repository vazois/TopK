#include "CudaHelper.h"
#include "time/Time.h"
#include "input/GInput.h"


int main(int argc, char **argv){
	GInput<float> input("data/d_1048576_4_i");

	input.init();
	input.sample();

	return 0;
}
