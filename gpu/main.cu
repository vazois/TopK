#include "CudaHelper.h"
#include "time/Time.h"
#include "input/GInput.h"
#include "GPA.h"

int main(int argc, char **argv){
	GInput<float> ginput("data/d_1048576_4_i");

	ginput.init();
	ginput.sample();

	GPA<float> gpa(&ginput);
	gpa.init();

	ginput.sample();

	gpa.findTopK(1000);

	return 0;
}
