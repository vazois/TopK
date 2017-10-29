#include "time/Time.h"
#include "input/Input.h"


int main(int argc, char **argv){
	char fname[] = "data/d_1048576_4_i";
	Time<msecs> t;
	Input<float> input("data/d_1048576_4_i");

	input.init();


	return 0;
}
