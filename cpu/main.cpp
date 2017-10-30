#include "time/Time.h"
#include "input/Input.h"

#include "cpu/AA.h"
#include "cpu/FA.h"

int main(int argc, char **argv){
	char fname[] = "data/d_1048576_4_i";
	Time<msecs> t;
	Input<float> input("data/d_16777216_4_i");

	input.init();

	FA<float> fa(&input);

	fa.init();
	fa.findTopK();

	return 0;
}
