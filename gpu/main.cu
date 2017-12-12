#include "GPA.h"
#include "input/File.h"

int main(int argc, char **argv){
	File<float> f("data/d_1048576_12_i",true);
	GPA<float> gpa;


	gpa.alloc(f.items(),f.rows());
	f.set_transpose(true);
	f.load(gpa.get_cdata());
	f.sample();
	gpa.init();
	gpa.findTopK(100);

	return 0;
}
