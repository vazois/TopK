#include "../tools/ArgParser.h"
#include "File.h"
#include <cstdint>
#include <string>
#include <cstdio>

#include <parallel/algorithm>
#include <unordered_set>

template<class T, class Z>
struct reorder_pair{
	Z id;
	T score;
};

template<class T,class Z>
static bool cmp_reorder_pair(const reorder_pair<T,Z> &a, const reorder_pair<T,Z> &b){ return a.score > b.score; };

template<class T,class Z>
void reorder_transpose(std::string fname,uint64_t n, uint64_t d){
	File<T> f(fname,false,n,d);
	f.set_transpose(true);
	T *cdata_in = (T*)malloc(sizeof(float)*n*d);

	std::cout << "Loading data from file for reordering !!!" <<std::endl;
	f.load(cdata_in);
	std::cout << "Reordering data !!!" <<std::endl;

	reorder_pair<T,Z> *lists = (reorder_pair<T,Z>*)malloc(sizeof(reorder_pair<T,Z>)*n*d);

	//Create lists
	for(uint8_t m = 0; m < d; m++){
		for(uint64_t i = 0; i < n; i++){
			lists[m*n + i].id = i;
			lists[m*n + i].score = cdata_in[m*n + i];
		}
	}
	//Sort lists
	for(uint8_t m = 0;m<d;m++){ __gnu_parallel::sort(&lists[m*n],(&lists[m*n]) + n,cmp_reorder_pair<T,Z>); }

	T *cdata_out = (T*)malloc(sizeof(T)*n*d);
	std::unordered_set<Z> eset;
	uint64_t ii = 0;
	for(uint64_t i = 0; i < n; i++){
		for(uint8_t m = 0; m < d; m++){
			reorder_pair<T,Z> p = lists[m*n + i];
			if(eset.find(p.id) == eset.end()){
				eset.insert(p.id);
				for(uint8_t j = 0; j < d; j++){ cdata_out[j * n + ii] = cdata_in[j * n + p.id]; }
				ii++;
			}
		}
	}

	std::string fname2 = fname + "_o";
	std::cout << "Storing to " << fname2 << std::endl;
	f.store(fname2,cdata_out);

	free(cdata_in);
	free(cdata_out);
	free(lists);
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

	uint64_t n = ap.getInt("-n");
	uint64_t d = ap.getInt("-d");

	//reorder(ap.getString("-f"),n,d);
	reorder_transpose<float,uint32_t>(ap.getString("-f"),n,d);

	return 0;
}
