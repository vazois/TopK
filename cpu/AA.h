#ifndef AA_H
#define AA_H

#include<vector>
#include<algorithm>
#include<map>

#include "../input/Input.h"


/*
 * Predicate structure
 */
template<class T>
struct pred{
	pred(){ tid = 0; attr = 0; }
	pred(uint64_t t, T a){ tid = t; attr = a; }
	uint64_t tid;
	T attr;
};

/*
 * Tuple structure
 */
template<class T>
struct tuple{
	tuple(){ tid = 0; score = 0; }
	tuple(uint64_t t, T s){ tid = t; score = s; }
	uint64_t tid;
	T score;
};

template<class T>
static bool cmp_score(const tuple<T> &a, const tuple<T> &b){ return a.score > b.score; };

template<class T>
static bool cmp_max_pred(const pred<T> &a, const pred<T> &b){ return a.attr > b.attr; };

/*
 * Base class for aggregation algorithm
 */
template<class T>
class AA{
	public:
		AA(Input<T>* input);
		~AA();

		void compare(AA<T> b);
		std::vector<tuple<T>> get_res(){ return this->res; };
		std::string get_algo(){ return this->algo; };

		void benchmark();

	protected:
		Input<T>* input;

		std::string algo;
		std::vector<tuple<T>> res;
		T *cdata;
		uint64_t n;
		uint64_t d;

		Time<msecs> t;
		double tt_init;//initialization time
		double tt_processing;//processing time
		uint64_t eval_count;//count tuples evaluated
};

template<class T>
AA<T>::AA(Input<T>* input){
	this->input = input;
	this->tt_init = 0;
	this->tt_processing = 0;
	this->eval_count = 0;
	this->n = this->input->get_n();
	this->d = this->input->get_d();
	this->cdata = this->input->get_dt();
}

template<class T>
AA<T>::~AA(){ }

/*
 * cmp results to check correctness
 */
template<class T>
void AA<T>::compare(AA<T> b){
	std::string cmp = "PASSED";
	std::map<uint64_t,T> tmap;

	/*create map with tuple ids*/
	for(uint64_t i = 0;i < this->res.size();i++){
		tmap[this->res[i].tid] = this->res[i].score;
	}

	/*check if ids in b exist in my result*/
	for(uint64_t i = 0;i < b.res.size();i++){
		if (tmap.find(b.get_res()[i].tid) == tmap.end()){//find if id of b does not exist in map
			std::cout <<"i:(" << i << ") "<<this->res[i].tid << " = " << this->res[i].score << "," << b.get_res()[i].tid << " = " << b.get_res()[i].score << std::endl;
			cmp  = "FAILED";
			std::cout << "(" <<this->algo <<") != (" << b.get_algo() << ") ( "<< cmp <<" )" << std::endl;
			exit(1);
		}
	}
	std::cout << "(" <<this->algo <<") compare result to (" << b.get_algo() << ") ( "<< cmp <<" )" << std::endl;
}

/*
 * List benchmarking information
 */
template<class T>
void AA<T>::benchmark(){
	std::cout << "< Benchmark for " << this->algo << " algorithm >" << std::endl;
	std::cout << "tt_init: " << this->tt_init << std::endl;
	std::cout << "tt_procesing: " << this->tt_processing << std::endl;

	std::cout << "eval_count: " << this->eval_count << std::endl;
}

#endif
