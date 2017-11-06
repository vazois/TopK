#ifndef AA_H
#define AA_H

#include<vector>
#include<algorithm>

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

		void cmp_res(AA<T> b);
		void get_res(){ return this->res; };

		void benchmark();

	protected:
		Input<T>* input;

		std::string algo;
		std::vector<tuple<T>> res;

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
}

template<class T>
AA<T>::~AA(){ }

/*
 * cmp results to check correctness
 */
template<class T>
void AA<T>::cmp_res(AA<T> b){

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
