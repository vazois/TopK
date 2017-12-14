#ifndef NA_H
#define NA_H

#include "AA.h"

/*
 * Naive algorithm for aggregation
 */
template<class T>
class NA : public AA<T>{
	public:
		NA(uint64_t n,uint64_t d) : AA<T>(n,d){ this->algo = "NA"; };

		void init();
		void findTopK(uint64_t k);

	protected:
		std::vector<tuple<T>> tuples;
};

/*
 * Calculate scores
 */
template<class T>
void NA<T>::init(){
	this->tuples.resize(this->n);
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++ ){
		T score = 0;
		for(uint64_t j = 0; j < this->d; j++ ){ score+= this->cdata[i * this->d + j]; }
		this->tuples.push_back(tuple<T>(i,score));
		this->eval_count+=this->d;
	}
	this->tt_init = this->t.lap();
}


/*
 * Sort and find top-K
 */
template<class T>
void NA<T>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";

	this->t.start();
	std::sort(this->tuples.begin(),this->tuples.end(),cmp_score<T>);
	//std::cout << std::endl;
	for(uint64_t i = 0;i <(k < this->tuples.size() ? k : this->tuples.size() ) ;i++){
		//std::cout << this->algo <<" : " << this->tuples[i].tid << "," << this->tuples[i].score <<std::endl;
		this->res.push_back(tuple<T>(this->tuples[i].tid,this->tuples[i].score));
	}
	this->tt_processing = this->t.lap();

	std::cout << " (" << this->res.size() << ")" << std::endl;
}

#endif
