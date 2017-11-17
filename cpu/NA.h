#ifndef NA_H
#define NA_H

#include "AA.h"

/*
 * Naive algorithm for aggregation
 */
template<class T>
class NA : public AA<T>{
	public:
		NA(Input<T>* input) : AA<T>(input){ this->algo = "NA"; };

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
	this->tuples.resize(this->input->get_n());
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
	std::cout << this->algo << " find topK ..." << std::endl;

	this->t.start();
	std::sort(this->tuples.begin(),this->tuples.end(),cmp_score<T>);
	for(uint64_t i = 0;i <(k < this->tuples.size() ? k : this->tuples.size() ) ;i++){
		this->res.push_back(tuple<T>(this->tuples[i].tid,this->tuples[i].score));
	}
	this->tt_processing = this->t.lap();
}

#endif
