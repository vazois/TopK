#ifndef NA_H
#define NA_H

#include "NA.h"

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

	uint64_t n = this->input->get_n();
	uint64_t d = this->input->get_d();
	T *data = this->input->get_dt();

	this->t.start();
	for(uint64_t i = 0; i < n; i++ ){
		T score = 0;
		for(uint64_t j = 0; j < d; j++ ){ score+= data[i * d + j]; }
		this->tuples.push_back(tuple<T>(i,score));
		this->eval_count++;
	}
	this->tt_init = this->t.lap();
	//std::cout << "tt_init: " << this->tt_init << std::endl;
}


/*
 * Sort and find top-K
 */
template<class T>
void NA<T>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ..." << std::endl;
	this->res.resize(k);

	this->t.start();
	std::sort(this->tuples.begin(),this->tuples.end(),cmp_score<T>);
	for(uint64_t i = 0;i <(k < this->tuples.size() ? k : this->tuples.size() ) ;i++){
		//std::cout << "t: " << this->tuples[i].tid << ", (" << this->tuples[i].score << ")" <<std::endl;
		this->res.push_back(this->tuples[i]);
	}
	this->tt_processing = this->t.lap();
	//std::cout << "tt_processing: " << this->tt_processing << std::endl;
}

#endif
