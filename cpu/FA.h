#ifndef FA_H
#define FA_H

#include "AA.h"
#include<set>

/*
 * Simple implementation of Fagin's algorithm
 */
template<class T>
class FA : public AA<T>{
	public:
		FA(Input<T>* input) : AA<T>(input){ this->algo = "FA"; };

		void init();
		void findTopK(uint64_t k);

	protected:
		std::vector<std::vector<pred<T>>> lists;
};

/*
 * Create m lists and sort them
 */
template<class T>
void FA<T>::init(){
	this->lists.resize(this->input->get_d());
	for(int i =0;i<this->input->get_d();i++){ this->lists[i].resize(this->input->get_n()); }

	this->t.start();
	for(uint64_t j=0;j<this->input->get_n();j++){
		for(int i =0;i<this->input->get_d();i++){
			uint64_t tid = j;
			T attr = (this->input->get_dt())[j*this->input->get_d() + i];
			this->lists[i].push_back(pred<T>(tid,attr));
		}
	}
	this->tt_init = this->t.lap();

	for(int i =0;i<this->input->get_d();i++){ std::sort(this->lists[i].begin(),this->lists[i].end(),cmp_max_pred<T>);}
}

/*
 * Iterate through lists and then evaluate tuples
 */
template<class T>
void FA<T>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ..." << std::endl;

	std::set<uint64_t> tset;
	uint64_t n = this->input->get_n();
	uint64_t d = this->input->get_d();
	T *data = this->input->get_dt();

	this->t.start();
	for(uint64_t i = 0; i < n;i++){
		for(uint64_t j = 0; j < d;j++){
			pred<T> p = this->lists[j][i];
			tset.insert(p.tid);
		}
		if(tset.size() >= k) break;
	}

	for(std::set<uint64_t>::iterator it = tset.begin();it!=tset.end();++it){
		uint64_t tid = *it;
		T score = 0;
		for(uint64_t j = 0; j < d; j++){ score+= data[tid * d + j]; }
		this->res.push_back(tuple<T>(tid,score));
		this->eval_count++;
	}
	this->tt_processing = this->t.lap();

	std::sort(this->res.begin(),this->res.end(),cmp_score<T>);
}

#endif

