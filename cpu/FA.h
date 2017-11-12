#ifndef FA_H
#define FA_H

#include "AA.h"
#include<map>

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
	uint64_t n = this->input->get_n();
	uint64_t d = this->input->get_d();
	T *data = this->input->get_dt();

	this->lists.resize(d);
	for(int i =0;i<d;i++){ this->lists[i].resize(n); }

	this->t.start();
	for(uint64_t i=0;i<n;i++){
		for(int j =0;j<d;j++){
			this->lists[j].push_back(pred<T>(i,data[i*d + j]));
		}
	}

	for(int i =0;i<this->input->get_d();i++){ std::sort(this->lists[i].begin(),this->lists[i].end(),cmp_max_pred<T>);}
	this->tt_init = this->t.lap();
}

/*
 * Iterate through lists and then evaluate tuples
 */
template<class T>
void FA<T>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ..." << std::endl;

	std::map<uint64_t,uint8_t> tmap;
	uint64_t n = this->input->get_n();
	uint64_t d = this->input->get_d();
	T *data = this->input->get_dt();
	//this->res.resize(k);

	uint64_t stop=0;
	this->t.start();

	for(uint64_t i = 0; i < n;i++){
		for(uint64_t j = 0; j < d;j++){
			pred<T> p = this->lists[j][i];
			if ( tmap.find(p.tid) == tmap.end() ){
				tmap.insert(std::pair<uint64_t,uint8_t>(p.tid,1));
			}else{
				tmap[p.tid]++;
				if( tmap[p.tid] == d ) stop++;
			}
		}
		if(stop >= k) break;
	}

	std::cout << "complete tuple: " << stop << std::endl;

	std::vector<tuple<T>> res;
	for(std::map<uint64_t,uint8_t>::iterator it = tmap.begin(); it!=tmap.end(); ++it){
		uint64_t tid = it->first;
		T score = 0;
		for(uint64_t j = 0; j < d; j++){ score+= data[tid * d + j]; }
		res.push_back(tuple<T>(it->first,score));
		this->eval_count++;
	}
	std::sort(res.begin(),res.end(),cmp_score<T>);
	this->tt_processing = this->t.lap();
	for(uint64_t i = 0;i < k ;i++){ this->res.push_back(res[i]); }

//	for(uint64_t i = 0;i <(k < this->res.size() ? k : this->res.size() ) ;i++){
//		if(i < 10) std::cout << "t: " << this->res[i].tid << ", (" << this->res[i].score << ")" <<std::endl;
//	}
}

#endif

