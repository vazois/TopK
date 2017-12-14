#ifndef FA_H
#define FA_H

#include "AA.h"

/*
 * Simple implementation of Fagin's algorithm
 */
template<class T>
class FA : public AA<T>{
	public:
		FA(uint64_t n,uint64_t d) : AA<T>(n,d){ this->algo = "FA"; };

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
	this->lists.resize(this->d);
	for(int i =0;i<this->d;i++){ this->lists[i].resize(this->n); }

	this->t.start();
	for(uint64_t i=0;i<this->n;i++){
		for(int j =0;j<this->d;j++){
			this->lists[j].push_back(pred<T>(i,this->cdata[i*this->d + j]));
		}
	}

	for(int i =0;i<this->d;i++){ std::sort(this->lists[i].begin(),this->lists[i].end(),cmp_max_pred<T>);}
	this->tt_init = this->t.lap();
}

/*
 * Iterate through lists and then evaluate tuples
 */
template<class T>
void FA<T>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	std::map<uint64_t,uint8_t> tmap;
	uint64_t stop=0;

	this->t.start();
	//Iterate Lists
	for(uint64_t i = 0; i < this->n;i++){
		for(uint64_t j = 0; j < this->d;j++){
			pred<T> p = this->lists[j][i];
			if ( tmap.find(p.tid) == tmap.end() ){
				tmap.insert(std::pair<uint64_t,uint8_t>(p.tid,1));
			}else{
				tmap[p.tid]++;
				if( tmap[p.tid] == this->d ) stop++;
			}
		}
		if(stop >= k) break;
	}

	//Gather results and evaluate scores
	std::vector<tuple<T>> res;
	for(std::map<uint64_t,uint8_t>::iterator it = tmap.begin(); it!=tmap.end(); ++it){
		uint64_t tid = it->first;
		T score = 0;
		for(uint64_t j = 0; j < this->d; j++){ score+= this->cdata[tid * this->d + j]; }
		res.push_back(tuple<T>(it->first,score));
		this->eval_count+=this->d;
	}
	std::sort(res.begin(),res.end(),cmp_score<T>);
	this->tt_processing = this->t.lap();
	for(uint64_t i = 0;i < k ;i++){ this->res.push_back(res[i]); }

	std::cout << " (" << this->res.size() << ")" << std::endl;
}

#endif

