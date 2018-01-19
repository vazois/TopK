#ifndef FA_H
#define FA_H

#include "AA.h"
#include <unordered_map>
#include <unordered_set>
#include <list>

template<class Z>
struct rrpred{
	rrpred(){ tid = 0; offset = 0; }
	rrpred(Z t, Z o){ tid = t; offset = o; }
	Z tid;
	Z offset;
};

/*
 * Simple implementation of Fagin's algorithm
 */
template<class T,class Z>
class FA : public AA<T,Z>{
	public:
		FA(uint64_t n,uint64_t d) : AA<T,Z>(n,d){ this->algo = "FA"; };

		void init();
		void findTopK(uint64_t k);

	protected:
		std::vector<std::vector<pred<T,Z>>> lists;
		std::vector<rrpred<Z>> rrlist;
};

/*
 * Create m lists and sort them
 */
template<class T,class Z>
void FA<T,Z>::init(){
	this->lists.resize(this->d);
	for(int i =0;i<this->d;i++){ this->lists[i].resize(this->n); }

	this->t.start();
	for(uint64_t i=0;i<this->n;i++){
		for(int j =0;j<this->d;j++){
			this->lists[j].push_back(pred<T,Z>(i,this->cdata[i*this->d + j]));
		}
	}
	//for(int i =0;i<this->d;i++){ std::sort(this->lists[i].begin(),this->lists[i].end(),cmp_max_pred<T,Z>);}
	for(int i =0;i<this->d;i++){
		__gnu_parallel::sort(this->lists[i].begin(),this->lists[i].end(),cmp_max_pred<T,Z>);
	}
	this->tt_init = this->t.lap();
}

template<class T,class Z>
void FA<T,Z>::findTopK(uint64_t k){
	std::unordered_map<Z,uint8_t> tmap;
	uint64_t stop=0;

	//Iterate Lists
	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		for(uint64_t j = 0; j < this->d;j++){
			pred<T,Z> p = this->lists[j][i];
			if ( tmap.find(p.tid) == tmap.end() ){
				tmap.insert(std::pair<Z,uint8_t>(p.tid,1));
			}else{
				tmap[p.tid]++;
				if( tmap[p.tid] == this->d ) stop++;
			}
		}
		if(stop >= k){
			//std::cout << "Stopped at: " << i << std::endl;
			break;
		}
	}
	this->tt_processing = this->t.lap("");

	//Gather results and evaluate scores
	std::vector<tuple<T,Z>> res;
	for(typename std::unordered_map<Z,uint8_t>::iterator it = tmap.begin(); it!=tmap.end(); ++it){
		uint64_t tid = it->first;
		T score = 0;
		for(uint64_t j = 0; j < this->d; j++){ score+= this->cdata[tid * this->d + j]; }
		res.push_back(tuple<T,Z>(it->first,score));
		this->pred_count+=this->d;
		this->tuple_count+=1;
	}
	std::sort(res.begin(),res.end(),cmp_score<T,Z>);
	for(uint64_t i = 0;i < k ;i++){ this->res.push_back(res[i]); }
}

#endif

