#ifndef TA_H
#define TA_H

#include "FA.h"
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>


template<class T,class Z>
class PQComparison{
	public:
		PQComparison(){};

		bool operator() (const tuple<T,Z>& lhs, const tuple<T,Z>& rhs) const{
			return (lhs.score>rhs.score);
		}
};

template<class T,class Z>
class TA : public FA<T,Z>{
	public:
		TA(uint64_t n,uint64_t d) : FA<T,Z>(n,d){ this->algo = "TA"; };

		void findTopK(uint64_t k);
	private:
		std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
		void seq_topk(uint64_t k);
		void seq_topk2(uint64_t k);
		void par_topk(uint64_t k);
};

template<class T,class Z>
void TA<T,Z>::seq_topk(uint64_t k){
	std::unordered_set<Z> tids_set;
	std::unordered_map<Z,T> tmap;

	T threshold=0;
	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		threshold=0;
		for(uint8_t j = 0; j < this->d;j++){
			pred<T,Z> p = this->lists[j][i];
			threshold+=p.attr;

			T score = 0;
			if(tmap.find(p.tid) == tmap.end()){// Only if we do not want to re-evaluate the score for tuples re-appearing in the lists
				for(uint8_t k = 0; k < this->d; k++){
					score+=this->cdata[p.tid * this->d + k];
				}
				this->pred_count+=this->d;
				this->tuple_count+=1;
				tmap.insert(std::pair<T,Z>(p.tid,score));
			}else{
				score = tmap[p.tid];
			}

			if(tids_set.find(p.tid) == tids_set.end()){//if does not exist in set / if tuple has not been evaluated yet
				if(this->q.size() < k){//insert if space in queue
					this->q.push(tuple<T,Z>(p.tid,score));
					tids_set.insert(p.tid);
				}else if(this->q.top().score<score){//delete smallest element if current score is bigger
					tids_set.erase(tids_set.find(q.top().tid));
					this->q.pop();
					this->q.push(tuple<T,Z>(p.tid,score));
					tids_set.insert(p.tid);
				}
			}
		}
		if(this->q.top().score >= threshold){
			//std::cout << "stopped at: " << i << std::endl;
			break;
		}
	}
	this->tt_processing = this->t.lap();
}

template<class T,class Z>
void TA<T,Z>::seq_topk2(uint64_t k){

	T threshold=0;
	for(uint64_t i = 0; i < this->n;i++){
		for(uint8_t j = 0; j < this->d;j++){
			pred<T,Z> p = this->lists[j][i];
			threshold+=p.attr;
		}

	}

}

template<class T,class Z>
void TA<T,Z>::par_topk(uint64_t k){

}


template<class T,class Z>
void TA<T,Z>::findTopK(uint64_t k){
	//Note: keep truck of ids so you will not re-insert the same tupple as your process them in order
	std::cout << this->algo << " find topK ...";

	this->t.start();
	if(this->topkp){
		this->par_topk(k);
	}else{
		this->seq_topk(k);
	}
	this->tt_processing = this->t.lap("");

	//std::cout << "q_size: " << q.size() << std::endl;

	//std::cout << std::endl;
	while(!this->q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(this->q.top());
		this->q.pop();
	}
	std::cout << " (" << this->res.size() << ")" << std::endl;
}
#endif
