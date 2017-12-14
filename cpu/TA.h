#ifndef TA_H
#define TA_H

#include "FA.h"
#include <queue>
#include <set>
#include <map>


template<class T>
class PQComparison{
	public:
		PQComparison(){};

		bool operator() (const tuple<T>& lhs, const tuple<T>& rhs) const{
			return (lhs.score>rhs.score);
		}
};

template<class T>
class TA : public FA<T>{
	public:
		TA(uint64_t n,uint64_t d) : FA<T>(n,d){ this->algo = "TA"; };

		void findTopK(uint64_t k);
};

template<class T>
void TA<T>::findTopK(uint64_t k){
	//Note: keep truck of ids so you will not re-insert the same tupple as your process them in order
	std::cout << this->algo << " find topK ...";

	std::set<uint64_t> tids_set;
	std::map<uint64_t,T> tmap;
	std::priority_queue<T, std::vector<tuple<T>>, PQComparison<T>> q;
	T threshold=0;
	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		threshold=0;
		for(uint8_t j = 0; j < this->d;j++){
			pred<T> p = this->lists[j][i];
			threshold+=p.attr;

			T score = 0;
			if(tmap.find(p.tid) == tmap.end()){// Only if we do not want to re-evaluate the score for tuples re-appering in the lists
				for(uint8_t k = 0; k < this->d; k++){
					score+=this->cdata[p.tid * this->d + k];
				}
				this->pred_count+=this->d;
				this->tuple_count+=1;
				tmap.insert(std::pair<uint64_t,T>(p.tid,score));
			}else{
				score = tmap[p.tid];
			}

			if(tids_set.find(p.tid) == tids_set.end()){//if does not exist in set / if tuple has not been evaluated yet
				if(q.size() < k){//insert if space in queue
					q.push(tuple<T>(p.tid,score));
					tids_set.insert(p.tid);
				}else if(q.top().score<score){//delete smallest element if current score is bigger
					tids_set.erase(tids_set.find(q.top().tid));
					q.pop();
					q.push(tuple<T>(p.tid,score));
					tids_set.insert(p.tid);
				}
			}
		}
		if(q.top().score >= threshold){
			//std::cout << "stopped at: " << i << std::endl;
			break;
		}
	}
	this->tt_processing = this->t.lap();
	//std::cout << "q_size: " << q.size() << std::endl;

	//std::cout << std::endl;
	while(!q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << " (" << this->res.size() << ")" << std::endl;
}

#endif
