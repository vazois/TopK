#ifndef TA_H
#define TA_H

#include "FA.h"
#include <queue>


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
	protected:
		std::vector<std::vector<pred<T>>> lists;

};

template<class T>
void TA<T>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ..." << std::endl;

	std::priority_queue<T, std::vector<tuple<T>>, PQComparison<T>> q;
	T threshold=0;
	for(uint64_t i = 0; i < this->n;i++){
		threshold=0;
		for(uint8_t j = 0; j < this->d;j++){
			pred<T> p = this->lists[j][i];
			threshold+=p.attr;

			T score = 0;
			for(uint8_t k = 0; k < this->d; k++){
				score+=this->cdata[p.tid * this->d + k];
			}

			if(q.size() < k){
				q.push(tuple<T>(p.tid,score));
			}else if(q.top().score<score){
				q.pop();
				q.push(tuple<T>(p.tid,score));
			}
		}
		if(q.top().score >= threshold){
			break;
		}
	}
}

#endif
