#ifndef BPA_H
#define BPA_H

#include "FA.h"
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>

template<class T>
class BPA : public FA<T>{
	public:
		BPA(uint64_t n,uint64_t d) : FA<T>(n,d){ this->algo = "BPA"; };
		void findTopK(uint64_t k);
};


template<class T>
void BPA<T>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	std::vector<std::unordered_map<uint64_t,uint64_t>*> pmap;//Quickly find position of tuple in list
	std::vector<std::set<uint64_t>*> pset;//Keep track of positions i have seen
	this->t.start();
	for(uint8_t j = 0; j < this->d;j++){
		pmap.push_back(new std::unordered_map<uint64_t,uint64_t>);
		pset.push_back(new std::set<uint64_t>);
	}

	for(uint64_t i = 0; i < this->n;i++){
		//pmap.push_back(new std::map<uint64_t,uint64_t>);
		for(uint8_t j = 0; j < this->d;j++){
			pred<T> p = this->lists[j][i];
			pmap[j]->insert(std::pair<uint64_t,uint64_t>(p.tid,i));//tupple-id position in j-th list
		}
	}

	std::unordered_map<uint64_t,T> tmap;//keep track of evaluated tupples
	std::unordered_set<uint64_t> tids_set;//make sure not to insert twice tupple into priority queue
	std::priority_queue<T, std::vector<tuple<T>>, PQComparison<T>> q;
	for(uint64_t i = 0; i < this->n;i++){
		for(uint8_t j = 0; j < this->d;j++){
			pred<T> p = this->lists[j][i];

			T score = 0;
			if(tmap.find(p.tid) == tmap.end()){// Only if we do not want to re-evaluate the score for tuples re-appearing in the lists
				for(uint8_t k = 0; k < this->d; k++){
					score+=this->cdata[p.tid * this->d + k];
					uint64_t pos = pmap[k]->at(p.tid);//find attribute position
					pset[k]->insert(pos);//keep track positions that i have seen
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

		//Find threshold
		T threshold = 0;
		for(uint64_t j = 0; j < this->d;j++){
			std::set<uint64_t>::iterator it = pset[j]->find(i);
			uint64_t pos=*it;
			++it;
			while(it != pset[j]->end()){
				if(pos - *it > 1){
					break;
				}
				pos = *it;
			}
			threshold+=this->lists[j][pos].attr;
		}

		if(q.top().score >= threshold){
			//std::cout << "stopped at: " << i << "," << threshold << std::endl;
			break;
		}
	}
	this->tt_processing = this->t.lap();

	//Gather results//
	while(!q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << " (" << this->res.size() << ")" << std::endl;


}

#endif
