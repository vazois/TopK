#ifndef HJR_H
#define HJR_H

#include "ARJ.h"

template<class Z, class T>
class HJR : public AARankJoin<Z,T>{
	public:
		HJR(RankJoinInstance<Z,T> *rj_inst) : AARankJoin<Z,T>(rj_inst){ };
		~HJR(){};

		void can_hash_join();
	private:
		std::unordered_multimap<Z,T> htR;
		std::unordered_multimap<Z,T> htS;
		std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>> q;
};

template<class Z, class T>
void HJR<Z,T>::can_hash_join(){
	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();

	//Build phase
	for(uint64_t i = 0;i < R->n; i++){
		Z key = R->ids[i];
		T score = 0;
		for(uint8_t j = 0; j < R->d; j++){
			score+= R->scores[j*R->n + i];
		}
		this->htR.emplace(key,score);
	}

	//Probe phase
	for(uint64_t i =0; i< S->n; i++){
		Z key = S->ids[i];
		auto range = this->htR.equal_range(key);
		if( range.first != range.second ){ // If probe match
			T score = 0;
			for(uint8_t j = 0; j < S->d; j++){ // Calculate Score
				score+= S->scores[j*S->n + i];
			}
			//TODO: Check if can score higher than threshold, break otherwise ?
			for(auto it = range.first; it != range.second; ++it){
				T combined_score = score + it->second;
				//std::cout << key << " = combined: " << score << "," << it->second << "," << combined_score << std::endl;

				if(q.size() < k){
					q.push(_tuple<Z,T>(key,combined_score));
				}else if(q.top().score < combined_score){
					q.pop();
					q.push(_tuple<Z,T>(key,combined_score));
				}
			}
		}
	}
}

#endif
