#ifndef HJR_H
#define HJR_H

#include "ARJ.h"

template<class Z, class T>
class HJR : public AARankJoin<Z,T>{
	public:
		HJR(RankJoinInstance<Z,T> *rj_inst) : AARankJoin<Z,T>(rj_inst){ };
		~HJR(){};

		void can_hash_join();
		void nop_hash_join();
		void prt_hash_join();
	private:
};

template<class Z, class T>
void HJR<Z,T>::can_hash_join(){
	this->set_algo("canonical hash join");
	this->reset_metrics();
	this->reset_aux_struct();

	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();

	//Build phase
	this->t.start();
	for(uint64_t i = 0;i < R->n; i++){
		Z primary_key = R->ids[i];
		T score = 0;
		for(uint8_t j = 0; j < R->d; j++){
			score+= R->scores[j*R->n + i];
		}
		this->htR.emplace(primary_key,score);
	}

	//Probe phase
	for(uint64_t i =0; i< S->n; i++){
		Z foreign_key = S->ids[i];
		auto range = this->htR.equal_range(foreign_key);
		if( range.first != range.second ){ // If probe match
			T score = 0;
			for(uint8_t j = 0; j < S->d; j++){ // Calculate Score
				score+= S->scores[j*S->n + i];
			}
			//TODO: Check if can score higher than threshold, break otherwise ?
			for(auto it = range.first; it != range.second; ++it){
				T combined_score = score + it->second;
				//std::cout << key << " = combined: " << score << "," << it->second << "," << combined_score << std::endl;

				if(this->q.size() < k){
					this->q.push(_tuple<Z,T>(i,combined_score));
				}else if(this->q.top().score < combined_score){
					this->q.pop();
					this->q.push(_tuple<Z,T>(i,combined_score));
				}
			}
		}
	}
	this->tt_join = this->t.lap();
}

template<class Z, class T>
void HJR<Z,T>::nop_hash_join(){
	this->set_algo("no partitition hash join (" + std::to_string(THREADS) + ")");
	this->reset_metrics();
	this->reset_aux_struct();

	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();

	omp_set_num_threads(THREADS);
	uint32_t stride = 1024;
	this->t.start();
	#pragma omp parallel
	{
		uint32_t tid = omp_get_thread_num();
		uint32_t offset = tid * stride;
		uint32_t step = THREADS * stride;

		//Build-Phase
		for(uint64_t i = offset; i < R->n; i+=step){
			for(uint64_t j = i; j < i+stride; j++){
				Z primary_key = R->ids[j];
				T score = 0;
				for(uint8_t m = 0; m < R->d; m++){
					score+= R->scores[m*R->n + j];
				}
				#pragma omp critical
				{
					this->htR.emplace(primary_key,score);
				}
			}
		}
		#pragma omp barrier
		//Probe-Phase
		for(uint64_t i = offset; i <S->n; i+=step){
			for(uint64_t j = i; j < i+stride; j++){
				Z foreign_key = S->ids[j];
				auto range = this->htR.equal_range(foreign_key);
				if( range.first != range.second ){ // If probe match
					T score = 0;
					for(uint8_t m = 0; m < S->d; m++){ // Calculate score
						score+= S->scores[m*S->n + j];
					}
					//TODO: Check if can score higher than threshold, break otherwise ?
					for(auto it = range.first; it != range.second; ++it){
						T combined_score = score + it->second;
						//std::cout << key << " = combined: " << score << "," << it->second << "," << combined_score << std::endl;

						if(this->tq[tid].size() < k){
							this->tq[tid].push(_tuple<Z,T>(j,combined_score));
						}else if(this->tq[tid].top().score < combined_score){
							this->tq[tid].pop();
							this->tq[tid].push(_tuple<Z,T>(j,combined_score));
						}
					}
				}
			}
		}
	}
	this->merge_qs();
	this->tt_join = this->t.lap();
}

template<class Z, class T>
void HJR<Z,T>::prt_hash_join(){

}

#endif
