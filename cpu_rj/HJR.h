#ifndef HJR_H
#define HJR_H

#include "ARJ.h"

template<class Z, class T>
struct args_t{
	TABLE<Z,T> *R;
	TABLE<Z,T> *S;
};

template<class Z, class T>
class HJR : public AARankJoin<Z,T>{
	public:
		HJR(RankJoinInstance<Z,T> *rj_inst) : AARankJoin<Z,T>(rj_inst){ };
		~HJR(){};

		void snop_hash_join();
		void st_nop_hash_rank_join();
		void mt_nop_hash_rank_join();

	private:
		void pshift(Z *arr, Z arr_n){
			for(uint32_t i = arr_n-1; i > 0; i--){ arr[i] = arr[i-1]; }
			arr[0] = 0;
		}

		void psum(Z *arr, Z arr_n){
			for(uint32_t i = 1; i < arr_n+1; i++){
				Z tmp = arr[0] + arr[i];
				arr[i] = arr[0];
				arr[0] = tmp;
			}
			arr[0] = 0;
		}
};

template<class Z, class T>
void HJR<Z,T>::snop_hash_join(){
	this->set_algo("single-thread no partition hash join");
	this->reset_metrics();
	this->reset_aux_struct();

	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();

	//Build phase
	this->t.start();
	for(uint64_t i = 0; i < R->n; i++){
		Z primary_key = R->keys[i];
		T score = 0;
		for(uint8_t j = 0; j < R->d; j++){ score+= R->scores[j*R->n + i]; }
		this->htR.emplace(primary_key,score);
	}

	//Probe phase
	for(uint64_t i =0; i< S->n; i++){
		Z id = S->ids[i];
		Z foreign_key = S->keys[i];
		auto range = this->htR.equal_range(foreign_key);
		if( range.first != range.second ){ // If probe match
			T score = 0;
			for(uint8_t j = 0; j < S->d; j++){ score+= S->scores[j*S->n + i]; }
			//TODO: Check if can score higher than threshold, break otherwise ?
			for(auto it = range.first; it != range.second; ++it){
				T combined_score = score + it->second;
				//std::cout << key << " = combined: " << score << "," << it->second << "," << combined_score << std::endl;
				this->tuple_count++;
				if(this->q[0].size() < k){
					this->q[0].push(_tuple<Z,T>(id,combined_score));
				}else if(this->q[0].top().score < combined_score){
					this->q[0].pop();
					this->q[0].push(_tuple<Z,T>(id,combined_score));
				}
			}
		}
	}
	this->t_join = this->t.lap();
}

template<class Z, class T>
void HJR<Z,T>::st_nop_hash_rank_join(){
	this->set_algo("st_nop_hash_rank_join");
	this->reset_metrics();
	this->reset_aux_struct();

	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();

	S_HashTable<Z,T> htR;
	htR.initialize(((R->n - 1) / S_HASHT_BUCKET_SIZE) + 1);

	this->t.start();
	htR.build_st(R);
	this->tuple_count = htR.probe_st(S,&this->q[0],k);
	this->t_join += this->t.lap();
}

template<class Z, class T>
void HJR<Z,T>::mt_nop_hash_rank_join(){

}

#endif
