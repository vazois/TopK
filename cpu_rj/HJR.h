#ifndef HJR_H
#define HJR_H

#include "ARJ.h"

template<class Z, class T>
class HJR : public AARankJoin<Z,T>{
	public:
		HJR(RankJoinInstance<Z,T> *rj_inst) : AARankJoin<Z,T>(rj_inst){ };
		~HJR(){};

		void snop_hash_join();
		void pnop_hash_join();
		void sprt_hash_join();
		void pprt_hash_join();

	private:
		void pshift(Z *arr, Z arr_n){
			for(uint32_t i = arr_n-1; i > 0; i--){ arr[i] = arr[i-1]; }
			arr[0] = 0;
		}

		void psum(Z *arr, Z arr_n){
			for(uint32_t i = 1; i < PNUM+1; i++){
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

				if(this->q.size() < k){
					this->q.push(_tuple<Z,T>(id,combined_score));
				}else if(this->q.top().score < combined_score){
					this->q.pop();
					this->q.push(_tuple<Z,T>(id,combined_score));
				}
			}
		}
	}
	this->tt_join = this->t.lap();
}

template<class Z, class T>
void HJR<Z,T>::pnop_hash_join(){
	this->set_algo("multi-thread no partitition hash join (" + std::to_string(THREADS) + ")");
	this->reset_metrics();
	this->reset_aux_struct();
	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();

	omp_set_num_threads(THREADS);
	uint32_t stride = BATCH;
	this->t.start();
	#pragma omp parallel
	{
		uint32_t tid = omp_get_thread_num();
		uint32_t offset = tid * stride;
		uint32_t step = THREADS * stride;

		//Build-Phase
		for(uint64_t i = offset; i < R->n; i+=step){
			for(uint64_t j = i; j < i+stride; j++){
				Z primary_key = R->keys[j];
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
				Z id = S->ids[i];
				Z foreign_key = S->keys[j];
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
							this->tq[tid].push(_tuple<Z,T>(id,combined_score));
						}else if(this->tq[tid].top().score < combined_score){
							this->tq[tid].pop();
							this->tq[tid].push(_tuple<Z,T>(id,combined_score));
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
void HJR<Z,T>::sprt_hash_join(){
	this->set_algo("single-thread partitioned hash join");
	this->reset_metrics();
	this->reset_aux_struct();
	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();
	this->rr_alloc();

	///////////////
	//Partition R//
	this->t.start();
	for(uint64_t i = 0; i < R->n; i++){
		Z primary_key = R->keys[i];
		Z p = PHASH(primary_key);
		this->part_sizeR[p]++;
	}
//
	psum(this->part_sizeR,PNUM);
	//for(uint32_t i = 0;i < PNUM+1; i++) std::cout << this->part_sizeR[0][i] << std::endl;
//
	for(uint64_t i = 0; i < R->n; i++){
		Z primary_key = R->keys[i];
		Z p = PHASH(primary_key);
		Z pos = this->part_sizeR[p]++;
		T score = 0;
		//this->part_sizeR[p]++;

		for(uint8_t j = 0; j < R->d; j++){ score+= R->scores[j*R->n + i]; }
		this->rrR.ids[pos] = R->ids[i];
		this->rrR.keys[pos] = primary_key;
		this->rrR.scores[pos] = score;
	}
	/////////////////////////////////////////////////////////////////////////////
	///////////////
	//Partition S//
	for(uint64_t i = 0; i < S->n; i++){
		Z foreign_key = S->keys[i];
		Z p = PHASH(foreign_key);
		this->part_sizeS[p]++;
	}
//
	psum(this->part_sizeS,PNUM);
	//std::cout <<"---\n"; for(uint32_t i = 0;i < PNUM+1; i++) std::cout << this->part_sizeS[0][i] << std::endl;
//
	for(uint64_t i = 0; i < S->n; i++){
		Z foreign_key = S->keys[i];
		Z p = PHASH(foreign_key);
		Z pos = this->part_sizeS[p]++;
		T score = 0;
		//this->part_sizeS[p]++;

		for(uint8_t j = 0; j < S->d; j++){ score+= S->scores[j*S->n + i]; }
		this->rrS.ids[pos] = S->ids[i];
		this->rrS.keys[pos] = foreign_key;
		this->rrS.scores[pos] = score;
	}
	/////////////////////////////////////////////////////////////////////////////
	///////////////
	//Build-Probe//
	pshift(this->part_sizeR,PNUM);
	//for(uint32_t i = 0;i < PNUM+1; i++) std::cout << this->part_sizeR[i] << std::endl;
	pshift(this->part_sizeS,PNUM);
	//std::cout << "---\n";for(uint32_t i = 0;i < PNUM+1; i++) std::cout << this->part_sizeS[i] << std::endl;
	//return;
	for(uint64_t p = 0; p < PNUM; p++){
		//std::cout << this->part_sizeR[p] << "," <<this->part_sizeR[p+1] << std::endl;
		for(uint64_t i = this->part_sizeR[p]; i < this->part_sizeR[p+1]; i++){
			Z primary_key = this->rrR.keys[i];
			T score = this->rrR.scores[i];
			this->phtR[p].emplace(primary_key,score);
		}

		for(uint64_t i = this->part_sizeS[p]; i < this->part_sizeS[p+1]; i++){
			Z id = this->rrS.ids[i];
			Z foreign_key = this->rrS.keys[i];
			T score = this->rrS.scores[i];

			auto range = this->phtR[p].equal_range(foreign_key);
			if(range.first != range.second){
				for(auto it = range.first; it != range.second; ++it){
					T combined_score = score + it->second;
					//std::cout << foreign_key << " = combined: " << score << "," << it->second << "," << combined_score << std::endl;

					if(this->q.size() < k){
						this->q.push(_tuple<Z,T>(id,combined_score));
					}else if(this->q.top().score < combined_score){
						this->q.pop();
						this->q.push(_tuple<Z,T>(id,combined_score));
					}
				}
			}
		}
	}
	this->tt_join = this->t.lap();
}

template<class Z, class T>
void HJR<Z,T>::pprt_hash_join(){
	this->set_algo("multi-thread partitioned hash join");
	this->reset_metrics();
	this->reset_aux_struct();
	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();

	omp_set_num_threads(THREADS);
	this->t.start();
	#pragma omp parallel
	{
		uint32_t tid = omp_get_thread_num();
		uint32_t offset = tid * BATCH;
		uint32_t stride = THREADS * BATCH;

		Z *h = &this->part_sizeR[tid*(PNUM+1)];
		for(uint64_t i = offset; i < R->n; i+=stride){
			for(uint64_t j = i; j < i + BATCH; j++){
				Z primary_key = R->keys[j];
				Z p = PHASH(primary_key);
				h[p]++;
			}
		}
		psum(h,PNUM);
		h = &this->part_sizeS[tid*(PNUM+1)];
		for(uint64_t i = offset; i < S->n; i+=stride){
			for(uint64_t j = i; j < i + BATCH; j++){
				Z foreign_key = S->keys[j];
				Z p = PHASH(foreign_key);
				h[p]++;
			}
		}
		psum(h,PNUM);
		#pragma omp barier

	}

	this->tt_join = this->t.lap();
}

#endif
