#ifndef ARJ_H
#define ARJ_H

#include "../common/common.h"
#include "../time/Time.h"
#include <limits>
#include <stdio.h>
#include <cstdint>
#include <stdio.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <iomanip>

#include <chrono>
#include <random>
#include <utility>

#include <unordered_map>
#include <queue>
#include <mutex>
#include <thread>

static uint8_t UNIFORM_KEY=0;
static uint8_t UNIFORM_SCORE=0;

#define BATCH 1024
#define PNUM 128
#define MASK (PNUM-1)
#define PHASH(X) (X & MASK)

template<class Z, class T>
struct _tuple{
	_tuple(){}
	_tuple(Z i, T s){
		id = i;
		score = s;
	}
	Z id;
	T score;
};

template<class Z, class T>
struct TABLE{
	Z n;//cardinality
	Z d;//dimensionality
	Z *ids;
	Z *keys;//tuple ids
	T *scores;//tuple scores
};

template<class Z, class T>
struct rrTABLE{
	Z *ids;
	Z *keys;
	T *scores;
};

template<class T,class Z>
class pq_descending{
	public:
		pq_descending(){};

		bool operator() (const _tuple<T,Z>& lhs, const _tuple<T,Z>& rhs) const{
			return (lhs.score>rhs.score);
		}
};

template<class Z, class T>
class RankJoinInstance{
	public:
		RankJoinInstance(Z n0, Z d0, Z n1, Z d1, Z k){
			this->R.n = n0; this->R.d = d0;
			this->R.ids = static_cast<Z*>(aligned_alloc(32,sizeof(Z)*n0));
			this->R.keys = static_cast<Z*>(aligned_alloc(32,sizeof(Z)*n0));
			this->R.scores = static_cast<T*>(aligned_alloc(32,sizeof(T)*n0*d0));
			this->S.n = n1; this->S.d = d1;
			this->S.ids = static_cast<Z*>(aligned_alloc(32,sizeof(Z)*n1));
			this->S.keys = static_cast<Z*>(aligned_alloc(32,sizeof(Z)*n1));
			this->S.scores = static_cast<T*>(aligned_alloc(32,sizeof(T)*n1*d1));
			this->k = k;
		};

		~RankJoinInstance(){
			if(this->R.ids != NULL) free(this->R.ids); if(this->R.keys != NULL) free(this->R.keys); if(this->R.scores != NULL) free(this->R.scores);
			if(this->S.ids != NULL) free(this->S.ids); if(this->S.keys != NULL) free(this->S.keys); if(this->S.scores != NULL) free(this->S.scores);
		};

		TABLE<Z,T>* getR(){ return &(this->R); }
		TABLE<Z,T>* getS(){ return &(this->S); }
		Z getK(){ return this->k; }

		void sample(){
			std::cout << " <<< R >>> " << std::endl;
			for(uint32_t i = 0; i < 16; i++){
				std::cout << std::setfill('0') << std::setw(6) << this->R.keys[i];
				for(uint32_t j = 0; j <this->R.d; j++){
					std::cout << " | "<< std::fixed << std::setprecision(4) << this->R.scores[j*this->R.n + i];
				}
				std::cout << std::endl;
			}
			std::cout << " <<< S >>> " << std::endl;
			for(uint32_t i = 0; i < 32; i++){
				std::cout << std::setfill('0') << std::setw(6) << this->S.keys[i];
				for(uint32_t j = 0; j < this->S.d; j++){
					std::cout << " | " << std::fixed << std::setprecision(4) << this->S.scores[j*this->R.n + i];
				}
				std::cout << std::endl;
			}
		}
	private:
		TABLE<Z,T> R;
		TABLE<Z,T> S;
		Z k;
};

template<class Z, class T>
class GenData{
	public:
		GenData(RankJoinInstance<Z,T> *rj_inst, uint8_t key_distr=0, uint8_t score_distr=0) :
			def_eng(std::chrono::system_clock::now().time_since_epoch().count())
			//def_eng(1234)
		{
			this->populate(rj_inst->getR(),rj_inst->getS());
			this->key_distr = key_distr;
			this->score_distr = score_distr;
		}
		~GenData(){};

	private:
		void populate(TABLE<Z,T> *R, TABLE<Z,T> *S);
		std::default_random_engine def_eng;
		uint8_t key_distr=0;
		uint8_t score_distr=0;

		Z gen_key(Z max);
		T gen_score();
};

template<class Z,class T>
void GenData<Z,T>::populate(TABLE<Z,T> *R, TABLE<Z,T> *S){
	std::cout << "Populating tables ... " << std::endl;
	float progress = 0.0;
	uint64_t step = 1024;
	uint64_t ii = 0;

	for(uint64_t i = 0; i < R->n; i++){
		R->ids[i] = i;
		R->keys[i] = i;
		for(uint64_t j =0; j < R->d; j++){
			R->scores[j*R->n + i] = this->gen_score();//column-wise initialization//
		}

		if((ii & (step - 1)) == 0){
			std::cout << "Progress: [" << int(progress * 100.0) << "] %\r";
			std::cout.flush();
			progress += ((float)step)/(R->n + S->n); // for demonstration only
		}
		ii++;
	}

	for(uint64_t i = 0; i < S->n; i++){
		S->ids[i] = i;
		S->keys[i] = this->gen_key(R->n);
		for(uint64_t j = 0; j < S->d; j++){
			S->scores[j*S->n + i] = this->gen_score();
		}

		if((ii & (step - 1)) == 0){
			std::cout << "Progress: [" << int(progress * 100.0) << "] %\r";
			std::cout.flush();
			progress += ((float)step)/(R->n + S->n); // for demonstration only
		}
		ii++;
	}
	std::cout << "Progress: [" << 100 << "] %\r";
	std::cout.flush();
}

template<class Z, class T>
Z GenData<Z,T>::gen_key(Z max){
	switch(this->key_distr){
		case 0:
			return std::uniform_int_distribution<Z>{0, max}(this->def_eng);
		default:
			return 0;
	}	
}

template<class Z, class T>
T GenData<Z,T>::gen_score(){
	switch(this->score_distr){
		case 0:
			return std::uniform_real_distribution<T>{0, 1}(this->def_eng);
		default:
			return 0.0f;
	}
}

template<class Z, class T>
class AARankJoin{
	public:
		AARankJoin(RankJoinInstance<Z,T> *rj_inst){
			this->rj_inst = rj_inst;
			this->reset_metrics();
			this->reset_aux_struct();
			this->rrR.keys = this->rrS.keys = NULL;
			this->rrR.scores = this->rrS.scores = NULL;
		};

		~AARankJoin(){
			if(this->rrR.ids != NULL) free(this->rrR.ids); if(this->rrR.keys != NULL) free(this->rrR.keys); if(this->rrR.scores != NULL) free(this->rrR.scores);
			if(this->rrS.ids != NULL) free(this->rrS.ids); if(this->rrS.keys != NULL) free(this->rrS.keys); if(this->rrS.scores != NULL) free(this->rrS.scores);
		};

		void join();
		void reset_metrics(){
			this->tt_init = 0;
			this->tt_join = 0;
			this->tt_build = 0;
			this->tt_probe = 0;
			this->tuple_count = 0;
			for(uint32_t i = 0; i < THREADS; i++) this->ttuple_count[i] = 0;
		}

		void reset_aux_struct(){
			this->q = std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>>();
			for(uint32_t i = 0; i < THREADS; i++) this->tq[i] = std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>>();
			this->htR.clear();
			this->htS.clear();
			for(uint64_t i = 0; i < PNUM+1; i++){
				if(i < PNUM){
					this->phtR[i].clear();
					this->phtS[i].clear();
				}
			}
			for(uint32_t j = 0; j < THREADS*(PNUM+1); j++){
				this->part_sizeR[j] = 0;
				this->part_sizeS[j] = 0;
			}
		}

		void merge_qs(){
			Z k = this->rj_inst->getK();
			for(uint32_t i = 0; i < THREADS; i++){
				while(!this->tq[i].empty()){
					if(this->q.size() < k){
						this->q.push(this->tq[i].top());
					}else if(this->q.top().score < this->tq[i].top().score){
						this->q.pop();
						this->q.push(this->tq[i].top());
					}
					this->tq[i].pop();
				}
			}
		}


		void benchmark();
	protected:
		void set_algo(std::string algo){ this->algo = algo; }
		std::string algo;
		RankJoinInstance<Z,T> *rj_inst;
		std::unordered_multimap<Z,T> htR;
		std::unordered_multimap<Z,T> htS;

		//Partitionjoin structures structures//
		std::unordered_multimap<Z,T> phtR[PNUM];
		std::unordered_multimap<Z,T> phtS[PNUM];
//		Z part_sizeR[THREADS][PNUM+1];
//		Z part_sizeS[THREADS][PNUM+1];
		Z part_sizeR[THREADS*(PNUM+1)];
		Z part_sizeS[THREADS*(PNUM+1)];
		std::mutex part_mtx[PNUM];
		rrTABLE<Z,T> rrR;
		rrTABLE<Z,T> rrS;

		void rr_alloc(){
			this->rrR.ids = static_cast<Z*>(aligned_alloc(32,sizeof(Z)*(this->rj_inst->getR()->n)));
			this->rrR.keys = static_cast<Z*>(aligned_alloc(32,sizeof(Z)*(this->rj_inst->getR()->n)));
			this->rrR.scores = static_cast<T*>(aligned_alloc(32,sizeof(T)*(this->rj_inst->getR()->n)));
			this->rrS.ids = static_cast<Z*>(aligned_alloc(32,sizeof(Z)*(this->rj_inst->getS()->n)));
			this->rrS.keys = static_cast<Z*>(aligned_alloc(32,sizeof(Z)*(this->rj_inst->getS()->n)));
			this->rrS.scores = static_cast<T*>(aligned_alloc(32,sizeof(T)*(this->rj_inst->getS()->n)));
		}

		//Ranking Structures//
		std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>> q;
		std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>> tq[THREADS];

		Time<msecs> t;
		Time<msecs> tt[THREADS];
		double tt_init;
		double tt_join;
		double tt_build;
		double tt_probe;
		double tuple_count;
		double ttuple_count[THREADS];
};

template<class Z, class T>
void AARankJoin<Z,T>::benchmark(){
	std::cout << "<<< " << this->algo << " >>>" << std::endl;
	std::cout << "join elapsed(ms): " << this->tt_join << std::endl;
	if(this->q.size() > 0) std::cout << "threshold (" << this->q.size() <<"): " << std::fixed << std::setprecision(4) << this->q.top().score << std::endl;
	std::cout << "----------------------------" << std::endl;
}

#endif
