#ifndef ARJ_H
#define ARJ_H

#include "../common/common.h"
#include "../time/Time.h"
#include<limits>
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

static uint8_t UNIFORM_KEY=0;
static uint8_t UNIFORM_SCORE=0;

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
	Z *ids;//tuple ids
	T *scores;//tuple scores
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
			this->R.scores = static_cast<T*>(aligned_alloc(32,sizeof(T)*n0*d0));
			this->S.n = n1; this->S.d = d1;
			this->S.ids = static_cast<Z*>(aligned_alloc(32,sizeof(Z)*n1));
			this->S.scores = static_cast<T*>(aligned_alloc(32,sizeof(T)*n1*d1));
			this->k = k;
		};

		~RankJoinInstance(){
			if(this->R.ids != NULL) free(this->R.ids); if(this->R.scores != NULL) free(this->R.scores);
			if(this->S.ids != NULL) free(this->S.ids); if(this->S.scores != NULL) free(this->S.scores);
		};

		TABLE<Z,T>* getR(){ return &(this->R); }
		TABLE<Z,T>* getS(){ return &(this->S); }
		Z getK(){ return this->k; }

		void sample(){
			std::cout << " <<< R >>> " << std::endl;
			for(uint32_t i = 0; i < 16; i++){
				std::cout << std::setfill('0') << std::setw(6) << this->R.ids[i];
				for(uint32_t j = 0; j <this->R.d; j++){
					std::cout << " | "<< std::fixed << std::setprecision(4) << this->R.scores[j*this->R.n + i];
				}
				std::cout << std::endl;
			}
			std::cout << " <<< S >>> " << std::endl;
			for(uint32_t i = 0; i < 32; i++){
				std::cout << std::setfill('0') << std::setw(6) << this->S.ids[i];
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
	for(uint64_t i = 0; i < R->n; i++){
		R->ids[i] = i;
		for(uint64_t j =0; j < R->d; j++){
			R->scores[j*R->n + i] = this->gen_score();//column-wise initialization//
		}
	}

	for(uint64_t i = 0; i < S->n; i++){
		S->ids[i] = this->gen_key(R->n);
		for(uint64_t j = 0; j < S->d; j++){
			S->scores[j*S->n + i] = this->gen_score();
		}
	}
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
		};

		~AARankJoin(){

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
	protected:
		RankJoinInstance<Z,T> *rj_inst;

		Time<msecs> t;
		Time<msecs> tt[THREADS];
		double tt_init;
		double tt_join;
		double tt_build;
		double tt_probe;
		double tuple_count;
		double ttuple_count[THREADS];
};
#endif
