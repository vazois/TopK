#ifndef RJ_TOOLS_H
#define RJ_TOOLS_H

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
#include <pthread.h>

static uint8_t UNIFORM_KEY=0;
static uint8_t UNIFORM_SCORE=0;

#define CACHE_LINE_SIZE 64

#define BATCH 1024
#define PNUM (32*1024)
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
	Z *ids = NULL;
	Z *keys = NULL;//tuple ids
	T *scores = NULL;//tuple scores
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
			this->R.ids = static_cast<Z*>(aligned_alloc(CACHE_LINE_SIZE,sizeof(Z)*n0));
			this->R.keys = static_cast<Z*>(aligned_alloc(CACHE_LINE_SIZE,sizeof(Z)*n0));
			this->R.scores = static_cast<T*>(aligned_alloc(CACHE_LINE_SIZE,sizeof(T)*n0*d0));
			this->S.n = n1; this->S.d = d1;
			this->S.ids = static_cast<Z*>(aligned_alloc(CACHE_LINE_SIZE,sizeof(Z)*n1));
			this->S.keys = static_cast<Z*>(aligned_alloc(CACHE_LINE_SIZE,sizeof(Z)*n1));
			this->S.scores = static_cast<T*>(aligned_alloc(CACHE_LINE_SIZE,sizeof(T)*n1*d1));
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
		S->keys[i] = this->gen_key(R->n-1);
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


#endif
