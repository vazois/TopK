#ifndef PBRJ_H
#define PBRJ_H

#include "ARJ.h"

template<class Z, class T>
struct tuple_t
{
	Z key;
	T value;
	T scores[DIMS];
};

template<class Z,class T>
static bool descending(const tuple_t<Z,T> &a, const tuple_t<Z,T> &b){ return a.value > b.value; };

/*
 * Pull/Bound Rank Join
 */
template<class Z, class T>
class PBRJ : public AARankJoin<Z,T>{
	public:
		PBRJ(RankJoinInstance<Z,T> *rj_inst) : AARankJoin<Z,T>(rj_inst){ };
		~PBRJ(){};

		void st_nop_pbrj_rr();

	private:
		static void sort(TABLE<Z,T> *rel);
};

template<class Z, class T>
void PBRJ<Z,T>::sort(TABLE<Z,T> *rel)
{
	tuple_t<Z,T> *data = (tuple_t<Z,T>*) malloc(sizeof(tuple_t<Z,T>) * rel->n);
	for(uint64_t i = 0; i < rel->n; i++)
	{
		T score = 0;
		for(uint8_t m = 0; m < rel->d; m++) score=std::max(score, rel->scores[m*rel->n + i]);
		data[i].key = rel->keys[i];
		data[i].value = score;
		for(uint64_t j = 0; j < rel->d; j++) data[i].scores[j] = rel->scores[j*rel->n + i];
	}
	//std::sort(&data[0],(&data[0]) + rel->n, descending<Z,T>);
	__gnu_parallel::sort(&data[0],(&data[0]) + rel->n, descending<Z,T>);
	for(uint64_t i = 0; i < rel->n; i++)
	{
		rel->keys[i] = data[i].key;
		for(uint64_t j = 0; j < rel->d; j++) rel->scores[j*rel->n + i] = data[i].scores[j];
	}
//
//	for(uint64_t i = 0; i < 10; i++)
//	{
//		std::cout << rel->keys[i] << ":";
//		for(uint64_t j = 0; j < rel->d; j++) std::cout << " " << rel->scores[j*rel->n + i];
//		std::cout << std::endl;
//	}
	free(data);
}

template<class Z, class T>
void PBRJ<Z,T>::st_nop_pbrj_rr()
{
	this->set_algo("single-thread no partition pull/bound rank join");
	this->reset_metrics();
	this->reset_aux_struct();

	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();

	this->t.start();
	PBRJ<Z,T>::sort(R);
	PBRJ<Z,T>::sort(S);
	this->t_init=this->t.lap();

	uint64_t idxR = 0, idxS = 0;
	T maxR = 0, maxS = 0;
	for(uint8_t m = 0; m < R->d; m++) maxR += R->scores[m*R->n];
	for(uint8_t m = 0; m < S->d; m++) maxR += S->scores[m*S->n];
	T scrR = 0, scrS = 0;
	do{
		Z pkey, fkey;
		T pscore = 0, fscore = 0;

		if(idxR < R->n)
		{
			pkey = R->keys[idxR];
			for(uint8_t m = 0; m < R->d; m++) pscore += R->scores[m*R->n + idxR];
			this->htR.emplace(pkey,pscore);
			scrR = pscore;
			idxR++;
		}

		if(idxS < S->n)
		{
			fkey = S->keys[idxS];
			auto range = this->htR.equal_range(fkey);
			if( range.first != range.second ){
				for(uint8_t m = 0; m < S->d; m++) fscore += S->scores[m*S->n + idxS];
				for(auto it = range.first; it != range.second; ++it){
					T combined_score = fscore + it->second;
					this->tuple_count++;
					if(this->q[0].size() < k){
						this->q[0].push(_tuple<Z,T>(idxS,combined_score));
					}else if(this->q[0].top().score < combined_score){
						this->q[0].pop();
						this->q[0].push(_tuple<Z,T>(idxS,combined_score));
					}
				}
			}
			idxS++;
			scrS = fscore;
		}

	}while(idxR < R->n || idxS < S->n);
}

#endif
