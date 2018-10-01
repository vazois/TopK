#ifndef PBRJ_H
#define PBRJ_H

#include "ARJ.h"

template<class Z, class T>
struct tuple_t
{
	Z key;
	T score;
};

template<class Z,class T>
static bool descending(const tuple_t<Z,T> &a, const tuple_t<Z,T> &b){ return a.score > b.score; };

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
		static void sort_fast(TABLE<Z,T> *rel);
		inline void updateQ(std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>> *q, Z key, T score, Z k);
};

template<class Z, class T>
void PBRJ<Z,T>::sort(TABLE<Z,T> *rel)
{
	tuple_t<Z,T> *data = (tuple_t<Z,T>*) malloc(sizeof(tuple_t<Z,T>) * rel->n);
	for(uint64_t i = 0; i < rel->n; i++)
	{
		data[i].key = rel->keys[i];
		data[i].score = rel->scores[i];
	}
	//std::sort(&data[0],(&data[0]) + rel->n, descending<Z,T>);
	__gnu_parallel::sort(&data[0],(&data[0]) + rel->n, descending<Z,T>);
	for(uint64_t i = 0; i < rel->n; i++)
	{
		rel->keys[i] = data[i].key;
		rel->scores[i] = data[i].score;
	}
	free(data);
}

template<class Z, class T>
inline void PBRJ<Z,T>::updateQ(std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>> *q, Z key, T score, Z k)
{
	if(q[0].size() < k){
		q[0].push(_tuple<Z,T>(key,score));
	}else if(this->q[0].top().score < score){
		q[0].pop();
		q[0].push(_tuple<Z,T>(key,score));
	}
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

	this->t.start();
	uint64_t idxR = 1, idxS = 1;
	Z pkey = 0, fkey = 0;
	T maxR = MAX_SCORE, maxS = MAX_SCORE;
	T minR = MAX_SCORE, minS = MAX_SCORE;
	T threshold = 0;

	while(idxR < R->n || idxS < S->n)
	{
		Z pkey, fkey;
		T pscore = 0, fscore = 0;

		if(idxR < R->n)
		{
			this->pull_count++;
			pkey = R->keys[idxR];
			pscore = R->scores[idxR];
			//for(uint8_t m = 0; m < R->d; m++) pscore += R->scores[m*R->n + idxR];
			idxR++;
		}

		if(idxS < S->n)
		{
			this->pull_count++;
			fkey = S->keys[idxS];
			fscore = S->scores[idxS];
			//for(uint8_t m = 0; m < S->d; m++) fscore += S->scores[m*S->n + idxS];
			idxS++;
		}

		//Join new objects
		T combined_score = 0;
		if(pkey == fkey)
		{
			this->join_count++;
			combined_score = pscore + fscore;
			this->updateQ(&this->q[0],pkey,combined_score,k);
		}

		//Join fkey to previous objects
		auto prange = this->htR.equal_range(fkey);
		for(auto it = prange.first; it != prange.second; ++it)
		{
			this->join_count++;
			combined_score = fscore + it->second;
			this->updateQ(&this->q[0],fkey,combined_score,k);
		}

		//Join pkey to previous objects
		auto frange = this->htS.equal_range(pkey);
		for(auto it = frange.first; it != frange.second; ++it)
		{
			this->join_count++;
			combined_score = pscore + it->second;
			this->updateQ(&this->q[0],pkey,combined_score,k);
		}

		this->htR.emplace(pkey,pscore);
		this->htS.emplace(fkey,fscore);
		minR = std::min(minR,pscore);
		minS = std::min(minS,fscore);

		threshold = std::max(maxR + minS, minR + maxS);
		if( this->q[0].size() == k && this->q[0].top().score >= threshold){
			break;
		}
	}
	this->t_join += this->t.lap();

}

#endif
