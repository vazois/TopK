#ifndef LARA_H
#define LARA_H

#include "AA.h"

template<class T>
struct lara_candidate
{
	T score;
	uint8_t seen;
};

template<class T,class Z>
static bool lara_pair_descending(const qpair<T,Z> &a, const qpair<T,Z> &b){ return a.score > b.score; };

template<class T, class Z>
class LARA : public AA<T,Z>
{
	public:
		LARA(uint64_t n, uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "LARA";
			this->lists = NULL;
		};

		~LARA()
		{
			if(this->lists!=NULL){ for(uint8_t m = 0;m < this->d;m++){ if(this->lists[m] != NULL){ free(this->lists[m]); } } }
			free(this->lists);
		};

		void init();
		void findTopK(uint64_t k, uint8_t qq, T *weights, uint8_t *attr);
		void findTopK2(uint64_t k, uint8_t qq, T *weights, uint8_t *attr);

	private:
		qpair<T,Z> ** lists;
};

template<class T, class Z>
void LARA<T,Z>::init()
{
	normalize<T,Z>(this->cdata, this->n, this->d);
	this->lists = (qpair<T,Z> **) malloc(sizeof(qpair<T,Z>*)*this->d);
	for(uint8_t m = 0; m < this->d; m++){ this->lists[m] = (qpair<T,Z>*) malloc(sizeof(qpair<T,Z>)*this->n); }

	this->t.start();
	for(uint64_t i = 0; i < this->n; i++)
	{
		for(uint8_t m = 0; m < this->d; m++)
		{
			this->lists[m][i].id = i;
			this->lists[m][i].score = this->cdata[i*this->d + m];
		}
	}
	for(uint8_t m = 0; m < this->d; m++) __gnu_parallel::sort(this->lists[m],lists[m]+this->n,lara_pair_descending<T,Z>);
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void LARA<T,Z>::findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr)
{
	std::cout << this->algo << " find top-" << k << " (" << (int)qq << "D) ...";
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(STATS_EFF) this->candidate_count=0;

	pqueue<T,Z,pqueue_desc<T,Z>> q(k);
	std::unordered_map<Z,lara_candidate<T>> lbmap;
	std::unordered_set<Z> c;

	///////////////////
	///Growing Phase///
	///////////////////
	this->t.start();
	uint64_t i = 0;
	for(i = 0; i < this->n; i++)
	{
		T threshold = 0;
		for(uint8_t m = 0; m < qq; m++)
		{
			uint8_t idx_a = attr[m];
			Z id = lists[idx_a][i].id;
			T aa = lists[idx_a][i].score;
			T weight = weights[idx_a];
			threshold+= weight * aa;
			if(STATS_EFF) this->accesses+=4;

			////////////////
			//Update score//
			auto it = lbmap.find(id);
			if(STATS_EFF) this->accesses+=1;
			if (it == lbmap.end()){
				lara_candidate<T> lc;
				lc.score = 0;
				lc.seen = 0;
				it = lbmap.emplace(id,lc).first;
				if(STATS_EFF) this->accesses+=1;
			}
			it->second.score+=weight * aa;//aggregate upper bound
			it->second.seen|=(1<<idx_a);//flag seen location
			if(STATS_EFF) this->accesses+=2;

			qpair<T,Z> p;
			p.id = id;
			p.score = it->second.score;
			if(STATS_EFF) this->accesses+=1;
			if(c.find(id) == c.end())//if object is not in Wk
			{
				if(q.size() < k){//insert if empty space in queue
					q.push(p);
					c.insert(id);
					if(STATS_EFF) this->accesses+=1;
				}else if(q.top().score<p.score){//delete smallest element if current score is larger
					c.erase(q.top().id);
					c.insert(id);
					q.pop();
					q.push(p);
					if(STATS_EFF) this->accesses+=4;
				}
			}else{
				if(STATS_EFF) this->accesses+=1;
				q.update(p);//update score in queue and rebalance
			}
		}
		if(q.size() >= k && ((q.top().score) >= threshold) ){ break; }
	}
	if(STATS_EFF) this->tuple_count = lbmap.size();
	if(STATS_EFF) this->candidate_count = lbmap.size();

	T trank[NUM_DIMS];//Initialize last ranks
	for(uint8_t m = 0; m < qq; m++)
	{
		uint8_t idx_a = attr[m];
		Z id = lists[idx_a][i-1].id;
		T aa = lists[idx_a][i-1].score;
		T weight = weights[idx_a];
		trank[idx_a] = weight * aa;
		if(STATS_EFF) this->accesses+=3;
	}

	//Initialize latice lower bounds
	uint32_t latt_size = (1 << NUM_DIMS);
	T uparray[latt_size];
	T lbarray[latt_size];
	std::memset(lbarray,0,sizeof(T)*latt_size);
	for(auto it = lbmap.begin(); it!=lbmap.end(); ++it)
	{
		uint8_t seen = it->second.seen;
		T lbscore = it->second.score;
		lbarray[seen] = std::max(lbarray[seen],lbscore);
		if(STATS_EFF) this->accesses+=3;
	}

	//////////////////////
	///Shrinking Phase///
	/////////////////////
	for( ; i < this->n; i++)
	{
		for(uint8_t m = 0; m < qq; m++)
		{
			uint8_t idx_a = attr[m];
			Z id = lists[idx_a][i].id;
			T aa = lists[idx_a][i].score;
			T weight = weights[idx_a];
			trank[idx_a] = weight * aa;
			if(STATS_EFF) this->accesses+=4;

			auto it = lbmap.find(id);
			if(STATS_EFF) this->accesses+=1;
			if(it != lbmap.end())
			{
				uint8_t seen = it->second.seen;
				T lbscore = it->second.score + trank[idx_a];
				if(STATS_EFF) this->accesses+=2;

				it->second.seen |= (1 << idx_a);
				it->second.score = lbscore;
				lbarray[it->second.seen] = std::max(lbarray[it->second.seen], lbscore);
				if(STATS_EFF) this->accesses+=2;

				qpair<T,Z> p;
				p.id = id;
				p.score = it->second.score;
				if(STATS_EFF) this->accesses+=1;
				if(c.find(id) == c.end())//if object is not in Wk
				{
					if(q.size() < k){//insert if empty space in queue
						q.push(p);
						c.insert(id);
						if(STATS_EFF) this->accesses+=2;
					}else if(q.top().score<p.score){//delete smallest element if current score is larger
						c.erase(q.top().id);
						c.insert(id);
						q.pop();
						q.push(p);
						if(STATS_EFF) this->accesses+=4;
					}
				}else{
					q.update(p);//update score in queue and rebalance
				}

				T max_ub = 0;
				for(uint32_t j = 1; j < latt_size; j++)
				{
					uint8_t ll = j^0xFF;
					T add = 0;
					for(uint8_t mm = 0; mm < qq; mm++)
					{
						if(ll & (1 << mm))
						{
							add+= weights[mm]*attr[mm];
						}
					}
					max_ub = std::max(lbarray[j] + add,max_ub);
					if(STATS_EFF) this->accesses+=1;
				}

				if(max_ub <= q.top().score){
					i = this->n;
					this->lvl = i;
					break;
				}
			}
		}
	}
	this->tt_processing += this->t.lap();

	T threshold = q.top().score;
	while(!q.empty())
	{
		tuple_<T,Z> t;
		t.tid=q.top().id;
		t.score=q.top().score;
		this->res.push_back(t);
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " partial threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
