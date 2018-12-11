#ifndef BPA_H
#define BPA_H

#include "AA.h"
#include <unordered_set>

template<class T,class Z>
static bool bpa_pair_descending(const qpair<T,Z> &a, const qpair<T,Z> &b){ return a.score > b.score; };

template<class T, class Z>
class BPA : public AA<T,Z>
{
	public:
		BPA(uint64_t n, uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "BPA";
			this->lists = NULL;
		};

		~BPA()
		{
			if(this->lists!=NULL){ for(uint8_t m = 0;m < this->d;m++){ if(this->lists[m] != NULL){ free(this->lists[m]); } } }
			free(this->lists);
		};

		void init();
		void findTopK(uint64_t k, uint8_t qq, T *weights, uint8_t *attr);

	private:
		qpair<T,Z> ** lists;
		std::unordered_map<Z,Z> pos[NUM_DIMS];
};

template<class T, class Z>
void BPA<T,Z>::init()
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
	for(uint8_t m = 0; m < this->d; m++){
		__gnu_parallel::sort(this->lists[m],lists[m]+this->n,bpa_pair_descending<T,Z>);
		for(uint64_t i = 0; i < this->n; i++) pos[m].emplace(lists[m][i].id,i);
	}
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void BPA<T,Z>::findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr)
{
	std::cout << this->algo << " find top-" << k << " (" << (int)qq << "D) ...";
	std::unordered_set<Z> eset;
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(STATS_EFF) this->candidate_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	uint64_t bp[NUM_DIMS];
	uint32_t **seen = (uint32_t**)malloc(sizeof(uint32_t*) * qq );
	Z bvsize = ((this->n - 1)/32 + 1);
	for(uint8_t m = 0; m < qq; m++){
		bp[m] = 0;
		seen[m] = (uint32_t*)malloc(sizeof(uint32_t) * ((this->n - 1)/32 + 1));
		memset(seen[m], 0, sizeof(uint32_t) * bvsize);
	}

	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		for(uint8_t m = 0; m < qq; m++)
		{
			Z id = lists[attr[m]][i].id;
			T score = 0;
			if(eset.find(id) == eset.end()){
				for(uint8_t mm = 0; mm < qq; mm++){
					uint8_t idx_a = attr[mm];
					Z p = pos[idx_a].find(id)->second;//find position in list
					score+=lists[idx_a][p].score;// find score for position
					seen[mm][(p >> 5)] |=  (1 << (p & 31));// set bit vector to indicate seen position
				}
				if(STATS_EFF) this->tuple_count++;
				eset.insert(id);
				if(q.size() < k){//insert if empty space in queue
					q.push(tuple_<T,Z>(id,score));
				}else if(q.top().score<score){//delete smallest element if current score is bigger
					q.pop();
					q.push(tuple_<T,Z>(id,score));
				}
			}
		}

		T threshold=0;
		for(uint8_t m = 0; m < qq; m++){
			while(bp[m] < this->n && ( ( ( (seen[m][(bp[m] >> 5)] >> (bp[m] & 31)) & 0x1) == 0x1) ) ) bp[m]++;
			threshold+=lists[attr[m]][bp[m]-1].score;
		}
		if(q.size() >= k && ((q.top().score) >= threshold) ){
			//std::cout << "break i: " << i << std::endl;
			break;
		}
	}
	this->tt_processing += this->t.lap();
	if(STATS_EFF) this->candidate_count=k;

	for(uint8_t m = 0; m < qq; m++) free(seen[m]);
	free(seen);
	T threshold = !q.empty() ? q.top().score : 1313;
	while(!q.empty()){
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}


#endif
