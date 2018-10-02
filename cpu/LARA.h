#ifndef LARA_H
#define LARA_H

template<class T, class Z>
struct lara_pair
{
	Z id;
	T score;
};

template<class T,class Z>
static bool lara_pair_descending(const lara_pair<T,Z> &a, const lara_pair<T,Z> &b){ return a.score < b.score; };

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

	private:
		lara_pair<T,Z> ** lists;
};


template<class T, class Z>
void LARA<T,Z>::init()
{
	normalize<T,Z>(this->cdata, this->n, this->d);
	this->lists = (lara_pair<T,Z> **) malloc(sizeof(lara_pair<T,Z>*)*this->d);
	for(uint8_t m = 0; m < this->d; m++){ this->lists[m] = (lara_pair<T,Z>*) malloc(sizeof(lara_pair<T,Z>)*this->n); }

	for(uint64_t i = 0; i < this->n; i++)
	{
		for(uint8_t m = 0; m < this->d; m++)
		{
			this->lists[m][i].id = i;
			this->lists[m][i].score = this->cdata[i*this->d + m];
		}
	}

	for(uint8_t m = 0; m < this->d; m++) __gnu_parallel::sort(this->lists[m],lists[m]+this->n,lara_pair_descending<T,Z>);
}

template<class T, class Z>
void LARA<T,Z>::findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr)
{
	std::cout << this->algo << " find top-" << k << " (" << (int)qq << "D) ...";
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	std::unordered_map<T,Z> lbmap;
	//Growing Phase
	for(uint64_t i = 0; i < this->n; i++)
	{
		T threshold = 0;
		for(uint8_t m = 0; m < qq; m++)
		{
			uint8_t idx_a = attr[m];
			Z id = lists[idx_a][i].id;
			T aa = lists[idx_a][i].score;
			T weight = weights[idx_a];
			threshold+= weight * aa;

			auto it = lbmap.find(id);
			if (it == lbmap.end()) it = lbmap.emplace(id,0).first;
			it->second+=aa*weights[idx_a];

			T lbscore = it->score;
			if(q.size() < k){//insert if empty space in queue
				q.push(tuple_<T,Z>(id,lbscore));
			}else if(q.top().score<lbscore){//delete smallest element if current score is bigger
				q.pop();
				q.push(tuple_<T,Z>(id,lbscore));
			}
		}
		if(q.size() >= k && ((q.top().score) >= threshold) ){ break; }
	}


	T threshold = 1313;
	while(!q.empty()){ this->res.push_back(q.top()); q.pop(); }
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}


#endif
