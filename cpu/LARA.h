#ifndef LARA_H
#define LARA_H

template<class T, class Z>
struct lara_pair
{
	Z id;
	T score;
};

template<class T>
struct lara_candidate
{
	T score;
	uint8_t seen;
};

template<class T,class Z>
static bool lara_pair_descending(const lara_pair<T,Z> &a, const lara_pair<T,Z> &b){ return a.score > b.score; };

template<class T,class Z>
struct laraq_desc{
	bool operator()(lara_pair<T,Z>& a,lara_pair<T,Z>& b) const{ return a.score>b.score; }
};

template<class T,class Z>
struct laraq_asc{
	bool operator()(lara_pair<T,Z>& a,lara_pair<T,Z>& b) const{ return a.score<b.score; }
};

template<class T, class Z, class CMP>
class pqueue{
	public:
		pqueue()
		{

		}

		pqueue(uint32_t max_capacity)
		{
			this->queue.resize(max_capacity);
			this->max_capacity = max_capacity;
		}
		~pqueue(){}

		void push(lara_pair<T,Z> t);
		void update(lara_pair<T,Z> t);
		void remove(lara_pair<T,Z> t);
		void pop();
		uint64_t size(){return this->queue.size();}
		const lara_pair<T,Z>& top();

	private:
		std::vector<lara_pair<T,Z>> queue;
		uint32_t max_capacity;
};

template<class T,class Z, class CMP>
void pqueue<T,Z,CMP>::push(lara_pair<T,Z> t)
{
	this->queue.push_back(t);
	std::push_heap(this->queue.begin(),this->queue.end(),CMP());
}

template<class T,class Z, class CMP>
void pqueue<T,Z,CMP>::pop()
{
	std::pop_heap(this->queue.begin(),this->queue.end(),CMP());
	this->queue.pop_back();
}

template<class T,class Z, class CMP>
const lara_pair<T,Z>& pqueue<T,Z,CMP>::top()
{
	return this->queue.front();
}

template<class T, class Z, class CMP>
void pqueue<T,Z,CMP>::update(lara_pair<T,Z> t)
{
	auto it = this->queue.begin();
	while (it != this->queue.end())
	{
		if(it->id == t.id){ it->score = t.score; break;}
		++it;
	}
	std::make_heap(this->queue.begin(),this->queue.end(), CMP());
}

template<class T, class Z, class CMP>
void pqueue<T,Z,CMP>::remove(lara_pair<T,Z> t)
{
	auto it = this->queue.begin();
	while (it != this->queue.end())
	{
		if(it->id == t.id){ this->queue.erase(it); break;}
		++it;
	}
}

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
void LARA<T,Z>::findTopK2(uint64_t k,uint8_t qq, T *weights, uint8_t *attr)
{
	std::cout << this->algo << " find top-" << k << " (" << (int)qq << "D) ...";
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	//std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	pqueue<T,Z,laraq_desc<T,Z>> q(k);
	std::unordered_map<Z,lara_candidate<T>> lbmap;
	std::unordered_set<Z> c;

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

			T score = 0;
			for(uint8_t j = 0; j < qq; j++){ score+=this->cdata[id* this->d + attr[j]] * weights[attr[j]]; }

			if(c.find(id) == c.end())
			{
				c.insert(id);
				lara_pair<T,Z> p;
				p.id = id;
				p.score = score;
				if(q.size()<k)
				{
					q.push(p);
				}else if(q.top().score < score)
				{
					//std::cout<< q.top().score << "," << score <<std::endl;
					q.pop();
					q.push(p);
				}
			}
		}
	}

	T threshold = q.top().score;
	//while(!q.empty()){ this->res.push_back(q.top()); q.pop(); }
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void LARA<T,Z>::findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr)
{
	std::cout << this->algo << " find top-" << k << " (" << (int)qq << "D) ...";
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::string s = "";
	for(uint8_t m = 0; m <qq;m++){ s+=std::to_string(attr[m]); }
	pqueue<T,Z,laraq_desc<T,Z>> q(k);
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

			////////////////
			//Update score//
			auto it = lbmap.find(id);
			if (it == lbmap.end()){
				lara_candidate<T> lc;
				lc.score = 0;
				lc.seen = 0;
				it = lbmap.emplace(id,lc).first;
			}
			it->second.score+=weight * aa;//aggregate upper bound
			it->second.seen|=(1<<idx_a);//flag seen location

			lara_pair<T,Z> p;
			p.id = id;
			p.score = it->second.score;
			if(c.find(id) == c.end())//if object is not in Wk
			{
				if(q.size() < k){//insert if empty space in queue
					q.push(p);
					c.insert(id);
				}else if(q.top().score<p.score){//delete smallest element if current score is larger
					c.erase(q.top().id);
					c.insert(id);
					q.pop();
					q.push(p);
				}
			}else{
				q.update(p);//update score in queue and rebalance
			}
		}
		if(q.size() >= k && ((q.top().score) >= threshold) ){ break; }
	}
	if(STATS_EFF) this->tuple_count = lbmap.size();

	T trank[NUM_DIMS];//Initialize last ranks
	for(uint8_t m = 0; m < qq; m++)
	{
		uint8_t idx_a = attr[m];
		Z id = lists[idx_a][i-1].id;
		T aa = lists[idx_a][i-1].score;
		T weight = weights[idx_a];
		trank[idx_a] = weight * aa;
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

			auto it = lbmap.find(id);
			if(it != lbmap.end())
			{
				uint8_t seen = it->second.seen;
				T lbscore = it->second.score + trank[idx_a];

				it->second.seen |= (1 << idx_a);
				it->second.score = lbscore;
				lbarray[it->second.seen] = std::max(lbarray[it->second.seen], lbscore);

				lara_pair<T,Z> p;
				p.id = id;
				p.score = it->second.score;
				if(c.find(id) == c.end())//if object is not in Wk
				{
					if(q.size() < k){//insert if empty space in queue
						q.push(p);
						c.insert(id);
					}else if(q.top().score<p.score){//delete smallest element if current score is larger
						c.erase(q.top().id);
						c.insert(id);
						q.pop();
						q.push(p);
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
				}

				if(max_ub <= q.top().score){
					i = this->n;
					break;
				}
			}
		}
	}
	this->tt_processing += this->t.lap();

	T threshold = q.top().score;
//	while(!q.empty()){ this->res.push_back(q.top()); q.pop(); }
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " partial threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}


#endif
