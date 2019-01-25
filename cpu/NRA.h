#ifndef NRA_H
#define NRA_H

#include "AA.h"
#include<unordered_map>

template<class T>
struct nra_pair
{
	uint8_t seen;
	T ub;
	T lb;
	nra_pair(T ub,T lb, uint8_t seen) : ub(ub) , lb(lb) , seen(seen){}
};

template<class T,class Z>
static bool nra_pair_descending(const qpair<T,Z> &a, const qpair<T,Z> &b){ return a.score > b.score; };

template<class T, class Z>
class NRA : public AA<T,Z>
{
	public:
		NRA(uint64_t n, uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "NRA";
			this->lists = NULL;
		};

		~NRA()
		{
			if(this->lists!=NULL){ for(uint8_t m = 0;m < this->d;m++){ if(this->lists[m] != NULL){ free(this->lists[m]); } } }
			free(this->lists);
		};

		void init();
		void findTopK(uint64_t k, uint8_t qq, T *weights, uint8_t *attr);

	private:
		qpair<T,Z> ** lists;
};

template<class T, class Z>
void NRA<T,Z>::init()
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
	for(uint8_t m = 0; m < this->d; m++) __gnu_parallel::sort(this->lists[m],lists[m]+this->n,nra_pair_descending<T,Z>);
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void NRA<T,Z>::findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr)
{
	std::cout << this->algo << " find top-" << k << " (" << (int)qq << "D) ...";
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(STATS_EFF) this->candidate_count=0;

	std::unordered_map<Z,nra_pair<T>> obj;
	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++)
	{
		for(uint8_t m = 0; m < qq; m++)
		{
			uint8_t idx_a = attr[m];//M{1}
			Z id = lists[idx_a][i].id;//M{1}
			T aa = lists[idx_a][i].score;//M{1}
			T weight = weights[idx_a];//M{1}
			if(STATS_EFF) this->accesses+=4;

			auto it = obj.find(id);//M{1}
			if(STATS_EFF) this->accesses+=2;
			if(it == obj.end())//if object is not seen before//M{1}
			{
				nra_pair<T> nrap(0,weight * aa,(1<<m));//upper bound = 0, initialize lower bound, set seen of attribute to 1
				obj.emplace(id,nrap);//insert object to map//M{1}
				if(STATS_EFF) this->accesses+=1;
			}else{//if object was seen
				it->second.lb += weight * aa;// update lower bound//M{1}
				it->second.seen |= (1 << m);// set seen object for attribute.//M{1}
				if(STATS_EFF) this->accesses+=2;
			}
		}

		T mx_ub = 0;
		std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> _q;
		for(auto it = obj.begin(); it!=obj.end(); ++it)//Update all upper bounds//M{2}
		{
			if(STATS_EFF) this->accesses+=2;
			nra_pair<T> *np = &(it->second);//M{1}
			T ub = np->lb;//M{1}
			if(STATS_EFF) this->accesses+=2;
			for(uint8_t m = 0; m < qq; m++)
			{
				if(((np->seen >> m) & 0x1) == 0x0)//if flag seen is not set
				{
					uint8_t idx_a = attr[m];//M{1}
					ub += lists[idx_a][i].score * weights[idx_a];// add upper bound score ////M{2}
					if(STATS_EFF) this->accesses+=3;
				}
			}
			np->ub = ub;// update upper bound//M{1}
			mx_ub = std::max(mx_ub,ub);//find maximum upper bound

			if(STATS_EFF) this->accesses+=1;
			if(_q.size() < k)//Update priority queue with lower bounds////M{1}
			{
				if(STATS_EFF) this->accesses+=3;
				_q.push(tuple_<T,Z>(it->first,np->lb));//M{3}
			}else if(_q.top().score < np->lb){
				_q.pop();//M{1}
				_q.push(tuple_<T,Z>(it->first,np->lb));//M{3}
				if(STATS_EFF) this->accesses+=4;
			}
		}
		_q.swap(q);//M{1}
		if(STATS_EFF) this->accesses+=3;
		if(q.size() == k && mx_ub<=q.top().score){//M{2}
			this->lvl = i;
			break;
		}// break if maximum upper bound smaller or equal to minimum lower bound
	}
	this->tt_processing += this->t.lap();
	if(STATS_EFF) this->tuple_count=obj.size();
	if(STATS_EFF) this->candidate_count=obj.size();

	//T threshold = q.top().score;
	T threshold = 0;
	for(uint8_t m = 0; m < qq; m++) threshold+=this->cdata[q.top().tid * this->d + attr[m]] * weights[attr[m]];//M{4}
	int i = 0;
	while(!q.empty())
	{
		tuple_<T,Z> t;
		t.tid=q.top().tid; //t.score=q.top().score;
		//std::cout << i << " : " << t.score << std::endl; i++;
		t.score=0; for(uint8_t m = 0; m < qq; m++) t.score+=this->cdata[t.tid * this->d + attr[m]] * weights[attr[m]];
		this->res.push_back(t);
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " partial threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
