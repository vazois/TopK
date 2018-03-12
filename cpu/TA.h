#ifndef TA_H
#define TA_H

#include "FA.h"
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "reorder_attr_cpu_c.h"


//TODO: what if only compute scores, until threshold and then gather results in priority queue//
//Precompute number of distinct items at each positional index
//Scan table in parallel without restructuring data
//For each distinct k look at precomputed values to determines threshold position

template<class T,class Z>
class TA : public AA<T,Z>{
	public:
		TA(uint64_t n,uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "TA";  this->capacity=0;
			this->alists = NULL;
		}

		~TA()
		{
			if(this->alists!=NULL)
			{
				for(uint64_t i = 0;i < this->d;i++){
					if(this->alists[i] != NULL)
					{
						free(this->alists[i]);
					}
				}
			}
		}

		void init();
		void findTopK(uint64_t k,uint8_t qq);
		void findTopKscalar(uint64_t k,uint8_t qq);

		std::vector<std::vector<pred<T,Z>>> lists;
		pred<T,Z> **alists;
	private:
		uint64_t capacity;
};

template<class T,class Z>
void TA<T,Z>::init(){
	this->lists.resize(this->d);
	for(int i =0;i<this->d;i++){ this->lists[i].resize(this->n); }
	for(uint64_t i=0;i<this->n;i++){
		for(uint8_t m =0;m<this->d;m++){
			this->lists[m].push_back(pred<T,Z>(i,this->cdata[i*this->d + m]));
		}
	}

	this->t.start();
	for(int i =0;i<this->d;i++){
		__gnu_parallel::sort(this->lists[i].begin(),this->lists[i].end(),cmp_max_pred<T,Z>);
		this->ax_bytes+= this->lists[i].capacity() * (sizeof(Z) + sizeof(T));
	}
	this->tt_init = this->t.lap();
	//this->ax_bytes+=(sizeof(Z) + sizeof(T)) * this->n * this->d;
}

template<class T,class Z>
void TA<T,Z>::findTopK(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topK (" << (int)qq << "D) ...";
	std::unordered_set<Z> eset;
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		T threshold=0;
		for(uint8_t j = 0; j < qq;j++){
			pred<T,Z> p = this->lists[j][i];
			threshold+=p.attr;

			if(eset.find(p.tid) == eset.end()){
				T score00 = 0;
				for(uint8_t m = 0; m < qq; m++){
					score00+=this->cdata[p.tid * this->d + m];
				}
				if(STATS_EFF) this->pred_count+=this->d;
				if(STATS_EFF) this->tuple_count+=1;
				eset.insert(p.tid);
				if(q.size() < k){//insert if empty space in queue
					q.push(tuple_<T,Z>(p.tid,score00));
				}else if(q.top().score<score00){//delete smallest element if current score is bigger
					q.pop();
					q.push(tuple_<T,Z>(p.tid,score00));
					if(STATS_EFF) this->pop_count+=1;
				}
			}
		}
		if(q.top().score >= threshold){
			//std::cout << "\nstopped at: " << i << ", threshold: " << threshold << std::endl;
			this->stop_pos = i;
			break;
		}
	}
	this->tt_processing += this->t.lap();

	T threshold = q.top().score;
	while(!q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T,class Z>
void TA<T,Z>::findTopKscalar(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topKscalar (" << (int)qq << "D) ...";
	std::unordered_set<Z> eset;
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		T threshold=0;
		for(uint8_t j = 0; j < qq;j++){
			pred<T,Z> p = this->lists[j][i];
			threshold+=p.attr;

			if(eset.find(p.tid) == eset.end()){
				T score00 = 0;
				for(uint8_t m = 0; m < qq; m++){
					score00+=this->cdata[p.tid * this->d + m];
				}
				if(STATS_EFF) this->pred_count+=this->d;
				if(STATS_EFF) this->tuple_count+=1;
				eset.insert(p.tid);
				if(q.size() < k){//insert if empty space in queue
					q.push(tuple_<T,Z>(p.tid,score00));
				}else if(q.top().score<score00){//delete smallest element if current score is bigger
					q.pop();
					q.push(tuple_<T,Z>(p.tid,score00));
					if(STATS_EFF) this->pop_count+=1;
				}
			}
		}
		if(q.top().score >= threshold){
			//std::cout << "\nstopped at: " << i << ", threshold: " << threshold << std::endl;
			this->stop_pos = i;
			break;
		}
	}
	this->tt_processing += this->t.lap();

	T threshold = q.top().score;
	while(!q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
