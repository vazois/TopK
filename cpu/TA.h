#ifndef TA_H
#define TA_H

#include "FA.h"
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>


#include "reorder_attr_cpu_c.h"

template<class T,class Z>
class PQComparison{
	public:
		PQComparison(){};

		bool operator() (const tuple<T,Z>& lhs, const tuple<T,Z>& rhs) const{
			return (lhs.score>rhs.score);
		}
};

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
		void init2();
		void findTopK(uint64_t k);
		void findTopK2(uint64_t k);

		std::vector<std::vector<pred<T,Z>>> lists;
		pred<T,Z> **alists;
	private:
		std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
		uint64_t capacity;
};

template<class T, class Z>
void TA<T,Z>::init2(){
	this->alists = (pred<T,Z> **)malloc(sizeof(pred<T,Z>*) * this->d);
	for(uint64_t i = 0;i < this->d;i++){
		this->alists[i] = (pred<T,Z>*)malloc(sizeof(pred<T,Z>) * this->n);
	}
	for(uint64_t i=0;i<this->n;i++){
		for(uint8_t m =0;m<this->d;m++){
			this->alists[m][i] = pred<T,Z>(i,this->cdata[i*this->d + m]);
		}
	}

	this->t.start();
	for(uint8_t m =0;m<this->d;m++){
		std::sort(this->alists[m],this->alists[m] + this->n,cmp_max_pred<T,Z>);
	}

//	for(int i =0;i<this->d;i++){
//		__gnu_parallel::sort(this->alists[i],this->alists[i] + this->n,cmp_max_pred<T,Z>);
//		this->ax_bytes+= this->lists[i].capacity() * (sizeof(Z) + sizeof(T));
//	}
	this->tt_init = this->t.lap();
}

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
	for(uint8_t m =0;m<this->d;m++){
		std::sort(this->lists[m].begin(),this->lists[m].end(),cmp_max_pred<T,Z>);
		this->ax_bytes+= this->lists[m].capacity() * (sizeof(Z) + sizeof(T));
	}
//	for(int i =0;i<this->d;i++){
//		__gnu_parallel::sort(this->lists[i].begin(),this->lists[i].end(),cmp_max_pred<T,Z>);
//		this->ax_bytes+= this->lists[i].capacity() * (sizeof(Z) + sizeof(T));
//	}
	this->tt_init = this->t.lap();
	//this->ax_bytes+=(sizeof(Z) + sizeof(T)) * this->n * this->d;
}

template<class T,class Z>
void TA<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	std::unordered_set<Z> eset;

	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		T threshold=0;
		for(uint8_t j = 0; j < this->d;j++){
			pred<T,Z> p = this->lists[j][i];
			threshold+=p.attr;

			if(eset.find(p.tid) == eset.end()){
				T score = 0;
				for(uint8_t m = 0; m < this->d; m++){
					score+=this->cdata[p.tid * this->d + m];
				}
				if(STATS_EFF) this->pred_count+=this->d;
				if(STATS_EFF) this->tuple_count+=1;
				eset.insert(p.tid);
				if(this->q.size() < k){//insert if empty space in queue
					this->q.push(tuple<T,Z>(p.tid,score));
				}else if(this->q.top().score<score){//delete smallest element if current score is bigger
					this->q.pop();
					this->q.push(tuple<T,Z>(p.tid,score));
				}
			}
		}
		if(this->q.top().score >= threshold){
//			std::cout << "stopped at: " << i << ", threshold: " << threshold << std::endl;
			break;
		}
	}
	this->tt_processing = this->t.lap();

	while(!this->q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(this->q.top());
		this->q.pop();
	}
	std::cout << " (" << this->res.size() << ")" << std::endl;
}

template<class T,class Z>
void TA<T,Z>::findTopK2(uint64_t k){
	std::cout << this->algo << " find topK ...";
	std::unordered_set<Z> eset;

	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		T threshold=0;
		for(uint8_t j = 0; j < this->d;j++){
			pred<T,Z> p = this->alists[j][i];
			threshold+=p.attr;

			if(eset.find(p.tid) == eset.end()){
				T score = 0;
				for(uint8_t m = 0; m < this->d; m++){
					score+=this->cdata[p.tid * this->d + m];
				}
				if(STATS_EFF) this->pred_count+=this->d;
				if(STATS_EFF) this->tuple_count+=1;
				eset.insert(p.tid);
				if(this->q.size() < k){//insert if empty space in queue
					this->q.push(tuple<T,Z>(p.tid,score));
				}else if(this->q.top().score<score){//delete smallest element if current score is bigger
					this->q.pop();
					this->q.push(tuple<T,Z>(p.tid,score));
				}
			}
		}
		if(this->q.top().score >= threshold){
//			std::cout << "stopped at: " << i << ", threshold: " << threshold << std::endl;
			break;
		}
	}
	this->tt_processing = this->t.lap();

	while(!this->q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(this->q.top());
		this->q.pop();
	}
	std::cout << " (" << this->res.size() << ")" << std::endl;
}
#endif
