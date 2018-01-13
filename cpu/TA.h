#ifndef TA_H
#define TA_H

#include "FA.h"
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>

#define alpha 128

#include "reorder_attr_cpu.h"

template<class T,class Z>
class PQComparison{
	public:
		PQComparison(){};

		bool operator() (const tuple<T,Z>& lhs, const tuple<T,Z>& rhs) const{
			return (lhs.score>rhs.score);
		}
};

template<class T,class Z>
class TA : public FA<T,Z>{
	public:
		TA(uint64_t n,uint64_t d) : FA<T,Z>(n,d){ this->algo = "TA"; };

		void findTopK(uint64_t k);
	private:
		std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
		void seq_topk(uint64_t k);
		void seq_topk2(uint64_t k);
		void par_topk(uint64_t k);
};

template<class T,class Z>
void TA<T,Z>::seq_topk(uint64_t k){
	std::unordered_set<Z> tids_set;
	std::unordered_map<Z,T> tmap;

	T threshold=0;
	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		threshold=0;
		for(uint8_t j = 0; j < this->d;j++){
			pred<T,Z> p = this->lists[j][i];
			threshold+=p.attr;

			T score = 0;
			if(tmap.find(p.tid) == tmap.end()){// Only if we do not want to re-evaluate the score for tuples re-appearing in the lists
				for(uint8_t k = 0; k < this->d; k++){
					score+=this->cdata[p.tid * this->d + k];
				}
				if(STATS_EFF) this->pred_count+=this->d;
				if(STATS_EFF) this->tuple_count+=1;
				tmap.insert(std::pair<T,Z>(p.tid,score));
			}else{
				score = tmap[p.tid];
			}

			if(tids_set.find(p.tid) == tids_set.end()){//if does not exist in set / if tuple has not been evaluated yet
				if(this->q.size() < k){//insert if space in queue
					this->q.push(tuple<T,Z>(p.tid,score));
					tids_set.insert(p.tid);
				}else if(this->q.top().score<score){//delete smallest element if current score is bigger
					tids_set.erase(tids_set.find(q.top().tid));
					this->q.pop();
					this->q.push(tuple<T,Z>(p.tid,score));
					tids_set.insert(p.tid);
				}
			}
		}
		if(this->q.top().score >= threshold){
			//std::cout << "stopped at: " << i << std::endl;
			break;
		}
	}
	this->tt_processing = this->t.lap();
}

template<class T,class Z>
void TA<T,Z>::seq_topk2(uint64_t k){
	std::unordered_map<Z,T> emap;

	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		T threshold=0;
		for(uint8_t j = 0; j < this->d;j++){
			pred<T,Z> p = this->lists[j][i];
			threshold+=p.attr;

			if(emap.find(p.tid) == emap.end()){
				T score = 0;
				for(uint8_t k = 0; k < this->d; k++){
					score+=this->cdata[p.tid * this->d + k];
				}
				if(STATS_EFF) this->pred_count+=this->d;
				if(STATS_EFF) this->tuple_count+=1;
				emap.insert(std::pair<T,Z>(p.tid,score));
				if(this->q.size() < k){//insert if empty space in queue
					this->q.push(tuple<T,Z>(p.tid,score));
				}else if(this->q.top().score<score){//delete smallest element if current score is bigger
					this->q.pop();
					this->q.push(tuple<T,Z>(p.tid,score));
				}
			}
		}
		if(this->q.top().score >= threshold){
			std::cout << "stopped at: " << i << std::endl;
			break;
		}
	}
	this->tt_processing = this->t.lap();

}

template<class T,class Z>
void TA<T,Z>::par_topk(uint64_t k){
	//Keep track seen tupples
	//Discard irrelevant tupples
	std::vector<std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>>> qs;
	std::vector<T> gattr;
	std::vector<T> stats;
	gattr.resize(this->d);
	omp_set_num_threads(this->d);
	qs.resize(this->d);

	if(STATS_EFF){//Gather statistics
		stats.resize(this->d);
		for(uint8_t m = 0; m < this->d; m++) stats[m]=0;
	}

	for(uint64_t i = 0;i < this->n;i+=alpha){
		#pragma omp parallel
		{
			uint32_t tid = omp_get_thread_num();
			pred<T,Z> p;
			for(uint64_t j = i; j < i + alpha; j++){
				p = this->lists[tid][j];//get tuple in my list
				T score = 0;
				for(uint8_t m = 0; m < this->d; m++){
					score+=this->cdata[p.tid * this->d + m];//calculate score for tuple
				}

				if(STATS_EFF){
					stats[tid]++;
				}

				//Try pushing tuple into my priority queue
				if(qs[tid].size() < k){//If it is not full
					qs[tid].push(tuple<T,Z>(p.tid,score));// Push tuple
				}else if(qs[tid].top().score<score){//If full and top score smaller than this score
					qs[tid].pop();
					qs[tid].push(tuple<T,Z>(p.tid,score));
				}
			}
			gattr[tid] = p.attr;
		}

		//calculate threshold for stopping
		T threshold = 0;
		T min_score = qs[0].top().score;
		for(uint8_t m = 0; m < this->d; m++){
			threshold+=gattr[m];
			min_score = MIN(min_score,qs[m].top().score);
		}

		//if top score in all queues greater or equal than threshold stop
		if(min_score >= threshold){
			//std::cout << "stopped at: " << i + alpha << std::endl;
			break;
		}
	}

	//Gather k-best tupples
	std::unordered_set<Z> eset;
	for(uint8_t m = 0; m < this->d; m++){
		if(STATS_EFF){
			this->tuple_count+=stats[m];
		}
		while(!qs[m].empty()){
			Z tid = qs[m].top().tid;
			T score = qs[m].top().score;
			if(eset.find(tid) == eset.end()){
				if(this->q.size() < k){
					this->q.push(qs[m].top());
				}else if(this->q.top().score < score){
					this->q.pop();
					this->q.push(qs[m].top());
				}
				eset.insert(tid);
			}
			qs[m].pop();
		}
	}

	if(STATS_EFF){
		this->pred_count=this->tuple_count * this->d;
	}

	//std::cout << "q.size: " << this->q.size() << std::endl;
}


template<class T,class Z>
void TA<T,Z>::findTopK(uint64_t k){
	//Note: keep truck of ids so you will not re-insert the same tupple as your process them in order
	std::cout << this->algo << " find topK ...";

	this->t.start();
	if(this->topkp){
		this->par_topk(k);
	}else{
		//this->seq_topk2(k);
		this->par_topk(k);
	}
	this->tt_processing = this->t.lap("");

	//std::cout << "q_size(2): " << q.size() << std::endl;

	//std::cout << std::endl;
	while(!this->q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(this->q.top());
		this->q.pop();
	}
	std::cout << " (" << this->res.size() << ")" << std::endl;
}
#endif
