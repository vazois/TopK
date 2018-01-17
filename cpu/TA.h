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
		std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> qs[THREADS];
		Z t_offset[THREADS];

		void seq_topk(uint64_t k);
		//void seq_topk2(uint64_t k);
		void par_topk(uint64_t k);
		void par_topk2(uint64_t k);
		void par_topk3(uint64_t k);

		void rrorder();
};

template<class T,class Z>
void TA<T,Z>::rrorder(){
	this->rrlist.resize(this->n);
	std::unordered_set<Z> ids;
	Z m = 0;
	for(uint64_t i=0;i<this->n;i++){
		for(int j =0;j<this->d;j++){
			pred<T,Z> p = this->lists[j][i];
			if(ids.find(p.tid) == ids.end()){
//				if(p.tid == 360774){
//					std::cout<< "position: " <<m << " , " << p.tid << std::endl;
//				}
				ids.insert(p.tid);
				this->rrlist[m] = rrpred<Z>(p.tid,i);
				m++;
			}
		}
	}
}

template<class T,class Z>
void TA<T,Z>::seq_topk(uint64_t k){
	std::unordered_map<Z,T> emap;
	std::unordered_set<Z> eset;

	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		T threshold=0;
		for(uint8_t j = 0; j < this->d;j++){
			pred<T,Z> p = this->lists[j][i];
			threshold+=p.attr;

			//if(emap.find(p.tid) == emap.end()){
			if(eset.find(p.tid) == eset.end()){
				T score = 0;
				for(uint8_t k = 0; k < this->d; k++){
					score+=this->cdata[p.tid * this->d + k];
				}
				if(STATS_EFF) this->pred_count+=this->d;
				if(STATS_EFF) this->tuple_count+=1;
				//emap.insert(std::pair<T,Z>(p.tid,score));
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

}

template<class T,class Z>
void TA<T,Z>::par_topk(uint64_t k){
	//Keep track seen tuples
	//Discard irrelevant tuples
	std::vector<T> gattr;
	std::vector<T> stats;
	gattr.resize(this->d);
	omp_set_num_threads(this->d);

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
				if(this->qs[tid].size() < k){//If it is not full
					this->qs[tid].push(tuple<T,Z>(p.tid,score));// Push tuple
				}else if(this->qs[tid].top().score<score){//If full and top score smaller than this score
					this->qs[tid].pop();
					this->qs[tid].push(tuple<T,Z>(p.tid,score));
				}
			}
			gattr[tid] = p.attr;
		}

		//calculate threshold for stopping
		T threshold = 0;
		T min_score = this->qs[0].top().score;
		for(uint8_t m = 0; m < this->d; m++){
			threshold+=gattr[m];
			min_score = MIN(min_score,this->qs[m].top().score);
		}

		//if top score in all queues greater or equal than threshold stop
		if(min_score >= threshold){
			std::cout << "stopped at: " << i + alpha << std::endl;
			break;
		}
	}

	//Gather k-best tupples
//	std::unordered_set<Z> eset;
//	for(uint8_t m = 0; m < this->d; m++){
//		if(STATS_EFF){
//			this->tuple_count+=stats[m];
//		}
//		while(!qs[m].empty()){
//			Z tid = qs[m].top().tid;
//			T score = qs[m].top().score;
//			if(eset.find(tid) == eset.end()){
//				if(this->q.size() < k){
//					this->q.push(qs[m].top());
//				}else if(this->q.top().score < score){
//					this->q.pop();
//					this->q.push(qs[m].top());
//				}
//				eset.insert(tid);
//			}
//			qs[m].pop();
//		}
//	}

	if(STATS_EFF){
		for(uint8_t m = 0; m < this->d; m++){
			if(STATS_EFF){
				this->tuple_count+=stats[m];
			}
		}
	}
	if(STATS_EFF){
		this->pred_count=this->tuple_count * this->d;
	}

	//std::cout << "q.size: " << this->q.size() << std::endl;
}

template<class T,class Z>
void TA<T,Z>::par_topk2(uint64_t k){
	this->rrorder();
	omp_set_num_threads(THREADS);

	Z step = alpha;
//	std::cout << "step: " << step << std::endl;
	for(uint64_t i = 0;i < this->n;i+=step){
		#pragma omp parallel
		{
			uint32_t tid = omp_get_thread_num();
			uint32_t gsize = THREADS;
			Z tuple_id;
			for(uint64_t j = i; j < i + step; j+=gsize){
				rrpred<Z> rrp = this->rrlist[j+tid];

//				if(rrp.tid == 30675){
//					std::string msg = "FOUND IT: "+std::to_string(tid)+" = "+std::to_string(rrp.tid)+"\n";
//					std::cout << msg;
//				}
				T score = 0;
				for(uint8_t m = 0; m < this->d; m++){
					score+=this->cdata[rrp.tid * this->d + m];//calculate score for tuple
				}

				if(this->qs[tid].size() < k){//If it is not full
					this->qs[tid].push(tuple<T,Z>(rrp.tid,score));// Push tuple
				}else if(this->qs[tid].top().score<score){//If full and top score smaller than this score
					this->qs[tid].pop();
					this->qs[tid].push(tuple<T,Z>(rrp.tid,score));
				}

				t_offset[tid]=rrp.offset;
			}
		}

		Z offset = t_offset[0];
		T threshold = 0;
		T score = this->qs[0].top().score;
		for(uint8_t m = 0; m < THREADS; m++){
//			threshold+= this->lists[m][i+step].attr;
			offset = MIN(t_offset[m],offset);
			score = MIN(score,this->qs[m].top().score);
		}
		for(uint8_t m = 0; m < this->d; m++){
			threshold+= this->lists[m][offset].attr;
		}

		if(score >= threshold){
//			std::cout << "stopped at: " << i + step << ", threshold: " << threshold << std::endl;
			break;
		}
	}
}

template<class T,class Z>
void TA<T,Z>::par_topk3(uint64_t k){
	this->rrorder();
	T tt[THREADS];
	Time<msecs> tds[THREADS];
	omp_set_num_threads(THREADS);
	for(uint8_t m = 0; m < THREADS; m++){
		tt[m]=0.0f;
	}
	Time<msecs> f;
	T ff=0.0f;
	Z count=0;
	for(uint64_t i = 0;i < this->n;i+=alpha){

		f.start();
		#pragma omp parallel
		{
			uint32_t tid = omp_get_thread_num();

			tds[tid].start();
			for(uint64_t j = i ; j < i + alpha; j+=THREADS){
				rrpred<Z> rrp = this->rrlist[j+tid];
				T score = 0;
				for(uint8_t m = 0; m < this->d; m++){
					score+=this->cdata[rrp.tid * this->d + m];//calculate score for tuple
				}

				#pragma omp critical
				{

					if(this->q.size() < k){
						this->q.push(tuple<T,Z>(rrp.tid,score));
					}else if(this->q.top().score < score){
						count++;
						this->q.pop();
						this->q.push(tuple<T,Z>(rrp.tid,score));
					}
				}
			}
			tt[tid]+=tds[tid].lap();
		}
		ff+=f.lap("");

		//std::cout << "i:" << i<< std::endl;
		T threshold = 0;
		rrpred<Z> rrp = this->rrlist[i+alpha];
		for(uint8_t m = 0; m < this->d; m++){
			threshold+= this->lists[m][rrp.offset].attr;
		}
		if(this->q.top().score >= threshold){
			std::cout << "stopped at: " << i + alpha << ", threshold: " << threshold << std::endl;
			break;
		}
	}

	for(uint32_t m = 0; m < THREADS; m++){
		std::cout <<"(" << m << ") : " << tt[m] << " count: " << count <<std::endl;
	}
	std::cout << "ff: " << ff <<std::endl;
}

template<class T,class Z>
void TA<T,Z>::findTopK(uint64_t k){
	//Note: keep truck of ids so you will not re-insert the same tupple as your process them in order
	std::cout << this->algo << " find topK ...";

	this->t.start();
	if(this->topkp){
		this->par_topk3(k);
	}else{
		this->seq_topk(k);
	}
	this->tt_processing = this->t.lap("");

	//Gather results for comparison//
	if(this->topkp){
		std::unordered_set<Z> eset;
		for(uint8_t m = 0; m < this->d; m++){
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
	}

	while(!this->q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(this->q.top());
		this->q.pop();
	}
	std::cout << " (" << this->res.size() << ")" << std::endl;
}
#endif
