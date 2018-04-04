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
		void findTopKthreads(uint64_t k,uint8_t qq);

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

		if(q.size() >= k && ((q.top().score) >= threshold) ){
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << threshold << std::endl;
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
void TA<T,Z>::findTopKthreads(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topKsimd (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	uint32_t threads = THREADS;
	uint32_t threads_shf = log2((float)threads);
//	for(uint8_t m = 0; m <threads; m++){
//		uint32_t gid = m % qq;//GROUP
//		uint32_t lid = m / qq;//LOCAL
//		uint32_t tig = (gid+1)*((THREADS-1)/qq + 1);
//		tig = tig > THREADS ? tig - THREADS : tig - (gid)*(THREADS/qq);
//		printf("(%d,%d,%d,%d)\n",m, gid,lid,tig);
//	}
	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q[threads];
	omp_set_num_threads(threads);
	this->t.start();
#pragma omp parallel
{
	uint32_t thread_id = omp_get_thread_num();
	uint32_t gid = thread_id % qq;//attribute list assignment
	uint32_t lid = thread_id / qq;//tuple range assignment
	uint32_t tig = (gid+1)*((THREADS-1)/qq + 1);
	tig = tig > THREADS ? tig - THREADS : tig - (gid)*(THREADS/qq);
	for(uint64_t i = 0; i < this->n; i++){
		Z id = this->lists[gid][i].tid;
		//if( (id % threads)  == lid ){
		if((((uint64_t) id * (uint64_t) threads) >> threads_shf) == lid){



		}
	}
	this->tt_processing += this->t.lap();
}

	T threshold = 1313;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << 0 << ")" << std::endl;
	this->threshold = threshold;
}

//template<class T, class Z>
//void TA<T,Z>::findTopKthreads(uint64_t k, uint8_t qq){
//	std::cout << this->algo << " find topKsimd (" << (int)qq << "D) ...";
//	if(STATS_EFF) this->tuple_count = 0;
//	if(STATS_EFF) this->pop_count=0;
//	if(this->res.size() > 0) this->res.clear();
//
//	omp_set_num_threads(THREADS);
//	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q[THREADS];
//	Z *pos = (Z*)malloc(sizeof(Z)*this->d);
//	for(uint8_t m = 0; m < this->d; m++) pos[m] = 0;
//	this->t.start();
//
//	std::cout << std::endl;
//	for(uint8_t m = 0; m <THREADS; m++){
//		uint32_t gid = m % qq;
//		uint32_t lid = m / qq;
//		uint32_t tig = (gid+1)*((THREADS-1)/qq + 1);
//		tig = tig > THREADS ? tig - THREADS : tig - (gid)*(THREADS/qq);
//		printf("(%d,%d,%d,%d)\n",m, gid,lid,tig);
//	}
//
//#pragma omp parallel
//{
//	uint32_t thread_id = omp_get_thread_num();
//	uint32_t gid = thread_id % qq;
//	uint32_t lid = thread_id / qq;
//	uint32_t tig = (gid+1)*(THREADS/qq);
//	tig = tig > THREADS ? tig - THREADS : tig - (gid)*(THREADS/qq);
//
//	for(uint64_t i = 0; i < this->n; i++){
//		Z id = list_id[i*this->d + gid];
//
//		if( (id % THREADS)  == lid ){
//			T score = 0;
//			for(uint8_t p = 0; p < qq; p++){
//				score+=this->cdata[id*this->d + p];
//			}
//			if(q[thread_id].size() < k){//insert if empty space in queue
//				q[thread_id].push(tuple_<T,Z>(id,score));
//			}else if(q[thread_id].top().score<score){//delete smallest element if current score is bigger
//				q[thread_id].pop();
//				q[thread_id].push(tuple_<T,Z>(id,score));
//			}
//			//pos[gid] = std::max(pos[gid],i);
//		}
//		T threshold = 0;
//		for(uint8_t p = 0; p < qq; p++){ threshold += list_score[i*this->d + p]; }
//	}
//}
//
//	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> _q;
//	for(uint32_t m = 0 ; m < THREADS; m++){
//		while(!q[m].empty()){
//			if(_q.size() < k){
//				_q.push(q[m].top());
//			}else if( _q.top().score < q[m].top().score ){
//				_q.pop();
//				_q.push(q[m].top());
//			}
//			q[m].pop();
//		}
//	}
//
//	this->tt_processing += this->t.lap();
//	free(pos);
//
//	T threshold = _q.top().score;
//	std::cout << std::fixed << std::setprecision(4);
//	std::cout << " threshold=[" << threshold <<"] (" << _q.size() << ")" << std::endl;
//	this->threshold = threshold;
//}
#endif
