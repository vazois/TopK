#ifndef GAA_H
#define GAA_H

#include "CudaHelper.h"
#include <inttypes.h>
#include <vector>
#include <queue>
#include <algorithm>

#define SHARED_MEM_BYTES 49152
#define THREADS_PER_BLOCK 256
#define U32_BYTES_PER_TUPLE 8
#define U64_BYTES_PER_TUPLE 12

#define MAX_ATTRIBUTES 8

__constant__ float gpu_weights[MAX_ATTRIBUTES];
__constant__ uint32_t gpu_query[MAX_ATTRIBUTES];

/*
 * Tuple structure
 */
template<class T,class Z>
struct ranked_tuple{
	ranked_tuple(){ tid = 0; score = 0; }
	ranked_tuple(Z t, T s){ tid = t; score = s; }
	Z tid;
	T score;
};

template<class T,class Z>
class MaxFirst{
	public:
		MaxFirst(){};

		bool operator() (const ranked_tuple<T,Z>& lhs, const ranked_tuple<T,Z>& rhs) const{
			return (lhs.score>rhs.score);
		}
};

template<class T, class Z>
static T find_threshold(T *cscores, uint64_t n, uint64_t k){
	std::priority_queue<T, std::vector<ranked_tuple<T,Z>>, MaxFirst<T,Z>> q;
	for(uint64_t i = 0; i < n; i++){
		if(q.size() < k){
			q.push(ranked_tuple<T,Z>(i,cscores[i]));
		}else if ( q.top().score < cscores[i] ){
			q.pop();
			q.push(ranked_tuple<T,Z>(i,cscores[i]));
		}
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "threshold: " << q.top().score << std::endl;
	return q.top().score;
}

template<class T, class Z>
static T find_partial_threshold(T *cscores, uint64_t n, uint64_t k){
	std::priority_queue<T, std::vector<ranked_tuple<T,Z>>, MaxFirst<T,Z>> q;
//	for(uint64_t j = 0; j < n; j+=(k*2)){
//		for(uint64_t i = 0; i < k;i++){
//			if(q.size() < k){
//				q.push(ranked_tuple<T,Z>(i+j,cscores[i+j]));
//			}else if ( q.top().score < cscores[i+j] ){
//				q.pop();
//				q.push(ranked_tuple<T,Z>(i+j,cscores[i+j]));
//			}
//		}
//	}

	for(uint64_t j = 0; j < n; j+=4096){
		for(uint64_t i = 0; i < 2048;i++){
			if(q.size() < k){
				q.push(ranked_tuple<T,Z>(i+j,cscores[i+j]));
			}else if ( q.top().score < cscores[i+j] ){
				q.pop();
				q.push(ranked_tuple<T,Z>(i+j,cscores[i+j]));
			}
		}
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "partial threshold: " << q.top().score << std::endl;
	return q.top().score;
}

template<class T, class Z>
class GAA{
	public:
		GAA<T,Z>(uint64_t n, uint64_t d){
			this->n = n;
			this->d = d;
			this->cdata = NULL;
			this->gdata = NULL;
			this->cids = NULL;
			this->gids = NULL;
			this->cscores = NULL;
			this->gscores = NULL;
			this->cpu_threshold = 0;
			this->gpu_threshold = 0;

			this->iter = 1;
			this->tt_init = 0;
			this->tt_processing = 0;
			this->pred_count = 0;
			this->tuple_count = 0;
			this->queries_per_second = 0;
		};

		~GAA<T,Z>(){
			if(this->cdata != NULL) cudaFreeHost(this->cdata);
			if(this->gdata != NULL) cudaFree(this->gdata);
			if(this->cids != NULL) cudaFreeHost(this->cids);
			if(this->gids != NULL) cudaFree(this->gids);
			if(this->cscores != NULL) cudaFreeHost(this->cscores);
			if(this->gscores != NULL) cudaFree(this->gscores);
		};

		std::string get_algo(){ return this->algo; }
		T get_thres(){return this->threshold;}
		T*& get_cdata(){ return this->cdata; }
		void set_cdata(T *cdata){ this->cdata = cdata; }
		T*& get_gdata(){ return this->gdata; }
		void set_gdata(T *gdata){ this->gdata = gdata; }

		void set_iter(uint64_t iter){ this->iter = iter; }

		void benchmark(){
			std::cout << std::fixed << std::setprecision(4);
			std::cout << "< Benchmark for " << this->algo << " algorithm >" << std::endl;
			std::cout << "tt_init: " << this->tt_init << std::endl;
			std::cout << "tt_procesing: " << this->tt_processing/this->iter << std::endl;
			std::cout << "tuples_per_second: " << (this->tt_processing == 0 ? 0 : WORKLOAD/(this->tt_processing/1000)) << std::endl;
			std::cout << "tuple_count: " << this->tuple_count << std::endl;
			std::cout << "cpu_threshold: " << this->cpu_threshold << std::endl;
			std::cout << "gpu_threshold: " << this->gpu_threshold << std::endl;
			std::cout << "< ---------------------------------------------- >" << std::endl;
			this->reset_stats();
		}

		void reset_stats(){
			this->tt_init = 0;
			this->tt_processing = 0;

			this->pred_count = 0;
			this->tuple_count = 0;
			this->queries_per_second = 0;
		}

	protected:
		std::string algo = "default";
		uint64_t n,d;// cardinality,dimensinality
		T *cdata;
		T *gdata;
		Z *cids;
		Z *gids;
		T *cscores;
		T *gscores;
		T cpu_threshold;
		T gpu_threshold;

		uint64_t iter;//experiment count
		double tt_init;
		double tt_processing;
		uint64_t pred_count;//count predicate evaluations
		uint64_t tuple_count;//count predicate evaluations
		uint64_t queries_per_second;

		Time<msecs> t;
};


#endif
