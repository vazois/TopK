#ifndef GAA_H
#define GAA_H

#include "CudaHelper.h"
#include <inttypes.h>

#define SHARED_MEM_BYTES 49152
#define THREADS_PER_BLOCK 256
#define U32_BYTES_PER_TUPLE 8
#define U64_BYTES_PER_TUPLE 12

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
			this->threshold = 0;

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
			std::cout << "< Benchmark for " << this->algo << " algorithm >" << std::endl;
			std::cout << "tt_init: " << this->tt_init << std::endl;
			std::cout << "tt_procesing: " << this->tt_processing/this->iter << std::endl;
			std::cout << "tuples_per_second: " << (this->tt_processing == 0 ? 0 : WORKLOAD/(this->tt_processing/1000)) << std::endl;
			std::cout << "tuple_count: " << this->tuple_count << std::endl;
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
		T threshold;

		uint64_t iter;//experiment count
		double tt_init;
		double tt_processing;
		uint64_t pred_count;//count predicate evaluations
		uint64_t tuple_count;//count predicate evaluations
		uint64_t queries_per_second;
};


#endif
