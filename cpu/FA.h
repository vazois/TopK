#ifndef FA_H
#define FA_H

#include "AA.h"
#include <unordered_map>
#include <unordered_set>
#include <list>

template<class Z>
struct rrpred{
	rrpred(){ tid = 0; offset = 0; }
	rrpred(Z t, Z o){ tid = t; offset = o; }
	Z tid;
	Z offset;
};

/*
 * Simple implementation of Fagin's algorithm
 */
template<class T,class Z>
class FA : public AA<T,Z>{
	public:
		FA(uint64_t n,uint64_t d) : AA<T,Z>(n,d){ this->algo = "FA"; };

		void init();
		void findTopK(uint64_t k);

	protected:
		std::vector<std::vector<pred<T,Z>>> lists;
		std::vector<rrpred<Z>> rrlist;

	private:
		void seq_topk(uint64_t k);
		void par_topk(uint64_t k);

		void seq_init();
		void par_init();
};

template<class T,class Z>
void FA<T,Z>::seq_init(){
	for(uint64_t i=0;i<this->n;i++){
		for(int j =0;j<this->d;j++){
			this->lists[j].push_back(pred<T,Z>(i,this->cdata[i*this->d + j]));
		}
	}
	for(int i =0;i<this->d;i++){ std::sort(this->lists[i].begin(),this->lists[i].end(),cmp_max_pred<T,Z>);}
}

template<class T,class Z>
void FA<T,Z>::par_init(){

	for(uint64_t i=0;i<this->n;i++){
		for(int j =0;j<this->d;j++){
			this->lists[j].push_back(pred<T,Z>(i,this->cdata[i*this->d + j]));
		}
	}

	for(int i =0;i<this->d;i++){
		__gnu_parallel::sort(this->lists[i].begin(),this->lists[i].end(),cmp_max_pred<T,Z>);
	}
}


/*
 * Create m lists and sort them
 */
template<class T,class Z>
void FA<T,Z>::init(){
	this->lists.resize(this->d);
	for(int i =0;i<this->d;i++){ this->lists[i].resize(this->n); }

	this->t.start();
	if(this->initp){
		this->par_init();
	}else{
		this->seq_init();
	}
	this->tt_init = this->t.lap();
}

template<class T,class Z>
void FA<T,Z>::seq_topk(uint64_t k){
	std::unordered_map<Z,uint8_t> tmap;
	uint64_t stop=0;

	//Iterate Lists
	for(uint64_t i = 0; i < this->n;i++){
		for(uint64_t j = 0; j < this->d;j++){
			pred<T,Z> p = this->lists[j][i];
			if ( tmap.find(p.tid) == tmap.end() ){
				tmap.insert(std::pair<Z,uint8_t>(p.tid,1));
			}else{
				tmap[p.tid]++;
				if( tmap[p.tid] == this->d ) stop++;
			}
		}
		if(stop >= k){
			//std::cout << "Stopped at: " << i << std::endl;
			break;
		}
	}

	//Gather results and evaluate scores
	std::vector<tuple<T,Z>> res;
	for(typename std::unordered_map<Z,uint8_t>::iterator it = tmap.begin(); it!=tmap.end(); ++it){
		uint64_t tid = it->first;
		T score = 0;
		for(uint64_t j = 0; j < this->d; j++){ score+= this->cdata[tid * this->d + j]; }
		res.push_back(tuple<T,Z>(it->first,score));
		this->pred_count+=this->d;
		this->tuple_count+=1;
	}
	std::sort(res.begin(),res.end(),cmp_score<T,Z>);
	for(uint64_t i = 0;i < k ;i++){ this->res.push_back(res[i]); }
}

template<class T,class Z>
void FA<T,Z>::par_topk(uint64_t k){
	uint32_t *bvec = (uint32_t *) malloc(sizeof(uint32_t) * this->d * (this->n / 32));//vectors tracking seen tupples
	uint32_t *gcount = (uint32_t *) malloc(sizeof(uint32_t) * this->d );//global count array to stop processing lists

	omp_set_num_threads(this->d);
	#pragma omp parallel//1
	{
		uint32_t tid = omp_get_thread_num();
		for(uint64_t i = 0;i<this->n/32; i++){ bvec[tid + i*this->d] = 0; } //initialize vectors to zero
	}

	for(uint64_t i = 0; i < this->n;i+=32){//for each block of ids
		#pragma omp parallel//2
		{
			uint32_t tid = omp_get_thread_num();
			uint32_t gid = omp_get_num_threads();

			for(uint64_t j = i; j < i+32;j++){ // process block of ids
				pred<T,Z> p = this->lists[tid][j];

				uint32_t offset = p.tid >> 5;//divide by 32 // find offset for id
				uint32_t bit = p.tid % 32;// p.tid & 31 // find bit for id

				bvec[tid + offset *this->d] ^= (0x1 << bit); // set bit to indicate id seen
			}

			#pragma omp barrier

			gcount[tid] = 0;//global count value
			for(uint64_t j = tid;j<this->n/32; j+=gid){//gather bit vectors
				uint32_t vec = 0xFFFFFFFF;
				for(uint32_t k = 0; k < this->d ;k++){
					vec = vec & (bvec[k + j*this->d]);
				}
				gcount[tid]+=__builtin_popcount(vec);//count population of bit vector from all lists
			}
		}
		uint32_t count = 0;
		for(uint64_t j = 0; j < this->d ; j++){ count+=gcount[j]; }//merge count from all threads
		if(count >= k){//check if elligible to stop
			std::cout << "Stopped at: " << i+32 << std::endl;
			break;
		}
	}


	//Can use more threads for evaluation
	//Not limited to number of attributes
	std::vector<std::list<tuple<T,Z>>> res;
	//std::vector<tuple<T>> res;
	res.resize(THREADS);
	omp_set_num_threads(THREADS);
	#pragma omp parallel//3
	{
		uint32_t tid = omp_get_thread_num();
		uint32_t gid = omp_get_num_threads();

		for(uint64_t j = tid;j<this->n/32; j+=gid){
			for(uint32_t k = 1; k < this->d ;k++){
				bvec[j*this->d] |= (bvec[k + j*this->d]);//gather all set bits
			}
			uint32_t vec = bvec[j*this->d];
			for(uint32_t k = 0; k < 32;k++){
				if( vec & (0x1 << k) ){// if tupple id has been seen
					uint32_t id = (j << 5) + k;//find tupple id
					T score = 0;
					for(uint64_t m = 0; m < this->d; m++){ score+= this->cdata[id * this->d + m]; }
					res[tid].push_back(tuple<T,Z>(id,score));
				}
			}
		}
	}

	free(bvec);
}

/*
 * Iterate through lists and then evaluate tuples
 */
template<class T,class Z>
void FA<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";

	this->t.start();
	if(this->topkp){
		this->par_topk(k);
	}else{
		this->seq_topk(k);
	}
	this->tt_processing = this->t.lap("");


	std::cout << " (" << this->res.size() << ")" << std::endl;
}

#endif

