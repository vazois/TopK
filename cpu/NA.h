#ifndef NA_H
#define NA_H

#include "AA.h"

/*
 * Naive algorithm for aggregation
 */
template<class T>
class NA : public AA<T>{
	public:
		NA(uint64_t n,uint64_t d) : AA<T>(n,d){ this->algo = "NA"; };

		void init();
		void initSIMD();
		void findTopK(uint64_t k);

	protected:
		std::vector<tuple<T>> tuples;
};

/*
 * Calculate scores
 */
template<class T>
void NA<T>::init(){
	this->tuples.resize(this->n);
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++ ){
		T score = 0;
		for(uint64_t j = 0; j < this->d; j++ ){ score+= this->cdata[i * this->d + j]; }
		this->tuples.push_back(tuple<T>(i,score));
		this->pred_count+=this->d;
		this->tuple_count+=1;
	}
	this->tt_init = this->t.lap();
}

//			__m256 vT0 =  _mm256_load_ps(&this->cdata[offset + this->n]);
//			__m256 vT1 =  _mm256_load_ps(&this->cdata[offset + this->n]);
//			__m256 vT2 =  _mm256_load_ps(&this->cdata[offset + this->n]);
//			__m256 vT3 =  _mm256_load_ps(&this->cdata[offset + this->n]);
			//__m256 vT0 =  _mm256_set_ps(0,0,0,0); // set manually
//__m128 acc = _mm_set_ps(0,0,0,0);
//__m256 acc = _mm256_set_ps(0,0,0,0);
template<class T>
void NA<T>::initSIMD(){
	this->tuples.resize(this->n);

	this->t.start();
	uint32_t bytes = sizeof(T) * this->d;
	__m256i one = _mm256_set1_epi32(sizeof(T));
	__m256i index = _mm256_set_epi32(bytes*7,bytes*6,bytes*5,bytes*4,bytes*3,bytes*2,bytes,0);
	//__m256i index = _mm256_setzero_si256();
	T score[8];
	for(uint64_t i = 0; i < this->n; i+=8 ){
		__m256 acc = _mm256_setzero_ps();

		T *offset = &this->cdata[i * this->d];
		__m256i rindex = index;
		for(uint64_t j = 0;j <this->d;j++){
			__m256 vT = _mm256_i32gather_ps(offset,rindex,1);
			acc = _mm256_add_ps(vT,acc);
			rindex = _mm256_add_epi32(rindex,one);
		}
		_mm256_store_ps(score,acc);

		this->tuples.push_back(tuple<T>(i,score[0]));
		this->tuples.push_back(tuple<T>(i+1,score[1]));
		this->tuples.push_back(tuple<T>(i+2,score[2]));
		this->tuples.push_back(tuple<T>(i+3,score[3]));
		this->tuples.push_back(tuple<T>(i+4,score[4]));
		this->tuples.push_back(tuple<T>(i+5,score[5]));
		this->tuples.push_back(tuple<T>(i+6,score[6]));
		this->tuples.push_back(tuple<T>(i+7,score[7]));
	}
	this->tt_init = this->t.lap();
}


/*
 * Sort and find top-K
 */
template<class T>
void NA<T>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";

	this->t.start();
	std::sort(this->tuples.begin(),this->tuples.end(),cmp_score<T>);
	//std::cout << std::endl;
	for(uint64_t i = 0;i <(k < this->tuples.size() ? k : this->tuples.size() ) ;i++){
		//std::cout << this->algo <<" : " << this->tuples[i].tid << "," << this->tuples[i].score <<std::endl;
		this->res.push_back(tuple<T>(this->tuples[i].tid,this->tuples[i].score));
	}
	this->tt_processing = this->t.lap();

	std::cout << " (" << this->res.size() << ")" << std::endl;
}

#endif
