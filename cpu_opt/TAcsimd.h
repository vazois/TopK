#ifndef TAc_SIMD_H
#define TAc_SIMD_H

#include "../cpu/AA.h"

template<class T, class Z>
struct ta_pair{
	Z id;
	T score;
};

template<class T,class Z>
static bool cmp_ta_pair(const ta_pair<T,Z> &a, const ta_pair<T,Z> &b){ return a.score > b.score; };

template<class T, class Z>
class TAcsimd : public AA<T,Z>{
	public:
		TAcsimd(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "TAsimd";
			this->tuples = NULL;
		}

		~TAcsimd(){
			if(this->tuples!=NULL) free(this->tuples);
		}

		void init();
		void findTopK(uint64_t k);
	private:
		ta_pair<T,Z> *tuples;
};

template<class T, class Z>
void TAcsimd<T,Z>::init(){
	ta_pair<T,Z> *lists = (ta_pair<T,Z>*)malloc(sizeof(ta_pair<T,Z>)*this->n*this->d);
	this->t.start();

	for(uint8_t m = 0; m < this->d; m++){
		for(uint64_t i = 0; i < this->n; i++){
			lists[m*this->n + i].id = i;
			lists[m*this->n + i].score = this->cdata[m*this->n + i];
		}
	}

	for(uint8_t m = 0;m<this->d;m++){
		__gnu_parallel::sort(&lists[m*this->n],(&lists[m*this->n]) + this->n,cmp_ta_pair<T,Z>);
	}

	for(uint64_t i = 0; i < this->n; i++){

	}

	this->tt_init = this->t.lap();
	free(lists);
}

template<class T,class Z>
void TAcsimd<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topKsimd ...";
	this->t.start();


	this->tt_processing = this->t.lap();

	T threshold = 1313;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
