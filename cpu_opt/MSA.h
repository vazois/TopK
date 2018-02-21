#ifndef MSA_H
#define MSA_H

#include "../cpu/AA.h"
#include <cmath>

template<class T>
struct msa_info{
	uint8_t a;
	T v;
};

template<class T, class Z>
class MSA : public AA<T,Z>{
	public:
		MSA(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "MSA";
			this->msa_values = NULL;
			this->query = (uint8_t*) malloc(sizeof(uint8_t)*this->d);
			this->query_len = this->d;
			for(uint8_t m = 0; m < this->d; m++) this->query[m] = m;
		}

		~MSA(){
			if(this->msa_values != NULL) free(this->msa_values);
			if(this->query != NULL) free(this->query);
		}

		void init();
		void findTopK(uint64_t k);

		static inline T max(T a, T b){
			return a > b ? a : b;
		}

		static inline bool cmp_msa_values(const msa_info<T> &a, const msa_info<T> &b){ return a.v > b.v; };

	private:
		msa_info<T> *msa_values;
		uint8_t *query;
		uint8_t query_len;
};

template<class T, class Z>
void MSA<T,Z>::init(){
	this->msa_values = (msa_info<T>*) malloc(sizeof(msa_info<T>) * this->d);

	this->t.start();
	for(uint8_t m = 0; m < this->d; m++){
		this->msa_values[m].a = m;
		this->msa_values[m].v = this->cdata[m*this->n];

		for(uint64_t i = 0; i < this->n; i++){
			this->msa_values[m].v = MSA::max(this->msa_values[m].v,this->cdata[m*this->n + i]);
		}
	}
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void MSA<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";

	msa_info<T> *q_msa_values = (msa_info<T>*) malloc(sizeof(msa_info<T>) * this->query_len);
	this->t.start();
	//Consider only query attributes
	for(uint8_t m = 0; m < this->query_len; m++){
		q_msa_values[m].a = this->msa_values[this->query[m]].a;
		q_msa_values[m].v = this->msa_values[this->query[m]].v;
	}
	std::sort(q_msa_values,q_msa_values + this->query_len,MSA<T,Z>::cmp_msa_values);
	std::cout << std::endl;
	for(uint8_t m = 0; m < this->query_len; m++){
		std::cout << "a" << std::setfill('0') << std::setw(2) <<
				(int)q_msa_values[m].a << ":" << q_msa_values[m].v << std::endl;
	}

	for(uint8_t m = this->d; m < this->d; m++){
		//this->msa_values[m];
	}

	this->tt_processing = this->t.lap();

	T threshold = 1313;
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;

	free(q_msa_values);
}

#endif
