#ifndef MSA_H
#define MSA_H

#include "AA.h"
#include <cmath>


template<class T, class Z>
struct msa_pair{
	Z id;
	T score;
};

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
			this->tuples = NULL;
			this->msa_values = NULL;
			this->query = (uint8_t*) malloc(sizeof(uint8_t)*this->d);
			this->query_len = this->d;
			for(uint8_t m = 0; m < this->d; m++) this->query[m] = m;
		}

		~MSA(){
			if(this->tuples != NULL) free(this->tuples);
			if(this->msa_values != NULL) free(this->msa_values);
			if(this->query != NULL) free(this->query);
		}

		void init();
		void findTopK(uint64_t k);

	private:
		msa_pair<T,Z> * partition(msa_pair<T,Z> *first, msa_pair<T,Z> *last,T threshold, T remainder);
		static inline T max(T a, T b){ return a > b ? a : b; };
		static inline bool cmp_msa_values(const msa_info<T> &a, const msa_info<T> &b){ return a.v > b.v; };

		msa_pair<T,Z> *tuples;
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
	this->tuples = (msa_pair<T,Z> *) malloc(sizeof(msa_pair<T,Z>)*this->n);
	for(uint64_t i = 0; i < this->n; i++){
		this->tuples[i].id = i;
		this->tuples[i].score = 0;
	}

	this->tt_init = this->t.lap();
}

template<class T, class Z>
msa_pair<T,Z>* MSA<T,Z>::partition(msa_pair<T,Z> *first, msa_pair<T,Z> *last,T threshold, T remainder){
	while(first < last){

		while(first->score + remainder >= threshold){
			++first;
			if(first == last) return first;
		}

		do{
			--last;
			if(first == last) return first;
		}while(last->score + remainder < threshold);
		std::swap(*first,*last);
		++first;
	}
	return first;
}

template<class T, class Z>
void MSA<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";

	msa_info<T> *q_msa_values = (msa_info<T>*) malloc(sizeof(msa_info<T>) * this->query_len+1);
	this->t.start();
	//Consider only query attributes
	for(uint8_t m = 0; m < this->query_len; m++){
		q_msa_values[m].a = this->msa_values[this->query[m]].a;
		q_msa_values[m].v = this->msa_values[this->query[m]].v;
	}
	q_msa_values[this->query_len].v = 0;
	q_msa_values[this->query_len].a = 99;
	std::sort(q_msa_values,q_msa_values + this->query_len,MSA<T,Z>::cmp_msa_values);

	for(int8_t m = this->query_len-2; m >=0 ; m--){ q_msa_values[m].v+=q_msa_values[m+1].v; }

	//DEBUG
//	std::cout << std::endl;
//	for(uint8_t m = 0; m < this->query_len+1; m++){
//		std::cout << "a" << std::setfill('0') << std::setw(2) <<
//				(int)q_msa_values[m].a << ":" << q_msa_values[m].v << std::endl;
//	}
	//

	msa_pair<T,Z> *first = this->tuples;
	msa_pair<T,Z> *last = this->tuples+this->n;
	for(uint8_t m = 0; m < this->query_len; m++){
		uint8_t a = q_msa_values[m].a;
		//accumulate score for next attribute
		first = this->tuples;
		while(first < last){
			first->score += this->cdata[ a * this->n  + first->id ];
			first++;
		}

		//Find threshold
		first = this->tuples;
		std::priority_queue<T, std::vector<T>, std::greater<T>> q;
		while(first < last){
			if(q.size() < k){
				q.push(first->score);
			}else if(q.top() < first->score){
				q.pop();
				q.push(first->score);
			}
			first++;
		}
		T threshold = q.top();

		//Partition data
		first = this->tuples;
		last = this->partition(first,last,threshold,q_msa_values[m+1].v);
	}
	this->tt_processing = this->t.lap();

	T threshold = this->tuples[0].score;
	for(uint64_t i = 0; i < k; i++){
		threshold = threshold < this->tuples[i].score ? threshold : this->tuples[i].score;
		this->res.push_back(tuple_<T,Z>(this->tuples[i].id,this->tuples[i].score));
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;

	free(q_msa_values);
}

#endif
