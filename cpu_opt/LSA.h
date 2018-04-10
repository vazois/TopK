#ifndef LSA_H
#define LSA_H

#include "../cpu/AA.h"
#include "../cpu/reorder_attr_cpu_c.h"

template<class T, class Z>
struct lsa_pair{
	Z id;
	T score;
};

template<class T, class Z>
class LSA : public AA<T,Z>{
	public:
		LSA(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "LSA";
			this->tuples = NULL;
			this->ids = NULL;
			this->scores = NULL;
		}

		~LSA(){
			if(this->tuples != NULL) free(this->tuples);
			if(this->ids != NULL) free(this->ids);
			if(this->scores != NULL) free(this->scores);
		}

		void init();
		void findTopK(uint64_t k);
		void findTopKscalar(uint64_t k);

	private:
		static inline bool cmp_lsa_pairs(const lsa_pair<T,Z> &a, const lsa_pair<T,Z> &b){ return a.score > b.score; };
		uint64_t partition(Z *ids, T *scores, T *curr, uint64_t n, uint8_t remainder, T threshold );
		lsa_pair<T,Z>* partition(lsa_pair<T,Z> *first, lsa_pair<T,Z> *last, T *curr, uint8_t remainder, T threshold);
		lsa_pair<T,Z> *tuples;
		Z *ids;
		T *scores;
};

template<class T, class Z>
void LSA<T,Z>::init(){
	this->t.start();
	switch(this->d){
		case 2:
			reorder_attr_2(this->cdata,this->n);
			break;
		case 4:
			reorder_attr_4(this->cdata,this->n);
			break;
		case 6:
			reorder_attr_6(this->cdata,this->n);
			break;
		case 8:
			reorder_attr_8(this->cdata,this->n);
			break;
		case 10:
			reorder_attr_10(this->cdata,this->n);
			break;
		case 12:
			reorder_attr_12(this->cdata,this->n);
			break;
		case 14:
			reorder_attr_14(this->cdata,this->n);
			break;
		case 16:
			reorder_attr_16(this->cdata,this->n);
			break;
		case 18:
			reorder_attr_18(this->cdata,this->n);
			break;
		case 20:
			reorder_attr_20(this->cdata,this->n);
			break;
		case 22:
			reorder_attr_22(this->cdata,this->n);
			break;
		case 24:
			reorder_attr_24(this->cdata,this->n);
			break;
		case 26:
			reorder_attr_26(this->cdata,this->n);
			break;
		case 28:
			reorder_attr_28(this->cdata,this->n);
			break;
		case 30:
			reorder_attr_30(this->cdata,this->n);
			break;
		case 32:
			reorder_attr_32(this->cdata,this->n);
			break;
		default:
			break;
	}
	this->tt_init = this->t.lap();
}

template<class T, class Z>
lsa_pair<T,Z>* LSA<T,Z>::partition(lsa_pair<T,Z> *first, lsa_pair<T,Z> *last, T *curr, uint8_t remainder, T threshold){
	while(first < last)
	{
		while(first->score + remainder * curr[first->id] >= threshold){
			++first;
			if(first == last) return first;
		}

		do{
			--last;
			if(first == last) return first;
		}while(last->score + remainder * curr[last->id] < threshold);
		std::swap(*first,*last);
		++first;
	}
	return first;
}

template<class T, class Z>
void LSA<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	this->res.clear();
	this->tuples = (lsa_pair<T,Z>*)malloc(sizeof(lsa_pair<T,Z>*) * this->n);
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++){
		this->tuples[i].id = i;
		this->tuples[i].score = this->cdata[i];
	}

	lsa_pair<T,Z> *first = this->tuples;
	lsa_pair<T,Z> *last = this->tuples + this->n;
	for(uint8_t m = 0; m < this->d; m++){
		//Find Threshold//
		std::priority_queue<T, std::vector<T>, std::greater<T>> q;
		first = this->tuples;
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
		//std::cout << "t: " << threshold << std::endl;

		//Partition data
		first = &this->tuples[0];
		uint32_t remainder = this->d-(m+1);
		last = this->partition(first,last,&this->cdata[m * this->n],remainder,threshold);

		//Accumulate scores//
		if(m < this->d-1){
			first = &this->tuples[0];
			uint64_t size=0;
			while(first < last){
				first->score+= this->cdata[(m+1)*this->n + first->id];
				first++;
			}
		}
	}
	this->tt_processing = this->t.lap();

	T threshold = this->tuples[0].score;
	for(uint64_t i = 0; i < k;i++){
		threshold = threshold < this->tuples[i].score ? threshold : this->tuples[i].score;
		this->res.push_back(tuple_<T,Z>(this->tuples[i].id,this->tuples[i].score));
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
uint64_t LSA<T,Z>::partition(Z *ids, T *scores, T *curr, uint64_t n, uint8_t remainder, T threshold ){
	uint64_t first = 0;
	uint64_t last = n;

	while(first < last){
		while ( scores[first] + curr[ids[first]]*remainder >= threshold ) {
			++first;
			if (first==last) return first;
		}

		do{
			--last;
			if (first==last) return first;
		}while(scores[last] + curr[ids[last]]*remainder < threshold);

		std::swap(ids[first],ids[last]);
		std::swap(scores[first],scores[last]);
	}
	return first;
}

template<class T, class Z>
void LSA<T,Z>::findTopKscalar(uint64_t k){
	std::cout << this->algo << " find topKscalar ...";
	this->res.clear();
	this->ids = (Z*)malloc(sizeof(Z) * this->n);
	this->scores = (T*)malloc(sizeof(T) * this->n);

	this->t.start();
	for(uint64_t i = 0; i < this->n; i++){
		this->ids[i] = i;
		this->scores[i] = this->cdata[i];
	}

	uint64_t end = this->n;
	for(uint8_t m = 0; m < this->d; m++){
		//Find threshold
		std::priority_queue<T, std::vector<T>, std::greater<T>> q;
		for(uint64_t i = 0; i < this->n; i++){
			if(q.size() < k){
				q.push(this->scores[i]);
			}else if(q.top()<this->scores[i]){
				q.pop();
				q.push(this->scores[i]);
			}
		}
		T threshold = q.top();
		//std::cout << "t: " << threshold << std::endl;

		//Partition data
		uint32_t remainder = this->d-(m+1);
		end = this->partition(ids,scores,&this->cdata[m * this->n],end,remainder,threshold);

		if( m < this->d-1 ){
			for(uint64_t i = 0; i < end;i++){
				scores[i] += this->cdata[(m+1) * this->n + ids[i]];
			}
		}
	}
	this->tt_processing = this->t.lap();

	T threshold = scores[0];
	for(uint64_t i = 0;i<k;i++){
		threshold = threshold < scores[i] ? threshold : scores[i];
		this->res.push_back(tuple_<T,Z>(ids[i],scores[i]));
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;

	this->threshold = threshold;
	free(this->ids); this->ids = NULL;
	free(this->scores); this->scores = NULL;
}

#endif
