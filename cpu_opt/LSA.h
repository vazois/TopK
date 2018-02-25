#ifndef LSA_H
#define LSA_H

#include "../cpu/AA.h"

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
		}

		~LSA(){
			if(this->tuples != NULL) free(this->tuples);
		}

		void init();
		void findTopK3(uint64_t k);
		void findTopK2(uint64_t k);
		void findTopK(uint64_t k);

	private:
		static inline bool cmp_lsa_pairs(const lsa_pair<T,Z> &a, const lsa_pair<T,Z> &b){ return a.score > b.score; };
		lsa_pair<T,Z>* partition(lsa_pair<T,Z> *first, lsa_pair<T,Z> *last, T *curr, uint8_t remainder, T threshold);
		lsa_pair<T,Z> *tuples;
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

//	lsa_pair<T,Z> *tt = (lsa_pair<T,Z>*) malloc(sizeof(lsa_pair<T,Z>) * this->n );
//	for(uint64_t i = 0;i <this->n;i++){
//		tt[i].id=i;
//		tt[i].score=this->cdata[(this->d-1) * this->n + i];
//	}
//	__gnu_parallel::sort(tt,tt + this->n,LSA<T,Z>::cmp_lsa_pairs);
//
//	T *cdata = (T*)malloc(sizeof(T) * this->n * this->d);
//	for(uint64_t i=0;i<this->n;i++){
//		for(uint8_t m =0; m < this->d;m++){
//			cdata[ m * this->n + i] = this->cdata[m*this->n + tt[i].id];
//		}
//	}
//	free(this->cdata);
//	this->cdata = cdata;
//
//	free(tt);
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
				//size++;
			}
			//std::cout << (int)m << " = " << size << std::endl;
		}
	}
	this->tt_processing = this->t.lap();

	T threshold = this->tuples[0].score;
	for(uint64_t i = 0; i < k;i++){
		//std::cout << i << std::endl;
		threshold = threshold < this->tuples[i].score ? threshold : this->tuples[i].score;
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void LSA<T,Z>::findTopK3(uint64_t k){
	std::cout << this->algo << " find topK3 ...";

	this->tuples = (lsa_pair<T,Z>*)malloc(sizeof(lsa_pair<T,Z>*) * this->n);
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++){
		this->tuples[i].id = i;
		this->tuples[i].score = 0;
	}

	lsa_pair<T,Z> *first = this->tuples;
	lsa_pair<T,Z> *last = this->tuples + this->n;
	for(uint8_t m = 0; m < this->d; m++){
		//Accumulate scores
		first = this->tuples;
		while(first < last){
			first->score += this->cdata[m * this->n + first->id];
			first++;
		}

		//Find Threshold
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
		//std::cout << "t: " << threshold << std::endl;

		first = this->tuples;
		uint8_t remainder = this->d-(m+1);
		last = this->partition(first,last,&this->cdata[m * this->n],remainder,threshold);
	}
	this->tt_processing = this->t.lap();

	T threshold = this->tuples[0].score;
	for(uint64_t i = 0; i < k;i++){
		//std::cout << i << std::endl;
		threshold = threshold < this->tuples[i].score ? threshold : this->tuples[i].score;
	}
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void LSA<T,Z>::findTopK2(uint64_t k){
	std::cout << this->algo << " find topK2 ...";

	this->tuples = (lsa_pair<T,Z>*)malloc(sizeof(lsa_pair<T,Z>*) * this->n);
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++){
		this->tuples[i].id = i;
		this->tuples[i].score = 0;
	}

	lsa_pair<T,Z> *first = this->tuples;
	lsa_pair<T,Z> *last = this->tuples + this->n;
	for(uint8_t m = 0; m < this->d; m++){
		//Accumulate scores & Find thresholds
		first = this->tuples;
		//std::cout<< first << "," << last << " : " << (last - first) <<std::endl;

		std::priority_queue<T, std::vector<T>, std::greater<T>> q;
		while(first < last){
			first->score += this->cdata[m * this->n + first->id];
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

		first = this->tuples;
		uint8_t remainder = this->d-(m+1);
		last = this->partition(first,last,&this->cdata[m * this->n],remainder,threshold);
	}
	this->tt_processing = this->t.lap();

	T threshold = this->tuples[0].score;
	for(uint64_t i = 0; i < k;i++){
		//std::cout << i << std::endl;
		threshold = threshold < this->tuples[i].score ? threshold : this->tuples[i].score;
	}
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
