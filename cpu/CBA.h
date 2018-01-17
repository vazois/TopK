#ifndef CBA_H
#define CBA_H

#include "AA.h"
#include "reorder_attr_cpu.h"

#include <list>
#include <queue>

template<class T,class Z>
struct cpred{
	cpred(){ tid = 0; curr_attr = 0; total = 0; }
	cpred(uint64_t t, T a){ tid = t; curr_attr = a; total = a;}
	bool compare(T threshold, Z r){ return (total + curr_attr * r) < threshold; }
	Z tid;
	T curr_attr;
	T total;
};

template<class T,class Z>
struct vpred{
	vpred(){ tid = 0; last = 0; total = 0; }
	vpred(Z t, T a){ tid = t; last = a; total = a;}

	bool keep(T threshold, uint64_t r){ return (total + last * r) >= threshold;}

	Z tid;
	T total;
	T last;
};

template<class T,class Z>
static bool cmp_max_cpred(const cpred<T,Z> &a, const cpred<T,Z> &b){ return a.total > b.total; };

template<class T,class Z>
class CBA : public AA<T,Z>{
	public:
		CBA(uint64_t n,uint64_t d) : AA<T,Z>(n,d){ this->algo = "CBA"; };

		void init();
		void findTopK(uint64_t k);

	protected:
		std::vector<vpred<T,Z>> vtupples;

	private:
		void seq_topk(uint64_t k);
		void par_topk(uint64_t k);

		void seq_init();
		void par_init();
		void check_order();

		float radix_select(T *data, uint64_t n,uint64_t k);
		T findAndPrune(uint64_t k,uint64_t r);

		typename std::vector<vpred<T,Z>>::iterator partition(typename std::vector<vpred<T,Z>>::iterator first, typename std::vector<vpred<T,Z>>::iterator last, T threshold, uint32_t mult);
};

template<class T,class Z>
void CBA<T,Z>::init(){
	//std::cout << this->algo <<" initialize: " << "(" << this->n << "," << this->d << ")"<< std::endl;

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
		default:
			break;
	}
	//this->tt_init = this->t.lap("");
	if(this->topkp){
		this->vtupples.resize(this->n);
		for(uint64_t i = 0; i < this->n; i++){ this->vtupples[i] = vpred<T,Z>(i,this->cdata[i]); }
	}else{
		//for(uint64_t i = 0; i < this->n; i++){ this->tupples.push_back(cpred<T,Z>(i,this->cdata[i])); }
		this->vtupples.resize(this->n);
		for(uint64_t i = 0; i < this->n; i++){ this->vtupples[i] = vpred<T,Z>(i,this->cdata[i]); }
	}
	this->tt_init = this->t.lap("");
	this->check_order();//TODO: Comment
}

template<class T,class Z>
void CBA<T,Z>::check_order(){
	std::string passed = "(PASSED)";
	for(uint64_t i = 0; i < this->n; i++){
		bool ordered = true;
		for(uint64_t j = 0; j < (this->d - 1); j++){ ordered &=(this->cdata[j * this->n + i] >= this->cdata[(j+1) * this->n + i]); }

		if(!ordered){
			passed = "(FAILED)";
			std::cout << "i: <" << i << "> ";
			for(uint64_t j = 0; j < this->d; j++) std::cout << this->cdata[j * this->n + i] << " ";
			std::cout << std::endl;
			std::cout << "check_order: " << passed << std::endl;
			exit(1);
		}
		//if(i < 10){ for(uint64_t j = 0; j < this->d; j++){ std::cout << this->cdata[j * this->n + i] << " "; } std::cout << std::endl;}
	}
}

template<class T,class Z>
float CBA<T,Z>::radix_select(T *data, uint64_t n,uint64_t k){
	uint32_t prefix=0x00000000;
	uint32_t prefix_mask=0x00000000;
	uint32_t digit_mask=0xF0000000;
	uint32_t digit_shf=28;

	uint64_t tmpK = k;
	for(int i = 0;i <8;i++){
		uint32_t bins[16];
		for(uint8_t i = 0;i < 16;i++) bins[i]=0;

		uint8_t digit;
		for(uint32_t j = 0; j < this->n; j++){
			uint32_t vi = *(uint32_t*)&(data[j]);
			digit = (vi & digit_mask) >> digit_shf;
			bins[digit]+= ((vi & prefix_mask) == prefix);
		}

		if (bins[0] > k){
			digit =  0x0;
		}else{
			for(uint8_t i = 1;i < 16;i++){
				bins[i]+=bins[i-1];
				if( bins[i] > k ){
					k = k-bins[i-1];
					digit = i;
					break;
				}
			}
		}

		prefix = prefix | (digit << digit_shf);
		prefix_mask=prefix_mask | digit_mask;
		digit_mask>>=4;
		digit_shf-=4;
	}

	return *(float*)&prefix;
}

template<class T,class Z>
typename std::vector<vpred<T,Z>>::iterator CBA<T,Z>::partition(typename std::vector<vpred<T,Z>>::iterator first, typename std::vector<vpred<T,Z>>::iterator last, T threshold, uint32_t mult){
	uint32_t keep_count = 0;
	while(first != last){
		while (first->keep(threshold,mult)) {
			++first;
			if (first==last) return first;
		}

		do{
			--last;
			if (first==last) return first;
		}while(!last->keep(threshold,mult));
		std::iter_swap(first,last);
		++first;
	}
	return first;
}

template<class T,class Z>
void CBA<T,Z>::seq_topk(uint64_t k){
	typename std::vector<vpred<T,Z>>::iterator first = this->vtupples.begin();
	typename std::vector<vpred<T,Z>>::iterator last = this->vtupples.end();
	for(uint64_t j = 0; j < this->d; j++){
		std::priority_queue<T, std::vector<T>, std::greater<T>> q;

		//Find threshold
		first = this->vtupples.begin();
		while( first != last  ){
			if(q.size() < k){
				q.push(first->total);
			}else if(q.top()<first->total){
				q.pop();
				q.push(first->total);
			}
			first++;
		}
		T threshold = q.top();
//		//std::cout << "threshold: " << threshold << std::endl;

//		//Partition data
		uint32_t mult =this->d-(j+1);
		last = this->partition(this->vtupples.begin(),last,threshold,mult);

//		//Update tupple scores
		if( j < this->d-1 ){
			Z size = 0;
			first = this->vtupples.begin();
			while( first != last && j < this->d-1){
				T next_attr = this->cdata[(j+1) * this->n + first->tid];
				first->total += next_attr;
				first->last = next_attr;
				first++;
				size++;
			}
			if(STATS_EFF) this->pred_count+=size;
			if(size <= k) break;
		}
	}
	if(STATS_EFF) this->tuple_count+=k;
}

template<class T,class Z>
void CBA<T,Z>::par_topk(uint64_t k){
	//TODO: Fix bug one iteration more//
	uint64_t tuples[THREADS];
	T thresholds[THREADS];
	typename std::priority_queue<T, std::vector<T>, std::greater<T>> qs[THREADS];
	typename std::vector<vpred<T,Z>>::iterator start_it[THREADS];
	typename std::vector<vpred<T,Z>>::iterator end_it[THREADS];
	omp_set_num_threads(THREADS);

	//std::cout << "sss < " << this->vtupples[0].total << "\n";
	#pragma omp parallel
	{
		uint32_t tid = omp_get_thread_num();
		uint32_t p = THREADS;
		T threshold = 0;
		typename std::vector<vpred<T,Z>>::iterator first = this->vtupples.begin() + (tid * this->n)/p;
		typename std::vector<vpred<T,Z>>::iterator last = this->vtupples.begin() + (((tid+1) * this->n )/p) - 1;
		start_it[tid] = first;
		end_it[tid] = last;
		std::string msg = "<" + std::to_string(tid) + "> = " + std::to_string(this->n) + " | " + std::to_string((tid * this->n)/p) + "\n";

		for(uint64_t j = 0; j < this->d; j++){
			if(tid == 0) std::cout << msg;
			std::priority_queue<T, std::vector<T>, std::greater<T>> q;

			first = start_it[tid];
			last = end_it[tid];
			while( first != last  ){//Find k-largest value//
				if(qs[tid].size() < k){
					//std::cout << first->total << " -- "<<msg ;
					qs[tid].push(first->total);
					//std::cout << "2"+msg;
				}else if(qs[tid].top()<first->total){
					qs[tid].pop();
					qs[tid].push(first->total);
				}
				first++;
			}
			#pragma omp barrier

			if(tid == 0){
				for(uint32_t m = 0;m < p;m++){
					while(!qs[m].empty()){
						if(q.size() < k){
							q.push(qs[m].top());
						}else if(q.top()<qs[m].top()){
							q.pop();
							q.push(qs[m].top());
						}
						qs[m].pop();
					}
				}
				threshold = q.top();
			}
			#pragma omp barrier

			//prune irrelevant tupples//
			first = start_it[tid];
			uint32_t mult =this->d-(j+1);
			last = this->partition(first,last,threshold,mult);
			end_it[tid] = last;

			//probe for next predicate//
			first = start_it[tid];
			if(j == (this->d-1)) tuples[tid] = 0;
			while( first != last  ){
				T next_attr = this->cdata[(j+1) * this->n + first->tid];
				first->total += next_attr;
				first->last = next_attr;
				first++;
				if(j == (this->d-1)) tuples[tid]++;
			}
		}
	}

//	for(uint32_t m = 1;m < THREADS;m++){
//		while( start_it[m] != end_it[m]  ){
//			end_it[m-1] = start_it[m];
//			start_it[m]++;
//			end_it[m-1]++;
//		}
//		end_it[m] = end_it[m-1];
//	}
}

template<class T,class Z>
void CBA<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	this->t.start();
	if(this->topkp){
		this->par_topk(k);
	}else{
		this->seq_topk(k);
	}
	this->tt_processing = this->t.lap();
	//this->tuple_count+=this->vtupples.size();

	for(uint32_t i = 0;i <k;i++){
		this->res.push_back(tuple<T,Z>(this->vtupples[i].tid,this->vtupples[i].total));
	}
	std::cout << " (" << this->res.size() << ")" << std::endl;
}

#endif
