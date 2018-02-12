#ifndef CBA_H
#define CBA_H

#include "AA.h"
#include "reorder_attr_cpu_c.h"

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
struct LocalMax{
	LocalMax(){ tid = 0; score = 0; }
	LocalMax(Z t, T s, T n){ tid = t; score = s; next = n; }

	Z tid;
	T score;
	T next;
};

template<class T,class Z>
class LocalMaxCmp{
	public:
		LocalMaxCmp(){};

		bool operator() (const LocalMax<T,Z>& lhs, const LocalMax<T,Z>& rhs) const{
			return (lhs.score>rhs.score);
		}
};

template<class T,class Z>
class CBA : public AA<T,Z>{
	public:
		CBA(uint64_t n,uint64_t d) : AA<T,Z>(n,d){ this->algo = "CBA"; };

		void init();
		void findTopK(uint64_t k);
		void findTopK2(uint64_t k);

	protected:
		std::vector<vpred<T,Z>> vtupples;

	private:
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
//	if(this->topkp){
//		this->vtupples.resize(this->n);
//		for(uint64_t i = 0; i < this->n; i++){ this->vtupples[i] = vpred<T,Z>(i,this->cdata[i]); }
//	}else{
	//for(uint64_t i = 0; i < this->n; i++){ this->tupples.push_back(cpred<T,Z>(i,this->cdata[i])); }
	this->vtupples.resize(this->n);
	for(uint64_t i = 0; i < this->n; i++){ this->vtupples[i] = vpred<T,Z>(i,this->cdata[i]); }

	this->tt_init = this->t.lap();
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
void CBA<T,Z>::findTopK(uint64_t k){
	typename std::vector<vpred<T,Z>>::iterator first = this->vtupples.begin();
	typename std::vector<vpred<T,Z>>::iterator last = this->vtupples.end();

	std::cout << this->algo << " find topK ...";
//	std::cout << "\nsize(" << 0 << ") :" << this->vtupples.size() << std::endl;
	this->t.start();
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
//		std::cout << "threshold: " << threshold << std::endl;

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
			//std::cout << "size(" << j+1 << ") :" << size << std::endl;
			if(STATS_EFF) this->pred_count+=size;
			if(size <= k) break;
		}
	}
	this->tt_processing = this->t.lap();
	if(STATS_EFF) this->tuple_count+=k;

	//Gather results for verification
	T threshold = this->vtupples[0].total;
	for(uint32_t i = 0;i <k;i++){
		this->res.push_back(tuple<T,Z>(this->vtupples[i].tid,this->vtupples[i].total));
		threshold = threshold > this->vtupples[i].total ? this->vtupples[i].total : threshold;
	}
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T,class Z>
void CBA<T,Z>::findTopK2(uint64_t k){
	std::cout << this->algo << " find topK ...";
	T *score = (T* )malloc(sizeof(T) * this->n);

	LocalMax<T,Z> k_0;
	LocalMax<T,Z> k_1;
	this->t.start();
	memset(score,0,sizeof(T)*this->n);
	uint8_t suffix_len=this->d-1;
	for(uint8_t m = 0; m < this->d; m++){
		std::priority_queue<T, std::vector<LocalMax<T,Z>>, LocalMaxCmp<T,Z>> q;
		for(uint64_t i = 0; i < this->n;i++){
			score[i]+=this->cdata[m * this->n + i];
			if(q.size() < k+1){
				q.push(LocalMax<T,Z>(i,score[i],this->cdata[m * this->n + i]));
			}else if( q.top().score < score[i]){
				q.pop();
				q.push(LocalMax<T,Z>(i,score[i],this->cdata[m * this->n + i]));
			}
		}

		k_1 =q.top(); q.pop(); k_0 =q.top();
		std::cout << "k_0: " <<k_0.score <<  std::endl;
		std::cout << "k_1: " <<k_1.score <<" + " << k_1.next << " * " << (int)suffix_len<<  std::endl;
		if(k_0.score >= k_1.score + k_1.next * suffix_len){
			std::cout << "stopped at:" <<(int)m << std::endl;
			break;
		}
		suffix_len--;
	}

	this->tt_processing = this->t.lap();
	free(score);

}

#endif
