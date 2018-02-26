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
			this->gt_array = NULL;
		}

		~TAcsimd(){
			if(this->tuples!=NULL) free(this->tuples);
			if(this->gt_array!=NULL) free(this->gt_array);
		}

		void init();
		void findTopK(uint64_t k);
		void findTopK2(uint64_t k);
	private:
		ta_pair<T,Z> *tuples;
		T *gt_array;
		T acc;
};

template<class T, class Z>
void TAcsimd<T,Z>::init(){
	ta_pair<T,Z> *lists = (ta_pair<T,Z>*)malloc(sizeof(ta_pair<T,Z>)*this->n*this->d);
	this->gt_array = (T*)malloc(sizeof(T)*this->n);
	this->tuples = (ta_pair<T,Z>*)malloc(sizeof(ta_pair<T,Z>)*this->n);
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++){
		this->tuples[i].id = i;
		this->tuples[i].score = 0;
	}
	//this->tt_init = this->t.lap();
	for(uint8_t m = 0; m < this->d; m++){
		for(uint64_t i = 0; i < this->n; i++){
			lists[m*this->n + i].id = i;
			lists[m*this->n + i].score = this->cdata[m*this->n + i];
		}
	}
	for(uint8_t m = 0;m<this->d;m++){ __gnu_parallel::sort(&lists[m*this->n],(&lists[m*this->n]) + this->n,cmp_ta_pair<T,Z>); }

	T *cdata = (T*)malloc(sizeof(T)*this->n*this->d);
	std::unordered_set<Z> eset;
	uint64_t ii = 0;
	for(uint64_t i = 0; i < this->n; i++){
		this->gt_array[i]=lists[i].score;
		for(uint8_t m = 0; m < this->d; m++){
			ta_pair<T,Z> p = lists[m*this->n + i];
			this->gt_array[i]=this->gt_array[i]> p.score ? this->gt_array[i] : p.score;
			if(eset.find(p.id) == eset.end()){
				eset.insert(p.id);
				for(uint8_t j = 0; j < this->d; j++){ cdata[j * this->n + ii] = this->cdata[j * this->n + p.id]; }
				ii++;
			}
		}
	}
	this->tt_init = this->t.lap();
	free(this->cdata);
	this->cdata = cdata;
	free(lists);
}

template<class T, class Z>
void TAcsimd<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topKscalar ...";

	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
	T t_array[16];
	__builtin_prefetch(t_array,1,3);
	this->t.start();
	for(uint64_t i = 0; i < this->n; i+=4){
		T score00 = 0;
		T score01 = 0;
		T score02 = 0;
		T score03 = 0;
//		T score04 = 0;
//		T score05 = 0;
//		T score06 = 0;
//		T score07 = 0;
		for(uint8_t m = 0; m < this->d; m++){
			uint64_t offset0 = m * this->n + i;
			score00+= this->cdata[offset0];
			score01+= this->cdata[offset0+1];
			score02+= this->cdata[offset0+2];
			score03+= this->cdata[offset0+3];
//			score04+= this->cdata[offset0+4];
//			score05+= this->cdata[offset0+5];
//			score06+= this->cdata[offset0+6];
//			score07+= this->cdata[offset0+7];
		}

		if(q.size() < k){//insert if empty space in queue
			q.push(tuple<T,Z>(i,score00));
			q.push(tuple<T,Z>(i+1,score01));
			q.push(tuple<T,Z>(i+2,score02));
			q.push(tuple<T,Z>(i+3,score03));
//			q.push(tuple<T,Z>(i+4,score04));
//			q.push(tuple<T,Z>(i+5,score05));
//			q.push(tuple<T,Z>(i+6,score06));
//			q.push(tuple<T,Z>(i+7,score07));
		}else{//delete smallest element if current score is bigger
			if(q.top().score < score00){ q.pop(); q.push(tuple<T,Z>(i,score00)); }
			if(q.top().score < score01){ q.pop(); q.push(tuple<T,Z>(i,score01)); }
			if(q.top().score < score02){ q.pop(); q.push(tuple<T,Z>(i,score02)); }
			if(q.top().score < score03){ q.pop(); q.push(tuple<T,Z>(i,score03)); }
//			if(q.top().score < score04){ q.pop(); q.push(tuple<T,Z>(i,score04)); }
//			if(q.top().score < score05){ q.pop(); q.push(tuple<T,Z>(i,score05)); }
//			if(q.top().score < score06){ q.pop(); q.push(tuple<T,Z>(i,score06)); }
//			if(q.top().score < score07){ q.pop(); q.push(tuple<T,Z>(i,score07)); }
		}

		if(q.top().score > (this->gt_array[i+3] * this->d) ){
			std::cout << "\nStopped at " << i << "= " << q.top().score << std::endl;
			break;
		}
	}
	this->tt_processing = this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
