#ifndef PTAP_H
#define PTAP_H

#include "../cpu/AA.h"

#define R 2

template<class T, class Z>
struct ptap_pair{
	Z id;
	T score;
};

template<class Z>
struct ptap_pair_ps{
	Z id;
	Z tpos;
};

template<class T,class Z>
static bool cmp_ptap_pair(const ptap_pair<T,Z> &a, const ptap_pair<T,Z> &b){ return a.score > b.score; };

template<class T,class Z>
static bool cmp_ptap_pair_ps(const ptap_pair_ps<Z> &a, const ptap_pair_ps<Z> &b){ return a.tpos < b.tpos; };

template<class T, class Z>
class PTAp : public AA<T,Z>{
	public:
		PTAp(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "PTAp";
			this->pos = NULL;
			this->lists = NULL;
			this->tarray=NULL;
		}

		~PTAp(){
			if(this->lists!=NULL){
				for(uint8_t m = 0; m < this->d; m++){ free(lists[m]); }
				free(lists);
			}
			if(this->pos != NULL){ free(this->pos); }
			if(this->tarray != NULL){ free(this->tarray); }
		}

		void init();
		void findTopK(uint64_t k,uint8_t qq);
		void findTopKscalar(uint64_t k,uint8_t qq);
		void findTopKsimd(uint64_t k,uint8_t qq);
		void findTopKthreads(uint64_t k,uint8_t qq);
	private:
		ptap_pair_ps<Z> *pos;
		ptap_pair<T,Z> **lists;
		T *tarray;
};

template<class T, class Z>
void PTAp<T,Z>::init(){
	lists = (ptap_pair<T,Z> **) malloc(sizeof(ptap_pair<T,Z>*) * this->d);
	for(uint8_t m = 0; m < this->d; m++){ lists[m] = (ptap_pair<T,Z> *) malloc(sizeof(ptap_pair<T,Z>)*this->n); }

	omp_set_num_threads(ITHREADS);
	this->t.start();
	for(uint8_t m = 0; m < this->d; m++){
		for(uint64_t i = 0; i < this->n; i++){
			lists[m][i].id = i;
			lists[m][i].score = this->cdata[m*this->n + i];
		}
		__gnu_parallel::sort(lists[m],(lists[m]) + this->n,cmp_ptap_pair<T,Z>);
	}

	pos = (ptap_pair_ps<Z>*)malloc(sizeof(ptap_pair_ps<Z>)*this->n);
	for(uint64_t i = 0; i < this->n; i++){
		pos[i].id = i;
		pos[i].tpos = this->n;
	}
//	for(uint8_t m = 0; m < this->d; m++){
//		for(uint64_t i = 0; i < this->n; i++){
//			ptap_pair<T,Z> p0 = lists[m][i];
//			pos[p0.id].tpos = pos[p0.id].tpos < i ? pos[p0.id].tpos : i;
//		}
//	}

	for(uint8_t m = 0; m < this->d-1; m++){
		for(uint64_t i = 0; i < this->n; i++){
			ptap_pair<T,Z> p0 = lists[m][i];
			ptap_pair<T,Z> p1 = lists[m+1][i];
//			ptap_pair<T,Z> p2 = lists[m+2][i];
//			ptap_pair<T,Z> p3 = lists[m+3][i];
			pos[p0.id].tpos = pos[p0.id].tpos < i ? pos[p0.id].tpos : i;
			pos[p1.id].tpos = pos[p1.id].tpos < i ? pos[p1.id].tpos : i;
//			pos[p2.id].tpos = pos[p2.id].tpos < i ? pos[p2.id].tpos : i;
//			pos[p3.id].tpos = pos[p3.id].tpos < i ? pos[p3.id].tpos : i;
		}
	}
	__gnu_parallel::sort(&pos[0],(&pos[0]) + this->n,cmp_ptap_pair_ps<T,Z>);


	this->tt_init = this->t.lap();
}

template<class T, class Z>
void PTAp<T,Z>::findTopK(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topK (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();

	for(uint64_t i = 0; i < this->n;i++){
		ptap_pair_ps<Z> p = pos[i];

		T score00 = 0;
		T threshold = 0;
		for(uint8_t m = 0; m < qq; m++){
			score00+=this->cdata[m*this->n + p.id];
			threshold+= this->lists[m][p.tpos].score;
		}

		if(q.size() < k){
			q.push(tuple_<T,Z>(p.id,score00));
		}else if(q.top().score < score00){
			q.pop();
			q.push(tuple_<T,Z>(p.id,score00));
		}

		if(STATS_EFF) this->tuple_count++;
		if(q.size() >= k && ((q.top().score) > threshold) ){
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << threshold << std::endl;
			break;
		}
	}


	this->tt_processing = this->t.lap();

	while(q.size() > 100){ q.pop(); }
	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}


#endif
