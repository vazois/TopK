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
			this->gt_array = NULL;
		}

		~TAcsimd(){
			if(this->gt_array!=NULL) free(this->gt_array);
		}

		void init();
		void init2();
		void findTopK(uint64_t k);
		void findTopKsimd(uint64_t k);
	private:
		T *gt_array;
		T acc;
};

template<class T, class Z>
void TAcsimd<T,Z>::init(){
	ta_pair<T,Z> *lists = (ta_pair<T,Z>*)malloc(sizeof(ta_pair<T,Z>)*this->n*this->d);
	this->gt_array = (T*)malloc(sizeof(T)*this->n);
	this->t.start();
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
		//T threshold=0;
		T threshold=lists[i].score;
		for(uint8_t m = 0; m < this->d; m++){
			ta_pair<T,Z> p = lists[m*this->n + i];
			//threshold+=p.score;
			threshold=threshold> p.score ? threshold : p.score;
		}

		for(uint8_t m = 0; m < this->d; m++){
			ta_pair<T,Z> p = lists[m*this->n + i];
			if(eset.find(p.id) == eset.end()){
				eset.insert(p.id);
				for(uint8_t j = 0; j < this->d; j++){ cdata[j * this->n + ii] = this->cdata[j * this->n + p.id]; }
				this->gt_array[ii] = threshold;
				ii++;
				//if(ii == this->n) break;
			}
		}
	}
	this->tt_init = this->t.lap();
	free(this->cdata); this->cdata = cdata;
	free(lists);
}

template<class T, class Z>
void TAcsimd<T,Z>::init2(){
	ta_pair<T,Z> *lists = (ta_pair<T,Z>*)malloc(sizeof(ta_pair<T,Z>)*this->n*this->d);
	this->gt_array = (T*)malloc(sizeof(T)*this->n);
	Z *p_array = (Z*)malloc(sizeof(Z)*this->n);
	this->t.start();
	//this->tt_init = this->t.lap();
	for(uint8_t m = 0; m < this->d; m++){
		for(uint64_t i = 0; i < this->n; i++){
			lists[m*this->n + i].id = i;
			lists[m*this->n + i].score = this->cdata[m*this->n + i];
		}
	}
	for(uint8_t m = 0;m<this->d;m++){ __gnu_parallel::sort(&lists[m*this->n],(&lists[m*this->n]) + this->n,cmp_ta_pair<T,Z>); }
	for(uint64_t i = 0; i < this->n; i++){ p_array[i] = this->n; }

	uint64_t ii = 0;
	for(uint64_t i = 0; i < this->n; i++){
		//T threshold=0;
		T threshold=lists[i].score;
		for(uint8_t m = 0; m < this->d; m++){
			ta_pair<T,Z> p = lists[m*this->n + i];
			//threshold+=p.score;
			threshold=threshold> p.score ? threshold : p.score;
		}
		for(uint8_t m = 0; m < this->d; m++){
			ta_pair<T,Z> p = lists[m*this->n + i];
			if(p_array[p.id] == this->n){
				p_array[p.id] = ii;
				this->gt_array[ii] = threshold;
				ii++;
				//if(ii == this->n){ m = this->d; i =this->n;}
			}
		}
	}

	free(lists);
	//T *cdata = (T*)malloc(sizeof(T)*this->n*this->d);
	T *cdata = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->n) * (this->d)));
	for(uint64_t i = 0; i < this->n; i++){
		for(uint8_t j = 0; j < this->d; j++){
			cdata[j * this->n + p_array[i]] = this->cdata[j * this->n + i];
		}
	}
	free(p_array);

	this->tt_init = this->t.lap();
	free(this->cdata); this->cdata = cdata;
}

template<class T, class Z>
void TAcsimd<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topKscalar ...";

	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
//	T t_array[16];
//	__builtin_prefetch(t_array,1,3);
	this->t.start();
	uint64_t step = 0;
	this->tt_ranking = 0;
	for(uint64_t i = 0; i < this->n; i+=4){
		T score00 = 0;
		T score01 = 0;
		T score02 = 0;
		T score03 = 0;
		for(uint8_t m = 0; m < this->d; m++){
			uint64_t offset0 = m * this->n + i;
			score00+= this->cdata[offset0];
			score01+= this->cdata[offset0+1];
			score02+= this->cdata[offset0+2];
			score03+= this->cdata[offset0+3];
		}
		if(q.size() < k){//insert if empty space in queue
			q.push(tuple<T,Z>(i,score00));
			q.push(tuple<T,Z>(i+1,score01));
			q.push(tuple<T,Z>(i+2,score02));
			q.push(tuple<T,Z>(i+3,score03));
		}else{//delete smallest element if current score is bigger
			if(q.top().score < score00){ q.pop(); q.push(tuple<T,Z>(i,score00)); }
			if(q.top().score < score01){ q.pop(); q.push(tuple<T,Z>(i+1,score01)); }
			if(q.top().score < score02){ q.pop(); q.push(tuple<T,Z>(i+2,score02)); }
			if(q.top().score < score03){ q.pop(); q.push(tuple<T,Z>(i+3,score03)); }
		}

		if((q.top().score) > (this->gt_array[i+3]*this->d) ){
			std::cout << "\nStopped at " << i << "= " << q.top().score << "," << this->gt_array[i+3] << std::endl;
			break;
		}
	}
	this->tt_processing = this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void TAcsimd<T,Z>::findTopKsimd(uint64_t k){
	std::cout << this->algo << " find topKsimd ...";

	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	uint64_t step = 0;
	this->tt_ranking = 0;
	float score[16] __attribute__((aligned(32)));
	__builtin_prefetch(score,1,3);
	for(uint64_t i = 0; i < this->n; i+=16){
		__m256 score00 = _mm256_setzero_ps();
		__m256 score01 = _mm256_setzero_ps();
		for(uint8_t m = 0; m < this->d; m++){
			uint64_t offset00 = m * this->n + i;
			uint64_t offset01 = m * this->n + i + 8;
			__m256 load00 = _mm256_load_ps(&this->cdata[offset00]);
			__m256 load01 = _mm256_load_ps(&this->cdata[offset01]);
			score00 = _mm256_add_ps(score00,load00);
			score01 = _mm256_add_ps(score01,load01);
		}
		_mm256_store_ps(&score[0],score00);
		_mm256_store_ps(&score[8],score01);
		if(q.size() < k){//insert if empty space in queue
			q.push(tuple<T,Z>(i,score[0]));
			q.push(tuple<T,Z>(i+1,score[1]));
			q.push(tuple<T,Z>(i+2,score[2]));
			q.push(tuple<T,Z>(i+3,score[3]));
			q.push(tuple<T,Z>(i+4,score[4]));
			q.push(tuple<T,Z>(i+5,score[5]));
			q.push(tuple<T,Z>(i+6,score[6]));
			q.push(tuple<T,Z>(i+7,score[7]));
			q.push(tuple<T,Z>(i+8,score[8]));
			q.push(tuple<T,Z>(i+9,score[9]));
			q.push(tuple<T,Z>(i+10,score[10]));
			q.push(tuple<T,Z>(i+11,score[11]));
			q.push(tuple<T,Z>(i+12,score[12]));
			q.push(tuple<T,Z>(i+13,score[13]));
			q.push(tuple<T,Z>(i+14,score[14]));
			q.push(tuple<T,Z>(i+15,score[15]));
		}else{//delete smallest element if current score is bigger
			if(q.top().score < score[0]){ q.pop(); q.push(tuple<T,Z>(i,score[0])); }
			if(q.top().score < score[1]){ q.pop(); q.push(tuple<T,Z>(i+1,score[1])); }
			if(q.top().score < score[2]){ q.pop(); q.push(tuple<T,Z>(i+2,score[2])); }
			if(q.top().score < score[3]){ q.pop(); q.push(tuple<T,Z>(i+3,score[3])); }
			if(q.top().score < score[4]){ q.pop(); q.push(tuple<T,Z>(i+4,score[4])); }
			if(q.top().score < score[5]){ q.pop(); q.push(tuple<T,Z>(i+5,score[5])); }
			if(q.top().score < score[6]){ q.pop(); q.push(tuple<T,Z>(i+6,score[6])); }
			if(q.top().score < score[7]){ q.pop(); q.push(tuple<T,Z>(i+7,score[7])); }
			if(q.top().score < score[8]){ q.pop(); q.push(tuple<T,Z>(i+8,score[8])); }
			if(q.top().score < score[9]){ q.pop(); q.push(tuple<T,Z>(i+9,score[9])); }
			if(q.top().score < score[10]){ q.pop(); q.push(tuple<T,Z>(i+10,score[10])); }
			if(q.top().score < score[11]){ q.pop(); q.push(tuple<T,Z>(i+11,score[11])); }
			if(q.top().score < score[12]){ q.pop(); q.push(tuple<T,Z>(i+12,score[12])); }
			if(q.top().score < score[13]){ q.pop(); q.push(tuple<T,Z>(i+13,score[13])); }
			if(q.top().score < score[14]){ q.pop(); q.push(tuple<T,Z>(i+14,score[14])); }
			if(q.top().score < score[15]){ q.pop(); q.push(tuple<T,Z>(i+15,score[15])); }
		}

		if((q.top().score) > (this->gt_array[i+15]*this->d) ){
//		if((q.top().score) > (this->gt_array[i+3]) ){
			if(STATS_EFF) this->tuple_count = i+15;
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << this->gt_array[i+7] << std::endl;
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
