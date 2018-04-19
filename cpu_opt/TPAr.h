#ifndef TPA_R_H
#define TPA_R_H

#include "../cpu/AA.h"

template<class T, class Z>
class  TPAr : public AA<T,Z>{
	public:
		TPAr(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "TPAr";
			this->scores = NULL;
		}

		~TPAr(){
			if(this->scores!=NULL) free(this->scores);
		}

		void init();
		void findTopKscalar(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);
		void findTopKsimd(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);
		void findTopKthreads(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);

	private:
		T *scores;
};

template<class T, class Z>
void TPAr<T,Z>::init(){
	this->scores = (T*) malloc(sizeof(T)*this->n);
	this->t.start();

	this->tt_init = this->t.lap();
}

template<class T, class Z>
void TPAr<T,Z>::findTopKscalar(uint64_t k,uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " scalar (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
	//boost::heap::priority_queue<tuple_<T,Z>,boost::heap::compare<MaxCMP<T,Z>>> q;
	this->t.start();
	for(uint64_t i = 0; i < this->n; i+=8){
		T score00 = 0; T score01 = 0; T score02 = 0; T score03 = 0;
		T score04 = 0; T score05 = 0; T score06 = 0; T score07 = 0;
		for(uint8_t m = 0; m < qq; m++){
			T weight = weights[attr[m]];
			uint64_t offset0 = i * this->d;
			uint64_t offset1 = (i+1) * this->d;
			uint64_t offset2 = (i+2) * this->d;
			uint64_t offset3 = (i+3) * this->d;
			uint64_t offset4 = (i+4) * this->d;
			uint64_t offset5 = (i+5) * this->d;
			uint64_t offset6 = (i+6) * this->d;
			uint64_t offset7 = (i+7) * this->d;
			score00 += this->cdata[offset0 + attr[m]]*weight;
			score01 += this->cdata[offset1 + attr[m]]*weight;
			score02 += this->cdata[offset2 + attr[m]]*weight;
			score03 += this->cdata[offset3 + attr[m]]*weight;
			score04 += this->cdata[offset4 + attr[m]]*weight;
			score05 += this->cdata[offset5 + attr[m]]*weight;
			score06 += this->cdata[offset6 + attr[m]]*weight;
			score07 += this->cdata[offset7 + attr[m]]*weight;
		}

		if(q.size() < k){
			q.push(tuple_<T,Z>(i,score00));
			q.push(tuple_<T,Z>(i+1,score01));
			q.push(tuple_<T,Z>(i+2,score02));
			q.push(tuple_<T,Z>(i+3,score03));
			q.push(tuple_<T,Z>(i+4,score04));
			q.push(tuple_<T,Z>(i+5,score05));
			q.push(tuple_<T,Z>(i+6,score06));
			q.push(tuple_<T,Z>(i+7,score07));
		}else{
			if(q.top().score < score00){ q.pop(); q.push(tuple_<T,Z>(i,score00)); }
			if(q.top().score < score01){ q.pop(); q.push(tuple_<T,Z>(i+1,score01)); }
			if(q.top().score < score02){ q.pop(); q.push(tuple_<T,Z>(i+2,score02)); }
			if(q.top().score < score03){ q.pop(); q.push(tuple_<T,Z>(i+3,score03)); }
			if(q.top().score < score04){ q.pop(); q.push(tuple_<T,Z>(i+4,score04)); }
			if(q.top().score < score05){ q.pop(); q.push(tuple_<T,Z>(i+5,score05)); }
			if(q.top().score < score06){ q.pop(); q.push(tuple_<T,Z>(i+6,score06)); }
			if(q.top().score < score07){ q.pop(); q.push(tuple_<T,Z>(i+7,score07)); }
		}
	}

	this->tt_processing += this->t.lap();
	if(STATS_EFF) this->tuple_count=this->n;

	while(q.size() > k){ q.pop(); }
	T threshold = q.top().score;
	while(!q.empty()){
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void TPAr<T,Z>::findTopKsimd(uint64_t k,uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " simd (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
	float score[16] __attribute__((aligned(32)));
	this->t.start();
	__builtin_prefetch(score,1,3);
	for(uint64_t i = 0; i < this->n; i+=16){
		T score0 = 0; T score1 = 0; T score2 = 0; T score3 = 0;
		T score4 = 0; T score5 = 0; T score6 = 0; T score7 = 0;
		__m256 score00 = _mm256_setzero_ps();
		__m256 score01 = _mm256_setzero_ps();

		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset0 = i * this->d;
			uint64_t offset1 = (i+1) * this->d;
			uint64_t offset2 = (i+2) * this->d;
			uint64_t offset3 = (i+3) * this->d;
			uint64_t offset4 = (i+4) * this->d;
			uint64_t offset5 = (i+5) * this->d;
			uint64_t offset6 = (i+6) * this->d;
			uint64_t offset7 = (i+7) * this->d;
			uint64_t offset8 = (i+8) * this->d;
			uint64_t offset9 = (i+9) * this->d;
			uint64_t offset10 = (i+10) * this->d;
			uint64_t offset11 = (i+11) * this->d;
			uint64_t offset12 = (i+12) * this->d;
			uint64_t offset13 = (i+13) * this->d;
			uint64_t offset14 = (i+14) * this->d;
			uint64_t offset15 = (i+15) * this->d;

			T weight = weights[attr[m]];
			__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
			__m256 load00 = _mm256_set_ps(
					this->cdata[offset0 + attr[m]],
					this->cdata[offset1 + attr[m]],
					this->cdata[offset2 + attr[m]],
					this->cdata[offset3 + attr[m]],
					this->cdata[offset4 + attr[m]],
					this->cdata[offset5 + attr[m]],
					this->cdata[offset6 + attr[m]],
					this->cdata[offset7 + attr[m]]);

			__m256 load01 = _mm256_set_ps(
					this->cdata[offset8 + attr[m]],
					this->cdata[offset9 + attr[m]],
					this->cdata[offset10 + attr[m]],
					this->cdata[offset11 + attr[m]],
					this->cdata[offset12 + attr[m]],
					this->cdata[offset13 + attr[m]],
					this->cdata[offset14 + attr[m]],
					this->cdata[offset15 + attr[m]]);

			load00 = _mm256_mul_ps(load00,_weight);
			load01 = _mm256_mul_ps(load01,_weight);
			score00 = _mm256_add_ps(score00,load00);
			score01 = _mm256_add_ps(score01,load01);
		}

		_mm256_store_ps(&score[0],score00);
		_mm256_store_ps(&score[8],score01);
		if(q.size() < k){//insert if empty space in queue
			q.push(tuple_<T,Z>(i,score[0]));
			q.push(tuple_<T,Z>(i+1,score[1]));
			q.push(tuple_<T,Z>(i+2,score[2]));
			q.push(tuple_<T,Z>(i+3,score[3]));
			q.push(tuple_<T,Z>(i+4,score[4]));
			q.push(tuple_<T,Z>(i+5,score[5]));
			q.push(tuple_<T,Z>(i+6,score[6]));
			q.push(tuple_<T,Z>(i+7,score[7]));
			q.push(tuple_<T,Z>(i+8,score[8]));
			q.push(tuple_<T,Z>(i+9,score[9]));
			q.push(tuple_<T,Z>(i+10,score[10]));
			q.push(tuple_<T,Z>(i+11,score[11]));
			q.push(tuple_<T,Z>(i+12,score[12]));
			q.push(tuple_<T,Z>(i+13,score[13]));
			q.push(tuple_<T,Z>(i+14,score[14]));
			q.push(tuple_<T,Z>(i+15,score[15]));
		}else{//delete smallest element if current score is bigger
			if(q.top().score < score[0]){ q.pop(); q.push(tuple_<T,Z>(i,score[0])); }
			if(q.top().score < score[1]){ q.pop(); q.push(tuple_<T,Z>(i+1,score[1])); }
			if(q.top().score < score[2]){ q.pop(); q.push(tuple_<T,Z>(i+2,score[2])); }
			if(q.top().score < score[3]){ q.pop(); q.push(tuple_<T,Z>(i+3,score[3])); }
			if(q.top().score < score[4]){ q.pop(); q.push(tuple_<T,Z>(i+4,score[4])); }
			if(q.top().score < score[5]){ q.pop(); q.push(tuple_<T,Z>(i+5,score[5])); }
			if(q.top().score < score[6]){ q.pop(); q.push(tuple_<T,Z>(i+6,score[6])); }
			if(q.top().score < score[7]){ q.pop(); q.push(tuple_<T,Z>(i+7,score[7])); }
			if(q.top().score < score[8]){ q.pop(); q.push(tuple_<T,Z>(i+8,score[8])); }
			if(q.top().score < score[9]){ q.pop(); q.push(tuple_<T,Z>(i+9,score[9])); }
			if(q.top().score < score[10]){ q.pop(); q.push(tuple_<T,Z>(i+10,score[10])); }
			if(q.top().score < score[11]){ q.pop(); q.push(tuple_<T,Z>(i+11,score[11])); }
			if(q.top().score < score[12]){ q.pop(); q.push(tuple_<T,Z>(i+12,score[12])); }
			if(q.top().score < score[13]){ q.pop(); q.push(tuple_<T,Z>(i+13,score[13])); }
			if(q.top().score < score[14]){ q.pop(); q.push(tuple_<T,Z>(i+14,score[14])); }
			if(q.top().score < score[15]){ q.pop(); q.push(tuple_<T,Z>(i+15,score[15])); }
		}
	}
	this->tt_processing += this->t.lap();
	if(STATS_EFF) this->tuple_count=this->n;

	while(q.size() > k){ q.pop(); }
	T threshold = q.top().score;
	while(!q.empty()){
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void TPAr<T,Z>::findTopKthreads(uint64_t k,uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " threads (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	omp_set_num_threads(THREADS);
	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q[THREADS];
	this->t.start();
#pragma omp parallel
{
	uint32_t thread_id = omp_get_thread_num();
	float score[16] __attribute__((aligned(32)));
	__builtin_prefetch(score,1,3);
	uint64_t start = ((uint64_t)thread_id)*(this->n)/THREADS;
	uint64_t end = ((uint64_t)(thread_id+1))*(this->n)/THREADS;
	for(uint64_t i = start; i < end; i+=16){
		T score0 = 0; T score1 = 0; T score2 = 0; T score3 = 0;
		T score4 = 0; T score5 = 0; T score6 = 0; T score7 = 0;
		__m256 score00 = _mm256_setzero_ps();
		__m256 score01 = _mm256_setzero_ps();

		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset0 = i * this->d;
			uint64_t offset1 = (i+1) * this->d;
			uint64_t offset2 = (i+2) * this->d;
			uint64_t offset3 = (i+3) * this->d;
			uint64_t offset4 = (i+4) * this->d;
			uint64_t offset5 = (i+5) * this->d;
			uint64_t offset6 = (i+6) * this->d;
			uint64_t offset7 = (i+7) * this->d;
			uint64_t offset8 = (i+8) * this->d;
			uint64_t offset9 = (i+9) * this->d;
			uint64_t offset10 = (i+10) * this->d;
			uint64_t offset11 = (i+11) * this->d;
			uint64_t offset12 = (i+12) * this->d;
			uint64_t offset13 = (i+13) * this->d;
			uint64_t offset14 = (i+14) * this->d;
			uint64_t offset15 = (i+15) * this->d;

			T weight = weights[attr[m]];
			__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
			__m256 load00 = _mm256_set_ps(
					this->cdata[offset0 + attr[m]],
					this->cdata[offset1 + attr[m]],
					this->cdata[offset2 + attr[m]],
					this->cdata[offset3 + attr[m]],
					this->cdata[offset4 + attr[m]],
					this->cdata[offset5 + attr[m]],
					this->cdata[offset6 + attr[m]],
					this->cdata[offset7 + attr[m]]);

			__m256 load01 = _mm256_set_ps(
					this->cdata[offset8 + attr[m]],
					this->cdata[offset9 + attr[m]],
					this->cdata[offset10 + attr[m]],
					this->cdata[offset11 + attr[m]],
					this->cdata[offset12 + attr[m]],
					this->cdata[offset13 + attr[m]],
					this->cdata[offset14 + attr[m]],
					this->cdata[offset15 + attr[m]]);

			load00 = _mm256_mul_ps(load00,_weight);
			load01 = _mm256_mul_ps(load01,_weight);
			score00 = _mm256_add_ps(score00,load00);
			score01 = _mm256_add_ps(score01,load01);
		}

		_mm256_store_ps(&score[0],score00);
		_mm256_store_ps(&score[8],score01);
		if(q[thread_id].size() < k){//insert if empty space in queue
			q[thread_id].push(tuple_<T,Z>(i,score[0]));
			q[thread_id].push(tuple_<T,Z>(i+1,score[1]));
			q[thread_id].push(tuple_<T,Z>(i+2,score[2]));
			q[thread_id].push(tuple_<T,Z>(i+3,score[3]));
			q[thread_id].push(tuple_<T,Z>(i+4,score[4]));
			q[thread_id].push(tuple_<T,Z>(i+5,score[5]));
			q[thread_id].push(tuple_<T,Z>(i+6,score[6]));
			q[thread_id].push(tuple_<T,Z>(i+7,score[7]));
			q[thread_id].push(tuple_<T,Z>(i+8,score[8]));
			q[thread_id].push(tuple_<T,Z>(i+9,score[9]));
			q[thread_id].push(tuple_<T,Z>(i+10,score[10]));
			q[thread_id].push(tuple_<T,Z>(i+11,score[11]));
			q[thread_id].push(tuple_<T,Z>(i+12,score[12]));
			q[thread_id].push(tuple_<T,Z>(i+13,score[13]));
			q[thread_id].push(tuple_<T,Z>(i+14,score[14]));
			q[thread_id].push(tuple_<T,Z>(i+15,score[15]));
		}else{//delete smallest element if current score is bigger
			if(q[thread_id].top().score < score[0]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i,score[0])); }
			if(q[thread_id].top().score < score[1]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+1,score[1])); }
			if(q[thread_id].top().score < score[2]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+2,score[2])); }
			if(q[thread_id].top().score < score[3]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+3,score[3])); }
			if(q[thread_id].top().score < score[4]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+4,score[4])); }
			if(q[thread_id].top().score < score[5]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+5,score[5])); }
			if(q[thread_id].top().score < score[6]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+6,score[6])); }
			if(q[thread_id].top().score < score[7]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+7,score[7])); }
			if(q[thread_id].top().score < score[8]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+8,score[8])); }
			if(q[thread_id].top().score < score[9]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+9,score[9])); }
			if(q[thread_id].top().score < score[10]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+10,score[10])); }
			if(q[thread_id].top().score < score[11]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+11,score[11])); }
			if(q[thread_id].top().score < score[12]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+12,score[12])); }
			if(q[thread_id].top().score < score[13]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+13,score[13])); }
			if(q[thread_id].top().score < score[14]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+14,score[14])); }
			if(q[thread_id].top().score < score[15]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+15,score[15])); }
		}
	}
}
	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> _q;
	for(uint32_t m = 0; m < THREADS; m++){
		while(!q[m].empty()){
			if(_q.size() < k){
				_q.push(q[m].top());
			}else if(_q.top().score < q[m].top().score){
				_q.pop();
				_q.push(q[m].top());
			}
			q[m].pop();
		}
	}
	this->tt_processing += this->t.lap();

	while(_q.size() > k){ _q.pop(); }
	T threshold = _q.top().score;
	while(!_q.empty()){
		this->res.push_back(_q.top());
		_q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
