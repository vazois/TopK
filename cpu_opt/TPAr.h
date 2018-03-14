#ifndef TPA_R_H
#define TPA_R_H

#include "../cpu/AA.h"

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
		void findTopKscalar(uint64_t k,uint8_t qq);
		void findTopKsimd(uint64_t k,uint8_t qq);
		void findTopKthreads(uint64_t k,uint8_t qq);

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
void TPAr<T,Z>::findTopKscalar(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topKscalar (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	this->t.start();
	for(uint64_t i = 0; i < this->n; i+=8){
		T score0 = 0; T score1 = 0; T score2 = 0; T score3 = 0;
		T score4 = 0; T score5 = 0; T score6 = 0; T score7 = 0;
		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset0 = i * this->d;
			uint64_t offset1 = (i+1) * this->d;
			uint64_t offset2 = (i+2) * this->d;
			uint64_t offset3 = (i+3) * this->d;
			uint64_t offset4 = (i+4) * this->d;
			uint64_t offset5 = (i+5) * this->d;
			uint64_t offset6 = (i+6) * this->d;
			uint64_t offset7 = (i+7) * this->d;
			score0 += this->cdata[offset0 + m];
			score1 += this->cdata[offset1 + m];
			score2 += this->cdata[offset2 + m];
			score3 += this->cdata[offset3 + m];
			score4 += this->cdata[offset4 + m];
			score5 += this->cdata[offset5 + m];
			score6 += this->cdata[offset6 + m];
			score7 += this->cdata[offset7 + m];
		}
		scores[i] = score0;
		scores[i+1] = score1;
		scores[i+2] = score2;
		scores[i+3] = score3;
		scores[i+4] = score4;
		scores[i+5] = score5;
		scores[i+6] = score6;
		scores[i+7] = score7;
	}
	this->tt_processing = this->t.lap();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0;i < this->n; i++){
		if(q.size() < k){//insert if empty space in queue
			q.push(tuple_<T,Z>(i,scores[i]));
			//count_insert++;
		}else if(q.top().score<scores[i]){//delete smallest element if current score is bigger
			q.pop();
			q.push(tuple_<T,Z>(i,scores[i]));
			//count_pop++;
		}
	}
	this->tt_ranking = this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void TPAr<T,Z>::findTopKsimd(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topKscalar (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	this->t.start();
	float score[16] __attribute__((aligned(32)));
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

			__m256 load00 = _mm256_set_ps(
					this->cdata[offset0 + m],
					this->cdata[offset1 + m],
					this->cdata[offset2 + m],
					this->cdata[offset3 + m],
					this->cdata[offset4 + m],
					this->cdata[offset5 + m],
					this->cdata[offset6 + m],
					this->cdata[offset7 + m]);

			__m256 load01 = _mm256_set_ps(
					this->cdata[offset8 + m],
					this->cdata[offset9 + m],
					this->cdata[offset10 + m],
					this->cdata[offset11 + m],
					this->cdata[offset12 + m],
					this->cdata[offset13 + m],
					this->cdata[offset14 + m],
					this->cdata[offset15 + m]);

			score00 = _mm256_add_ps(score00,load00);
			score01 = _mm256_add_ps(score01,load01);
		}

		_mm256_store_ps(&score[0],score00);
		_mm256_store_ps(&score[8],score01);
		scores[i] = score[0];
		scores[i+1] = score[1];
		scores[i+2] = score[2];
		scores[i+3] = score[3];
		scores[i+4] = score[4];
		scores[i+5] = score[5];
		scores[i+6] = score[6];
		scores[i+7] = score[7];
		scores[i+8] = score[8];
		scores[i+9] = score[9];
		scores[i+10] = score[10];
		scores[i+11] = score[11];
		scores[i+12] = score[12];
		scores[i+13] = score[13];
		scores[i+14] = score[14];
		scores[i+15] = score[15];
	}
	this->tt_processing = this->t.lap();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0;i < this->n; i++){
		if(q.size() < k){//insert if empty space in queue
			q.push(tuple_<T,Z>(i,scores[i]));
			//count_insert++;
		}else if(q.top().score<scores[i]){//delete smallest element if current score is bigger
			q.pop();
			q.push(tuple_<T,Z>(i,scores[i]));
			//count_pop++;
		}
	}
	this->tt_ranking = this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void TPAr<T,Z>::findTopKthreads(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topKscalar (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	omp_set_num_threads(THREADS);
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

			__m256 load00 = _mm256_set_ps(
					this->cdata[offset0 + m],
					this->cdata[offset1 + m],
					this->cdata[offset2 + m],
					this->cdata[offset3 + m],
					this->cdata[offset4 + m],
					this->cdata[offset5 + m],
					this->cdata[offset6 + m],
					this->cdata[offset7 + m]);

			__m256 load01 = _mm256_set_ps(
					this->cdata[offset8 + m],
					this->cdata[offset9 + m],
					this->cdata[offset10 + m],
					this->cdata[offset11 + m],
					this->cdata[offset12 + m],
					this->cdata[offset13 + m],
					this->cdata[offset14 + m],
					this->cdata[offset15 + m]);

			score00 = _mm256_add_ps(score00,load00);
			score01 = _mm256_add_ps(score01,load01);
		}

		_mm256_store_ps(&score[0],score00);
		_mm256_store_ps(&score[8],score01);
		scores[i] = score[0];
		scores[i+1] = score[1];
		scores[i+2] = score[2];
		scores[i+3] = score[3];
		scores[i+4] = score[4];
		scores[i+5] = score[5];
		scores[i+6] = score[6];
		scores[i+7] = score[7];
		scores[i+8] = score[8];
		scores[i+9] = score[9];
		scores[i+10] = score[10];
		scores[i+11] = score[11];
		scores[i+12] = score[12];
		scores[i+13] = score[13];
		scores[i+14] = score[14];
		scores[i+15] = score[15];
	}
}
	this->tt_processing = this->t.lap();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0;i < this->n; i++){
		if(q.size() < k){//insert if empty space in queue
			q.push(tuple_<T,Z>(i,scores[i]));
			//count_insert++;
		}else if(q.top().score<scores[i]){//delete smallest element if current score is bigger
			q.pop();
			q.push(tuple_<T,Z>(i,scores[i]));
			//count_pop++;
		}
	}
	this->tt_ranking = this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
