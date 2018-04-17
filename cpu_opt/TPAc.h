#ifndef TPA_C_F
#define TPA_C_F

#include "../cpu/AA.h"

template<class T, class Z>
class  TPAc : public AA<T,Z>{
	public:
		TPAc(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "TPAc";
			this->scores = NULL;
		}

		~TPAc(){
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
void TPAc<T,Z>::init(){
	std::cout << this->algo << " Init ..." << std::endl;
	this->scores = (T*) malloc(sizeof(T)*this->n);
	this->t.start();

	this->tt_init = this->t.lap();
}

template<class T, class Z>
void TPAc<T,Z>::findTopKscalar(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find top-" << k << " scalar (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	this->t.start();
	for(uint64_t i = 0; i < this->n; i+=8){
		T score00 = 0;
		T score01 = 0;
		T score02 = 0;
		T score03 = 0;
		T score04 = 0;
		T score05 = 0;
		T score06 = 0;
		T score07 = 0;

		uint64_t offset0 = i;
		uint64_t offset1 = i + 1;
		uint64_t offset2 = i + 2;
		uint64_t offset3 = i + 3;
		uint64_t offset4 = i + 4;
		uint64_t offset5 = i + 5;
		uint64_t offset6 = i + 6;
		uint64_t offset7 = i + 7;

		for(uint8_t m = 0; m < qq; m++){
			score00+= this->cdata[offset0];
			score01+= this->cdata[offset1];
			score02+= this->cdata[offset2];
			score03+= this->cdata[offset3];
			score04+= this->cdata[offset4];
			score05+= this->cdata[offset5];
			score06+= this->cdata[offset6];
			score07+= this->cdata[offset7];

			offset0+=this->n;
			offset1+=this->n;
			offset2+=this->n;
			offset3+=this->n;
			offset4+=this->n;
			offset5+=this->n;
			offset6+=this->n;
			offset7+=this->n;
		}
		scores[i] = score00;
		scores[i+1] = score01;
		scores[i+2] = score02;
		scores[i+3] = score03;
		scores[i+4] = score04;
		scores[i+5] = score05;
		scores[i+6] = score06;
		scores[i+7] = score07;
	}
	this->tt_processing += this->t.lap();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0;i < this->n; i++){
		if(q.size() < k){//insert if empty space in queue
			q.push(tuple_<T,Z>(i,scores[i]));
		}else if(q.top().score<scores[i]){//delete smallest element if current score is bigger
			q.pop();
			q.push(tuple_<T,Z>(i,scores[i]));
			if(STATS_EFF) this->pop_count++;
		}
	}
	this->tt_ranking += this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void TPAc<T,Z>::findTopKsimd(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find top-" << k << " simd (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	this->t.start();
	float score[16] __attribute__((aligned(32)));
	__builtin_prefetch(score,1,3);
	for(uint64_t i = 0; i < this->n; i+=16){
		__m256 score00 = _mm256_setzero_ps();
		__m256 score01 = _mm256_setzero_ps();
		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset00 = m * this->n + i;
			uint64_t offset01 = m * this->n + i + 8;
			__m256 load00 = _mm256_load_ps(&this->cdata[offset00]);
			__m256 load01 = _mm256_load_ps(&this->cdata[offset01]);
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
	this->tt_processing += this->t.lap();

	if(STATS_EFF) this->tuple_count=this->n;
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
	this->tt_ranking += this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	while(!q.empty()){
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void TPAc<T,Z>::findTopKthreads(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find top-" << k << " threads (" << (int)qq << "D) ...";
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
		__m256 score00 = _mm256_setzero_ps();
		__m256 score01 = _mm256_setzero_ps();
		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset00 = m * this->n + i;
			uint64_t offset01 = m * this->n + i + 8;
			__m256 load00 = _mm256_load_ps(&this->cdata[offset00]);
			__m256 load01 = _mm256_load_ps(&this->cdata[offset01]);
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
		scores[i+15]= score[15];
	}
}
	this->tt_processing += this->t.lap();

	if(STATS_EFF) this->tuple_count=this->n;
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
	this->tt_ranking += this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	while(!q.empty()){
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
