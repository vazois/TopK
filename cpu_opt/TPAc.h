#ifndef TPA_C_F
#define TPA_C_F

#include "../cpu/AA.h"

template<class T, class Z>
struct tpac_pair{
	Z id;
	T score;
};

template<class T, class Z>
class  TPAc : public AA<T,Z>{
	public:
		TPAc(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "TPAc";
			this->tuples = NULL;
			this->scores = NULL;
		}

		~TPAc(){
			if(this->tuples!=NULL) free(this->tuples);
			if(this->scores!=NULL) free(this->scores);
		}

		void init();
		void findTopKscalar(uint64_t k,uint8_t qq);
		void findTopKsimd(uint64_t k,uint8_t qq);
		void findTopKthreads(uint64_t k,uint8_t qq);
	private:
		tpac_pair<T,Z> *tuples;
		T *scores;
};

template<class T, class Z>
void TPAc<T,Z>::init(){
	std::cout << this->algo << " Init ..." << std::endl;
	this->tuples = (tpac_pair<T,Z>*) malloc(sizeof(tpac_pair<T,Z>)*this->n);
	this->scores = (T*) malloc(sizeof(T)*this->n);
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++){
		this->tuples[i].id = i;
		this->tuples[i].score = 0;
	}
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void TPAc<T,Z>::findTopKscalar(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topKscalar (" << (int)qq << "D) ...";
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

		//for(uint8_t m = 0; m < qq; m+=2){
		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset0 = m * this->n + i;
			//uint64_t offset1 = (m+1) * this->n + i;
			score00+= this->cdata[offset0];// + this->cdata[offset1];
			score01+= this->cdata[offset0+1];// + this->cdata[offset1+1];
			score02+= this->cdata[offset0+2];// + this->cdata[offset1+2];
			score03+= this->cdata[offset0+3];// + this->cdata[offset1+3];
			score04+= this->cdata[offset0+4];// + this->cdata[offset1+4];
			score05+= this->cdata[offset0+5];// + this->cdata[offset1+5];
			score06+= this->cdata[offset0+6];// + this->cdata[offset1+6];
			score07+= this->cdata[offset0+7];// + this->cdata[offset1+7];
		}
		this->tuples[i].score = score00;
		this->tuples[i+1].score = score01;
		this->tuples[i+2].score = score02;
		this->tuples[i+3].score = score03;
		this->tuples[i+4].score = score04;
		this->tuples[i+5].score = score05;
		this->tuples[i+6].score = score06;
		this->tuples[i+7].score = score07;
	}
	this->tt_processing += this->t.lap();

	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0;i < this->n; i++){
		if(q.size() < k){//insert if empty space in queue
			q.push(tuple<T,Z>(this->tuples[i].id,this->tuples[i].score));
		}else if(q.top().score<this->tuples[i].score){//delete smallest element if current score is bigger
			q.pop();
			q.push(tuple<T,Z>(this->tuples[i].id,this->tuples[i].score));
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
	std::cout << this->algo << " find topKsimd (" << (int)qq << "D) ...";
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
		this->tuples[i].score = score[0];
		this->tuples[i+1].score = score[1];
		this->tuples[i+2].score = score[2];
		this->tuples[i+3].score = score[3];
		this->tuples[i+4].score = score[4];
		this->tuples[i+5].score = score[5];
		this->tuples[i+6].score = score[6];
		this->tuples[i+7].score = score[7];
		this->tuples[i+8].score = score[8];
		this->tuples[i+9].score = score[9];
		this->tuples[i+10].score = score[10];
		this->tuples[i+11].score = score[11];
		this->tuples[i+12].score = score[12];
		this->tuples[i+13].score = score[13];
		this->tuples[i+14].score = score[14];
		this->tuples[i+15].score = score[15];
	}
	this->tt_processing += this->t.lap();

	if(STATS_EFF) this->tuple_count=this->n;
	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0;i < this->n; i++){
		if(q.size() < k){//insert if empty space in queue
			q.push(tuple<T,Z>(this->tuples[i].id,this->tuples[i].score));
			//count_insert++;
		}else if(q.top().score<this->tuples[i].score){//delete smallest element if current score is bigger
			q.pop();
			q.push(tuple<T,Z>(this->tuples[i].id,this->tuples[i].score));
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
	std::cout << this->algo << " find topKthreads (" << (int)qq << "D) ...";
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
	//std::cout << thread_id << ": " << start << "," << end << std::endl;
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
	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0;i < this->n; i++){
		if(q.size() < k){//insert if empty space in queue
			q.push(tuple<T,Z>(i,scores[i]));
			//count_insert++;
		}else if(q.top().score<scores[i]){//delete smallest element if current score is bigger
			q.pop();
			q.push(tuple<T,Z>(i,scores[i]));
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
