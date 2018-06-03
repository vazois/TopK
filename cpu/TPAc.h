#ifndef TPA_C_F
#define TPA_C_F

#include "AA.h"

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
		void findTopKscalar(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);
		void findTopKsimd(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);
		void findTopKthreads(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);
		void findTopKsimdMQ(uint64_t k,uint8_t qq, T *weights, uint8_t *attr, uint32_t tid);
	private:
		T *scores;
};

template<class T, class Z>
void TPAc<T,Z>::init(){
	normalize_transpose<T,Z>(this->cdata, this->n, this->d);
	this->t.start();
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void TPAc<T,Z>::findTopKscalar(uint64_t k,uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " scalar (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
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

		uint64_t offset0 = attr[0] * this->n + i;
		uint64_t offset1 = attr[0] * this->n + i + 1;
		uint64_t offset2 = attr[0] * this->n + i + 2;
		uint64_t offset3 = attr[0] * this->n + i + 3;
		uint64_t offset4 = attr[0] * this->n + i + 4;
		uint64_t offset5 = attr[0] * this->n + i + 5;
		uint64_t offset6 = attr[0] * this->n + i + 6;
		uint64_t offset7 = attr[0] * this->n + i + 7;

		for(uint8_t m = 0; m < qq; m++){
			T weight = weights[attr[m]];
			score00+= this->cdata[offset0]*weight;
			score01+= this->cdata[offset1]*weight;
			score02+= this->cdata[offset2]*weight;
			score03+= this->cdata[offset3]*weight;
			score04+= this->cdata[offset4]*weight;
			score05+= this->cdata[offset5]*weight;
			score06+= this->cdata[offset6]*weight;
			score07+= this->cdata[offset7]*weight;

			offset0+=this->n;
			offset1+=this->n;
			offset2+=this->n;
			offset3+=this->n;
			offset4+=this->n;
			offset5+=this->n;
			offset6+=this->n;
			offset7+=this->n;
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
void TPAc<T,Z>::findTopKsimd(uint64_t k,uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " simd (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
	float score[16] __attribute__((aligned(32)));
	this->t.start();
	__m256 dim_num = _mm256_set_ps(qq,qq,qq,qq,qq,qq,qq,qq);
	__builtin_prefetch(score,1,3);
	for(uint64_t i = 0; i < this->n; i+=16){
		__m256 score00 = _mm256_setzero_ps();
		__m256 score01 = _mm256_setzero_ps();
		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset00 = attr[m] * this->n + i;
			uint64_t offset01 = attr[m] * this->n + i + 8;
			T weight = weights[attr[m]];
			__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
			__m256 load00 = _mm256_load_ps(&this->cdata[offset00]);
			__m256 load01 = _mm256_load_ps(&this->cdata[offset01]);
			load00 = _mm256_mul_ps(load00,_weight);
			load01 = _mm256_mul_ps(load01,_weight);
			score00 = _mm256_add_ps(score00,load00);
			score01 = _mm256_add_ps(score01,load01);
			#if LD == 2
				score00 = _mm256_div_ps(score00,dim_num);
				score01 = _mm256_div_ps(score01,dim_num);
			#endif
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

//		for(uint8_t l = 0; l < 16; l++){
//			if(q.size() < k){
//				q.push(tuple_<T,Z>(l,score[l]));
//			}else if(q.top().score < score[l]){
//				q.pop(); q.push(tuple_<T,Z>(l,score[l]));
//			}
//		}
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
void TPAc<T,Z>::findTopKsimdMQ(uint64_t k,uint8_t qq, T *weights, uint8_t *attr, uint32_t tid){
	Time<msecs> t;
	//std::cout << this->algo << " find top-" << k << " simdMQ (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
	float score[16] __attribute__((aligned(32)));
	t.start();
	__builtin_prefetch(score,1,3);
	__m256 dim_num = _mm256_set_ps(qq,qq,qq,qq,qq,qq,qq,qq);
	for(uint64_t i = 0; i < this->n; i+=16){
		__m256 score00 = _mm256_setzero_ps();
		__m256 score01 = _mm256_setzero_ps();
		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset00 = attr[m] * this->n + i;
			uint64_t offset01 = attr[m] * this->n + i + 8;
			T weight = weights[attr[m]];
			__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
			__m256 load00 = _mm256_load_ps(&this->cdata[offset00]);
			__m256 load01 = _mm256_load_ps(&this->cdata[offset01]);
			load00 = _mm256_mul_ps(load00,_weight);
			load01 = _mm256_mul_ps(load01,_weight);
			score00 = _mm256_add_ps(score00,load00);
			score01 = _mm256_add_ps(score01,load01);

			#if LD == 2
				score00 = _mm256_div_ps(score00,dim_num);
				score01 = _mm256_div_ps(score01,dim_num);
			#endif
		}
		_mm256_store_ps(&score[0],score00);
		_mm256_store_ps(&score[8],score01);

		for(uint8_t l = 0; l < 16; l++){
			if(q.size() < k){
				q.push(tuple_<T,Z>(l,score[l]));
			}else if(q.top().score < score[l]){
				q.pop(); q.push(tuple_<T,Z>(l,score[l]));
			}
		}
	}
	this->tt_array[tid] += t.lap();
}

template<class T, class Z>
void TPAc<T,Z>::findTopKthreads(uint64_t k,uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " threads (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	omp_set_num_threads(THREADS);
	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q[THREADS];
	//boost::heap::priority_queue<tuple_<T,Z>,boost::heap::compare<MaxCMP<T,Z>>> q[THREADS];
	this->t.start();
#pragma omp parallel
{
	uint32_t thread_id = omp_get_thread_num();
	float score[16] __attribute__((aligned(32)));
	__builtin_prefetch(score,1,3);
	uint64_t start = ((uint64_t)thread_id)*(this->n)/THREADS;
	uint64_t end = ((uint64_t)(thread_id+1))*(this->n)/THREADS;

	__m256 dim_num = _mm256_set_ps(qq,qq,qq,qq,qq,qq,qq,qq);
	for(uint64_t i = start; i < end; i+=16){
		__m256 score00 = _mm256_setzero_ps();
		__m256 score01 = _mm256_setzero_ps();
		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset00 = attr[m] * this->n + i;
			uint64_t offset01 = attr[m] * this->n + i + 8;
			T weight = weights[attr[m]];
			__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
			__m256 load00 = _mm256_load_ps(&this->cdata[offset00]);
			__m256 load01 = _mm256_load_ps(&this->cdata[offset01]);
			load00 = _mm256_mul_ps(load00,_weight);
			load01 = _mm256_mul_ps(load01,_weight);
			score00 = _mm256_add_ps(score00,load00);
			score01 = _mm256_add_ps(score01,load01);

			#if LD == 2
				score00 = _mm256_div_ps(score00,dim_num);
				score01 = _mm256_div_ps(score01,dim_num);
			#endif
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
