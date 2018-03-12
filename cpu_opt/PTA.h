#ifndef PTA_H
#define PTA_H

#include "../cpu/AA.h"

template<class T, class Z>
struct pta_pair{
	Z id;
	T score;
};

template<class T, class Z>
struct pta_pt{
	Z id;
	Z pos;
	T score;
};

template<class Z>
struct pta_ps{
	Z id;
	Z pos;
};

template<class T,class Z>
static bool cmp_pta_pair(const pta_pair<T,Z> &a, const pta_pair<T,Z> &b){ return a.score > b.score; };

template<class T,class Z>
static bool cmp_pta_pt_pos(const pta_pt<T,Z> &a, const pta_pt<T,Z> &b){ return a.pos < b.pos; };

template<class Z>
static bool cmp_pta_ps(const pta_ps<Z> &a, const pta_ps<Z> &b){ return a.pos < b.pos; };

template<class T,class Z>
static bool cmp_pta_pt_pos_score(const pta_pt<T,Z> &a, const pta_pt<T,Z> &b){
	if(a.pos == b.pos){
		return a.score > b.score;
	}else{
		return a.pos < b.pos;
	}

};

template<class T, class Z>
class PTA : public AA<T,Z>{
	public:
		PTA(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "PTA";
			this->pt = NULL;
			this->tarray=NULL;
		}

		~PTA(){
			if(this->pt != NULL) free(this->pt);
			if(this->tarray != NULL){ free(this->tarray); }
		}

		void init();
		void findTopKscalar(uint64_t k,uint8_t qq);
		void findTopKsimd(uint64_t k,uint8_t qq);
		void findTopKthreads(uint64_t k,uint8_t qq);
	private:
		pta_pt<T,Z> *pt;
		T *tarray;
};

template<class T, class Z>
void PTA<T,Z>::init(){
	pta_pair<T,Z> *list = (pta_pair<T,Z>*)malloc(sizeof(pta_pair<T,Z>)*this->n);
	pta_ps<Z> *tpos = (pta_ps<Z>*)malloc(sizeof(pta_ps<Z>)*this->n);

	this->t.start();
	//Find seen position for each tuple//
	for(uint64_t i = 0; i < this->n; i++){ tpos[i].id = i; tpos[i].pos = this->n; }
	for(uint8_t m = 0; m < this->d; m++){
		for(uint64_t i = 0; i < this->n; i++){
			list[i].id = i;
			list[i].score = this->cdata[m*this->n + i];
		}
		__gnu_parallel::sort(&list[0],(&list[0]) + this->n,cmp_pta_pair<T,Z>);//data based on attribute//

		for(uint64_t i = 0; i < this->n; i++){
			pta_pair<T,Z> p = list[i];
			if(tpos[p.id].pos > i){ tpos[p.id].pos = i; }
		}
	}
	free(list);
	///////////////////////////////////////////////////////////////////////
	omp_set_num_threads(ITHREADS);
	//Reorder base table using seen position//
	__gnu_parallel::sort(&tpos[0],(&tpos[0]) + this->n,cmp_pta_ps<Z>);//data based on attribute//

	T *cdata = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->n) * (this->d)));
	#pragma omp parallel
	{
		uint32_t thread_id = omp_get_thread_num();
		uint64_t start = ((uint64_t)thread_id)*(this->n)/ITHREADS;
		uint64_t end = ((uint64_t)(thread_id+1))*(this->n)/ITHREADS;
		for(uint64_t i = start; i < end; i++){
			for(uint8_t m = 0; m < this->d; m++){
				cdata[m * this->n + i] = this->cdata[m * this->n + tpos[i].id];
			}
		}
	}
	free(this->cdata); this->cdata = cdata;
	///////////////////////////////////////////////////////////////////////
//	for(uint64_t i = 0; i < 25; i++){
//		std::cout << std::dec << std::setfill('0') << std::setw(4);
//		std::cout << "<" << tpos[i].pos << ">";
//		std::cout << std::fixed << std::setprecision(4);
//		std::cout << " || ";
//		for(uint8_t m = 0; m < this->d; m++){
//			std::cout << this->cdata[m * this->n + i] << " ";
//		}
//		std::cout << " || ";
//		std::cout << std::endl;
//	}

	//Create Threshold array
	this->tarray = (T*)malloc(sizeof(T)*(this->n >> 4) * this->d);
	list = (pta_pair<T,Z>*)malloc(sizeof(pta_pair<T,Z>)*this->n);
	for(uint8_t m = 0; m < this->d; m++){
		for(uint64_t i = 0; i < this->n; i++){
			list[i].id = i;
			list[i].score = this->cdata[m*this->n + i];
		}
		__gnu_parallel::sort(&list[0],(&list[0]) + this->n,cmp_pta_pair<T,Z>);
		uint64_t ii = 0;
		for(uint64_t i = 0; i < this->n; i+=16){
			this->tarray[m*(this->n >> 4) + ii] = list[tpos[i].pos].score;
			ii++;
		}
	}

	free(tpos);
	free(list);
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void PTA<T,Z>::findTopKscalar(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topKscalar (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	this->tt_ranking = 0;
	uint64_t ii = 0;
	for(uint64_t i = 0; i < this->n; i+=16){
		T score00 = 0; T score01 = 0; T score02 = 0; T score03 = 0; T score04 = 0; T score05 = 0; T score06 = 0; T score07 = 0;
		T score08 = 0; T score09 = 0; T score10 = 0; T score11 = 0; T score12 = 0; T score13 = 0; T score14 = 0; T score15 = 0;

		T threshold = 0;
		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset0 = m * this->n + i;
			T a00 = this->cdata[offset0+0]; T a01 = this->cdata[offset0+1]; T a02 = this->cdata[offset0+2]; T a03 = this->cdata[offset0+3];
			T a04 = this->cdata[offset0+4]; T a05 = this->cdata[offset0+5]; T a06 = this->cdata[offset0+6]; T a07 = this->cdata[offset0+7];
			T a08 = this->cdata[offset0+8]; T a09 = this->cdata[offset0+9]; T a10 = this->cdata[offset0+10]; T a11 = this->cdata[offset0+11];
			T a12 = this->cdata[offset0+12]; T a13 = this->cdata[offset0+13]; T a14 = this->cdata[offset0+14]; T a15 = this->cdata[offset0+15];
			score00+= a00; score01+= a01; score02+= a02; score03+= a03; score04+= a04; score05+= a05; score06+= a06; score07+= a07;
			score08+= a08; score09+= a09; score10+= a10; score11+= a11; score12+= a12; score13+= a13; score14+= a14; score15+= a15;

			uint64_t toffset0 = m * (this->n >> 4) + ii;
			threshold += this->tarray[toffset0];
		}
		ii++;

		if(q.size() < k){//insert if empty space in queue
			q.push(tuple_<T,Z>(i,score00)); q.push(tuple_<T,Z>(i+1,score01)); q.push(tuple_<T,Z>(i+2,score02)); q.push(tuple_<T,Z>(i+3,score03));
			q.push(tuple_<T,Z>(i+4,score04)); q.push(tuple_<T,Z>(i+5,score05)); q.push(tuple_<T,Z>(i+6,score06)); q.push(tuple_<T,Z>(i+7,score07));
			q.push(tuple_<T,Z>(i+8,score08)); q.push(tuple_<T,Z>(i+9,score09)); q.push(tuple_<T,Z>(i+10,score10)); q.push(tuple_<T,Z>(i+11,score11));
			q.push(tuple_<T,Z>(i+12,score12)); q.push(tuple_<T,Z>(i+13,score13)); q.push(tuple_<T,Z>(i+14,score14)); q.push(tuple_<T,Z>(i+15,score15));
		}else{//delete smallest element if current score is bigger
			if(q.top().score < score00){ q.pop(); q.push(tuple_<T,Z>(i,score00)); } if(q.top().score < score01){ q.pop(); q.push(tuple_<T,Z>(i+1,score01)); }
			if(q.top().score < score02){ q.pop(); q.push(tuple_<T,Z>(i+2,score02)); } if(q.top().score < score03){ q.pop(); q.push(tuple_<T,Z>(i+3,score03)); }
			if(q.top().score < score04){ q.pop(); q.push(tuple_<T,Z>(i+4,score04)); } if(q.top().score < score05){ q.pop(); q.push(tuple_<T,Z>(i+5,score05)); }
			if(q.top().score < score06){ q.pop(); q.push(tuple_<T,Z>(i+6,score06)); } if(q.top().score < score07){ q.pop(); q.push(tuple_<T,Z>(i+7,score07)); }

			if(q.top().score < score08){ q.pop(); q.push(tuple_<T,Z>(i+8,score08)); } if(q.top().score < score09){ q.pop(); q.push(tuple_<T,Z>(i+9,score09)); }
			if(q.top().score < score10){ q.pop(); q.push(tuple_<T,Z>(i+10,score10)); } if(q.top().score < score11){ q.pop(); q.push(tuple_<T,Z>(i+11,score11)); }
			if(q.top().score < score12){ q.pop(); q.push(tuple_<T,Z>(i+12,score12)); } if(q.top().score < score13){ q.pop(); q.push(tuple_<T,Z>(i+13,score13)); }
			if(q.top().score < score14){ q.pop(); q.push(tuple_<T,Z>(i+14,score14)); } if(q.top().score < score15){ q.pop(); q.push(tuple_<T,Z>(i+15,score15)); }
			if(STATS_EFF) this->pop_count+=16;
		}
		if(STATS_EFF) this->tuple_count+=16;
		if((q.top().score) > threshold ){
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << threshold << std::endl;
			break;
		}
	}
	this->tt_processing += this->t.lap();

	while(q.size() > 100){ q.pop(); }
	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void PTA<T,Z>::findTopKsimd(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topKsimd (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	uint64_t step = 0;
	this->tt_ranking = 0;
	float score[16] __attribute__((aligned(32)));
	__builtin_prefetch(score,1,3);
	uint64_t ii = 0;
	for(uint64_t i = 0; i < this->n; i+=16){
		__m256 score00 = _mm256_setzero_ps();
		__m256 score01 = _mm256_setzero_ps();
		T threshold = 0;
		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset00 = m * this->n + i;
			uint64_t offset01 = m * this->n + i + 8;
			__m256 load00 = _mm256_load_ps(&this->cdata[offset00]);
			__m256 load01 = _mm256_load_ps(&this->cdata[offset01]);
			score00 = _mm256_add_ps(score00,load00);
			score01 = _mm256_add_ps(score01,load01);

			uint64_t toffset0 = m * (this->n >> 4) + ii;
			threshold += this->tarray[toffset0];
		}
		ii++;

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
			if(STATS_EFF) this->pop_count+=16;
		}

		if(STATS_EFF) this->tuple_count+=16;
		if((q.top().score) > threshold ){
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << threshold << std::endl;
			break;
		}
	}
	this->tt_processing += this->t.lap();

	while(q.size() > 100){ q.pop(); }
	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void PTA<T,Z>::findTopKthreads(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topKthreads (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q[THREADS];
	this->t.start();
	omp_set_num_threads(THREADS);
	Z tuple_count[THREADS];
	for(uint32_t m = 0; m < THREADS; m++) tuple_count[m] = 0;
#pragma omp parallel
{
	uint32_t thread_id = omp_get_thread_num();
	uint64_t start = ((uint64_t)thread_id)*(this->n)/THREADS;
	uint64_t end = ((uint64_t)(thread_id+1))*(this->n)/THREADS;
	uint64_t ii = ((uint64_t)thread_id)*(this->n >> 4)/THREADS;
	float score[16] __attribute__((aligned(32)));
	T tp_count = 0;
	__builtin_prefetch(score,1,3);
	for(uint64_t i = start; i < end; i+=(16)){
		__m256 score00 = _mm256_setzero_ps();
		__m256 score01 = _mm256_setzero_ps();
		T threshold = 0;
		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset00 = m * this->n + i;
			uint64_t offset01 = m * this->n + i + 8;
			__m256 load00 = _mm256_load_ps(&this->cdata[offset00]);
			__m256 load01 = _mm256_load_ps(&this->cdata[offset01]);
			score00 = _mm256_add_ps(score00,load00);
			score01 = _mm256_add_ps(score01,load01);

			uint64_t toffset0 = m * (this->n >> 4) + ii;
			threshold += this->tarray[toffset0];
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
		if(STATS_EFF) tp_count+=16;
		if((q[thread_id].top().score) > threshold ){
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << threshold << std::endl;
			break;
		}
	}
	if(STATS_EFF) tuple_count[thread_id]=tp_count;
}
	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> _q;
	for(uint32_t m = 0 ; m < THREADS; m++){
		while(!q[m].empty()){
			if(_q.size() < k) _q.push(q[m].top());
			else if( _q.top().score < q[m].top().score ){
				_q.pop();
				_q.push(q[m].top());
			}
			q[m].pop();
		}
	}
	this->tt_processing += this->t.lap();

	if(STATS_EFF){
		for(uint32_t m = 1; m < THREADS; m++) tuple_count[0] += tuple_count[m];
		this->tuple_count = tuple_count[0];
	}
	T threshold = _q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q[0].size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
