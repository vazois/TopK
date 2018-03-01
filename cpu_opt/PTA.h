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

template<class T,class Z>
static bool cmp_pta_pair(const pta_pair<T,Z> &a, const pta_pair<T,Z> &b){ return a.score > b.score; };

template<class T,class Z>
static bool cmp_pta_pt_pos(const pta_pt<T,Z> &a, const pta_pt<T,Z> &b){ return a.pos < b.pos; };

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
			this->gt_array = NULL;
			this->pt = NULL;
		}

		~PTA(){
			if(this->gt_array!=NULL) free(this->gt_array);
			if(this->pt != NULL) free(this->pt);
		}

		void init();
		void findTopKscalar(uint64_t k,uint8_t qq);
		void findTopKsimd(uint64_t k,uint8_t qq);
		void findTopKthreads(uint64_t k,uint8_t qq);
	private:
		T *gt_array;
		pta_pt<T,Z> *pt;
		T acc;
};

template<class T, class Z>
void PTA<T,Z>::init(){
	pta_pair<T,Z> *list = (pta_pair<T,Z>*)malloc(sizeof(pta_pair<T,Z>)*this->n);
	this->pt = (pta_pt<T,Z>*)malloc(sizeof(pta_pt<T,Z>)*this->n);
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++){
		this->pt[i].id = i;
		this->pt[i].pos = this->n;
		this->pt[i].score = 0;
	}

	for(uint8_t m = 0; m < this->d; m++){
		for(uint64_t i = 0; i < this->n; i++){
			list[i].id = i;
			list[i].score = this->cdata[m*this->n + i];
		}
		__gnu_parallel::sort(&list[0],(&list[0]) + this->n,cmp_pta_pair<T,Z>);//data based on attribute//

		//Find Minimum Position and threshold for that position//
		for(uint64_t i = 0; i < this->n; i++){
			pta_pair<T,Z> p = list[i];
			this->pt[p.id].pos = pt[p.id].pos < i ? pt[p.id].pos : i;// Best position according to lists
			this->pt[p.id].score = pt[p.id].score > p.score ? pt[p.id].score : p.score;// Maximum attribute for threshold calculation
		}
		//
	}
	free(list);
	__gnu_parallel::sort(&pt[0],(&pt[0]) + this->n,cmp_pta_pt_pos_score<T,Z>);

//	for(uint64_t i = 0; i < 25; i++){
//		std::cout << std::dec << std::setfill('0') << std::setw(2);
//		std::cout << i << ": ";
//		std::cout << std::dec << std::setfill('0') << std::setw(8);
//		std::cout << this->pt[i].id << ",";
//		std::cout << std::dec << std::setfill('0') << std::setw(8);
//		std::cout << this->pt[i].pos << ",";
//		std::cout << std::fixed << std::setprecision(4);
//		std::cout << this->pt[i].score;
//
//		std::cout << " || ";
//		for(uint8_t m = 0; m < this->d; m++){
//			std::cout << this->cdata[m * this->n + pt[i].id] << " ";
//		}
//		std::cout << " || ";
//		std::cout << std::endl;
//	}

	//Reordered data based on min position//
	T *cdata = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->n) * (this->d)));
	for(uint64_t i = 0; i < this->n; i++){
		pta_pt<T,Z> p = this->pt[i];
		for(uint8_t m = 0; m < this->d; m++){
			cdata[m * this->n + i] = this->cdata[m * this->n + p.id];
		}
	}
	free(this->cdata); this->cdata = cdata;
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void PTA<T,Z>::findTopKscalar(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topKscalar (" << (int)qq << "D) ...";

	this->tuple_count = 0;
	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	uint64_t step = 0;
	this->tt_ranking = 0;
	for(uint64_t i = 0; i < this->n; i+=8){
		T score00 = 0;
		T score01 = 0;
		T score02 = 0;
		T score03 = 0;
		T score04 = 0;
		T score05 = 0;
		T score06 = 0;
		T score07 = 0;
		for(uint8_t m = 0; m < qq; m++){
			uint64_t offset0 = m * this->n + i;
			score00+= this->cdata[offset0];
			score01+= this->cdata[offset0+1];
			score02+= this->cdata[offset0+2];
			score03+= this->cdata[offset0+3];
			score04+= this->cdata[offset0+4];
			score05+= this->cdata[offset0+5];
			score06+= this->cdata[offset0+6];
			score07+= this->cdata[offset0+7];
		}
		if(q.size() < k){//insert if empty space in queue
			q.push(tuple<T,Z>(i,score00));
			q.push(tuple<T,Z>(i+1,score01));
			q.push(tuple<T,Z>(i+2,score02));
			q.push(tuple<T,Z>(i+3,score03));
			q.push(tuple<T,Z>(i+4,score04));
			q.push(tuple<T,Z>(i+5,score05));
			q.push(tuple<T,Z>(i+6,score06));
			q.push(tuple<T,Z>(i+7,score07));
		}else{//delete smallest element if current score is bigger
			if(q.top().score < score00){ q.pop(); q.push(tuple<T,Z>(i,score00)); }
			if(q.top().score < score01){ q.pop(); q.push(tuple<T,Z>(i+1,score01)); }
			if(q.top().score < score02){ q.pop(); q.push(tuple<T,Z>(i+2,score02)); }
			if(q.top().score < score03){ q.pop(); q.push(tuple<T,Z>(i+3,score03)); }
			if(q.top().score < score04){ q.pop(); q.push(tuple<T,Z>(i+4,score04)); }
			if(q.top().score < score05){ q.pop(); q.push(tuple<T,Z>(i+5,score05)); }
			if(q.top().score < score06){ q.pop(); q.push(tuple<T,Z>(i+6,score06)); }
			if(q.top().score < score07){ q.pop(); q.push(tuple<T,Z>(i+7,score07)); }
		}

		if(STATS_EFF) this->tuple_count+=4;
		if((q.top().score) > (this->pt[i+7].score*this->d) ){
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << this->pt[i+7].score << std::endl;
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

	this->tuple_count = 0;
	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	uint64_t step = 0;
	this->tt_ranking = 0;
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

		if(STATS_EFF) this->tuple_count+=16;
		//if((q.top().score) > (this->gt_array[i+15]*this->d) ){
		if((q.top().score) > (this->pt[i+15].score*this->d) ){
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << this->gt_array[i+7] << std::endl;
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

	this->tuple_count = 0;
	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q[THREADS];
	this->t.start();
	omp_set_num_threads(THREADS);
#pragma omp parallel
{
	uint32_t thread_id = omp_get_thread_num();
	uint64_t start = ((uint64_t)thread_id)*(this->n)/THREADS;
	uint64_t end = ((uint64_t)(thread_id+1))*(this->n)/THREADS;
	float score[16] __attribute__((aligned(32)));
	__builtin_prefetch(score,1,3);
	for(uint64_t i = start; i < this->n; i+=(16)){
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
		if(q[thread_id].size() < k){//insert if empty space in queue
			q[thread_id].push(tuple<T,Z>(i,score[0]));
			q[thread_id].push(tuple<T,Z>(i+1,score[1]));
			q[thread_id].push(tuple<T,Z>(i+2,score[2]));
			q[thread_id].push(tuple<T,Z>(i+3,score[3]));
			q[thread_id].push(tuple<T,Z>(i+4,score[4]));
			q[thread_id].push(tuple<T,Z>(i+5,score[5]));
			q[thread_id].push(tuple<T,Z>(i+6,score[6]));
			q[thread_id].push(tuple<T,Z>(i+7,score[7]));
			q[thread_id].push(tuple<T,Z>(i+8,score[8]));
			q[thread_id].push(tuple<T,Z>(i+9,score[9]));
			q[thread_id].push(tuple<T,Z>(i+10,score[10]));
			q[thread_id].push(tuple<T,Z>(i+11,score[11]));
			q[thread_id].push(tuple<T,Z>(i+12,score[12]));
			q[thread_id].push(tuple<T,Z>(i+13,score[13]));
			q[thread_id].push(tuple<T,Z>(i+14,score[14]));
			q[thread_id].push(tuple<T,Z>(i+15,score[15]));
		}else{//delete smallest element if current score is bigger
			if(q[thread_id].top().score < score[0]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i,score[0])); }
			if(q[thread_id].top().score < score[1]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+1,score[1])); }
			if(q[thread_id].top().score < score[2]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+2,score[2])); }
			if(q[thread_id].top().score < score[3]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+3,score[3])); }
			if(q[thread_id].top().score < score[4]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+4,score[4])); }
			if(q[thread_id].top().score < score[5]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+5,score[5])); }
			if(q[thread_id].top().score < score[6]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+6,score[6])); }
			if(q[thread_id].top().score < score[7]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+7,score[7])); }
			if(q[thread_id].top().score < score[8]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+8,score[8])); }
			if(q[thread_id].top().score < score[9]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+9,score[9])); }
			if(q[thread_id].top().score < score[10]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+10,score[10])); }
			if(q[thread_id].top().score < score[11]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+11,score[11])); }
			if(q[thread_id].top().score < score[12]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+12,score[12])); }
			if(q[thread_id].top().score < score[13]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+13,score[13])); }
			if(q[thread_id].top().score < score[14]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+14,score[14])); }
			if(q[thread_id].top().score < score[15]){ q[thread_id].pop(); q[thread_id].push(tuple<T,Z>(i+15,score[15])); }
		}

		if(STATS_EFF) this->tuple_count+=16;
		if((q[thread_id].top().score) > (this->pt[i+15].score*this->d) ){
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << this->gt_array[i+7] << std::endl;
			break;
		}
	}
}
	this->tt_processing += this->t.lap();

	while(q[0].size() > 100){ q[0].pop(); }
	T threshold = q[0].top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q[0].size() << ")" << std::endl;
	this->threshold = threshold;
}


#endif
