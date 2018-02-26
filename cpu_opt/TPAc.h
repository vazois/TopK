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
		}

		~TPAc(){
			if(this->tuples!=NULL) free(this->tuples);
		}

		void init();
		void findTopK(uint64_t k);
		void findTopKsimd(uint64_t k);
	private:
		tpac_pair<T,Z> *tuples;

};

template<class T, class Z>
void TPAc<T,Z>::init(){
	this->tuples = (tpac_pair<T,Z>*) malloc(sizeof(tpac_pair<T,Z>)*this->n);
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++){
		this->tuples[i].id = i;
		this->tuples[i].score = 0;
	}
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void TPAc<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
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
//		T score08 = 0;
//		T score09 = 0;
//		T score10 = 0;
//		T score11 = 0;
//		T score12 = 0;
//		T score13 = 0;
//		T score14 = 0;
//		T score15 = 0;

		for(uint8_t m = 0; m < this->d; m+=2){
			uint64_t offset0 = m * this->n + i;
			uint64_t offset1 = (m+1) * this->n + i;
			score00+= this->cdata[offset0] + this->cdata[offset1];
			score01+= this->cdata[offset0+1] + this->cdata[offset1+1];
			score02+= this->cdata[offset0+2] + this->cdata[offset1+2];
			score03+= this->cdata[offset0+3] + this->cdata[offset1+3];
			score04+= this->cdata[offset0+4] + this->cdata[offset1+4];
			score05+= this->cdata[offset0+5] + this->cdata[offset1+5];
			score06+= this->cdata[offset0+6] + this->cdata[offset1+6];
			score07+= this->cdata[offset0+7] + this->cdata[offset1+7];
//			score08+= this->cdata[offset0+8] + this->cdata[offset1+8];
//			score09+= this->cdata[offset0+9] + this->cdata[offset1+9];
//			score10+= this->cdata[offset0+10] + this->cdata[offset1+10];
//			score11+= this->cdata[offset0+11] + this->cdata[offset1+11];
//			score12+= this->cdata[offset0+12] + this->cdata[offset1+12];
//			score13+= this->cdata[offset0+13] + this->cdata[offset1+13];
//			score14+= this->cdata[offset0+14] + this->cdata[offset1+14];
//			score15+= this->cdata[offset0+15] + this->cdata[offset1+15];
		}
		this->tuples[i].score = score00;
		this->tuples[i+1].score = score01;
		this->tuples[i+2].score = score02;
		this->tuples[i+3].score = score03;
		this->tuples[i+4].score = score04;
		this->tuples[i+5].score = score05;
		this->tuples[i+6].score = score06;
		this->tuples[i+7].score = score07;
//		this->tuples[i+8].score = score08;
//		this->tuples[i+9].score = score09;
//		this->tuples[i+10].score = score10;
//		this->tuples[i+11].score = score11;
//		this->tuples[i+12].score = score12;
//		this->tuples[i+13].score = score13;
//		this->tuples[i+14].score = score14;
//		this->tuples[i+15].score = score15;
	}
	this->tt_processing = this->t.lap();

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
	this->tt_ranking = this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void TPAc<T,Z>::findTopKsimd(uint64_t k){
	std::cout << this->algo << " find topKsimd ...";
	this->t.start();
	T score[16];
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
		_mm256_store_ps(score,score00);
		this->tuples[i].score = score[0];
		this->tuples[i+1].score = score[1];
		this->tuples[i+2].score = score[2];
		this->tuples[i+3].score = score[3];
		this->tuples[i+4].score = score[4];
		this->tuples[i+5].score = score[5];
		this->tuples[i+6].score = score[6];
		this->tuples[i+7].score = score[7];
		_mm256_store_ps(score,score01);
		this->tuples[i+8].score = score[0];
		this->tuples[i+9].score = score[1];
		this->tuples[i+10].score = score[2];
		this->tuples[i+11].score = score[3];
		this->tuples[i+12].score = score[4];
		this->tuples[i+13].score = score[5];
		this->tuples[i+14].score = score[6];
		this->tuples[i+15].score = score[7];
	}
	this->tt_processing = this->t.lap();

	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
//	uint64_t count_insert = 0;
//	uint64_t count_pop = 0;
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
	this->tt_ranking = this->t.lap();
//	std::cout << std::endl << "count_insert = " <<count_insert << std::endl;
//	std::cout << std::endl << "count_pop = " <<count_pop << std::endl;

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}


#endif
