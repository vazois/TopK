#ifndef TPA_R_H
#define TPA_R_H

#include "../cpu/AA.h"

#include "../cpu/AA.h"

template<class T, class Z>
struct tpar_pair{
	Z id;
	T score;
};

template<class T, class Z>
class  TPAr : public AA<T,Z>{
	public:
		TPAr(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "TPAr";
			this->tuples = NULL;
		}

		~TPAr(){
			if(this->tuples!=NULL) free(this->tuples);
		}

		void init();
		void findTopK(uint64_t k);
	private:
		tpar_pair<T,Z> *tuples;

};

template<class T, class Z>
void TPAr<T,Z>::init(){
	this->tuples = (tpar_pair<T,Z>*) malloc(sizeof(tpar_pair<T,Z>)*this->n);
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++){
		this->tuples[i].id = i;
		this->tuples[i].score = 0;
	}
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void TPAr<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	this->t.start();
	for(uint64_t i = 0; i < this->n; i+=8){
		T score0 = 0;
		T score1 = 0;
		T score2 = 0;
		T score3 = 0;
		T score4 = 0;
		T score5 = 0;
		T score6 = 0;
		T score7 = 0;
		for(uint8_t m = 0; m < this->d; m++){
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
		this->tuples[i].score = score0;
		this->tuples[i+1].score = score1;
		this->tuples[i+2].score = score2;
		this->tuples[i+3].score = score3;
		this->tuples[i+4].score = score4;
		this->tuples[i+5].score = score5;
		this->tuples[i+6].score = score6;
		this->tuples[i+7].score = score7;
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

#endif
