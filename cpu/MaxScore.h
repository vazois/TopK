#ifndef C_MAX_SCORE_H
#define C_MAX_SCORE_H

#include "AA.h"

template<class T>
struct MxAttr{
	MxAttr(){ ii = 0; attr = 0; }
	MxAttr(uint8_t i, T a){ ii = i; attr = a; }
	uint8_t ii;
	T attr;
};

template<class T,class Z>
struct MxPred{
	MxPred(){ tid = 0; score = 0; }
	MxPred(Z t, T s){ tid = t; score = s; }

	Z tid;
	T score;
	T  next;
};

template<class T>
static inline bool _cmp_mx_attr(const MxAttr<T> &a, const MxAttr<T> &b){ return a.attr > b.attr; };

template<class T, class Z>
class MaxScore : public AA<T,Z>{
	public:
	MaxScore(uint64_t n,uint64_t d) : AA<T,Z>(n,d){
			this->algo = "MaxScore";
			this->mxr = NULL;
		};
		~MaxScore(){
			if(this->mxr != NULL) free(this->mxr);
		};

		void init();
		void findTopK(uint64_t k);
		void findTopK2(uint64_t k);

	private:
		std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
		//std::vector<vpred<T,Z>> vtupples;
		std::vector<MxPred<T,Z>> mxtuples;
		MxAttr<T> *mxr;

		static inline T max(T a, T b){
			return a > b ? a : b;
		}
		typename std::vector<MxPred<T,Z>>::iterator partition( typename std::vector<MxPred<T,Z>>::iterator first, typename std::vector<MxPred<T,Z>>::iterator last, T threshold, T remainder);
};

template<class T, class Z>
void MaxScore<T,Z>::init(){
	this->mxr = (MxAttr<T> *)malloc(sizeof(MxAttr<T>)*this->d+1);
	this->mxr[this->d].ii = this->d+1;
	this->mxr[this->d].attr = 0;

	this->t.start();
	for(uint64_t m = 0; m < this->d; m++){
		T mx = 0;
		for(uint64_t i = 0; i < this->n;i++){
			mx=MaxScore::max(mx,this->cdata[m * this->n + i]);
		}
		this->mxr[m].ii = m;
		this->mxr[m].attr = mx;
	}

	std::sort(this->mxr,this->mxr + this->d, _cmp_mx_attr<T>);
	//this->vtupples.resize(this->n);
	this->tt_init = this->t.lap();
//	for(uint8_t m = 0; m < this->d; m++){
//		std::cout << "< "<< (int)this->mxr[m].ii <<"," << this->mxr[m].attr << std::endl;
//	}
}

template<class T, class Z>
void MaxScore<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	T *score = (T* )malloc(sizeof(T) * this->n);
	for(int8_t m = this->d-2; m >=0; m--){ this->mxr[m].attr += this->mxr[m+1].attr; }

	tuple<T,Z> k_0;
	tuple<T,Z> k_1;
	this->t.start();
	memset(score,0,sizeof(T)*this->n);
	for(uint8_t m = 0; m < this->d; m++){
		std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
		uint8_t ii = this->mxr[m].ii;
		for(uint64_t i = 0; i < this->n;i++){
			score[i]+=this->cdata[ii * this->n + i];
			if(q.size() < k+1){
				q.push(tuple<T,Z>(i,score[i]));
			}else if(q.top().score < score[i]){
				q.pop();
				q.push(tuple<T,Z>(i,score[i]));
			}
		}
		if(STATS_EFF) this->pred_count+=this->n;

		k_1 =q.top(); q.pop(); k_0 =q.top();
		if(k_0.score >= k_1.score + this->mxr[m+1].attr){
			std::swap(q,this->q);
			break;
		}else if(m == this->d-1){
			if(STATS_EFF) this->tuple_count+=this->n;
			std::swap(q,this->q);
		}

//		if(STATS_EFF) this->pred_count+=size;
//		if(STATS_EFF && m == (this->d-1)) this->tuple_count+=size;
	}
	this->tt_processing = this->t.lap();
	free(score);

	//Gather results for verification
	T threshold = q.top().score;
	while(!this->q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(this->q.top());
		this->q.pop();
	}
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T,class Z>
typename std::vector<MxPred<T,Z>>::iterator MaxScore<T,Z>::partition(
		typename std::vector<MxPred<T,Z>>::iterator first,
		typename std::vector<MxPred<T,Z>>::iterator last,
		T threshold,
		T remainder){
	uint32_t keep_count = 0;
	while(first != last)
	{
		while (first->score + remainder >= threshold) {
			++first;
			if (first==last) return first;
		}
		do{
			--last;
			if (first==last) return first;
		}while(!(last->score + remainder >= threshold));
		std::iter_swap(first,last);
		++first;
	}
	return first;
}

template<class T, class Z>
void MaxScore<T,Z>::findTopK2(uint64_t k){
	std::cout << this->algo << " find topK2 ...";
	for(int8_t m = this->d-2; m >=0; m--){ this->mxr[m].attr += this->mxr[m+1].attr; }
//	std::cout << "size: " << this->mxtuples.size() << std::endl;
//	return ;

	this->mxtuples.resize(this->n);
	uint8_t ii = this->mxr[0].ii;
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++){ this->mxtuples[i] = MxPred<T,Z>(i,this->cdata[ii * this->n + i]); }
	typename std::vector<MxPred<T,Z>>::iterator first = this->mxtuples.begin();
	typename std::vector<MxPred<T,Z>>::iterator last = this->mxtuples.end();

	for(uint8_t m = 0; m < this->d; m++){
		std::priority_queue<T, std::vector<T>, std::greater<T>> q;
		first = this->mxtuples.begin();
		while( first != last  ){
			if(q.size() < k){
				q.push(first->score);
			}else if(q.top()<first->score){
				q.pop();
				q.push(first->score);
			}
			first++;
		}
		T threshold = q.top();

		//Partition//
		last = this->partition(this->mxtuples.begin(),last,threshold,this->mxr[m+1].attr);

		if( m < this->d-1 ){
			uint64_t size = 0;
			first = this->mxtuples.begin();
			while( first != last ){
				first->score+=this->cdata[(m+1) * this->n + first->tid];
				first->next=this->cdata[(m+1) * this->n + first->tid];
				size++;
				first++;
			}
			//std::cout << std::endl << "MaxScore: "<<(int) m << ": " << size << std::endl;
			if(STATS_EFF) this->pred_count+=size;
			if(STATS_EFF && m == (this->d-1)) this->tuple_count+=size;
			if(size <= k) break;
		}
	}
	this->tt_processing = this->t.lap();

	T threshold = this->mxtuples[0].score;
	for(uint32_t i = 0;i <k;i++){
		this->res.push_back(tuple<T,Z>(this->mxtuples[i].tid,this->mxtuples[i].score));
		threshold = threshold > this->mxtuples[i].score ? this->mxtuples[i].score : threshold;
	}
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
