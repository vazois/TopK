#ifndef CBA_OPT
#define CBA_OPT

#include "../cpu/AA.h"

template<class T, class Z>
struct cba_pair{
	T score;
	Z id;
};

template<class T,class Z>
class CBAopt : public AA<T,Z>{
	public:
		CBAopt(uint64_t n,uint64_t d) : AA<T,Z>(n,d){
			this->algo = "CBAopt";
			this->ids = NULL;
			this->scores = NULL;
			this->tuples = NULL;
		};

		~CBAopt(){
			if(this->ids!=NULL) free(this->ids);
			if(this->scores!=NULL) free(this->scores);
			if(this->tuples!=NULL) free(this->tuples);
		}

		void init();
		void findTopK(uint64_t k);
		void findTopK2(uint64_t k);
		void findTopK3(uint64_t k);
		void findTopK4(uint64_t k);
		void findTopK5(uint64_t k);

	private:
		uint64_t partition(Z *ids, T *scores, T *curr, uint64_t n, uint8_t remainder, T threshold );
		uint64_t partition(cba_pair<T,Z>* tuples, T *curr, uint64_t n, uint8_t remainder, T threshold );
		cba_pair<T,Z>* partition( cba_pair<T,Z>* first, cba_pair<T,Z>* last, T *curr, uint8_t remainder, T threshold );
		cba_pair<T,Z>* partition2( cba_pair<T,Z>* first, cba_pair<T,Z>* last, T *next, uint8_t remainder , uint64_t k );

		cba_pair<T,Z> *tuples;
		Z *ids;
		T *scores;
};

template<class T,class Z>
void CBAopt<T,Z>::init(){
	//std::cout << this->algo <<" initialize: " << "(" << this->n << "," << this->d << ")"<< std::endl;

	this->t.start();
	switch(this->d){
		case 2:
			reorder_attr_2(this->cdata,this->n);
			break;
		case 4:
			reorder_attr_4(this->cdata,this->n);
			break;
		case 6:
			reorder_attr_6(this->cdata,this->n);
			break;
		case 8:
			reorder_attr_8(this->cdata,this->n);
			break;
		case 10:
			reorder_attr_10(this->cdata,this->n);
			break;
		case 12:
			reorder_attr_12(this->cdata,this->n);
			break;
		case 14:
			reorder_attr_14(this->cdata,this->n);
			break;
		case 16:
			reorder_attr_16(this->cdata,this->n);
			break;
		case 18:
			reorder_attr_18(this->cdata,this->n);
			break;
		case 20:
			reorder_attr_20(this->cdata,this->n);
			break;
		case 22:
			reorder_attr_22(this->cdata,this->n);
			break;
		case 24:
			reorder_attr_24(this->cdata,this->n);
			break;
		case 26:
			reorder_attr_26(this->cdata,this->n);
			break;
		case 28:
			reorder_attr_28(this->cdata,this->n);
			break;
		case 30:
			reorder_attr_30(this->cdata,this->n);
			break;
		case 32:
			reorder_attr_32(this->cdata,this->n);
			break;
		default:
			break;
	}
	this->tt_init = this->t.lap();
}

template<class T, class Z>
uint64_t CBAopt<T,Z>::partition(Z *ids, T *scores, T *curr, uint64_t n, uint8_t remainder, T threshold ){
	uint64_t first = 0;
	uint64_t last = n;

	while(first < last){
		while ( scores[first] + curr[ids[first]]*remainder >= threshold ) {
			++first;
			if (first==last) return first;
		}

		do{
			--last;
			if (first==last) return first;
		}while(scores[last] + curr[ids[last]]*remainder < threshold);

		std::swap(ids[first],ids[last]);
		std::swap(scores[first],scores[last]);
	}
	return first;
}

template<class T, class Z>
void CBAopt<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";

	//Initialize
	this->ids = (Z*)malloc(sizeof(Z) * this->n);
	this->scores = (T*)malloc(sizeof(T) * this->n);
	for(uint64_t i = 0; i < this->n ; i++){
		this->ids[i] = i;
		this->scores[i] = this->cdata[i];
	}

	this->t.start();
	uint64_t end = this->n;
	for(uint8_t m =0;m < this->d;m++){
		std::priority_queue<T, std::vector<T>, std::greater<T>> q;

		//(1) Find threshold
		for(uint64_t i = 0; i < end;i++){
			if(q.size() < k){
				q.push(this->scores[i]);
			}else if(q.top()<this->scores[i]){
				q.pop();
				q.push(this->scores[i]);
			}
		}
		T threshold = q.top();
		//std::cout << "t: " << threshold << std::endl;

		//(2)
		uint32_t remainder = this->d-(m+1);
		end = this->partition(ids,scores,&this->cdata[m * this->n],end,remainder,threshold);


		if( m < this->d-1 ){
			for(uint64_t i = 0; i < end;i++){
				scores[i] += this->cdata[(m+1) * this->n + ids[i]];
			}
		}

	}
	this->tt_processing = this->t.lap();

	T threshold = scores[0];
	for(uint64_t i = 0;i<k;i++){
		threshold = threshold < scores[i] ? threshold : scores[i];
		this->res.push_back(tuple<T,Z>(ids[i],scores[i]));
	}
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
uint64_t CBAopt<T,Z>::partition( cba_pair<T,Z>* tuples, T *curr, uint64_t n, uint8_t remainder, T threshold ){
	uint64_t first = 0;
	uint64_t last = n;

	while(first < last){
		while ( tuples[first].score + curr[tuples[first].id]*remainder >= threshold ){
			++first;
			if (first==last) return first;
		}

		do{
			--last;
			if (first==last) return first;
		}while(tuples[last].score + curr[tuples[last].id]*remainder < threshold);
		//std::swap(tuples[first],tuples[last]);
		tuples[first] = tuples[last];
		first++;
	}
	return first;
}

template<class T, class Z>
void CBAopt<T,Z>::findTopK2(uint64_t k){
	std::cout << this->algo << " find topK2 ...";

	this->tuples =(cba_pair<T,Z>*)malloc(sizeof(cba_pair<T,Z>) * this->n);
	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		tuples[i].id = i;
		tuples[i].score = this->cdata[i];
	}
	uint64_t end = this->n;
	for(uint8_t m =0;m < this->d;m++){
		std::priority_queue<T, std::vector<T>, std::greater<T>> q;

		//(1) Find threshold
		for(uint64_t i = 0; i < end;i++){
			if(q.size() < k){
				q.push(this->tuples[i].score);
			}else if(q.top()<this->tuples[i].score){
				q.pop();
				q.push(this->tuples[i].score);
			}
		}
		T threshold = q.top();
		//std::cout << "t: " << threshold << std::endl;

		//(2)
		uint32_t remainder = this->d-(m+1);
		end = this->partition(this->tuples,&this->cdata[m * this->n],end,remainder,threshold);

		//(3)
		if( m < this->d-1 ){
			for(uint64_t i = 0; i < end; i++){
				this->tuples[i].score += this->cdata[(m+1) * this->n + this->tuples[i].id];
			}
		}
		if(end <= k) break;
	}
	this->tt_processing = this->t.lap();

	T threshold = this->tuples[0].score;
	for(uint64_t i = 0;i < k;i++){
		threshold = threshold < this->tuples[i].score ? threshold : this->tuples[i].score;
		this->res.push_back(tuple<T,Z>(this->tuples[i].id,this->tuples[i].score));
	}
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
cba_pair<T,Z>* CBAopt<T,Z>::partition( cba_pair<T,Z>* first,cba_pair<T,Z>* last, T *curr, uint8_t remainder, T threshold ){

	while(first < last){
		while ( first->score + curr[first->id]*remainder >= threshold ){
			++first;
			if (first==last) return first;
		}
		--last;//TODO: safe?
		//do{
		while(last->score + curr[last->id]*remainder < threshold){
			--last;
			if (first==last) return first;
		}
		std::swap(*first,*last);
		//tuples[first] = tuples[last];
		++first;
	}
	return first;
}

template<class T, class Z>
void CBAopt<T,Z>::findTopK3(uint64_t k){
	std::cout << this->algo << " find topK3 ...";

	this->tuples =(cba_pair<T,Z>*)malloc(sizeof(cba_pair<T,Z>) * this->n+1);

	//Initialize
	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		tuples[i].id = i;
		tuples[i].score = this->cdata[i];
	}
	cba_pair<T,Z> *first = &this->tuples[0];
	cba_pair<T,Z> *last = &this->tuples[this->n];
	for(uint8_t m =0;m < this->d;m++){
		std::priority_queue<T, std::vector<T>, std::greater<T>> q;

		first = &this->tuples[0];
		while(first < last){
			if(q.size() < k){
				q.push(first->score);
			}else if(q.top() < first->score){
				q.pop();
				q.push(first->score);
			}
			first++;
		}
		T threshold = q.top();
		//std::cout << "t: " << threshold << std::endl;

		//(2)
		first = &this->tuples[0];
		uint32_t remainder = this->d-(m+1);
		last = this->partition(first,last,&this->cdata[m * this->n],remainder,threshold);

		if(m < this->d-1){
			first = &this->tuples[0];
			while(first < last){
				first->score+= this->cdata[(m+1)*this->n + first->id];
				first++;
			}
		}
	}
	this->tt_processing = this->t.lap();

	T threshold = this->tuples[0].score;
	for(uint64_t i = 0; i < k; i++){
		threshold = threshold < this->tuples[i].score ? threshold : this->tuples[i].score;
		this->res.push_back(tuple<T,Z>(this->tuples[i].id,this->tuples[i].score));
	}
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
cba_pair<T,Z>* CBAopt<T,Z>::partition2( cba_pair<T,Z>* first, cba_pair<T,Z>* last, T *next, uint8_t remainder , uint64_t k ){
	std::priority_queue<T, std::vector<T>, std::greater<T>> q;
	T threshold = this->threshold;
	while(first < last){
		while ( first->score + next[first->id]*remainder >= threshold ){
			first->score+= next[first->id];
			if(q.size() < k){
				q.push(first->score);
			}else if(q.top() < first->score){
				q.pop(); q.push(first->score);
			}
			++first;
			if (first==last){
				this->threshold = q.top();
				return first;
			}
		}

		--last;
		while(last->score + next[last->id]*remainder < threshold){
			last->score+= next[last->id];
			if(q.size() < k){
				q.push(last->score);
			}else if(q.top() < last->score){
				q.pop(); q.push(last->score);
			}
			--last;
			if (first==last){
				this->threshold = q.top();
				return first;
			}
		}
		std::swap(*first,*last);
		//++first;
		//tuples[first] = tuples[last];
	}
	this->threshold = q.top();
	return first;
}

template<class T, class Z>
void CBAopt<T,Z>::findTopK4(uint64_t k){
	std::cout << this->algo << " find topK4 ...";

	this->tuples =(cba_pair<T,Z>*)malloc(sizeof(cba_pair<T,Z>) * this->n+1);
	for(uint64_t i = 0; i < this->n;i++){
		tuples[i].id = i;
		tuples[i].score = this->cdata[i];
	}

	//Initialize
	this->t.start();
	cba_pair<T,Z> *first = &this->tuples[0];
	cba_pair<T,Z> *last = &this->tuples[this->n];

	std::priority_queue<T, std::vector<T>, std::greater<T>> q;
	for(uint64_t i = 0; i < this->n; i++){
		while(first != last){
			if(q.size() < k){
				q.push(first->score);
			}else if (q.top() < first->score){
				q.pop(); q.push(first->score);
			}
			first++;
		}
	}

	this->threshold = q.top();
	first = &this->tuples[0];
	std::cout << "t: " << this->threshold << std::endl;
	uint8_t remainder = this->d-3;
	for(uint8_t m =1;m < this->d;m++){
		first = &this->tuples[0];
		last = this->partition2(first,last,&this->cdata[m * this->n],remainder,k);
		std::cout << "t: " << this->threshold << std::endl;
		//partition(first,last,&this->cdata[m * this->n],remainder,threshold);
		remainder--;
	}
	this->tt_processing = this->t.lap();

	T threshold = this->tuples[0].score;
	for(uint64_t i =0 ; i < k ; i++){
		threshold = threshold < this->tuples[i].score ? threshold : this->tuples[i].score;
		this->res.push_back(tuple<T,Z>(this->tuples[i].id,this->tuples[i].score));
	}
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
