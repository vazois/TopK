#ifndef LARA_H
#define LARA_H

#include "../cpu/AA.h"
#include <unordered_map>

template<class T, class Z>
struct lara_pair{
	Z id;
	T score;
};

template<class T>
struct lara_info{
	uint8_t bitmap;//Seen in lists
	uint8_t pos;//pos in priority_queue
	T score;//partial score
};

static uint8_t s[8] = {
		0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
};

static uint8_t shf[8] = {
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
};

template<class T,class Z>
static bool cmp_lara_pair(const lara_pair<T,Z> &a, const lara_pair<T,Z> &b){ return a.score > b.score; };

template<class T,class Z>
class LARA : public AA<T,Z>{
	public:
		LARA(uint64_t n,uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "LARA";
			this->lists = NULL;
		}

		~LARA(){
			if(this->lists!=NULL)
			{
				for(uint64_t i = 0;i < this->d;i++){
					if(this->lists[i] != NULL)
					{
						free(this->lists[i]);
					}
				}
				free(this->lists);
			}

		}

		void init();
		void findTopK(uint64_t k,uint8_t qq);
		void findTopKscalar(uint64_t k,uint8_t qq);

	private:
		lara_pair<T,Z> **lists;

		void growing_phase(uint64_t k,uint8_t qq,std::unordered_map<Z,lara_info<T>> &seen,std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> &q);
		void shrinking_phase(uint64_t k,uint8_t qq,std::unordered_map<Z,lara_info<T>> &seen,std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> &q);
};

template<class T, class Z>
void LARA<T,Z>::init(){
	this->lists = (lara_pair<T,Z>**)malloc(sizeof(lara_pair<T,Z>*)*this->d);
	for(uint32_t m = 0; m < this->d; m++){ this->lists[m] = (lara_pair<T,Z>*)malloc(sizeof(lara_pair<T,Z>)*this->n); }

	this->t.start();
	for(uint64_t i=0;i<this->n;i++){
		for(uint8_t m =0;m<this->d;m++){
			this->lists[m][i].id =i;
			this->lists[m][i].score = this->cdata[i*this->d + m];
		}
	}

	for(uint32_t m =0;m<this->d;m++){ __gnu_parallel::sort(this->lists[m],this->lists[m]+this->n,cmp_lara_pair<T,Z>); }
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void LARA<T,Z>::growing_phase(uint64_t k,uint8_t qq,std::unordered_map<Z,lara_info<T>> &seen,std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> &q){
	for(uint64_t i = 0; i < this->n; i++){
		T threshold=0;

		for(uint8_t m = 0; m < qq; m++){
			lara_pair<T,Z> lp = this->lists[m][i];
			threshold += lp.score;

			//Update seen
			if(seen.find(lp.id) == seen.end()){
				seen[lp.id].pos = k + 1;
				seen[lp.id].bitmap = s[m];
				seen[lp.id].score = lp.score;
			}else{
				seen[lp.id].bitmap |= s[m];
				seen[lp.id].score += lp.score;
			}

			//Update priority queue
			if( seen[lp.id].pos == (k+1) ){
				if(q.size() < k){
					seen[lp.id].pos = 0;
					q.push(tuple_<T,Z>(lp.id,seen[lp.id].score));
				}else if(q.top().score<seen[lp.id].score){
					seen[q.top().tid].pos = k + 1;
					q.pop();
					q.push(tuple_<T,Z>(lp.id,seen[lp.id].score));
				}
			}
			//
		}

		if(q.size() >= k && ((q.top().score) >= threshold) ){
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << threshold << std::endl;
			break;
		}
	}

	std::cout << "seen_size: " << seen.size() << std::endl;
}

template<class T, class Z>
void LARA<T,Z>::shrinking_phase(uint64_t k,uint8_t qq,std::unordered_map<Z,lara_info<T>> &seen,std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> &q){
	std::cout << "seen_size(2): " << seen.size() << std::endl;
	for(uint64_t i = 0; i < this->n; i++){
		if(seen.find(i) != seen.end()){

		}
	}
}

template<class T, class Z>
void LARA<T,Z>::findTopK(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topK (" << (int)qq << "D) ...";
	std::unordered_set<Z> eset;
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::unordered_map<Z,lara_info<T>> seen;
	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	this->growing_phase(k,qq,seen,q);
	//this->shrinking_phase(k,qq,seen,q);

	this->tt_processing += this->t.lap();

	//T threshold = q.top().score;
	T threshold = 1313;
	while(!q.empty()){
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
