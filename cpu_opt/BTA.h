#ifndef BTA_H
#define BTA_H

/*
* Blocked Threshold Aggregation
*/

#define BLOCK_SIZE 4
#define BLOCK_SHF 2

template<class T, class Z>
struct bta_pair{
	T id;
	Z score;
};

template<class T, class Z>
struct bta_block{
	Z offset[BLOCK_SIZE];
	T score;
	//T tarray[NUM_DIMS];
};

template<class T,class Z>
static bool cmp_bta_pair(const bta_pair<T,Z> &a, const bta_pair<T,Z> &b){ return a.score > b.score; };

template<class T,class Z>
class BTA : public AA<T,Z>{
	public:
		BTA(uint64_t n,uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "BTA";
			this->list_id = NULL;
			this->list_score = NULL;
		}

		~BTA(){
			if(list_id!=NULL) free(list_id);
			if(list_score!=NULL) free(list_score);
		}
		void init();
		void findTopKscalar(uint64_t k,uint8_t qq);

	private:
		Z *list_id;
		T *list_score;
};

template<class T, class Z>
void BTA<T,Z>::init(){
	bta_pair<T,Z> *list = (bta_pair<T,Z>*)malloc(sizeof(bta_pair<T,Z>)*this->n);
	list_id = (Z*)malloc(sizeof(Z)*this->n*this->d);
	list_score = (T*)malloc(sizeof(T)*this->n*this->d);

	this->t.start();
	for(uint8_t m = 0; m < this->d; m++){
		for(uint64_t i = 0; i < this->n; i++){
			list[i].id = i;
			list[i].score = this->cdata[i*this->d + m];
			//list[i].score = this->cdata[m*this->n + i];
		}
		__gnu_parallel::sort(&list[m],(&list[m]) + this->n,cmp_bta_pair<T,Z>);

		for(uint64_t i = 0; i < this->n; i++){
			list_id[i*this->d + m] = list[i].id;
			list_score[i*this->d + m] = list[i].score;
		}
	}

	this->tt_init = this->t.lap();
	free(list);
}

template<class T, class Z>
void BTA<T,Z>::findTopKscalar(uint64_t k, uint8_t qq){
	std::cout << this->algo << " find topKsimd (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	omp_set_num_threads(THREADS);
	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q[THREADS];
	Z *pos = (Z*)malloc(sizeof(Z)*this->d);
	for(uint8_t m = 0; m < this->d; m++) pos[m] = 0;
	this->t.start();

	std::cout << std::endl;
	for(uint8_t m = 0; m <THREADS; m++){
		uint32_t gid = m % qq;
		uint32_t lid = m / qq;
		uint32_t tig = (gid+1)*((THREADS-1)/qq + 1);
		tig = tig > THREADS ? tig - THREADS : tig - (gid)*(THREADS/qq);
		printf("(%d,%d,%d,%d)\n",m, gid,lid,tig);
	}

#pragma omp parallel
{
	uint32_t thread_id = omp_get_thread_num();
	uint32_t gid = thread_id % qq;
	uint32_t lid = thread_id / qq;
	uint32_t tig = (gid+1)*(THREADS/qq);
	tig = tig > THREADS ? tig - THREADS : tig - (gid)*(THREADS/qq);

	for(uint64_t i = 0; i < this->n; i++){
		Z id = list_id[i*this->d + gid];

		if( (id % THREADS)  == lid ){
			T score = 0;
			for(uint8_t p = 0; p < qq; p++){
				score+=this->cdata[id*this->d + p];
			}
			if(q[thread_id].size() < k){//insert if empty space in queue
				q[thread_id].push(tuple_<T,Z>(id,score));
			}else if(q[thread_id].top().score<score){//delete smallest element if current score is bigger
				q[thread_id].pop();
				q[thread_id].push(tuple_<T,Z>(id,score));
			}
			pos[gid] = std::max(pos[gid],i);
		}
		T threshold = 0;
		for(uint8_t p = 0; p < qq; p++){ threshold += list_score[i*this->d + p]; }
	}
}

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> _q;
	for(uint32_t m = 0 ; m < THREADS; m++){
		while(!q[m].empty()){
			if(_q.size() < k){
				_q.push(q[m].top());
			}else if( _q.top().score < q[m].top().score ){
				_q.pop();
				_q.push(q[m].top());
			}
			q[m].pop();
		}
	}

	this->tt_processing = this->t.lap();
	free(pos);

	T threshold = _q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << _q.size() << ")" << std::endl;
	this->threshold = threshold;
}



#endif
