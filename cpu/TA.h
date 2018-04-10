#ifndef TA_H
#define TA_H

#include <queue>
#include <unordered_set>

template<class T,class Z>
class TA : public AA<T,Z>{
	public:
		TA(uint64_t n,uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "TA";  this->capacity=0;
			this->alists = NULL;
		}

		~TA()
		{
			if(this->alists!=NULL)
			{
				for(uint64_t i = 0;i < this->d;i++){
					if(this->alists[i] != NULL)
					{
						free(this->alists[i]);
					}
				}
			}
		}

		void init();
		void findTopK(uint64_t k,uint8_t qq);

		pred<T,Z> **alists;
	private:
		uint64_t capacity;
};

template<class T,class Z>
void TA<T,Z>::init(){
	//this->lists.resize(this->d);
	this->alists = (pred<T,Z>**)malloc(sizeof(pred<T,Z>*)*this->d);
	for(uint32_t m = 0; m < this->d; m++){ this->alists[m] = (pred<T,Z>*)malloc(sizeof(pred<T,Z>)*this->n); }
	for(uint64_t i=0;i<this->n;i++){
		for(uint8_t m =0;m<this->d;m++){
			this->alists[m][i] = pred<T,Z>(i,this->cdata[i*this->d + m]);
		}
	}

	this->t.start();
	for(uint32_t m =0;m<this->d;m++){
		__gnu_parallel::sort(this->alists[m],this->alists[m]+this->n,cmp_max_pred<T,Z>);
	}
	this->tt_init = this->t.lap();
}

template<class T,class Z>
void TA<T,Z>::findTopK(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find topK (" << (int)qq << "D) ...";
	std::unordered_set<Z> eset;
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		T threshold=0;
		for(uint32_t m =0;m<this->d;m++){
			pred<T,Z> p = this->alists[m][i];
			threshold+=p.attr;

			if(eset.find(p.tid) == eset.end()){
				T score00 = 0;
				for(uint8_t m = 0; m < qq; m++){ score00+=this->cdata[p.tid * this->d + m]; }

				if(STATS_EFF) this->pred_count+=this->d;
				if(STATS_EFF) this->tuple_count+=1;
				eset.insert(p.tid);
				if(q.size() < k){//insert if empty space in queue
					q.push(tuple_<T,Z>(p.tid,score00));
				}else if(q.top().score<score00){//delete smallest element if current score is bigger
					q.pop();
					q.push(tuple_<T,Z>(p.tid,score00));
					if(STATS_EFF) this->pop_count+=1;
				}
			}
		}

		if(q.size() >= k && ((q.top().score) >= threshold) ){
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << threshold << std::endl;
			break;
		}
	}
	this->tt_processing += this->t.lap();

	T threshold = q.top().score;
	while(!q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
