#ifndef TA_H
#define TA_H

#include "AA.h"
#include <queue>
#include <unordered_set>

#define TA_BLOCK 1024

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
			free(this->alists);
		}

		void init();
		void findTopK(uint64_t k,uint8_t qq);
		void findTopKscalar(uint64_t k,uint8_t qq);

		pred<T,Z> **alists;
	private:
		uint64_t capacity;
};

template<class T,class Z>
void TA<T,Z>::init(){
	//this->lists.resize(this->d);
	this->alists = (pred<T,Z>**)malloc(sizeof(pred<T,Z>*)*this->d);
	for(uint32_t m = 0; m < this->d; m++){ this->alists[m] = (pred<T,Z>*)malloc(sizeof(pred<T,Z>)*this->n); }

	this->t.start();
	for(uint64_t i=0;i<this->n;i++){
		for(uint8_t m =0;m<this->d;m++){
			this->alists[m][i] = pred<T,Z>(i,this->cdata[i*this->d + m]);
		}
	}

	for(uint32_t m =0;m<this->d;m++){
		__gnu_parallel::sort(this->alists[m],this->alists[m]+this->n,cmp_max_pred<T,Z>);
	}
	this->tt_init = this->t.lap();
}

template<class T,class Z>
void TA<T,Z>::findTopK(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find top-" << k << " (" << (int)qq << "D) ...";
	std::unordered_set<Z> eset;
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0; i < this->n;i++){
		T threshold=0;
		for(uint32_t m =0;m<qq;m++){
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

template<class T,class Z>
void TA<T,Z>::findTopKscalar(uint64_t k,uint8_t qq){
	std::cout << this->algo << " find top-" << k << " scalar (" << (int)qq << "D) ...";
	std::unordered_set<Z> eset;
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	Z process[ NUM_DIMS * TA_BLOCK ];
	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0; i < this->n;i+=TA_BLOCK){
//	for(uint64_t i = 0; i < this->n;){

		Z pnum = 0;
		for(uint64_t j = 0; j < TA_BLOCK; j++){
			for(uint32_t m =0;m<qq;m++){
				pred<T,Z> p = this->alists[m][i+j];
				if(eset.find(p.tid) == eset.end()){
					eset.insert(p.tid);
					process[pnum] = p.tid;
					pnum++;
				}
			}
		}

//		while (pnum < TA_BLOCK ){
//			for(uint32_t m =0;m<qq;m++){
//				pred<T,Z> p = this->alists[m][i];
//				if(eset.find(p.tid) == eset.end()){
//					eset.insert(p.tid);
//					process[pnum] = p.tid;
//					pnum++;
//				}
//			}
//			i++;
//		}

		//__builtin_prefetch(process,1,3);
		//Evaluate block//
		//std::cout << "pnum:" << pnum <<std::endl;
		Z remain =  (pnum & 7);
		pnum = pnum - remain;
		for(uint64_t j = 0; j < pnum; j+=8){
			T score00 = 0, score01 = 0, score02 = 0, score03 = 0, score04 = 0, score05 = 0, score06 = 0, score07 = 0;
			for(uint32_t m =0;m<qq;m++){
				score00+=this->cdata[process[j] * this->d + m];
				score01+=this->cdata[process[j+1] * this->d + m];
				score02+=this->cdata[process[j+2] * this->d + m];
				score03+=this->cdata[process[j+3] * this->d + m];
				score04+=this->cdata[process[j+4] * this->d + m];
				score05+=this->cdata[process[j+5] * this->d + m];
				score06+=this->cdata[process[j+6] * this->d + m];
				score07+=this->cdata[process[j+7] * this->d + m];
			}

			if(q.size() < k){//insert if empty space in queue
				q.push(tuple_<T,Z>(process[j],score00));
				q.push(tuple_<T,Z>(process[j+1],score01));
				q.push(tuple_<T,Z>(process[j+2],score02));
				q.push(tuple_<T,Z>(process[j+3],score03));
				q.push(tuple_<T,Z>(process[j+4],score04));
				q.push(tuple_<T,Z>(process[j+5],score05));
				q.push(tuple_<T,Z>(process[j+6],score06));
				q.push(tuple_<T,Z>(process[j+7],score07));
			}else{//delete smallest element if current score is bigger
				if(q.top().score < score00){ q.pop(); q.push(tuple_<T,Z>(process[j],score00)); }
				if(q.top().score < score01){ q.pop(); q.push(tuple_<T,Z>(process[j+1],score01)); }
				if(q.top().score < score02){ q.pop(); q.push(tuple_<T,Z>(process[j+2],score02)); }
				if(q.top().score < score03){ q.pop(); q.push(tuple_<T,Z>(process[j+3],score03)); }
				if(q.top().score < score04){ q.pop(); q.push(tuple_<T,Z>(process[j+4],score04)); }
				if(q.top().score < score05){ q.pop(); q.push(tuple_<T,Z>(process[j+5],score05)); }
				if(q.top().score < score06){ q.pop(); q.push(tuple_<T,Z>(process[j+6],score06)); }
				if(q.top().score < score07){ q.pop(); q.push(tuple_<T,Z>(process[j+7],score07)); }
			}
		}

		//Evaluate remaining in block
		for(uint64_t j = pnum; j < pnum+remain; j++){
			T score00 = 0;
			for(uint32_t m =0;m<qq;m++){ score00+=this->cdata[process[j] * this->d + m]; }

			if(q.size() < k){//insert if empty space in queue
				q.push(tuple_<T,Z>(process[j],score00));
			}else if(q.top().score<score00){//delete smallest element if current score is bigger
				q.pop();
				q.push(tuple_<T,Z>(process[j],score00));
			}
		}
		if(STATS_EFF) this->tuple_count+=pnum+remain;

		//Break if threshold <= top.score
		T threshold = 0;
		for(uint32_t m =0;m<qq;m++){ threshold+= this->alists[m][i+TA_BLOCK-1].attr; }
		//for(uint32_t m =0;m<qq;m++){ threshold+= this->alists[m][i-1].attr; }
		if(q.size() >= k && ((q.top().score) >= threshold) ){
			//std::cout << "\nStopped at " << i << "= " << q.top().score << "," << threshold << std::endl;
			break;
		}
	}
	this->tt_processing += this->t.lap();

	while(q.size() > 100){ q.pop(); }
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
