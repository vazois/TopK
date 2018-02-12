#ifndef C_FA_H
#define C_FA_H

#include "AA.h"
#include <unordered_map>
#include <unordered_set>
#include <list>

#include "build_attr_index_r.h"
#include <inttypes.h>

template<class T, class Z>
class cFA : public AA<T,Z>{
	public:
		cFA(uint64_t n,uint64_t d) : AA<T,Z>(n,d){
			this->algo = "cFA";
			this->TT = NULL;
			this->II = NULL;
			this->R = NULL;
		};
		~cFA(){
			if(this->TT != NULL) free(TT);
			if(this->II != NULL) free(II);
			if(this->R != NULL ) free(R);
		};

		void init();
		void findTopK(uint64_t k);

	protected:
		void create_lists();
		void base_table();
		void build_index();
	private:
		std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
		pred<T,Z> **lists;
		Z *TT;//Base table with appearance order for each tuple within each list
		uint64_t *II;// Index indicating attribute order //
		Z *R;
};

template<class T, class Z>
void cFA<T,Z>::create_lists(){
	//Create and Sort Lists//
	for(uint64_t i=0;i<this->n;i++){
		for(uint8_t m =0;m<this->d;m++){
			//lists[m][i] = pred<T,Z>(i,this->cdata[i*this->d + m]);
			lists[m][i].tid = i;
			lists[m][i].attr = this->cdata[i*this->d + m];
		}
	}
	for(uint8_t m =0;m<this->d;m++){ __gnu_parallel::sort(lists[m],lists[m] + this->n,cmp_max_pred<T,Z>); }
	/////////////////////////

	if(this->n<50){
	std::cout << std::endl << "<Lists>" << std::endl;
	for(uint64_t i=0;i<this->n;i++){
		std::cout << i << ": ";
		for(uint8_t m =0;m<this->d;m++){
			pred<T,Z> p = lists[m][i];
			std::cout << std::dec << std::setfill('0') << std::setw(3) << p.tid << " ";
		}
		std::cout << std::endl;
	}
	}
}

template<class T,class Z>
void cFA<T,Z>::init(){
	lists = (pred<T,Z> **)malloc(sizeof(pred<T,Z>*) * this->d);
	for(uint64_t i = 0;i < this->d;i++) lists[i] = (pred<T,Z>*)malloc(sizeof(pred<T,Z>) * this->n);
	TT = (Z *)malloc(sizeof(Z) * this->d * this->n);
	II = (uint64_t *)malloc(sizeof(uint64_t) * this->n);

	this->t.start();
	//1
	this->create_lists();
	//2
	this->base_table();
	for(uint64_t i = 0;i < this->d;i++) free(lists[i]);
	free(lists);
	//3
	this->build_index();

//	if(this->n < 50){
//	std::cout << "<<index>>"<< std::endl;
//	for(uint64_t i = 0;i<25;i++){
//		std::cout << std::dec << std::setfill('0') << std::setw(6) << i << ": ";
//		for(uint8_t m = 0;m <this->d;m++){
//			std::cout << std::dec << std::setfill('0') << std::setw(6) << TT[i*this->d + m] << " ";
//		}
//		std::cout << " | 0x" <<  std::hex << std::setfill('0') << std::setw(16) << II[i] << std::endl;
//		//printf ("| 0x%llx\n",II[i]) ;
//
//	}
//	std::cout << std::dec;
//	}
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void cFA<T,Z>::base_table(){
	for(uint64_t i=0;i<this->n;i++){
		for(uint8_t m =0;m<this->d;m++){
			pred<T,Z> p = lists[m][i];
			TT[p.tid * this->d + m] = i;
		}
	}

//	if(this->n < 50){
//	std::cout << std::endl << "<TT>" << std::endl;
//	for(uint64_t i=0;i<25;i++){
//		std::cout << std::dec << std::setfill('0') << std::setw(6) << i << ": ";
//		for(uint8_t m =0;m<this->d;m++){
//			std::cout << std::dec << std::setfill('0') << std::setw(6) << TT[ i * this->d + m] << " ";
//		}
//		std::cout << std::endl;
//	}
//	}
}

template<class T, class Z>
void cFA<T,Z>::build_index(){
	build_attr_index_r<Z>(II,TT,this->n,this->d);
}

template<class T, class Z>
void cFA<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	R = (Z *)malloc(sizeof(Z)*this->n);
	Z *Tm = (Z *)malloc(sizeof(Z)*this->n);
	Z *Tw = (Z *)malloc(sizeof(Z)*this->n);
	for(uint64_t i = 0;i <this->n;i++) R[i] = 0;

	this->t.start();
	//
	//prepare for prefix sum
	Z seen_pos;
	uint8_t tm_shf = (this->d-1)*4;
	uint8_t tw_shf = 0;
	//std::cout << std::endl << "<RR>" << std::endl;
	//this->t.start();
	for(uint64_t i = 0; i < this->n; i++){
		//Tm[i] = TT[(II[i] & (0xF << tm_shf)) >> tm_shf];
		seen_pos = TT[i * this->d + ((II[i] & ((uint64_t)0xF << tm_shf)) >> tm_shf)];
		Tw[i] = (TT[i * this->d + (II[i] & (0xF))] == 0) ?  1 : TT[i * this->d + (II[i] & (0xF))];
		//std::cout << i <<": " << seen_pos<<  "," << Tw[i] <<std::endl;
		R[seen_pos]++;
	}

//	if(this->n < 50){
//		std::cout <<std::endl << "<R>" << std::endl;
//		for(uint64_t i = 0; i < this->n; i++){
//			std::cout << i<<  ": " << R[i] <<std::endl;
//		}
//	}

	//Prefix sum
	uint64_t upper = 0;
	R[0] = 1;
	for(uint64_t i = 1; i < this->n; i++){
		R[i] = R[i-1] + R[i];
		upper = i;// Find offset of prefix that is less than k
		if(R[i] >= k) break;// >= or > ?
	}

//	if(this->n < 50){
//		std::cout <<std::endl << "<RS>" << std::endl;
//		for(uint64_t i = 0; i < this->n; i++){
//			std::cout << i<<  ": " << R[i] <<std::endl;
//		}
//	}

	//Process only tuples participating in top-K
	for(uint64_t i = 0; i <this->n; i++){
		if(Tw[i]-1 <= upper){
			Z id = i;
			T score = 0;
			for(uint8_t m = 0; m < this->d; m++){
				score+=this->cdata[id * this->d + m];
			}
			if(STATS_EFF) this->tuple_count++;
			if(STATS_EFF) this->pred_count+=this->d;

			if(this->q.size() < k){//insert if empty space in queue
				this->q.push(tuple<T,Z>(id,score));
			}else if(this->q.top().score<score){//delete smallest element if current score is bigger
				this->q.pop();
				this->q.push(tuple<T,Z>(id,score));
			}
		}
	}
	this->tt_processing = this->t.lap();
	free(Tm);
	free(Tw);

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

#endif
