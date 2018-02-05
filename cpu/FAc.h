#ifndef FA_C_H
#define FA_C_H

#include "AA.h"
#include <unordered_map>
#include <unordered_set>
#include <list>

#include "build_attr_index_r.h"

template<class T, class Z>
class FAc : public AA<T,Z>{
	public:
		FAc(uint64_t n,uint64_t d) : AA<T,Z>(n,d){
			this->algo = "FAc";
			this->TT = NULL;
			this->II = NULL;
			this->R = NULL;
		};
		~FAc(){
			if(this->TT != NULL) free(TT);
			if(this->II != NULL) free(II);
			if(this->R != NULL ) free(R);
		};

		void init();
		void init2();
		void findTopK(uint64_t k);
		void findTopK2(uint64_t k);

	protected:
		void create_lists();
		void build_seen();
		void compute_prefix_sum();
		void compute_participation();//In the computatation of top-K result

		void base_table();
		void build_index();
	private:
		std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
		Z *tids;
		Z *R;
		Z *M;
		Z *test;

		pred<T,Z> **lists;
		Z *TT;//Base table with appearance order for each tuple within each list
		uint64_t *II;
};

template<class T, class Z>
void FAc<T,Z>::create_lists(){
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

template<class T, class Z>
void FAc<T,Z>::build_seen(){
	//Count seen count per level//
	std::unordered_map<Z,uint8_t> tmap;
	for(uint64_t i = 0; i < this->n;i++){
		uint64_t seen_count=0;
		for(uint8_t m =0;m<this->d;m++){
			pred<T,Z> p = lists[m][i];
			if ( tmap.find(p.tid) == tmap.end() ){
				tmap.insert(std::pair<Z,uint8_t>(p.tid,1));
			}else{
				tmap[p.tid]++;
				if( tmap[p.tid] == this->d ){
					seen_count++;
					//tmap.erase(p.tid);
				}
			}
			test[i*(this->d+2) + m] = p.tid;
		}
		M[i] = seen_count;
		test[i*(this->d+2) + this->d] = M[i];
	}
	//////////////////////////////
}

template<class T, class Z>
void FAc<T,Z>::compute_prefix_sum(){
	Z prev = M[0];
	Z tmp = 0;
	M[0]=1;
	test[0*(this->d+2) + this->d + 1] = M[0];
	for(uint64_t i = 1; i < this->n;i++){
		tmp=M[i];
		M[i] = M[i-1] + prev;
		prev=tmp;
		test[i*(this->d+2) + this->d + 1] = M[i];
	}
}

template<class T, class Z>
void FAc<T,Z>::compute_participation(){
	tids = (Z *)malloc(sizeof(Z)*this->n);
	R = (Z *)malloc(sizeof(Z)*this->n);

	uint64_t rows=0;
	std::unordered_set<Z> tset;
	for(uint64_t i = 0; i < this->n;i++){
		uint64_t seen_count=0;
		for(uint8_t m =0;m<this->d;m++){
			pred<T,Z> p = lists[m][i];
			if ( tset.find(p.tid) == tset.end() ){
				tset.insert(p.tid);
				tids[rows]=p.tid;
				R[rows]=M[i];
				rows++;
			}
		}
	}
}

template<class T, class Z>
void FAc<T,Z>::init(){
	lists = (pred<T,Z> **)malloc(sizeof(pred<T,Z>*) * this->d);
	M = (Z*)malloc(sizeof(Z)*this->n);
	test = (Z*)malloc(sizeof(Z)*this->n*(this->d+2));
	for(uint64_t i = 0;i < this->d;i++) lists[i] = (pred<T,Z>*)malloc(sizeof(pred<T,Z>) * this->n);

	this->t.start();

	this->create_lists();
	this->build_seen();
	this->compute_prefix_sum();
	this->compute_participation();

	//Prefix sum calculation//
	this->tt_init = this->t.lap();
	//

	//Print results//
	if(this->n < 50){
		std::cout << "\t";
		for(uint32_t m =0;m<this->d;m++){ std::cout << "L"<<m<<"\t"; } std::cout << "e\tM";
		std::cout << std::endl;
		for(uint64_t i = 0; i < this->n ;i++){
			std::cout<<i<<":\t";
			for(uint8_t m =0;m<this->d;m++){
				std::cout << test[i*(this->d+2) + m] << "\t";
			}
			std::cout << test[i*(this->d+2) + this->d] << "\t";
			std::cout << test[i*(this->d+2) + this->d+1] << "\t";
			std::cout << std::endl;
		}
		for(uint64_t i = 0; i < this->n ;i++){
			std::cout << "t (" << tids[i] << "," << R[i] <<")" << std::endl;
		}
	}
	///////////////


	free(M);
	free(test);
	for(uint64_t i = 0;i < this->d;i++) free(lists[i]);
	free(lists);
}

template<class T,class Z>
void FAc<T,Z>::init2(){
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

	if(this->n < 50){
	std::cout << "<<index>>"<< std::endl;
	for(uint64_t i = 0;i<this->n;i++){
		std::cout << std::dec << std::setfill('0') << std::setw(3) << i << ": ";
		for(uint8_t m = 0;m <this->d;m++){
			std::cout << std::dec << std::setfill('0') << std::setw(3) << TT[i*this->d + m] << " ";
		}
		std::cout << " | 0x" <<  std::hex << std::setfill('0') << std::setw(16) << II[i] << std::endl;
	}
	std::cout << std::dec;
	}
	this->tt_init = this->t.lap();

	//for(uint64_t i = 0;i < this->d;i++) free(lists[i]);
	//free(lists);
	//free(TT);
	//free(II);
}

template<class T, class Z>
void FAc<T,Z>::base_table(){
	for(uint64_t i=0;i<this->n;i++){
		for(uint8_t m =0;m<this->d;m++){
			pred<T,Z> p = lists[m][i];
			TT[p.tid * this->d + m] = i;
		}
	}

	if(this->n < 50){
	std::cout << std::endl << "<TT>" << std::endl;
	for(uint64_t i=0;i<this->n;i++){
		std::cout << std::dec << std::setfill('0') << std::setw(3) << i << ": ";
		for(uint8_t m =0;m<this->d;m++){
			std::cout << std::dec << std::setfill('0') << std::setw(3) << TT[ i * this->d + m] << " ";
		}
		std::cout << std::endl;
	}
	}
}

template<class T, class Z>
void FAc<T,Z>::build_index(){
//	switch(this->d){
//		case 4:
//			build_attr_index_r_4<Z>(II,TT,this->n,this->d);
//			break;
//		default:
//			break;
//	}
	build_attr_index_r<Z>(II,TT,this->n,this->d);
}

template<class T, class Z>
void FAc<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";

	this->t.start();
	for(uint64_t i = 0; i <this->n; i++){
		//std::cout << tids[i] <<  " : " << R[i] << std::endl;
		if(R[i] <= k){
			Z id = tids[i];
			T score = 0;
			for(uint8_t m = 0; m < this->d; m++){
				score+=this->cdata[id * this->d + m];
			}
			if(this->q.size() < k){//insert if empty space in queue
				this->q.push(tuple<T,Z>(id,score));
			}else if(this->q.top().score<score){//delete smallest element if current score is bigger
				this->q.pop();
				this->q.push(tuple<T,Z>(id,score));
			}
		}else{
			break;
		}
	}
	this->tt_processing = this->t.lap();
	while(!this->q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(this->q.top());
		this->q.pop();
	}
	std::cout << " (" << this->res.size() << ")" << std::endl;
}

template<class T, class Z>
void FAc<T,Z>::findTopK2(uint64_t k){
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
		seen_pos = TT[i * this->d + ((II[i] & (0xF << tm_shf)) >> tm_shf)];
		Tw[i] = (TT[i * this->d + (II[i] & (0xF))] == 0) ?  1 : TT[i * this->d + (II[i] & (0xF))];
		//std::cout << i <<": " << seen_pos<<  "," << Tw[i] <<std::endl;
		R[seen_pos]++;
	}

	if(this->n < 50){
		std::cout <<std::endl << "<R>" << std::endl;
		for(uint64_t i = 0; i < this->n; i++){
			std::cout << i<<  ": " << R[i] <<std::endl;
		}
	}

	//Prefix sum
	uint64_t upper = 0;
	for(uint64_t i = 1; i < this->n; i++){
		R[i] = R[i-1] + R[i];
		upper = i;// Find offset of prefix that is less than k
		if(R[i] >= k) break;// >= or > ?
	}

	if(this->n < 50){
		std::cout <<std::endl << "<RS>" << std::endl;
		for(uint64_t i = 0; i < this->n; i++){
			std::cout << i<<  ": " << R[i] <<std::endl;
		}
	}

	for(uint64_t i = 0; i <this->n; i++){
		//Z r = R[Tw[i]-1];
		//std::cout << i <<  " : " << r << std::endl;
		//if(r <= k){
		if(Tw[i]-1 <= upper){
			Z id = i;
			T score = 0;
			for(uint8_t m = 0; m < this->d; m++){
				score+=this->cdata[id * this->d + m];
			}
			if(this->q.size() < k){//insert if empty space in queue
				this->q.push(tuple<T,Z>(id,score));
			}else if(this->q.top().score<score){//delete smallest element if current score is bigger
				this->q.pop();
				this->q.push(tuple<T,Z>(id,score));
			}
		}
	}
	this->tt_processing = this->t.lap();
	while(!this->q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(this->q.top());
		this->q.pop();
	}
	std::cout << " (" << this->res.size() << ")" << std::endl;
	//free(R);
	free(Tm);
	free(Tw);
}

#endif
