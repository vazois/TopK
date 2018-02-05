#ifndef PFAC_H
#define PFAC_H

#include "AA.h"
#include <unordered_map>
#include <unordered_set>
#include <list>

template<class T, class Z>
class PFAc : public AA<T,Z>{
	public:
		PFAc(uint64_t n,uint64_t d) : AA<T,Z>(n,d){ this->algo = "PFAc";  this->I=NULL; };

		void init();
		void findTopK(uint64_t k);
	protected:
		void create_lists();
		void create_indices();

	private:
		pred<T,Z> **lists;
		Z **I;

};

template<class T, class Z>
void PFAc<T,Z>::create_lists(){
	//Create and Sort Lists//
	for(uint64_t i=0;i<this->n;i++){
		for(uint8_t m =0;m<this->d;m++){
			lists[m][i] = pred<T,Z>(i,this->cdata[i*this->d + m]);
		}
	}
	for(uint8_t m =0;m<this->d;m++){ __gnu_parallel::sort(lists[m],lists[m] + this->n,cmp_max_pred<T,Z>); }
	/////////////////////////
}

template<class T, class Z>
void PFAc<T,Z>::create_indices(){
	I = (Z**)malloc(sizeof(Z*)*this->d);
	for(uint8_t m =0;m<this->d;m++){
		I[m] = (Z*)malloc(sizeof(Z)*this->n);
		for(uint64_t i = 0; i < this->n;i++){
			pred<T,Z> p = lists[m][i];
			I[m][p.tid] = i;
		}
	}
}

template<class T, class Z>
void PFAc<T,Z>::init(){
	lists = (pred<T,Z> **)malloc(sizeof(pred<T,Z>*) * this->d);
	for(uint64_t i = 0;i < this->d;i++) lists[i] = (pred<T,Z>*)malloc(sizeof(pred<T,Z>) * this->n);
	this->t.start();
	this->create_lists();
	this->create_indices();
	this->tt_init = this->t.lap();

	for(uint64_t i = 0;i < this->d;i++) free(lists[i]);
	free(lists);
}

template<class T, class Z>
void PFAc<T,Z>::findTopK(uint64_t){
	std::cout << this->algo << " find topK ...";


	if(I!=NULL){
		for(uint8_t m =0;m<this->d;m++) free(I[m]);
		free(I);
	}
}

#endif
