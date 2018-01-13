#ifndef NRA_H
#define NRA_H

#include "cpu/FA.h"

template<class T, class Z>
class NRA : public FA<T,Z>{
	public:
		NRA(uint64_t n,uint64_t d) : FA<T,Z>(n,d){ this->algo = "TA"; };

		void findTopK(uint64_t k);
	protected:

	private:
		void seq_topk(uint64_t);
		void par_topk(uint64_t);

};


template<class T,class Z>
void NRA<T,Z>::findTopK(uint64_t k){
	//Note: keep truck of ids so you will not re-insert the same tupple as your process them in order
	std::cout << this->algo << " find topK ...";

	this->t.start();
	if(this->topkp){
		this->par_topk(k);
	}else{
		this->seq_topk(k);
	}
	this->tt_processing = this->t.lap("");
}

template<class T, class Z>
void NRA<T,Z>::seq_topk(uint64_t k){

}

template<class T, class Z>
void NRA<T,Z>::par_topk(uint64_t k){

}

#endif
