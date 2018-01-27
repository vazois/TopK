#ifndef T2S_H
#define T2S_H

#include "TA.h"

#define alpha 128

template<class T,class Z>
class T2S : public AA<T,Z>{
	public:
		T2S(uint64_t n,uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "T2S";
			this->alists = NULL;
			this->rrlist = NULL;
		}

		~T2S()
		{
			this->freeLists();
		}

		pred<T,Z> **alists;
		pred<T,Z> *rrlist;
		void init();
		void findTopK(uint64_t k);

	private:
		void freeLists();
		void printBaseTable(bool ordered);

};


template<class T, class Z>
void T2S<T,Z>::freeLists()
{
	if(this->alists!=NULL){
		for(uint64_t i = 0;i < this->d;i++){
			if(this->alists[i] != NULL){
				free(this->alists[i]);
			}
		}
	}
	this->alists = NULL;
}

template<class T, class Z>
void T2S<T,Z>::printBaseTable(bool ordered)
{
	if(!ordered){
		std::cout << "<Not Ordered Base Table>" << std::endl;
		for(uint64_t i = 0;i <10;i++){
			pred<T,Z> p = this->rrlist[i];
			std::cout << "(" <<p.tid << ") -- ";
			for(uint8_t m =0;m<this->d;m++){
				std::cout << this->cdata[p.tid * this->d + m] << " ";
			}
			std::cout << std::endl;
		}
	}else{
		std::cout << "<Ordered Base Table>" << std::endl;
		for(uint64_t i = 0;i <10;i++){
			pred<T,Z> p = this->rrlist[i];
			std::cout << "(" <<p.tid << ") -- ";
			for(uint8_t m =0;m<this->d;m++){
				std::cout << this->cdata[i * this->d + m] << " ";
			}
			std::cout << std::endl;
		}
	}
}

template<class T, class Z>
void T2S<T,Z>::init(){
	//Allocate and initialize lists
	this->alists = (pred<T,Z> **)malloc(sizeof(pred<T,Z>*) * this->d);
	this->rrlist = (pred<T,Z>*)malloc(sizeof(pred<T,Z>) * this->n);
	for(uint64_t i = 0;i < this->d;i++){ this->alists[i] = (pred<T,Z>*)malloc(sizeof(pred<T,Z>) * this->n); }
	for(uint64_t i=0;i<this->n;i++){
		for(uint8_t m =0;m<this->d;m++){
			this->alists[m][i] = pred<T,Z>(i,this->cdata[i*this->d + m]);
		}
	}

	this->t.start();
	for(uint8_t m =0;m<this->d;m++){
		__gnu_parallel::sort(this->alists[m],this->alists[m] + this->n,cmp_max_pred<T,Z>);
	}

	//Reorder Base Table
	this->t.start();
	std::unordered_set<Z> eset;
	Z nn = 0;
	for(uint64_t i=0;i<this->n;i++){
		T threshold = 0;
		for(uint8_t m =0;m<this->d;m++){
			pred<T,Z> p = this->alists[m][i];
			threshold+=p.attr;
		}
		for(uint8_t m =0;m<this->d;m++){
			pred<T,Z> p = this->alists[m][i];
			if(eset.find(p.tid) == eset.end()){
				this->rrlist[nn] = pred<T,Z>(p.tid,threshold);
				eset.insert(p.tid);
				nn++;
			}
			if(nn >= this->n) break;
		}
		if(nn >= this->n) break;
	}
	this->t.lap("<2>");
	eset.clear();
	this->freeLists();
	this->printBaseTable(false);

	this->t.start();
	T *tmp = (T*)malloc(sizeof(T)*this->n*this->d);
	for(uint64_t i=0;i<this->n;i++){
		pred<T,Z> p = this->rrlist[i];
//		if(i < 10){
//			std::cout << "<<<>>>>" << std::endl;
//			for(uint8_t m =0;m<this->d;m++){
//				std::cout << this->cdata[p.tid * this->d + m] << " ";
//			}
//			std::cout << std::endl;
//		}
		memcpy(&tmp[i * this->d] ,&this->cdata[p.tid * this->d],sizeof(T)*this->d);
//		if(i < 10){
//			for(uint8_t m =0;m<this->d;m++){
//				std::cout << tmp[i * this->d + m] << " ";
//			}
//			std::cout << std::endl;
//			std::cout << "<<<--------------->>>>" << std::endl;
//		}
	}
	memcpy(this->cdata,tmp,sizeof(T)*this->d*this->n);
	this->t.lap("<3>");
	this->printBaseTable(true);
	free(tmp);
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void T2S<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	T attr[THREADS];


	omp_set_num_threads(THREADS);
	#pragma omp parallel
	{
		uint32_t tid = omp_get_thread_num();
		uint32_t gsize = omp_get_num_threads();

		for(uint64_t i = tid; i < this->n; i+=gsize*alpha){
			for(uint64_t j = i; j < i + gsize*alpha ; j+=gsize){
				pred<T,Z> p = this->rrlist[j];

				T score = 0;
				for(uint8_t m = 0; m < this->d; m++){
					score+=this->cdata[j * this->d + m];
				}
				attr[tid] = p.attr;
			}
		}
	}
}

#endif
