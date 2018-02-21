#ifndef GSA_H
#define GSA_H

#include "../cpu/AA.h"
#include <cmath>

template<class T, class Z>
class GSA : public AA<T,Z>{
	public:
		GSA(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "GSA";
			this->bins = 16;
			this->bits = ceil(log2((float)this->bins));
			this->len = ceil((float)(this->bits * this->d)/(64));
		}

		~GSA(){

		}

		void init();
		void findTopK(uint64_t k);

	private:
		uint64_t bins;
		uint64_t bits;
		uint64_t len;
};

template<class T, class Z>
void GSA<T,Z>::init(){
	this->t.start();

	std::cout << this->bins << " , " << this->bits << " , " << this->len << std::endl;

	this->tt_init = this->t.lap();
}

#endif
