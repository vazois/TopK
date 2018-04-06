#ifndef PVTA_H
#define PVTA_H

#include "../cpu/AA.h"


template<class T,class Z>
class PVTA : public AA<T,Z>{
	public:
		PVTA(uint64_t n,uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "PVTA";
		}

		~PVTA(){

		}
		void init();
		void findTopKscalar(uint64_t k,uint8_t qq);
		void findTopKsimd(uint64_t k,uint8_t qq);
		void findTopKthreads(uint64_t k,uint8_t qq);

	private:
		void polar(T *&pdata);

};

template<class T, class Z>
void PVTA<T,Z>::polar(T *&pdata){
	float score[16] __attribute__((aligned(32)));
	__m256 curr = _mm256_load(&this->cdata[0]);
	__m256 next;
	__m256 sum = _mm256_setzero_ps();
	__m256 f = _mm256_setzero_ps();

	for(uint32_t m = 0; m < this->d-1;m++){
		next = _mm256_load(&this->cdata[(m+1)*this->n]);
		sum = _mm256_add_ps(sum,_mm256_mul_ps(curr,curr));
		f = _mm256_sqrt_ps(sum);
#ifdef GNU
		_mm256_store_ps(&score[m*this->n],f);
#else
		f = _mm256_atan2_ps(f,next);
		_mm256_store_ps(&score[m*this->n],f);
#endif
		curr = next;
	}
}

#endif
