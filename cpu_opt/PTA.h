#ifndef PTA_H
#define PTA_H
/*
* Partitioned Threshold Aggregation
*/

#include "../cpu/AA.h"
#include <math.h>

#define PBLOCK_SIZE 1024
#define PBLOCK_SHF 2
#define PPARTITIONS (1)

#define PSLITS 4
#define PI 3.1415926535

template<class T, class Z>
struct pta_pair{
	Z id;
	T score;
};

template<class Z>
struct pta_pos{
	Z id;
	Z pos;
};

template<class T, class Z>
struct pta_block{
	Z offset;
	Z tuple_num;
	T tarray[NUM_DIMS] __attribute__((aligned(32)));
	T tuples[PBLOCK_SIZE * NUM_DIMS] __attribute__((aligned(32)));
};

template<class T, class Z>
struct pta_partition{
	Z offset;
	Z size;
	Z block_num;
	pta_block<T,Z> *blocks;
};

template<class T,class Z>
static bool cmp_pta_pos(const pta_pos<Z> &a, const pta_pos<Z> &b){ return a.pos < b.pos; };

template<class T,class Z>
static bool cmp_pta_pair(const pta_pair<T,Z> &a, const pta_pair<T,Z> &b){ return a.score > b.score; };

template<class T,class Z>
class PTA : public AA<T,Z>{
	public:
		PTA(uint64_t n,uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "PTA";
			this->splits = NULL;
		}

		~PTA(){
			if(this->splits!=NULL) free(this->splits);
		}
		void init();
		void findTopKscalar(uint64_t k,uint8_t qq);
		void findTopKsimd(uint64_t k,uint8_t qq);
		void findTopKthreads(uint64_t k,uint8_t qq);

	private:
		pta_partition<T,Z> parts[PPARTITIONS];
		T *splits;
		void polar();
};

template<class T, class Z>
void PTA<T,Z>::polar(){
	T *pdata = static_cast<T*>(aligned_alloc(32,sizeof(T)*this->n * (this->d-1)));

	__m256 pi = _mm256_set1_ps(PI);
	__m256 _180 = _mm256_set1_ps(180.0);
	__m256 abs = _mm256_set1_ps(0x7FFFFFFF);
	__m256 one = _mm256_set1_ps(1.1);

	for(uint64_t i = 0; i < this->n; i+=16){
		__m256 sum = _mm256_setzero_ps();
		__m256 f = _mm256_setzero_ps();
		__m256 curr = _mm256_load(&this->cdata[i]);
		__m256 next;

		curr = _mm256_sub_ps(curr,one);
		for(uint32_t m = 0; m < this->d-1;m++){
			next = _mm256_load(&this->cdata[(m+1)*this->n + i]);
			next = _mm256_sub_ps(next,one);
			sum = _mm256_add_ps(sum,_mm256_mul_ps(curr,curr));//(sum +=x_i ^ 2)
			f = _mm256_sqrt_ps(sum);

			#ifdef GNU
				f = _mm256_div_ps(sum,next);
				_mm256_store_ps(&pdata[m*this->n + i],f);
				uint64_t offset = m*this->n + i;
				pdata[offset] = atan(pdata[offset]);
				pdata[offset+1] = atan(pdata[offset+1]);
				pdata[offset+2] = atan(pdata[offset+2]);
				pdata[offset+3] = atan(pdata[offset+3]);
				pdata[offset+4] = atan(pdata[offset+4]);
				pdata[offset+5] = atan(pdata[offset+5]);
				pdata[offset+6] = atan(pdata[offset+6]);
				pdata[offset+7] = atan(pdata[offset+7]);
				pdata[offset+8] = atan(pdata[offset+8]);
				pdata[offset+9] = atan(pdata[offset+9]);
				pdata[offset+10] = atan(pdata[offset+10]);
				pdata[offset+11] = atan(pdata[offset+11]);
				pdata[offset+12] = atan(pdata[offset+12]);
				pdata[offset+13] = atan(pdata[offset+13]);
				pdata[offset+14] = atan(pdata[offset+14]);
				pdata[offset+15] = atan(pdata[offset+15]);
			#elif
				f = _mm256_atan2_ps(f,next);
				f = _mm256_and_ps(_mm256_div_ps(_mm256_mul_ps(f,_180),pi),abs);
				_mm256_store_ps(&pdata[m*this->n + i],f);
			#endif
			curr = next;
		}
	}

	free(pdata);
}

template<class T, class Z>
void PTA<T,Z>::init(){
	this->splits =(T*)malloc(sizeof(this->d)*PSLITS);
	this->t.start();



	this->tt_init = this->t.lap();
}

template<class T, class Z>
void PTA<T,Z>::findTopKscalar(uint64_t k, uint8_t qq){

}

template<class T, class Z>
void PTA<T,Z>::findTopKsimd(uint64_t k, uint8_t qq){

}

#endif
