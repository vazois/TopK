#ifndef PTA_H
#define PTA_H
/*
* Partitioned Threshold Aggregation
*/

#include "../cpu/AA.h"
#include <math.h>
#include <map>

#define PBLOCK_SIZE 1024
#define PBLOCK_SHF 2
#define PPARTITIONS (1)

#define PSLITS 2
#define PI 3.1415926535
#define PI_2 (180.0f/PI)

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
			this->part_id = NULL;
		}

		~PTA(){
			if(this->splits!=NULL) free(this->splits);
			if(this->part_id!=NULL) free(this->part_id);
		}
		void init();
		void findTopKscalar(uint64_t k,uint8_t qq);
		void findTopKsimd(uint64_t k,uint8_t qq);
		void findTopKthreads(uint64_t k,uint8_t qq);

	private:
		pta_partition<T,Z> parts[PPARTITIONS];
		T *splits;
		Z *part_id;
		void polar();
};

template<class T, class Z>
void PTA<T,Z>::polar(){
	T *pdata = static_cast<T*>(aligned_alloc(32,sizeof(T)*this->n * (this->d-1)));
	pta_pair<T,Z> *pp = (pta_pair<T,Z>*)malloc(sizeof(pta_pair<T,Z>)*this->n);
	this->part_id = static_cast<Z*>(aligned_alloc(32,sizeof(Z)*this->n));

	__m256 pi_2 = _mm256_set1_ps(PI_2);
	__m256 abs = _mm256_set1_ps(0x7FFFFFFF);
	__m256 one = _mm256_set1_ps(1.0);
	for(uint64_t i = 0; i < this->n; i+=8){
		__m256 sum = _mm256_setzero_ps();
		__m256 f = _mm256_setzero_ps();
		__m256 curr = _mm256_load_ps(&this->cdata[(this->d-1)*this->n + i]);
		__m256 next;

		curr = _mm256_sub_ps(curr,one);//x_i=x_i - 1.0
		for(uint32_t m = this->d-1; m > 0;m--){
			next = _mm256_load_ps(&this->cdata[(m-1)*this->n + i]);//x_(i-1)
			next = _mm256_sub_ps(next,one);
			sum = _mm256_add_ps(sum,_mm256_mul_ps(curr,curr));//(sum +=x_i ^ 2)
			f = _mm256_sqrt_ps(sum);//sqrt(x_i^2+x_(i-1)^2+...)

			#ifdef GNU
				f = _mm256_div_ps(f,next);//sqrt(x_i^2+x_(i-1)^2+...+x_(i-k)/x_(i-k+1)
				uint64_t offset = (m-1)*this->n + i;
				_mm256_store_ps(&pdata[offset],f);
				pdata[offset] = fabs(atan(pdata[offset])*PI_2);
				pdata[offset+1] = fabs(atan(pdata[offset+1])*PI_2);
				pdata[offset+2] = fabs(atan(pdata[offset+2])*PI_2);
				pdata[offset+3] = fabs(atan(pdata[offset+3])*PI_2);
				pdata[offset+4] = fabs(atan(pdata[offset+4])*PI_2);
				pdata[offset+5] = fabs(atan(pdata[offset+5])*PI_2);
				pdata[offset+6] = fabs(atan(pdata[offset+6])*PI_2);
				pdata[offset+7] = fabs(atan(pdata[offset+7])*PI_2);
				pdata[offset+8] = fabs(atan(pdata[offset+8])*PI_2);
				pdata[offset+9] = fabs(atan(pdata[offset+9])*PI_2);
				pdata[offset+10] = fabs(atan(pdata[offset+10])*PI_2);
				pdata[offset+11] = fabs(atan(pdata[offset+11])*PI_2);
				pdata[offset+12] = fabs(atan(pdata[offset+12])*PI_2);
				pdata[offset+13] = fabs(atan(pdata[offset+13])*PI_2);
				pdata[offset+14] = fabs(atan(pdata[offset+14])*PI_2);
				pdata[offset+15] = fabs(atan(pdata[offset+15])*PI_2);
			#else
				f = _mm256_atan2_ps(f,next);
				f = _mm256_and_ps(_mm256_mul_ps(f,pi_2),abs);
				_mm256_store_ps(&pdata[m*this->n + i],f);
			#endif
			curr = next;//x_i = x_{i-1)
		}
	}

//	for(uint64_t i = 0; i < 16; i++){
//		std::cout << std::fixed;
//		std::cout << "[";
//		for(uint32_t m = 0; m < this->d; m++){ std::cout << std::setprecision(4) << this->cdata[m*this->n + i] << " , "; }
//		std::cout << "] ";
//		std::cout << "[";
//		for(uint32_t m = 0; m < this->d-1; m++){ std::cout << std::setprecision(4) << pdata[m*this->n + i] << " , "; }
//		std::cout << "]";
//		std::cout << std::endl;
//	}

	uint64_t mod = (this->n / PSLITS);
	uint64_t mul = 1;
	for(uint64_t i = 0; i < this->n; i++) this->part_id[i] = 0;
	for(uint32_t m = 0; m < this->d-1; m++){
		for(uint64_t i = 0; i < this->n; i++){
			pp[i].id = i;
			pp[i].score = pdata[m*this->n + i];
		}
		__gnu_parallel::sort(&pp[0],(&pp[0]) + this->n,cmp_pta_pair<T,Z>);

		for(uint64_t i = 0; i < this->n; i++){ this->part_id[pp[i].id]+=(mul*(i / mod)); }
		mul*=PSLITS;
	}

	std::map<Z,Z> mm;
	for(uint64_t i = 0; i < this->n; i++){
		Z pid = this->part_id[i];
		if( mm.find(pid) == mm.end()){ mm.insert(std::pair<Z,Z>(pid,0)); }
		mm[pid]+=1;
	}
	std::cout << "mm_size: " << mm.size() << std::endl;
	for(typename std::map<Z,Z>::iterator it = mm.begin(); it != mm.end(); ++it){
		std::cout << "g(" << it->first << "):" << std::setfill('0') << std::setw(8) << it->second << std::endl;
	}

	free(pp);
	free(pdata);
}

template<class T, class Z>
void PTA<T,Z>::init(){
	this->splits =(T*)malloc(sizeof(this->d)*PSLITS);
	this->t.start();

	this->polar();

	this->tt_init = this->t.lap();
}

template<class T, class Z>
void PTA<T,Z>::findTopKscalar(uint64_t k, uint8_t qq){

}

template<class T, class Z>
void PTA<T,Z>::findTopKsimd(uint64_t k, uint8_t qq){

}

#endif
