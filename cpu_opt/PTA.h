#ifndef PTA_H
#define PTA_H
/*
* Partitioned Threshold Aggregation
*/

#include "../cpu/AA.h"
#include <cmath>
#include <map>

#define PSLITS 2
#define PI 3.1415926535
#define PI_2 (180.0f/PI)
#define SIMD_GROUP 16

#define PBLOCK_SIZE 1024
#define PPARTITIONS (((uint64_t)pow(PSLITS,NUM_DIMS-1)))

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

template<class Z>
static bool cmp_pta_pos(const pta_pos<Z> &a, const pta_pos<Z> &b){ return a.pos < b.pos; };

template<class T,class Z>
static bool cmp_pta_pair(const pta_pair<T,Z> &a, const pta_pair<T,Z> &b){ return a.score > b.score; };

template<class T,class Z>
class PTA : public AA<T,Z>{
	public:
		PTA(uint64_t n,uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "PTA";
			this->part_id = NULL;
		}

		~PTA(){
			if(this->part_id!=NULL) free(this->part_id);
		}

		void init();
		void findTopKscalar(uint64_t k,uint8_t qq);
		void findTopKsimd(uint64_t k,uint8_t qq);
		void findTopKthreads(uint64_t k,uint8_t qq);

	private:
		pta_partition<T,Z> parts[PPARTITIONS];
		Z *part_id;
		Z max_part_size;

		void polar();
		void create_partitions();
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
		for(uint64_t i = 0; i < this->n; i++){ pp[i].id = i; pp[i].score = pdata[m*this->n + i]; }
		__gnu_parallel::sort(&pp[0],(&pp[0]) + this->n,cmp_pta_pair<T,Z>);
		for(uint64_t i = 0; i < this->n; i++){ this->part_id[pp[i].id]+=(mul*(i / mod)); }
		mul*=PSLITS;
	}

	//
	std::map<Z,Z> mm;
	for(uint64_t i = 0; i < PPARTITIONS;i++) mm.insert(std::pair<Z,Z>(i,0));
	for(uint64_t i = 0; i < this->n; i++){
		Z pid = this->part_id[i];
//		if( mm.find(pid) == mm.end()){ mm.insert(std::pair<Z,Z>(pid,0)); }
		mm[pid]+=1;
	}

	std::cout << "mm_size: " << mm.size() << " --> " << PPARTITIONS<< std::endl;
	this->max_part_size = 0;
	for(typename std::map<Z,Z>::iterator it = mm.begin(); it != mm.end(); ++it){
		uint64_t psize = it->second + (PBLOCK_SIZE - (it->second % PBLOCK_SIZE));
		std::cout << "g(" << it->first << "):" << std::setfill('0') << std::setw(8) << it->second << " < "
				<< psize << " [ " << ((float)psize)/PBLOCK_SIZE << " , " << ((float)psize)/SIMD_GROUP << " ] " << std::endl;

		this->parts[it->first].size = it->second;
		this->parts[it->first].block_num = ((float)psize)/PBLOCK_SIZE;
		this->parts[it->first].blocks = static_cast<pta_block<T,Z>*>(aligned_alloc(32,sizeof(pta_block<T,Z>)*this->parts[it->first].block_num));
		this->max_part_size = std::max(this->max_part_size,it->second);
	}
	//
	free(pp);
	free(pdata);
}

template<class T, class Z>
void PTA<T,Z>::create_partitions(){
	pta_pos<Z> *ppos = (pta_pos<Z>*)malloc(sizeof(pta_pos<Z>)*this->n);
	for(uint64_t i = 0; i < this->n; i++){ ppos[i].id =i; ppos[i].pos = this->part_id[i]; }
	__gnu_parallel::sort(&ppos[0],(&ppos[0]) + this->n,cmp_pta_pos<Z>);

	uint64_t gindex = 0;
	pta_pos<Z> *pos = (pta_pos<Z>*)malloc(sizeof(pta_pos<Z>)*this->max_part_size);
	pta_pair<T,Z> **lists = (pta_pair<T,Z>**)malloc(sizeof(pta_pair<T,Z>*)*this->d);
	for(uint32_t m=0; m<this->d;m++) lists[m] = (pta_pair<T,Z>*)malloc(sizeof(pta_pair<T,Z>)*this->max_part_size);
	for(uint32_t m = 0; m < this->d; m++){ for(uint64_t j = 0; j < this->max_part_size;j++){ lists[m][j].id = 0; lists[m][j].score = 0; } }

	for(uint64_t i = 0; i < PPARTITIONS;i++){
		for(uint64_t j = 0; j < this->max_part_size;j++){ pos[j].id = j; pos[j].pos = this->n; }//Initialize to max possible position
		for(uint32_t m = 0; m < this->d; m++){//Initialize lists for given partition//
			for(uint64_t j = 0; j < this->parts[i].size;j++){
				Z id = ppos[(gindex + j)].id;//global id
				lists[m][j].id = j;//local id//
				lists[m][j].score = this->cdata[m*this->n + id];
			}
			__gnu_parallel::sort(lists[m],(lists[m]) + this->parts[i].size,cmp_pta_pair<T,Z>);

			for(uint64_t j = 0; j < this->parts[i].size;j++){
				Z id = lists[m][j].id;
				pos[id].pos = std::min(pos[id].pos,(Z)j);//Find minimum position of appearance in list//
			}
		}
		__gnu_parallel::sort(&pos[0],(&pos[0]) + this->parts[i].size,cmp_pta_pos<Z>);//Sort local ids by minimum position//

		uint64_t b = 0;
		for(uint64_t j = 0; j < this->parts[i].size;j+=PBLOCK_SIZE){
			Z upper = (j + PBLOCK_SIZE <=  this->parts[i].size) ? PBLOCK_SIZE : this->parts[i].size - j + 1;
			this->parts[i].blocks[b].tuple_num = upper;
			this->parts[i].blocks[b].offset = j;
			for(uint64_t l = 0; l < upper; l++){
				Z lid = pos[j+l].id;
				Z gid = ppos[(gindex + lid)].id;
				for(uint32_t m = 0; m < this->d; m++){
					this->parts[i].blocks[b].tuples[ m * PBLOCK_SIZE + l] = this->cdata[ m * this->n + gid ];
					//this->parts[i].blocks[b].tuple_num = 13;
				}
			}

			Z p = pos[upper - 1].pos;
			for(uint32_t m = 0; m < this->d; m++){ this->parts[i].blocks[b].tarray[m] = lists[m][p].score; }
			b++;
		}
		this->parts[i].offset = gindex;
		gindex+=this->parts[i].size;
	}

	free(ppos);
	free(pos);
	for(uint32_t m = 0; m < this->d; m++){ free(lists[m]); }
	free(lists);
}

template<class T, class Z>
void PTA<T,Z>::init(){
	this->t.start();

	this->polar();
	this->create_partitions();

	this->tt_init = this->t.lap();
}

template<class T, class Z>
void PTA<T,Z>::findTopKscalar(uint64_t k, uint8_t qq){

}

template<class T, class Z>
void PTA<T,Z>::findTopKsimd(uint64_t k, uint8_t qq){

}

#endif
