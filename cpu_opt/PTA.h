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
static bool cmp_pta_pair_asc(const pta_pair<T,Z> &a, const pta_pair<T,Z> &b){ return a.score < b.score; };

template<class T>
inline void sortnet_8(T *&data){
	T a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
	T a4 = data[4], a5 = data[5], a6 = data[6], a7 = data[7];
	T t0;

	t0 = std::max(a0,a1); a1 = std::min(a0,a1); a0 = t0;
	t0 = std::max(a2,a3); a3 = std::min(a2,a3); a2 = t0;
	t0 = std::max(a4,a5); a5 = std::min(a4,a5); a4 = t0;
	t0 = std::max(a6,a7); a7 = std::min(a6,a7); a6 = t0;

	t0 = std::max(a0,a2); a2 = std::min(a0,a2); a0 = t0;
	t0 = std::max(a1,a3); a3 = std::min(a1,a3); a1 = t0;
	t0 = std::max(a4,a6); a6 = std::min(a4,a6); a4 = t0;
	t0 = std::max(a5,a7); a7 = std::min(a5,a7); a5 = t0;

	t0 = std::max(a1,a2); a2 = std::min(a1,a2); a1 = t0;
	t0 = std::max(a5,a6); a6 = std::min(a5,a6); a5 = t0;

	t0 = std::max(a0,a4); a4 = std::min(a0,a4); a0 = t0;
	t0 = std::max(a1,a5); a5 = std::min(a1,a5); a1 = t0;
	t0 = std::max(a2,a6); a6 = std::min(a2,a6); a2 = t0;
	t0 = std::max(a3,a7); a7 = std::min(a3,a7); a3 = t0;

	t0 = std::max(a2,a4); a4 = std::min(a2,a4); a2 = t0;
	t0 = std::max(a3,a5); a5 = std::min(a3,a5); a3 = t0;

	t0 = std::max(a1,a2); a2 = std::min(a1,a2); a1 = t0;
	t0 = std::max(a3,a4); a4 = std::min(a3,a4); a3 = t0;
	t0 = std::max(a5,a6); a6 = std::min(a5,a6); a5 = t0;

	data[0] = a0; data[1] = a1; data[2] = a2; data[3] = a3;
	data[4] = a4; data[5] = a5; data[6] = a6; data[7] = a7;
}

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
		void findTopKscalar(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);
		void findTopKsimd(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);
		void findTopKthreads(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);
		void findTopKthreads2(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);

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
	__m256 one = _mm256_set1_ps(1.1);
	for(uint64_t i = 0; i < this->n; i+=8){//Calculate hyperspherical coordinates for each point
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

	uint64_t mod = (this->n / PSLITS);
	uint64_t mul = 1;
	for(uint64_t i = 0; i < this->n; i++) this->part_id[i] = 0;
	for(uint32_t m = 0; m < this->d-1; m++){//For each hyperspherical coordinate
		for(uint64_t i = 0; i < this->n; i++){ pp[i].id = i; pp[i].score = pdata[m*this->n + i]; }
		__gnu_parallel::sort(&pp[0],(&pp[0]) + this->n,cmp_pta_pair_asc<T,Z>);//determine splitting points
		for(uint64_t i = 0; i < this->n; i++){
			this->part_id[pp[i].id]+=(mul*(i / mod));
		}// Assign partition id
		mul*=PSLITS;
	}

//	for(uint64_t i = 0; i < 16; i++){
//		std::cout << std::fixed;
//		std::cout << "[";
//		for(uint32_t m = 0; m < this->d; m++){ std::cout << std::setprecision(4) << this->cdata[m*this->n + i] << " , "; }
//		std::cout << "] ";
//		std::cout << "[";
//		for(uint32_t m = 0; m < this->d-1; m++){ std::cout << std::setprecision(4) << pdata[m*this->n + i] << " , "; }
//		std::cout << "] ";
//		std::cout << this->part_id[i];
//		std::cout << std::endl;
//	}

	//Count and verify number of points inside each partition//
	std::map<Z,Z> mm;
	for(uint64_t i = 0; i < PPARTITIONS;i++) mm.insert(std::pair<Z,Z>(i,0));
	for(uint64_t i = 0; i < this->n; i++){
		Z pid = this->part_id[i];
		mm[pid]+=1;
	}

	//std::cout << "mm_size: " << mm.size() << " --> " << PPARTITIONS<< std::endl;
	this->max_part_size = 0;
	uint64_t count_full_parts = 0;
	for(typename std::map<Z,Z>::iterator it = mm.begin(); it != mm.end(); ++it){//Initialize partitions//
		uint64_t psize = it->second + (PBLOCK_SIZE - (it->second % PBLOCK_SIZE));
//		std::cout << "g(" << it->first << "):" << std::setfill('0') << std::setw(8) << it->second << " < "
//				<< psize << " [ " << ((float)psize)/PBLOCK_SIZE << " , " << ((float)psize)/SIMD_GROUP << " ] " << std::endl;

		this->parts[it->first].size = it->second;//True partition size
		this->parts[it->first].block_num = ((float)psize)/PBLOCK_SIZE;//Maximum number of blocks//
		this->parts[it->first].blocks = static_cast<pta_block<T,Z>*>(aligned_alloc(32,sizeof(pta_block<T,Z>)*this->parts[it->first].block_num));
		this->max_part_size = std::max(this->max_part_size,it->second);
		if(it->second > 0 ) count_full_parts++;
	}
	//std::cout << "max_part_size: " << this->max_part_size << std::endl;
	std::cout << "count_full_parts: " << count_full_parts << std::endl;
	free(pp);
	free(pdata);
}

template<class T, class Z>
void PTA<T,Z>::create_partitions(){
	pta_pos<Z> *ppos = (pta_pos<Z>*)malloc(sizeof(pta_pos<Z>)*this->n);//tupple id, tied to partition id
	for(uint64_t i = 0; i < this->n; i++){
		ppos[i].id =i; ppos[i].pos = this->part_id[i];
	}
	__gnu_parallel::sort(&ppos[0],(&ppos[0]) + this->n,cmp_pta_pos<Z>);//Sort tuples based on their assigned partitions

//	for(uint64_t i = 0; i < 16; i++){
//		std::cout << std::setfill('0') << std::setw(2) << ppos[i].pos << " < ";
//		std::cout << std::setfill('0') << std::setw(8) << ppos[i].id ;
//		std::cout << std::endl;
//	}

	//return ;
	uint64_t gindex = 0;
	pta_pos<Z> *pos = (pta_pos<Z>*)malloc(sizeof(pta_pos<Z>)*this->max_part_size);
	pta_pair<T,Z> **lists = (pta_pair<T,Z>**)malloc(sizeof(pta_pair<T,Z>*)*this->d);
	for(uint32_t m=0; m<this->d;m++) lists[m] = (pta_pair<T,Z>*)malloc(sizeof(pta_pair<T,Z>)*this->max_part_size);
	for(uint32_t m = 0; m < this->d; m++){ for(uint64_t j = 0; j < this->max_part_size;j++){ lists[m][j].id = 0; lists[m][j].score = 0; } }

	for(uint64_t i = 0; i < PPARTITIONS;i++){//Build Partitions
		//INITIALIZE//
		for(uint64_t j = 0; j < this->max_part_size;j++){ pos[j].id = j; pos[j].pos = this->parts[i].size; }//Initialize to max possible position
		//for(uint32_t m = 0; m < this->d; m++){ for(uint64_t j = 0; j < this->max_part_size;j++){ lists[m][j].id = 0; lists[m][j].score = 0; } }

		for(uint32_t m = 0; m < this->d; m++){//Initialize lists for given partition//
			for(uint64_t j = 0; j < this->parts[i].size;j++){
				Z gid = ppos[(gindex + j)].id;//global tuple id
				lists[m][j].id = j;//local tuple id//
				lists[m][j].score = this->cdata[m*this->n + gid];
			}
			__gnu_parallel::sort(lists[m],(lists[m]) + this->parts[i].size,cmp_pta_pair<T,Z>);//Sort to create lists

			for(uint64_t j = 0; j < this->parts[i].size;j++){
				Z lid = lists[m][j].id;
				//if( lid >= this->parts[i].size ) std::cout << "ERROR(1) " << lid << "," << this->parts[i].size << std::endl;
				//if( lid == 914 ) std::cout << "914: "<< j << std::endl;
				pos[lid].pos = std::min(pos[lid].pos,(Z)j);//Find minimum local position of appearance in list//
			}
		}
		__gnu_parallel::sort(&pos[0],(&pos[0]) + this->parts[i].size,cmp_pta_pos<Z>);//Sort local tuple ids by minimum position//

		uint64_t b = 0;
		for(uint64_t j = 0; j < this->parts[i].size;j+=PBLOCK_SIZE){//For each block
			Z upper = ((j + PBLOCK_SIZE) <  this->parts[i].size) ? PBLOCK_SIZE : this->parts[i].size - j;
			this->parts[i].blocks[b].tuple_num = upper;
			this->parts[i].blocks[b].offset = j;
			//std::cout << i << " , " << j << " , " << PBLOCK_SIZE << " , " << this->parts[i].size <<" upper: " << upper << std::endl;
			for(uint64_t l = 0; l < upper; l++){
				Z lid = pos[j+l].id;//Find local tuple id
				//if( (gindex + lid) >= this->n ) std::cout << "ERROR(2) " << (gindex + lid) <<" ,gindex " << gindex << " ,lid "<< lid << " ,j+l " << j+l << " , " << this->n << std::endl;
				Z gid = ppos[(gindex + lid)].id;//Find global tuple id
				for(uint32_t m = 0; m < this->d; m++){//Assign attributes to partitions
					this->parts[i].blocks[b].tuples[ m * PBLOCK_SIZE + l] = this->cdata[ m * this->n + gid ];
				}
			}

			Z p = pos[j + upper - 1].pos;//Find last point position
			for(uint32_t m = 0; m < this->d; m++){ this->parts[i].blocks[b].tarray[m] = lists[m][p].score; }//Extract threshold
			b++;
		}
		this->parts[i].offset = gindex;//Update global index
		gindex+=this->parts[i].size;
	}

	free(ppos);
	free(pos);
	for(uint32_t m = 0; m < this->d; m++){ free(lists[m]); }
	free(lists);
	free(this->cdata);
	this->cdata = NULL;
}

template<class T, class Z>
void PTA<T,Z>::init(){
	this->t.start();

	std::cout << "computing polar coordinates ..." << std::endl;
	this->polar();
	std::cout << "creating partitions ..." << std::endl;
	this->create_partitions();

	this->tt_init = this->t.lap();
}

template<class T, class Z>
void PTA<T,Z>::findTopKscalar(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " scalar (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
	this->t.start();
	for(uint64_t p = 0; p < PPARTITIONS; p++){
		if(this->parts[p].size == 0) continue;
		Z poffset = this->parts[p].offset;
		//uint32_t count = 0;
		for(uint64_t b = 0; b < this->parts[p].block_num; b++){
			Z id = this->parts[p].offset + poffset;
			T *tuples = this->parts[p].blocks[b].tuples;
			//std::cout << b <<"< tuple_num: " << this->parts[p].blocks[b].tuple_num << std::endl;
			for(uint64_t t = 0; t < this->parts[p].blocks[b].tuple_num; t+=8){
				T score00 = 0; T score01 = 0;
				T score02 = 0; T score03 = 0;
				T score04 = 0; T score05 = 0;
				T score06 = 0; T score07 = 0;
				for(uint8_t m = 0; m < qq; m++){
					T weight = weights[attr[m]];
					uint32_t offset = attr[m]*VBLOCK_SIZE + t;
					score00+=tuples[offset]*weight;
					score01+=tuples[offset+1]*weight;
					score02+=tuples[offset+2]*weight;
					score03+=tuples[offset+3]*weight;
					score04+=tuples[offset+4]*weight;
					score05+=tuples[offset+5]*weight;
					score06+=tuples[offset+6]*weight;
					score07+=tuples[offset+7]*weight;
				}
				if(q.size() < k){
					q.push(tuple_<T,Z>(id,score00));
					q.push(tuple_<T,Z>(id+1,score01));
					q.push(tuple_<T,Z>(id+2,score02));
					q.push(tuple_<T,Z>(id+3,score03));
					q.push(tuple_<T,Z>(id+4,score04));
					q.push(tuple_<T,Z>(id+5,score05));
					q.push(tuple_<T,Z>(id+6,score06));
					q.push(tuple_<T,Z>(id+7,score07));
				}else{
					if(q.top().score < score00){ q.pop(); q.push(tuple_<T,Z>(id,score00)); }
					if(q.top().score < score01){ q.pop(); q.push(tuple_<T,Z>(id+1,score01)); }
					if(q.top().score < score02){ q.pop(); q.push(tuple_<T,Z>(id+2,score02)); }
					if(q.top().score < score03){ q.pop(); q.push(tuple_<T,Z>(id+3,score03)); }
					if(q.top().score < score04){ q.pop(); q.push(tuple_<T,Z>(id+4,score04)); }
					if(q.top().score < score05){ q.pop(); q.push(tuple_<T,Z>(id+5,score05)); }
					if(q.top().score < score06){ q.pop(); q.push(tuple_<T,Z>(id+6,score06)); }
					if(q.top().score < score07){ q.pop(); q.push(tuple_<T,Z>(id+7,score07)); }
				}
				if(STATS_EFF) this->tuple_count+=8;
			}

			T threshold = 0;
			T *tarray = this->parts[p].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			if(q.size() >= k && q.top().score >= threshold){ break; }
		}
		//std::cout << "p: " <<p << " = " << count << std::endl;
	}
	this->tt_processing += this->t.lap();

	while(q.size() > k){ q.pop(); }
	T threshold = q.top().score;
	while(!q.empty()){
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void PTA<T,Z>::findTopKsimd(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " simd (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
	float score[16] __attribute__((aligned(32)));
	this->t.start();
	for(uint64_t p = 0; p < PPARTITIONS; p++){
		if(this->parts[p].size == 0) continue;
		Z poffset = this->parts[p].offset;
		//uint32_t count = 0;
		for(uint64_t b = 0; b < this->parts[p].block_num; b++){
			__builtin_prefetch(score,1,3);
			Z id = this->parts[p].offset + poffset;
			T *tuples = this->parts[p].blocks[b].tuples;
			//std::cout << b <<"< tuple_num: " << this->parts[p].blocks[b].tuple_num << std::endl;
			for(uint64_t t = 0; t < this->parts[p].blocks[b].tuple_num; t+=16){
				__m256 score00 = _mm256_setzero_ps();
				__m256 score01 = _mm256_setzero_ps();
				for(uint8_t m = 0; m < qq; m++){
					T weight = weights[attr[m]];
					uint32_t offset = attr[m]*VBLOCK_SIZE + t;
					__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
					__m256 load00 = _mm256_load_ps(&tuples[offset]);
					__m256 load01 = _mm256_load_ps(&tuples[offset+8]);
					load00 = _mm256_mul_ps(load00,_weight);
					load01 = _mm256_mul_ps(load01,_weight);
					score00 = _mm256_add_ps(score00,load00);
					score01 = _mm256_add_ps(score01,load01);
				}
				_mm256_store_ps(&score[0],score00);
				_mm256_store_ps(&score[8],score01);
				for(uint8_t l = 0; l < 16; l++){
					if(q.size() < k){
						q.push(tuple_<T,Z>(id,score[l]));
					}else if(q.top().score < score[l]){
						q.pop(); q.push(tuple_<T,Z>(id,score[l]));
					}
				}
				if(STATS_EFF) this->tuple_count+=16;
			}

			T threshold = 0;
			T *tarray = this->parts[p].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			if(q.size() >= k && q.top().score >= threshold){ break; }
		}
		//std::cout << "p: " <<p << " = " << count << std::endl;
	}
	this->tt_processing += this->t.lap();

	while(q.size() > k){ q.pop(); }
	T threshold = q.top().score;
	while(!q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void PTA<T,Z>::findTopKthreads(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	uint32_t threads = THREADS < PPARTITIONS ? THREADS : PPARTITIONS;
	Z tt_count[threads];
	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> _q[threads];
	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
	omp_set_num_threads(threads);

	std::cout << this->algo << " find top-" << k << " threads x "<< threads <<" (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	this->t.start();
#pragma omp parallel
{
	float score[16] __attribute__((aligned(32)));
	uint32_t tid = omp_get_thread_num();
	Z tuple_count = 0;
	__builtin_prefetch(score,1,3);
	for(uint64_t p = tid; p < PPARTITIONS; p+=threads){
		if(this->parts[p].size == 0) continue;
		Z poffset = this->parts[p].offset;
		for(uint64_t b = 0; b < this->parts[p].block_num; b++){
			Z id = this->parts[p].offset + poffset;
			T *tuples = this->parts[p].blocks[b].tuples;

			for(uint64_t t = 0; t < this->parts[p].blocks[b].tuple_num; t+=16){
				__m256 score00 = _mm256_setzero_ps();
				__m256 score01 = _mm256_setzero_ps();
				for(uint8_t m = 0; m < qq; m++){
					T weight = weights[attr[m]];
					uint32_t offset = attr[m]*VBLOCK_SIZE + t;
					__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
					__m256 load00 = _mm256_load_ps(&tuples[offset]);
					__m256 load01 = _mm256_load_ps(&tuples[offset+8]);
					load00 = _mm256_mul_ps(load00,_weight);
					load01 = _mm256_mul_ps(load01,_weight);
					score00 = _mm256_add_ps(score00,load00);
					score01 = _mm256_add_ps(score01,load01);
				}
				_mm256_store_ps(&score[0],score00);
				_mm256_store_ps(&score[8],score01);

				for(uint8_t l = 0; l < 16; l++){
					if(_q[tid].size() < k){
						_q[tid].push(tuple_<T,Z>(id,score[l]));
					}else if(_q[tid].top().score < score[l]){
						_q[tid].pop(); _q[tid].push(tuple_<T,Z>(id,score[l]));
					}
				}
				if(STATS_EFF) tuple_count+=16;
			}

			T threshold = 0;
			T *tarray = this->parts[p].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			if(_q[tid].size() >= k && _q[tid].top().score >= threshold){ break; }
		}
		//std::cout << "p: " <<p << " = " << count << std::endl;
	}
	if(STATS_EFF) tt_count[tid] = tuple_count;
}
	for(uint32_t m = 0; m < threads; m++){
		while(!_q[m].empty()){
			if(q.size() < k){
				q.push(_q[m].top());
			}else if(q.top().score < _q[m].top().score){
				q.pop();
				q.push(_q[m].top());
			}
			_q[m].pop();
		}
	}
	this->tt_processing += this->t.lap();

	if(STATS_EFF){ for(uint32_t i = 0; i < threads; i++) this->tuple_count +=tt_count[i]; }
	T threshold = q.top().score;
	while(!q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void PTA<T,Z>::findTopKthreads2(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	uint32_t threads = THREADS < PPARTITIONS ? THREADS : PPARTITIONS;
	Z tt_count[threads];
	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> _q[threads];
	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
//	boost::heap::skew_heap<tuple_<T,Z>,boost::heap::compare<MaxCMP<T,Z>>> _q[threads];
//	boost::heap::skew_heap<tuple_<T,Z>,boost::heap::compare<MaxCMP<T,Z>>> q;
	omp_set_num_threads(threads);

	uint8_t first = FIRST(this->d,qq);
	uint8_t last = LAST(this->d,qq);
	std::cout << this->algo << " find top-" << k << " threads x "<< threads <<" (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	this->t.start();
#pragma omp parallel
{
	float score[16] __attribute__((aligned(32)));
	uint32_t tid = omp_get_thread_num();
	Z tuple_count = 0;
	__builtin_prefetch(score,1,3);
	for(uint64_t p = tid; p < PPARTITIONS; p+=threads){
		if(this->parts[p].size == 0) continue;
		Z poffset = this->parts[p].offset;
		for(uint64_t b = 0; b < this->parts[p].block_num; b++){
			Z id = this->parts[p].offset + poffset;
			T *tuples = this->parts[p].blocks[b].tuples;

			for(uint64_t t = 0; t < this->parts[p].blocks[b].tuple_num; t+=16){
				__m256 score00 = _mm256_setzero_ps();
				__m256 score01 = _mm256_setzero_ps();
				for(uint8_t m = 0; m < qq; m++){
					T weight = weights[attr[m]];
					uint32_t offset = attr[m]*VBLOCK_SIZE + t;
					__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
					__m256 load00 = _mm256_load_ps(&tuples[offset]);
					__m256 load01 = _mm256_load_ps(&tuples[offset+8]);
					load00 = _mm256_mul_ps(load00,_weight);
					load01 = _mm256_mul_ps(load01,_weight);
					score00 = _mm256_add_ps(score00,load00);
					score01 = _mm256_add_ps(score01,load01);
				}
				_mm256_store_ps(&score[0],score00);
				_mm256_store_ps(&score[8],score01);

				for(uint8_t l = 0; l < 16; l++){
					if(_q[tid].size() < k){
						_q[tid].push(tuple_<T,Z>(id,score[l]));
					}else if(_q[tid].top().score < score[l]){
						_q[tid].pop(); _q[tid].push(tuple_<T,Z>(id,score[l]));
					}
				}
				if(STATS_EFF) tuple_count+=16;
			}

			T threshold = 0;
			T *tarray = this->parts[p].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			if(_q[tid].size() >= k && _q[tid].top().score >= threshold){ break; }
		}
		//std::cout << "p: " <<p << " = " << count << std::endl;
	}
	if(STATS_EFF) tt_count[tid] = tuple_count;
}
	for(uint32_t m = 0; m < threads; m++){
		while(!_q[m].empty()){
			if(q.size() < k){
				q.push(_q[m].top());
			}else if(q.top().score < _q[m].top().score){
				q.pop();
				q.push(_q[m].top());
			}
			_q[m].pop();
		}
	}
	this->tt_processing += this->t.lap();

	if(STATS_EFF){ for(uint32_t i = 0; i < threads; i++) this->tuple_count +=tt_count[i]; }
	T threshold = q.top().score;
	while(!q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
