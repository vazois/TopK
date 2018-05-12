#ifndef BTA_H
#define BTA_H

/*
* Vectorized Threshold Aggregation
*/

#include "../cpu/AA.h"

#define VBLOCK_SIZE 1024
#define VSPLITS 2
#define VPARTITIONS (IMP == 2 ? ((uint64_t)pow(VSPLITS,NUM_DIMS-1)) : 1)
//#define VPARTITIONS (THREADS)

template<class T, class Z>
struct vta_pair{
	Z id;
	T score;
};

template<class Z>
struct vta_pos{
	Z id;
	Z pos;
};

template<class T, class Z>
struct vta_block{
	Z offset;
	Z tuple_num;
	T tarray[NUM_DIMS] __attribute__((aligned(32)));
	T tuples[VBLOCK_SIZE * NUM_DIMS] __attribute__((aligned(32)));
};

template<class T, class Z>
struct vta_partition{
	Z offset;
	Z size;
	Z block_num;
	vta_block<T,Z> *blocks;
};


template<class T,class Z>
static bool cmp_vta_pos(const vta_pos<Z> &a, const vta_pos<Z> &b){ return a.pos < b.pos; };

template<class T,class Z>
static bool cmp_vta_pair(const vta_pair<T,Z> &a, const vta_pair<T,Z> &b){ return a.score > b.score; };

template<class T,class Z>
class VTA : public AA<T,Z>{
	public:
		VTA(uint64_t n,uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "VTA";
		}

		~VTA(){

		}
		void init();
		void findTopKscalar(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);
		void findTopKsimd(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);
		void findTopKthreads(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);
		void findTopKsimdMQ(uint64_t k,uint8_t qq, T *weights, uint8_t *attr, uint32_t tid);

	private:
		vta_partition<T,Z> parts[VPARTITIONS];
};

template<class T, class Z>
void VTA<T,Z>::init(){
	normalize_transpose<T,Z>(this->cdata, this->n, this->d);
	this->t.start();
	//Allocate partition & blocks //
	uint64_t part_offset = 0;
	uint64_t part_size = ((this->n - 1) / VPARTITIONS) + 1;
	for(uint64_t i = 0; i < VPARTITIONS; i++){
		uint64_t first = (i*part_size);
		uint64_t last = (i+1)*part_size;
		//std::cout << "<<" << first << "," << last << ">>" << std::endl;
		last = last < this->n ? last : this->n;
		parts[i].offset = part_offset;
		parts[i].size = last - first;
		parts[i].block_num = ((parts[i].size - 1) / VBLOCK_SIZE) + 1;
		parts[i].blocks = static_cast<vta_block<T,Z>*>(aligned_alloc(32,sizeof(vta_block<T,Z>)*parts[i].block_num));

		uint64_t block_offset = 0;
		uint64_t block_size = (parts[i].size - 1)/parts[i].block_num + 1;
		for(uint64_t j = 0; j <parts[i].block_num; j++){
			uint64_t ftuple = j * block_size;
			uint64_t ltuple = (j+1)*block_size;
			ltuple = ltuple < parts[i].size ? ltuple : parts[i].size;
			//std::cout <<"("<< j <<")[" << ftuple << "," << ltuple << "]" << std::endl;
			parts[i].blocks[j].offset = block_offset;
			parts[i].blocks[j].tuple_num = ltuple - ftuple;
			//std::cout << "j< " << j << "=" << ftuple  << "," << ltuple << std::endl;
			block_offset+= ltuple - ftuple;
		}
		part_offset += last - first;
	}
	//////////////////////////////////////////////////////////

	//Initialize Partitions and Blocks//
	uint64_t max_part_size = (((this->n - 1)/VPARTITIONS) + 1);
	vta_pair<T,Z> **lists = (vta_pair<T,Z>**)malloc(sizeof(vta_pair<T,Z>*)*this->d);
	for(uint8_t m = 0; m < this->d; m++){ lists[m] = (vta_pair<T,Z>*)malloc(sizeof(vta_pair<T,Z>)*max_part_size); }
	vta_pos<Z> *order = (vta_pos<Z>*)malloc(sizeof(vta_pos<Z>)*max_part_size);
	uint64_t poffset = 0;
	omp_set_num_threads(THREADS);

	for(uint64_t i = 0; i < VPARTITIONS; i++){
		//Initialize structure to determine relative order inside partition//
		for(uint64_t j = 0; j < parts[i].size; j++){
			order[j].id = j;
			order[j].pos = parts[i].size;//Maximum appearance position//
		}
		//Find order of partition
		for(uint8_t m = 0; m < this->d; m++){
			//Create lists for partition//
			for(uint64_t j = 0; j < parts[i].size; j++){
				lists[m][j].id = j;
				lists[m][j].score = this->cdata[m*this->n + (poffset + j)];
			}
			__gnu_parallel::sort(lists[m],(lists[m]) + parts[i].size,cmp_vta_pair<T,Z>);

			//Find minimum position appearance
			for(uint64_t j = 0; j < parts[i].size; j++){
				Z id = lists[m][j].id;
				order[id].pos = std::min(order[id].pos,(Z)j);//Minimum appearance position
			}
		}
		__gnu_parallel::sort(&order[0],(&order[0]) + parts[i].size,cmp_vta_pos<T,Z>);

		//Split partition into blocks//
		uint64_t bnum = 0;
		for(uint64_t j = 0; j < parts[i].size; ){
			uint64_t jj;
			for(jj = 0; jj < parts[i].blocks[bnum].tuple_num; jj++){//For each block//
				Z id = order[j+jj].id;//Get next tuple in order
				for(uint8_t m = 0; m < this->d; m++){ parts[i].blocks[bnum].tuples[m*VBLOCK_SIZE + jj] = this->cdata[m*this->n + (poffset + id)]; }
			}
			Z pos = order[j+jj-1].pos;
			for(uint8_t m = 0; m < this->d; m++){ parts[i].blocks[bnum].tarray[m] = lists[m][pos].score; }
			j+=parts[i].blocks[bnum].tuple_num;
			bnum++;
		}
		poffset += parts[i].size;
	}

	free(this->cdata); this->cdata = NULL;
	free(order);
	for(uint8_t m = 0; m < this->d; m++){ free(lists[m]); }
	free(lists);
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void VTA<T,Z>::findTopKscalar(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " scalar (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0; i < VPARTITIONS; i++){
		for(uint64_t b = 0; b < parts[i].block_num; b++){
			Z tuple_num = parts[i].blocks[b].tuple_num;
			T *tuples = parts[i].blocks[b].tuples;
			uint64_t id = parts[i].offset + parts[i].blocks[b].offset;
			for(uint64_t t = 0; t < tuple_num; t+=8){
				id+=t;
				T score00 = 0; T score01 = 0; T score02 = 0; T score03 = 0; T score04 = 0; T score05 = 0; T score06 = 0; T score07 = 0;
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
			T *tarray = parts[i].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			if(q.size() >= k && q.top().score >= threshold){ break; }
		}
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
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void VTA<T,Z>::findTopKsimd(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " simd (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	float score[16] __attribute__((aligned(32)));
	__builtin_prefetch(score,1,3);
	__m256 dim_num = _mm256_set_ps(qq,qq,qq,qq,qq,qq,qq,qq);
	this->t.start();
	for(uint64_t i = 0; i < VPARTITIONS; i++){
		for(uint64_t b = 0; b < parts[i].block_num; b++){
			Z tuple_num = parts[i].blocks[b].tuple_num;
			T *tuples = parts[i].blocks[b].tuples;
			uint64_t id = parts[i].offset + parts[i].blocks[b].offset;
			for(uint64_t t = 0; t < tuple_num; t+=16){
				id+=t;
				__m256 score00 = _mm256_setzero_ps();
				__m256 score01 = _mm256_setzero_ps();
				for(uint8_t m = 0; m < qq; m++){
					T weight = weights[attr[m]];
					uint64_t offset = attr[m]*VBLOCK_SIZE + t;
					__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
					__m256 load00 = _mm256_load_ps(&tuples[offset]);
					__m256 load01 = _mm256_load_ps(&tuples[offset+8]);
					load00 = _mm256_mul_ps(load00,_weight);
					load01 = _mm256_mul_ps(load01,_weight);
					score00 = _mm256_add_ps(score00,load00);
					score01 = _mm256_add_ps(score01,load01);

					#if LD == 2
						score00 = _mm256_div_ps(score00,dim_num);
						score01 = _mm256_div_ps(score01,dim_num);
					#endif
				}
				_mm256_store_ps(&score[0],score00);
				_mm256_store_ps(&score[8],score01);
				if(q.size() < k){//insert if empty space in queue
					q.push(tuple_<T,Z>(id,score[0]));
					q.push(tuple_<T,Z>(id+1,score[1]));
					q.push(tuple_<T,Z>(id+2,score[2]));
					q.push(tuple_<T,Z>(id+3,score[3]));
					q.push(tuple_<T,Z>(id+4,score[4]));
					q.push(tuple_<T,Z>(id+5,score[5]));
					q.push(tuple_<T,Z>(id+6,score[6]));
					q.push(tuple_<T,Z>(id+7,score[7]));
					q.push(tuple_<T,Z>(id+8,score[8]));
					q.push(tuple_<T,Z>(id+9,score[9]));
					q.push(tuple_<T,Z>(id+10,score[10]));
					q.push(tuple_<T,Z>(id+11,score[11]));
					q.push(tuple_<T,Z>(id+12,score[12]));
					q.push(tuple_<T,Z>(id+13,score[13]));
					q.push(tuple_<T,Z>(id+14,score[14]));
					q.push(tuple_<T,Z>(id+15,score[15]));
				}else{//delete smallest element if current score is bigger
					if(q.top().score < score[0]){ q.pop(); q.push(tuple_<T,Z>(id,score[0])); }
					if(q.top().score < score[1]){ q.pop(); q.push(tuple_<T,Z>(id+1,score[1])); }
					if(q.top().score < score[2]){ q.pop(); q.push(tuple_<T,Z>(id+2,score[2])); }
					if(q.top().score < score[3]){ q.pop(); q.push(tuple_<T,Z>(id+3,score[3])); }
					if(q.top().score < score[4]){ q.pop(); q.push(tuple_<T,Z>(id+4,score[4])); }
					if(q.top().score < score[5]){ q.pop(); q.push(tuple_<T,Z>(id+5,score[5])); }
					if(q.top().score < score[6]){ q.pop(); q.push(tuple_<T,Z>(id+6,score[6])); }
					if(q.top().score < score[7]){ q.pop(); q.push(tuple_<T,Z>(id+7,score[7])); }
					if(q.top().score < score[8]){ q.pop(); q.push(tuple_<T,Z>(id+8,score[8])); }
					if(q.top().score < score[9]){ q.pop(); q.push(tuple_<T,Z>(id+9,score[9])); }
					if(q.top().score < score[10]){ q.pop(); q.push(tuple_<T,Z>(id+10,score[10])); }
					if(q.top().score < score[11]){ q.pop(); q.push(tuple_<T,Z>(id+11,score[11])); }
					if(q.top().score < score[12]){ q.pop(); q.push(tuple_<T,Z>(id+12,score[12])); }
					if(q.top().score < score[13]){ q.pop(); q.push(tuple_<T,Z>(id+13,score[13])); }
					if(q.top().score < score[14]){ q.pop(); q.push(tuple_<T,Z>(id+14,score[14])); }
					if(q.top().score < score[15]){ q.pop(); q.push(tuple_<T,Z>(id+15,score[15])); }
				}
				if(STATS_EFF) this->tuple_count+=16;
			}

			T threshold = 0;
			T *tarray = parts[i].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			if(q.size() >= k && q.top().score >= threshold) break;
		}
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
void VTA<T,Z>::findTopKsimdMQ(uint64_t k, uint8_t qq, T *weights, uint8_t *attr, uint32_t tid){
	Time<msecs> t;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	float score[16] __attribute__((aligned(32)));
	__builtin_prefetch(score,1,3);
	t.start();
	__m256 dim_num = _mm256_set_ps(qq,qq,qq,qq,qq,qq,qq,qq);
	for(uint64_t i = 0; i < VPARTITIONS; i++){
		for(uint64_t b = 0; b < parts[i].block_num; b++){
			Z tuple_num = parts[i].blocks[b].tuple_num;
			T *tuples = parts[i].blocks[b].tuples;
			uint64_t id = parts[i].offset + parts[i].blocks[b].offset;
			for(uint64_t t = 0; t < tuple_num; t+=16){
				id+=t;
				__m256 score00 = _mm256_setzero_ps();
				__m256 score01 = _mm256_setzero_ps();
				for(uint8_t m = 0; m < qq; m++){
					T weight = weights[attr[m]];
					uint64_t offset = attr[m]*VBLOCK_SIZE + t;
					__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
					__m256 load00 = _mm256_load_ps(&tuples[offset]);
					__m256 load01 = _mm256_load_ps(&tuples[offset+8]);
					load00 = _mm256_mul_ps(load00,_weight);
					load01 = _mm256_mul_ps(load01,_weight);
					score00 = _mm256_add_ps(score00,load00);
					score01 = _mm256_add_ps(score01,load01);

					#if LD == 2
						score00 = _mm256_div_ps(score00,dim_num);
						score01 = _mm256_div_ps(score01,dim_num);
					#endif
				}
				_mm256_store_ps(&score[0],score00);
				_mm256_store_ps(&score[8],score01);
				if(q.size() < k){//insert if empty space in queue
					q.push(tuple_<T,Z>(id,score[0]));
					q.push(tuple_<T,Z>(id+1,score[1]));
					q.push(tuple_<T,Z>(id+2,score[2]));
					q.push(tuple_<T,Z>(id+3,score[3]));
					q.push(tuple_<T,Z>(id+4,score[4]));
					q.push(tuple_<T,Z>(id+5,score[5]));
					q.push(tuple_<T,Z>(id+6,score[6]));
					q.push(tuple_<T,Z>(id+7,score[7]));
					q.push(tuple_<T,Z>(id+8,score[8]));
					q.push(tuple_<T,Z>(id+9,score[9]));
					q.push(tuple_<T,Z>(id+10,score[10]));
					q.push(tuple_<T,Z>(id+11,score[11]));
					q.push(tuple_<T,Z>(id+12,score[12]));
					q.push(tuple_<T,Z>(id+13,score[13]));
					q.push(tuple_<T,Z>(id+14,score[14]));
					q.push(tuple_<T,Z>(id+15,score[15]));
				}else{//delete smallest element if current score is bigger
					if(q.top().score < score[0]){ q.pop(); q.push(tuple_<T,Z>(id,score[0])); }
					if(q.top().score < score[1]){ q.pop(); q.push(tuple_<T,Z>(id+1,score[1])); }
					if(q.top().score < score[2]){ q.pop(); q.push(tuple_<T,Z>(id+2,score[2])); }
					if(q.top().score < score[3]){ q.pop(); q.push(tuple_<T,Z>(id+3,score[3])); }
					if(q.top().score < score[4]){ q.pop(); q.push(tuple_<T,Z>(id+4,score[4])); }
					if(q.top().score < score[5]){ q.pop(); q.push(tuple_<T,Z>(id+5,score[5])); }
					if(q.top().score < score[6]){ q.pop(); q.push(tuple_<T,Z>(id+6,score[6])); }
					if(q.top().score < score[7]){ q.pop(); q.push(tuple_<T,Z>(id+7,score[7])); }
					if(q.top().score < score[8]){ q.pop(); q.push(tuple_<T,Z>(id+8,score[8])); }
					if(q.top().score < score[9]){ q.pop(); q.push(tuple_<T,Z>(id+9,score[9])); }
					if(q.top().score < score[10]){ q.pop(); q.push(tuple_<T,Z>(id+10,score[10])); }
					if(q.top().score < score[11]){ q.pop(); q.push(tuple_<T,Z>(id+11,score[11])); }
					if(q.top().score < score[12]){ q.pop(); q.push(tuple_<T,Z>(id+12,score[12])); }
					if(q.top().score < score[13]){ q.pop(); q.push(tuple_<T,Z>(id+13,score[13])); }
					if(q.top().score < score[14]){ q.pop(); q.push(tuple_<T,Z>(id+14,score[14])); }
					if(q.top().score < score[15]){ q.pop(); q.push(tuple_<T,Z>(id+15,score[15])); }
				}
				if(STATS_EFF) this->tuple_count+=16;
			}

			T threshold = 0;
			T *tarray = parts[i].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			if(q.size() >= k && q.top().score >= threshold) break;
		}
	}
	this->tt_array[tid] += t.lap();
}

template<class T, class Z>
void VTA<T,Z>::findTopKthreads(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	uint32_t threads = THREADS < VPARTITIONS ? THREADS : VPARTITIONS;
	Z tt_count[threads];
	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q[threads];
	omp_set_num_threads(threads);

	std::cout << this->algo << " find top-" << k << " threads x "<< threads <<" (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	this->t.start();
#pragma omp parallel
{
	float score[16] __attribute__((aligned(32)));
	__builtin_prefetch(score,1,3);
	uint32_t tid = omp_get_thread_num();
	Z tuple_count = 0;
	__m256 dim_num = _mm256_set_ps(qq,qq,qq,qq,qq,qq,qq,qq);
	for(uint64_t i = tid; i < VPARTITIONS; i+=threads){
		for(uint64_t b = 0; b < parts[i].block_num; b++){
			Z tuple_num = parts[i].blocks[b].tuple_num;
			T *tuples = parts[i].blocks[b].tuples;
			uint64_t id = parts[i].offset + parts[i].blocks[b].offset;
			for(uint64_t t = 0; t < tuple_num; t+=16){
				id+=t;
				__m256 score00 = _mm256_setzero_ps();
				__m256 score01 = _mm256_setzero_ps();
				for(uint8_t m = 0; m < qq; m++){
					T weight = weights[attr[m]];
					uint64_t offset = attr[m]*VBLOCK_SIZE + t;
					__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
					__m256 load00 = _mm256_load_ps(&tuples[offset]);
					__m256 load01 = _mm256_load_ps(&tuples[offset+8]);
					load00 = _mm256_mul_ps(load00,_weight);
					load01 = _mm256_mul_ps(load01,_weight);
					score00 = _mm256_add_ps(score00,load00);
					score01 = _mm256_add_ps(score01,load01);
					#if LD == 2
						score00 = _mm256_div_ps(score00,dim_num);
						score01 = _mm256_div_ps(score01,dim_num);
					#endif
				}
				_mm256_store_ps(&score[0],score00);
				_mm256_store_ps(&score[8],score01);

				for(uint8_t l = 0; l < 16; l++){
					if(q[tid].size() < k){
						q[tid].push(tuple_<T,Z>(id,score[l]));
					}else if(q[tid].top().score < score[l]){
						q[tid].pop(); q[tid].push(tuple_<T,Z>(id,score[l]));
					}
				}
				if(STATS_EFF) tuple_count+=16;
			}

			T threshold = 0;
			T *tarray = parts[i].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			if(q[tid].size() >= k && q[tid].top().score >= threshold) break;
		}
	}
	if(STATS_EFF) tt_count[tid] = tuple_count;
}

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> _q;
	for(uint32_t m = 0; m < threads; m++){
		while(!q[m].empty()){
			if(_q.size() < k){
				_q.push(q[m].top());
			}else if(_q.top().score < q[m].top().score){
				_q.pop();
				_q.push(q[m].top());
			}
			q[m].pop();
		}
	}
	this->tt_processing += this->t.lap();

	if(STATS_EFF){ for(uint32_t i = 0; i < threads; i++) this->tuple_count +=tt_count[i]; }
	T threshold = _q.top().score;
	while(!_q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(_q.top());
		_q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
