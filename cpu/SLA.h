#ifndef SLA_H
#define SLA_H

/*
 * Skyline Layered Aggregation
 */

#include <unordered_map>
#include "../skyline/hybrid/hybrid.h"
#include "AA.h"

#define SLA_THREADS 16
#define SLA_ALPHA 1024
#define SLA_QSIZE 8

#define SBLOCK_SIZE 1024

template<class T, class Z>
struct sla_pair{
	Z id;
	T score;
};

template<class Z>
struct sla_pos{
	Z id;
	Z pos;
};

template<class T, class Z>
struct sla_block{
	Z offset;
	Z tuple_num;
	T tarray[NUM_DIMS] __attribute__((aligned(32)));
	T tuples[SBLOCK_SIZE * NUM_DIMS] __attribute__((aligned(32)));
};

template<class T, class Z>
struct sla_partition{
	Z offset;
	Z size;
	Z block_num;
	sla_block<T,Z> *blocks;
};

template<class T,class Z>
static bool cmp_sla_pair(const sla_pair<T,Z> &a, const sla_pair<T,Z> &b){ return a.score > b.score; };

template<class Z>
static bool cmp_sla_pos(const sla_pos<Z> &a, const sla_pos<Z> &b){ return a.pos < b.pos; };

template<class T, class Z>
class SLA : public AA<T,Z>{
	public:
		SLA(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "SLA";
			this->parts = NULL;
		};

		~SLA(){
			if(this->parts!=NULL){
				for(uint64_t i = 0; i < this->layer_num; i++) free(this->parts[i].blocks);
				free(this->parts);
			}
		};

		void init();
		void findTopKscalar(uint64_t k, uint8_t qq, T *weights, uint8_t *attr);
		void findTopKsimd(uint64_t k, uint8_t qq, T *weights, uint8_t *attr);
		void findTopKthreads(uint64_t k, uint8_t qq, T *weights, uint8_t *attr);
		void findTopKthreads2(uint64_t k, uint8_t qq, T *weights, uint8_t *attr);

	private:
		std::vector<std::vector<Z>> layers;
		sla_partition<T,Z> *parts;
		uint64_t layer_num;
		uint64_t max_part_size;

		T** sky_data(T **cdata);
		void build_layers(T **cdata);
		uint64_t partition_table(uint64_t first, uint64_t last, std::unordered_set<uint64_t> layer_set, T **cdata, Z *offset);

		void create_lists();
};

template<class T, class Z>
T** SLA<T,Z>::sky_data(T **cdata){
	if(cdata == NULL){
		cdata = static_cast<T**>(aligned_alloc(32, sizeof(T*) * (this->n)));
		for(uint64_t i = 0; i < this->n; i++) cdata[i] = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->d)));
	}
	for(uint64_t i = 0; i < this->n; i++){
		for(uint8_t m = 0; m < this->d; m++){
			//cdata[i][m] = this->cdata[m*this->n + i];
			cdata[i][m] = (1.0f - this->cdata[m*this->n + i]);//Calculate maximum skyline // Read column-major to row-major
		}
	}
	return cdata;
}

template<class T, class Z>
void SLA<T,Z>::build_layers(T **cdata){
	Z *offset = (Z*)malloc(sizeof(Z)*this->n);
	for(uint64_t i = 0; i < this->n; i++) offset[i]=i;
	uint64_t first = 0;
	uint64_t last = this->n;

	while(last > 100)
	{
		//Compute skyline only on first to last tuples//
		SkylineI* skyline = new Hybrid( SLA_THREADS, (uint64_t)(last), (uint64_t)(this->d), SLA_ALPHA, SLA_QSIZE );
		skyline->Init( cdata );
		std::vector<uint64_t> layer = skyline->Execute();//Find ids of tuples that belong to the skyline
		delete skyline;
//		std::cout << std::dec << std::setfill('0') << std::setw(10);
//		std::cout << last << " - ";
//		std::cout << std::dec << std::setfill('0') << std::setw(10);
//		std::cout << layer.size() << " = ";

		std::unordered_set<uint64_t> layer_set;
		for(uint64_t i = 0; i <layer.size(); i++){
			layer_set.insert(layer[i]);
			layer[i] = offset[layer[i]];//Real tuple id
		}

		//Split table to non-skyline points (beginning) and skyline points//
		last = this->partition_table(first, last, layer_set, cdata, offset);
//		std::cout << std::dec << std::setfill('0') << std::setw(10);
//		std::cout << last << std::endl;
		this->layers.push_back(layer);
	}

	if( last > 0 ){
		std::vector<uint64_t> layer;
		for(uint64_t i = 0; i < last; i++) layer.push_back(offset[i]);
		this->layers.push_back(layer);
	}
	this->layer_num = this->layers.size();
	std::cout << "Layer count: " << this->layer_num << std::endl;
	free(offset);

	//DEBUG//
	std::unordered_set<uint64_t> st;
	for(uint32_t i = 0; i < this->layers.size();i++){
//		uint64_t expected_size = (this->n-1)/SPARTITIONS + 1;
//		uint64_t splits = (this->layers[i].size() - 1) / (expected_size) + 1;
//		std::cout << "Layer (" << i << ")" << " = " << this->layers[i].size() << "," << expected_size << "," << splits << std::endl;
//
//		std::vector<uint64_t> layer;
//		for(uint64_t j = 0; j < this->layers[i].size;j+=expected_size){
//			uint64_t upper = j+expected_size < this->layers[i].size ? expected_size : this->layers[i].size() - j;
//
//			for(uint64_t m = 0; m < upper; m++){
//
//			}
//		}

//		for(uint32_t j = 0; j < this->layers[i].size();j++){ st.insert(this->layers[i][j]); }
	}
//	exit(1);
//	std::cout << "set values: <" << st.size() << " ? " << this->n << ">" << std::endl;
}

template<class T, class Z>
uint64_t SLA<T,Z>::partition_table(uint64_t first, uint64_t last, std::unordered_set<uint64_t> layer_set, T **cdata, Z *offset){
	while(first < last){
		while(layer_set.find(first) == layer_set.end()){//Find a skyline point
			++first;
			if(first == last) return first;
		}

		do{//Find a non-skyline point
			--last;
			if(first == last) return first;
		}while(layer_set.find(last) != layer_set.end());
		offset[first] = offset[last];// Copy real-id of non-skyline point to the beginning of the array
		memcpy(cdata[first],cdata[last],sizeof(T)*this->d);// Copy non-skyline point to beginning of the array
		++first;
	}
	return first;
}

template<class T, class Z>
void SLA<T,Z>::create_lists(){
	//Initialize partitions based on skyline layers//
	this->parts = static_cast<sla_partition<T,Z>*>(aligned_alloc(32,sizeof(sla_partition<T,Z>)*this->layers.size()));
	uint64_t offset = 0;
	this->max_part_size = 0;
	for(uint64_t i = 0; i < this->layer_num; i++){
		uint64_t size = this->layers[i].size();
		uint64_t block_num = (size - 1)/SBLOCK_SIZE + 1;

//		std::cout << "Layer <<<" << i << ">>>" << std::endl;
//		std::cout << "size: " << size << std::endl;
//		std::cout << "block_num: " << block_num << std::endl;
//		std::cout << "total blocks size: " << block_num * SBLOCK_SIZE << std::endl;

		this->parts[i].offset = offset;
		this->parts[i].size = size;
		this->parts[i].block_num = block_num;
		this->parts[i].blocks = static_cast<sla_block<T,Z>*>(aligned_alloc(32,sizeof(sla_block<T,Z>)*this->parts[i].block_num));

		//Set Partition Blocks to Zero
		for(uint64_t j = 0; j < this->parts[i].block_num; j++){
			for(uint64_t m = 0; m < SBLOCK_SIZE * NUM_DIMS; m++){ this->parts[i].blocks[j].tuples[m] = 0; }
			for(uint64_t m = 0; m < this->d; m++){ this->parts[i].blocks[j].tarray[m] = 0; }
		}
		offset+=size;
		this->max_part_size = std::max(size,this->max_part_size);
	}

	//Build sorted lists for each layer//
	sla_pos<Z> *pos = (sla_pos<Z>*)malloc(sizeof(sla_pos<Z>)*this->max_part_size);
	sla_pair<T,Z> **lists = (sla_pair<T,Z>**)malloc(sizeof(sla_pair<T,Z>*)*this->d);
	for(uint64_t m=0; m<this->d;m++) lists[m] = (sla_pair<T,Z>*)malloc(sizeof(sla_pair<T,Z>)*this->max_part_size);
	for(uint64_t i = 0; i < this->layer_num; i++){
		for(uint64_t j = 0; j < this->max_part_size;j++){ pos[j].id = j; pos[j].pos = this->parts[i].size; }//Initialize to max possible position

		for(uint64_t m = 0; m < this->d; m++){
			for(uint64_t j = 0; j < this->parts[i].size; j++){
				Z id = this->layers[i][j];
				lists[m][j].id = j;
				lists[m][j].score = this->cdata[m*this->n + id];
			}
			__gnu_parallel::sort(lists[m],(lists[m]) + this->parts[i].size,cmp_sla_pair<T,Z>);//Sort to create lists

			for(uint64_t j = 0; j < this->parts[i].size; j++){
				Z lid = lists[m][j].id;
				pos[lid].pos = std::min(pos[lid].pos,(Z)j);//Find minimum local position of appearance in list//
			}
		}
		__gnu_parallel::sort(&pos[0],(&pos[0]) + this->parts[i].size,cmp_sla_pos<Z>);//Sort local tuple ids by minimum position//

		uint64_t b = 0;
		for(uint64_t j = 0; j < this->parts[i].size; j+=SBLOCK_SIZE){
			Z upper = ((j + SBLOCK_SIZE) <  this->parts[i].size) ? SBLOCK_SIZE : this->parts[i].size - j;
			this->parts[i].blocks[b].tuple_num = upper;
			this->parts[i].blocks[b].offset = j;

			for(uint64_t l = 0; l < upper; l++){
				Z lid = pos[j+l].id;//Find local tuple id
				Z gid = this->layers[i][lid];//Find global tuple id inside layer
				for(uint32_t m = 0; m < this->d; m++){
					this->parts[i].blocks[b].tuples[ m * SBLOCK_SIZE + l] = this->cdata[ m * this->n + gid ];
				}
			}

			Z p = pos[j + upper - 1].pos;//Find last point position
			for(uint32_t m = 0; m < this->d; m++){ this->parts[i].blocks[b].tarray[m] = lists[m][p].score; }//Extract threshold
			b++;
		}

	}
	for(uint32_t m=0; m<this->d;m++) free(lists[m]);
	free(lists);
}

template<class T, class Z>
void SLA<T,Z>::init(){
	normalize_transpose<T,Z>(this->cdata, this->n, this->d);
	///////////////////////////////////////
	//Copy data to compute skyline layers//
	//////////////////////////////////////
	T **cdata = NULL;
	cdata = this->sky_data(cdata);//Preprocess data to new table

	///////////////////////////
	//Compute Skyline Layers//
	//////////////////////////
	this->t.start();

	this->build_layers(cdata);//Assign tuples to different layers
	for(uint64_t i = 0; i < this->n; i++) free(cdata[i]);
	free(cdata);

	////////////////////////////
	//Create Lists for Layers//
	///////////////////////////
	this->create_lists();

	this->tt_init = this->t.lap();
}

template<class T, class Z>
void SLA<T,Z>::findTopKscalar(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " scalar (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
	this->t.start();
	for(uint64_t l = 0; l < this->layer_num; l++){
		if ( l  > k ) break;
		Z poffset = this->parts[l].offset;
		for(uint64_t b = 0; b < this->parts[l].block_num; b++){
			Z id = this->parts[l].offset + poffset;
			T *tuples = this->parts[l].blocks[b].tuples;

			for(uint64_t t = 0; t < this->parts[l].blocks[b].tuple_num; t+=8){
				id+=t;
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
			T *tarray = this->parts[l].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			if(q.size() >= k && q.top().score >= threshold){ break; }
		}
	}
	this->tt_processing += this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void SLA<T,Z>::findTopKsimd(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " simd (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
	float score[16] __attribute__((aligned(32)));
	this->t.start();
	__m256 dim_num = _mm256_set_ps(qq,qq,qq,qq,qq,qq,qq,qq);
	for(uint64_t l = 0; l < this->layer_num; l++){
		if ( l  > k ) break;
		Z poffset = this->parts[l].offset;
		for(uint64_t b = 0; b < this->parts[l].block_num; b++){
			Z id = this->parts[l].offset + poffset;
			T *tuples = this->parts[l].blocks[b].tuples;

			for(uint64_t t = 0; t < this->parts[l].blocks[b].tuple_num; t+=16){
				id+=t;
				__m256 score00 = _mm256_setzero_ps();
				__m256 score01 = _mm256_setzero_ps();
				for(uint8_t m = 0; m < qq; m++){
					T weight = weights[attr[m]];
					uint32_t offset = attr[m]*SBLOCK_SIZE + t;
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
			T *tarray = this->parts[l].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			if(q.size() >= k && q.top().score >= threshold){ break; }
		}
	}
	this->tt_processing+=this->t.lap();

	while(q.size() > k){ q.pop(); }
	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void SLA<T,Z>::findTopKthreads(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	uint32_t threads = THREADS;
	Z tt_count[threads];
	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> _q[threads];
	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
	omp_set_num_threads(threads);

	std::cout << this->algo << " find top-" << k << " threads (2) x "<< threads <<" (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	T threshold = 0;
	this->t.start();
#pragma omp parallel
{
	float score[16] __attribute__((aligned(32)));
	uint32_t tid = omp_get_thread_num();
	Z tuple_count = 0;
	__builtin_prefetch(score,1,3);
	__m256 dim_num = _mm256_set_ps(qq,qq,qq,qq,qq,qq,qq,qq);
	for(uint64_t l = 0; l < this->layer_num; l++){
		if ( l  > k ) break;
		Z poffset = this->parts[l].offset;
		for(uint64_t b = 0; b < this->parts[l].block_num; b++){
			Z id = this->parts[l].offset + poffset;
			T *tuples = this->parts[l].blocks[b].tuples;

			uint64_t chunk =(this->parts[l].blocks[b].tuple_num - 1)/threads + 1;
			uint64_t start = tid*(chunk);
			uint64_t end = (tid+1)*(chunk);

			for(uint64_t t = start; t < end; t+=16){
				id+=t;
				__m256 score00 = _mm256_setzero_ps();
				__m256 score01 = _mm256_setzero_ps();
				for(uint8_t m = 0; m < qq; m++){
					T weight = weights[attr[m]];
					uint32_t offset = attr[m]*SBLOCK_SIZE + t;
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

			if(tid==0){
				threshold = 0;
				T *tarray = this->parts[l].blocks[b].tarray;
				for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			}
			#pragma omp_barrier
			for(uint8_t m = 0; m < qq; m++) if(_q[m].size() >= k && _q[m].top().score >= threshold){ break; }
			//if(_q[tid].size() >= k && _q[tid].top().score >= threshold){ break; }
		}
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
	this->tt_processing+=this->t.lap();

	if(STATS_EFF){ for(uint32_t i = 0; i < threads; i++) this->tuple_count +=tt_count[i]; }
	threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
void SLA<T,Z>::findTopKthreads2(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	uint32_t threads = THREADS;
	Z tt_count[threads];
	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> _q[threads];
	std::priority_queue<T, std::vector<tuple_<T,Z>>, MaxCMP<T,Z>> q;
	omp_set_num_threads(threads);

	std::cout << this->algo << " find top-" << k << " threads (2) x "<< threads <<" (" << (int)qq << "D) ...";
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
	__m256 dim_num = _mm256_set_ps(qq,qq,qq,qq,qq,qq,qq,qq);
	for(uint64_t l = 0; l < this->layer_num; l++){
		if ( l  > k ) break;
		Z poffset = this->parts[l].offset;
		for(uint64_t b = 0; b < this->parts[l].block_num; b++){
			Z id = this->parts[l].offset + poffset;
			T *tuples = this->parts[l].blocks[b].tuples;

			uint64_t chunk =(this->parts[l].blocks[b].tuple_num - 1)/threads + 1;
			uint64_t start = tid*(chunk);
			uint64_t end = (tid+1)*(chunk);

			for(uint64_t t = start; t < end; t+=16){
				id+=t;
				__m256 score00 = _mm256_setzero_ps();
				__m256 score01 = _mm256_setzero_ps();
				for(uint8_t m = 0; m < qq; m++){
					T weight = weights[attr[m]];
					uint32_t offset = attr[m]*SBLOCK_SIZE + t;
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
			T *tarray = this->parts[l].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			//for(uint8_t m = 0; m < qq; m++) if(_q[m].size() >= k && _q[m].top().score >= threshold){ break; }
			if(_q[tid].size() >= k && _q[tid].top().score >= threshold){ break; }
		}
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
	this->tt_processing+=this->t.lap();

	if(STATS_EFF){ for(uint32_t i = 0; i < threads; i++) this->tuple_count +=tt_count[i]; }
	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
