#ifndef T2S_H
#define T2S_H

#include "AA.h"

#define TBLOCK_SIZE 1024
#define TSPLITS 2
#define TPARTITIONS (IMP == 2 ? (THREADS) : 1)

template<class T,class Z>
struct t2s_pair{
	Z id;
	T score;
};

template<class Z>
struct t2s_pos{
	Z id;
	Z pos;
};

template<class T, class Z>
struct t2s_block{
	Z offset;
	Z tuple_num;
	T tarray[NUM_DIMS] __attribute__((aligned(32)));
	T tuples[VBLOCK_SIZE * NUM_DIMS] __attribute__((aligned(32)));
};

template<class T, class Z>
struct t2s_partition{
	Z offset;
	Z size;
	Z block_num;
	t2s_block<T,Z> *blocks;
};

template<class T,class Z>
static bool cmp_t2s_pos(const t2s_pos<Z> &a, const t2s_pos<Z> &b){ return a.pos < b.pos; };

template<class T,class Z>
static bool cmp_t2s_pair(const t2s_pair<T,Z> &a, const t2s_pair<T,Z> &b){ return a.score > b.score; };

template<class T, class Z>
class T2S : public AA<T,Z>{
	public:
		T2S(uint64_t n, uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "T2S";
		};

		~T2S()
		{

		}

		void init();
		void findTopK(uint64_t k, uint8_t qq, T *weights, uint8_t *attr);

	private:
		t2s_partition<T,Z> parts[TPARTITIONS];
};

template<class T, class Z>
void T2S<T,Z>::init()
{
	normalize_transpose<T,Z>(this->cdata, this->n, this->d);
	this->t.start();
	uint64_t part_offset = 0;
	uint64_t part_size = ((this->n - 1) / TPARTITIONS) + 1;
	for(uint64_t i = 0; i < TPARTITIONS; i++){
		uint64_t first = (i*part_size);
		uint64_t last = (i+1)*part_size;
		last = last < this->n ? last : this->n;
		parts[i].offset = part_offset;
		parts[i].size = last - first;
		parts[i].block_num = ((parts[i].size-1)/TBLOCK_SIZE) + 1;
		parts[i].blocks = static_cast<t2s_block<T,Z>*>(aligned_alloc(32,sizeof(t2s_block<T,Z>)*parts[i].block_num));

		uint64_t block_offset = 0;
		uint64_t block_size = ((parts[i].size - 1)/parts[i].block_num) + 1;
		for(uint64_t j = 0; j <parts[i].block_num; j++){
			uint64_t ftuple = j * block_size;
			uint64_t ltuple = (j+1)*block_size;
			ltuple = ltuple < parts[i].size ? ltuple : parts[i].size;
			parts[i].blocks[j].offset = block_offset;
			parts[i].blocks[j].tuple_num = ltuple - ftuple;
			block_offset+= ltuple - ftuple;
		}
		part_offset += last - first;
	}

	//Initialize Blocks//
	uint64_t max_part_size = (((this->n - 1)/TPARTITIONS) + 1);
	t2s_pair<T,Z> **lists = (t2s_pair<T,Z>**)malloc(sizeof(t2s_pair<T,Z>*)*this->d);
	for(uint8_t m = 0; m < this->d; m++){ lists[m] = (t2s_pair<T,Z>*)malloc(sizeof(t2s_pair<T,Z>)*max_part_size); }
	t2s_pos<Z> *order = (t2s_pos<Z>*)malloc(sizeof(t2s_pos<Z>)*max_part_size);
	uint64_t poffset = 0;
	omp_set_num_threads(THREADS);
	for(uint64_t i = 0; i < TPARTITIONS; i++){
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
			__gnu_parallel::sort(lists[m],(lists[m]) + parts[i].size,cmp_t2s_pair<T,Z>);

			//Find minimum position appearance
			for(uint64_t j = 0; j < parts[i].size; j++){
				Z id = lists[m][j].id;
				order[id].pos = std::min(order[id].pos,(Z)j);//Minimum appearance position
			}
		}
		__gnu_parallel::sort(&order[0],(&order[0]) + parts[i].size,cmp_t2s_pos<T,Z>);

		//Split partition into blocks//
		uint64_t bnum = 0;
		for(uint64_t j = 0; j < parts[i].size; ){
			uint64_t jj;
			for(jj = 0; jj < parts[i].blocks[bnum].tuple_num; jj++){//For each block//
				Z id = order[j+jj].id;//Get next tuple in order
				for(uint8_t m = 0; m < this->d; m++){
					//parts[i].blocks[bnum].tuples[m*TBLOCK_SIZE + jj] = this->cdata[m*this->n + (poffset + id)];
					parts[i].blocks[bnum].tuples[jj*this->d + m] = this->cdata[m*this->n + (poffset + id)];
				}
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
void T2S<T,Z>::findTopK(uint64_t k, uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " scalar (" << (int)qq << "D) ...";
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(this->res.size() > 0) this->res.clear();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();
	for(uint64_t i = 0; i < TPARTITIONS; i++){
		for(uint64_t b = 0; b < parts[i].block_num; b++){
			Z tuple_num = parts[i].blocks[b].tuple_num;
			T *tuples = parts[i].blocks[b].tuples;
			uint64_t id = parts[i].offset + parts[i].blocks[b].offset;
			for(uint64_t t = 0; t < tuple_num; t++){
				id+=t;
				T score00 = 0;
				for(uint8_t m = 0; m < qq; m++){
					T weight = weights[attr[m]];
					//uint32_t offset = attr[m]*VBLOCK_SIZE + t;
					uint32_t offset = t*this->d + attr[m];
					score00+=tuples[offset]*weight;
				}
				if(q.size() < k){
					q.push(tuple_<T,Z>(id,score00));
				}else if(q.top().score < score00){
					q.pop(); q.push(tuple_<T,Z>(id,score00));
				}
				if(STATS_EFF) this->tuple_count++;
			}
			T threshold = 0;
			T *tarray = parts[i].blocks[b].tarray;
			for(uint8_t m = 0; m < qq; m++) threshold+=tarray[attr[m]]*weights[attr[m]];
			//if(q.size() >= k && q.top().score >= threshold){ i = TPARTITIONS;break; }
		}
	}

	this->tt_processing += this->t.lap();
	while(q.size() > k){ q.pop(); }
	T threshold = q.empty() ? 1313 : q.top().score;
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
