#ifndef GPTA_H
#define GPTA_H
/*
 * GPU Parallel Threshold Algorithm
 */

#include "GAA.h"
#include "tools.h"
#include <map>
#define GPTA_PI 3.1415926535
#define GPTA_PI_2 (180.0f/PI)

#define GPTA_SPLITS 2
#define GPTA_PARTS (((uint64_t)pow(GPTA_SPLITS,NUM_DIMS-1)))
#define GPTA_PART_BITS ((uint64_t)log2f(GPTA_SPLITS))

#define GPTA_DATA_BLOCK_SIZE 1024
//A generic approach to efficient simd vectorization of Top-K Selection
template<class T, class Z>
struct GTG{ //GPU Threshold Group
	T *data;//raw data
	T *tvector;// threshold vector
	Z size;
};

template<class T, class Z>
class GPTA : public GAA<T,Z>{
	public:
		GPTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "GPTA";
			this->gtg_list = NULL;
			this->max_gtg_count = 0;
			this->min_gtg_count = 0;
			this->max_gtg_size = 0;
		};
		~GPTA(){
			if(this->gtg_list != NULL){
				for(uint64_t i = 0; i < this->max_gtg_count; i++){
					if(this->gtg_list[i].data != NULL){ cudaFreeHost(this->gtg_list[i].data); }
					if(this->gtg_list[i].tvector != NULL){ cudaFreeHost(this->gtg_list[i].tvector); }
				}

				free(this->gtg_list);
			}
		};

		void alloc();
		void init(T *weights, uint32_t *query);
		void findTopK(uint64_t k, uint64_t qq);

	private:
		void polar_partition();
		void build_gtg_list();

		GTG<T,Z> *gtg_list;
		Z max_gtg_count;
		Z min_gtg_count;
		Z max_gtg_size;
		std::map<Z,Z> mm;
};


template<class T, class Z>
void GPTA<T,Z>::polar_partition(){
	Z *part_id = (Z*)malloc(sizeof(Z) * this->n);
	T *sum_pcoord = (T*)malloc(sizeof(T) * this->n);
	gpta_pair<T,Z> *curr_pcoord = (gpta_pair<T,Z>*)malloc(sizeof(gpta_pair<T,Z>) * this->n);

	for(uint64_t i = 0; i < this->n; i++){
		T curr = this->cdata[(this->d-1)*this->n + i] - 1.1;
		sum_pcoord[i] = curr * curr;
		part_id[i] = 0;
	}

	uint64_t chunk_size = ((this->n - 1) /GPTA_SPLITS) + 1;
	uint64_t shift = 0;
	for(uint64_t m = this->d-1 ; m > 0; m--){
		//#pragma omp parallel for schedule(dynamic)
		for(uint64_t i = 0; i < this->n; i++){
			T next = fabs(this->cdata[(m-1)*this->n + i] - 1.1);
			T sum = sum_pcoord[i];

			curr_pcoord[i].id = i;
			curr_pcoord[i].score = atan((sqrt(sum) / next));
			sum_pcoord[i] += (next * next);
		}
		//std::sort(&curr_pcoord[0],(&curr_pcoord[0])+this->n,cmp_gpta_pair_asc<T,Z>);
		psort<T,Z>(curr_pcoord,this->n,true);//sort ascending

		for(uint64_t i = 0; i < this->n; i++){
			Z id = curr_pcoord[i].id;
			uint32_t part = i / chunk_size;
			//if(id==0){ std::cout << "part:" << part << std::endl; }
			part_id[id] = part_id[id] | (part << shift);
		}
		shift+=GPTA_PART_BITS;
	}
	free(sum_pcoord);
	free(curr_pcoord);

	Z *psum = (Z*)malloc(sizeof(Z)*GPTA_PARTS);
	for(uint64_t i = 0; i < GPTA_PARTS;i++) this->mm.insert(std::pair<Z,Z>(i,0));
	for(uint64_t i = 0; i < this->n; i++){
		Z pid = part_id[i];
		this->mm[pid]+=1;
	}
	psum[0] = 0;
	this->max_gtg_count = (this->mm[0]-1)/GPTA_DATA_BLOCK_SIZE + 1;
	this->min_gtg_count = (this->mm[0]-1)/GPTA_DATA_BLOCK_SIZE + 1;
	this->max_gtg_size = this->mm[0];
	for(uint64_t i = 1; i < GPTA_PARTS;i++){
		psum[i] = psum[i-1] + this->mm[i-1];
		Z gtg_count = (this->mm[i]-1)/GPTA_DATA_BLOCK_SIZE + 1;
		this->max_gtg_count = std::max(this->max_gtg_count,gtg_count);
		this->min_gtg_count = std::min(this->min_gtg_count,gtg_count);
		this->max_gtg_size = std::max(this->max_gtg_size, this->mm[i]);
	}

	//Debug Only//
	//for(uint64_t i = 0; i < 10; i++){ std::cout << i << "," << part_id[i] << std::endl; }
	for(uint64_t i = 0; i < GPTA_PARTS; i++){
		std::cout << "g(" << std::setfill('0') << std::setw(4) <<i << "):";
		std::cout << std::setfill('0') << std::setw(8) << this->mm[i] << ",";
		std::cout << std::setfill('0') << std::setw(8) << psum[i] << std::endl;
	}
	std::cout << "GPTA_PARTS:" << GPTA_PARTS << "," << chunk_size <<"," << GPTA_PART_BITS << std::endl;
	std::cout << "max_gtg_count: " << this->max_gtg_count << ", min_gtg_count: " << this->min_gtg_count << "" << std::endl;
	std::cout << "max_gtg_size: " << this->max_gtg_size << std::endl;
	compute_threshold<T,Z>(this->cdata,this->n,this->d,(uint64_t)KKS);
	//

	//Copy iteratively to save memory space when reordering the data
	Z *psize = (Z*)malloc(sizeof(Z)*GPTA_PARTS);
	T *column = (T*)malloc(sizeof(T) * this->n);
	for(uint64_t m = 0; m < this->d; m++){
		for(uint64_t i = 0; i < GPTA_PARTS;i++) psize[i] = 0;
		for(uint64_t i = 0; i < this->n; i++){
			Z pid = part_id[i];
			Z offset = psum[pid] + psize[pid];
			column[offset] = this->cdata[m*this->n + i];
			psize[pid]+=1;
		}
		std::memcpy(&this->cdata[m*this->n], column, sizeof(T)*this->n);
	}
	free(column);
	free(psize);
	free(psum);
	free(part_id);
	//DEBUG
	compute_threshold<T,Z>(this->cdata,this->n,this->d,(uint64_t)KKS);
}

template<class T,class Z>
void GPTA<T,Z>::build_gtg_list(){
	gpta_pair<T,Z> *tscore =(gpta_pair<T,Z>*) malloc(sizeof(gpta_pair<T,Z>)*this->max_gtg_size);
	gpta_pos<Z> *tpos = (gpta_pos<Z>*)malloc(sizeof(gpta_pos<Z>)*this->max_gtg_size);
	T *part_data = (T*)malloc(sizeof(T)*NUM_DIMS*this->max_gtg_size);

	Z offset=0;
	for(uint64_t i = 0; i <GPTA_PARTS;i++){
		for(uint64_t j = 0; j < this->mm[i]; j++){
			tpos[j].id = j;//relative id
			tpos[j].pos = this->mm[i];//relative maximum
		}
		for(uint64_t m = 0; m < this->d; m++){
			for(uint64_t j = 0; j < this->mm[i]; j++){
				tscore[j].id = j;//relative id
				tscore[j].score = this->cdata[m*this->n + (j+offset)];//actual attribute
			}
			psort<T,Z>(tscore,this->mm[i],false);//sort descending

			for(uint64_t j = 0; j < this->mm[i]; j++){
				Z id = tscore[j].id;
				tpos[id].pos = std::min(tpos[id].pos,j);//find minimum relative pos
			}
		}
		ppsort<Z>(tpos,this->mm[i]);

		//Reorder data using a buffer//
		for(uint64_t m = 0; m < this->d; m++){
			for(uint64_t j = 0; j < this->mm[i]; j++){
				Z id = tpos[j].id;
				part_data[m*this->max_gtg_size + j] = this->cdata[m*this->n + (offset + id)];
			}
		}
		for(uint64_t m = 0; m < this->d; m++){
			std::memcpy(&this->cdata[m*this->n + offset],
					&part_data[m*this->max_gtg_size],sizeof(T)*this->mm[i]);
		}
		offset+=this->mm[i];
	}
	compute_threshold<T,Z>(this->cdata,this->n,this->d,(uint64_t)KKS);

	free(tscore);
	free(tpos);
	free(part_data);
}

template<class T, class Z>
void GPTA<T,Z>::alloc(){
	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");// Allocate cpu data memory
	//cutil::safeMalloc<T,uint64_t>(&(this->gdata),sizeof(T)*this->n*this->d,"gdata alloc");//Allocate gpu data memory

	cutil::safeMallocHost<T,uint64_t>(&(this->cscores),sizeof(T)*this->n,"cscores alloc");//Allocate cpu scores memory
	cutil::safeMalloc<T,uint64_t>(&(this->gscores),sizeof(T)*this->n,"gscores alloc");//Allocate gpu scores memory
}

template<class T, class Z>
void GPTA<T,Z>::init(T *weights, uint32_t *query){
	//normalize_transpose<T>(this->cdata, this->n, this->d);
	this->t.start();
	this->polar_partition();
	this->build_gtg_list();
	this->tt_init=this->t.lap();
}

#endif
