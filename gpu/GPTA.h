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

template<class T, class Z>
class GPTA : public GAA<T,Z>{
	public:
		GPTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "GPTA";
		};

		void alloc();
		void init(T *weights, uint32_t *query);
		void findTopK(uint64_t k, uint64_t qq);

	private:
		void polar_partition();
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
		psort<T,Z>(curr_pcoord,this->n);

		for(uint64_t i = 0; i < this->n; i++){
			Z id = curr_pcoord[i].id;
			uint32_t part = i / chunk_size;
			//if(id==0){ std::cout << "part:" << part << std::endl; }
			part_id[id] = part_id[id] | (part << shift);
		}
		shift+=GPTA_PART_BITS;
	}

	std::cout << "GPTA_PARTS:" << GPTA_PARTS << "," << chunk_size <<"," << GPTA_PART_BITS << std::endl;
	for(uint64_t i = 0; i < 10; i++){ std::cout << i << "," << part_id[i] << std::endl; }

	std::map<Z,Z> mm;
	for(uint64_t i = 0; i < GPTA_PARTS;i++) mm.insert(std::pair<Z,Z>(i,0));
	for(uint64_t i = 0; i < this->n; i++){
		Z pid = part_id[i];
		mm[pid]+=1;
	}

	for(uint64_t i = 0; i < GPTA_PARTS; i++){
		std::cout << "g(" << std::setfill('0') << std::setw(4) <<i << "):";
		std::cout << std::setfill('0') << std::setw(8) << mm[i] << std::endl;
	}

	free(part_id);
	free(sum_pcoord);
	free(curr_pcoord);
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
	normalize_transpose<T>(this->cdata, this->n, this->d);
	this->t.start();
	this->polar_partition();
	this->tt_init=this->t.lap();
}

#endif
