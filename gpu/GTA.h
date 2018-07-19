#ifndef GTA_H
#define GTA_H

#include "GAA.h"
#include "../tools/CudaHelper.h"


template<class T, class Z>
class GTA : public GAA<T,Z>{
	public:
		GTA(uint64_t n, uint64_t d) : GAA<T,Z>(n,d){
			this->algo = "GTA";
		};

		~GTA(){

		};

		void alloc();
		void init(T *weights, uint32_t *query);
		void findTopK(uint64_t k, uint64_t qq);

	private:

};

template<class T, class Z>
void GTA<T,Z>::alloc(){
	cutil::safeMallocHost<T,uint64_t>(&(this->cdata),sizeof(T)*this->n*this->d,"cdata alloc");// Allocate cpu data memory
}

template<class T, class Z>
void GTA<T,Z>::init(T *weights, uint32_t *query){

}

#endif
