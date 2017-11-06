#ifndef GINPUT_H
#define GINPUT_H

#include "Input.h"
#include "../gpu/CudaHelper.h"

template<class T>
class GInput : public Input<T>{
	public:
		GInput(std::string fname) : Input<T>(fname){ this->gpu=true; };
		~GInput();

		void init();
};

template<class T>
GInput<T>::~GInput(){
	if(this->data != NULL){
		cudaFreeHost(this->data);
	}
}

template<class T>
void GInput<T>::init(){
	std::cout << " Calling GPU init " << std::endl;

	this->count();

	cutil::safeMallocHost<T,uint64_t>(&(this->data),(this->n) * (this->d) * sizeof(T),"data host alloc");

}

#endif
