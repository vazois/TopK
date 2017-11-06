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
		void read_scanf();//Read transpose

		void sample(){ this->sample(10); };
		void sample(uint64_t limit);

	protected:


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

	Time<msecs> t;
	t.start();
	this->read_scanf();
	t.lap("Read elapsed time (ms)!!!");
}

template<class T>
void GInput<T>::read_scanf(){
	std::cout << "Read scanf transpose..." << std::endl;
	FILE *f;
	f = fopen(this->fname.c_str(), "r");
	uint64_t i = 0;

	float *buffer = (T*)malloc(sizeof(T) * this->d);
	float *ptr = &(this->data[i]);
	while(this->fetch(buffer,this->d,f) > 0){
		for(uint64_t j = 0; j < this->d; j++){
			ptr[ j * this->n + i ] = buffer[j];
		}
		i++;
	}

	fclose(f);
}

template<class T>
void GInput<T>::sample(uint64_t limit){
	std::cout << "Sample transpose..." << std::endl;
	for(uint64_t i = 0; i < ( limit < this->n ? limit : this->n); i++){
		for(uint64_t j = 0; j < this->d; j++){
			std::cout << this->data[ j * this->n + i ] << " ";
		}
		std::cout << std::endl;
	}
}

#endif
