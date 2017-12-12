#ifndef INPUT_H
#define INPUT_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <cstring>

#include "../time/Time.h"


template<class T>
class Input{
public:
	Input(){};
	Input(std::string fname);
	~Input();

	void init();
	void sample(){ this->sample(10); };
	void sample(uint64_t limit);
	void transpose(bool tenable){ this->tenable = tenable; }

	uint64_t get_n(){ return this->n; }
	uint64_t get_d(){ return this->d; }
	T* get_dt(){ return this->data; }

protected:
	void line_specifier();
	void read_scanf();
	void read_scanf_t();
	void count();
	int fetch(float *&p, uint64_t d, FILE *f);

	T *data;
	std::string fname;
	uint64_t n;
	uint64_t d;
	bool tenable;
	bool gpu;
	char delimiter;
	std::string frow;
	const char *fetch_row;

};

template<class T>
Input<T>::Input(std::string fname){
	this->fname=fname;
	this->n=0;
	this->d=0;
	this->data=NULL;
	this->tenable=false;
	this->gpu = false;
	this->delimiter=',';
}

template<class T>
Input<T>::~Input(){
	if(this->data!=NULL && !this->gpu){ free(data); }
}

template<class T>
void Input<T>::init(){
	this->count();
	this->data = (T*)malloc(sizeof(T) * (this->n) * (this->d));

	Time<msecs> t;
	t.start();
	if(!tenable) this->read_scanf();
	else this->read_scanf_t();
	t.lap("Read elapsed time (ms)!!!");
}

template<class T>
void Input<T>::line_specifier(){
	//create fscanf line specifier
	this->frow="";
	for(uint64_t i = 0; i<d-1;i++){
		if( std::is_same<T,float>::value ){
			this->frow+="%f,";
		}else if( std::is_same<T,uint64_t>::value ){
			this->frow+="%d,";
		}
	}
	if( std::is_same<T,float>::value ){
		this->frow+="%f";
	}else if( std::is_same<T,uint64_t>::value ){
		this->frow+="%d";
	}
	this->fetch_row=this->frow.c_str();
}

template<class T>
void Input<T>::count(){
	FILE *fp = fopen(this->fname.c_str(), "r");
	int size = 16*1024*1024;
	int bytes = 0;
	char *buffer = (char*) malloc (size);

	bytes = fread(buffer,sizeof(char),size, fp);
	(this->d)++;
	for(int i = 0; i < bytes; i++){//Count D
		if( buffer[i] == ',' ) (this->d)++;
		if( buffer[i] == '\n' ) break;
	}
	do{//Count N
		for(int i = 0; i < bytes; i++) if( buffer[i] == '\n') (this->n)++;
	}while( (bytes = fread(buffer,sizeof(char),size, fp)) > 0 );

	fclose(fp);
	free(buffer);
	std::cout << "dim: (" << (this->n) << "," << (this->d) << ")"<<std::endl;
}

template<class T>
void Input<T>::read_scanf(){
	FILE *f;
	f = fopen(this->fname.c_str(), "r");
	uint64_t i = 0;

	float *ptr = &data[i];
	while(fetch(ptr,d,f) > 0){
		i+=(d);
		ptr = &data[i];
	}

	fclose(f);
}

template<class T>
void Input<T>::read_scanf_t(){
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
int Input<T>::fetch(float *&p, uint64_t d, FILE *f){
	int count = 0;
	switch(d){
		case 2:
			count = fscanf(f,"%f,%f",&p[0],&p[1]);
		case 4:
			count = fscanf(f,"%f,%f,%f,%f",&p[0],&p[1],&p[2],&p[3]);
			break;
		case 6:
			count = fscanf(f,"%f,%f,%f,%f,%f,%f",&p[0],&p[1],&p[2],&p[3],&p[4],&p[5]);
			break;
		case 8:
			count = fscanf(f,"%f,%f,%f,%f,%f,%f,%f,%f",&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7]);
			break;
		case 10:
			count = fscanf(f,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9]);
			break;
		case 12:
			count = fscanf(f,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11]);
			break;
		case 14:
			count = fscanf(f,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13]);
			break;
		case 16:
			count = fscanf(f,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15]);
			break;
		default:
			count = 0;
			break;
	}

	return count;
}

template<class T>
void Input<T>::sample(uint64_t limit){
	if(!this->tenable){
		std::cout << "Sample row-wise ... " << std::endl;
		for(uint64_t i = 0; i < ( limit < this->n ? limit : this->n ); i++){
			for(uint64_t j = 0; j < this->d; j++){
				std::cout << this->data[i * this->d + j] << " ";
			}
			std::cout << std::endl;
		}
	}else{
		std::cout << "Sample column-wise ... " << std::endl;
		for(uint64_t i = 0; i < ( limit < this->n ? limit : this->n); i++){
			for(uint64_t j = 0; j < this->d; j++){
				std::cout << this->data[ j * this->n + i ] << " ";
			}
			std::cout << std::endl;
		}
	}
}

#endif
