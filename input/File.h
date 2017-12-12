#ifndef FILE_H
#define FILE_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <cstring>

#include "../time/Time.h"


template<class T>
class File{
	public:
		File(){}
		File(std::string fname,bool gpu){
			this->fname = fname;
			this->delimiter=',';
			this->transpose=false;
			this->count();
			this->gpu=gpu;
		}
		File(std::string fname,char delimiter,bool transpose,bool gpu){
			this->fname=fname;
			this->delimiter=delimiter;
			this->transpose=transpose;
			this->count();
			this->gpu=gpu;
		}

		~File(){
			if(this->data!=NULL && !this->gpu){ free(data); }
		}

		void load();
		void load(T *&data);
		uint64_t items(){return this->d;}
		uint64_t rows(){return this->n;}
		T* get_dt(){ return this->data; }
		void set_transpose(bool transpose){ this->transpose = transpose; }

		//testing
		void sample(){ this->sample(10); };
		void sample(uint64_t limit);

	protected:
		void count();
		void line_specifier();
		void read_scanf();
		void read_scanf_t();
		int fetch(T *&p, uint64_t d, FILE *f);

	private:
		T *data;
		uint64_t d;
		uint64_t n;
		std::string frow;
		const char *fetch_row;
		std::string code;

		std::string fname;
		char delimiter;
		bool transpose;
		bool gpu;
};


template<class T>
void File<T>::line_specifier(){
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

	//std::cout << "fetch_row: " << this->frow << std::endl;
	this->fetch_row=this->frow.c_str();
}

template<class T>
void File<T>::count(){
	FILE *fp = fopen(this->fname.c_str(), "r");
	int size = 16*1024*1024;
	int bytes = 0;
	char *buffer = (char*) malloc (size);
	this->d=0;
	this->n=0;

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
	//std::cout << "dim: (" << (this->n) << "," << (this->d) << ")"<<std::endl;
}

template<class T>
int File<T>::fetch(T *&p, uint64_t d, FILE *f){
	int count = 0;
	switch(d){
		case 2:
			count = fscanf(f,this->fetch_row,&p[0],&p[1]);
		case 4:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3]);
			break;
		case 6:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5]);
			break;
		case 8:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7]);
			break;
		case 10:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9]);
			break;
		case 12:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11]);
			break;
		case 14:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13]);
			break;
		case 16:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15]);
			break;
		default:
			count = 0;
			break;
	}

	return count;
}

template<class T>
void File<T>::read_scanf(){
	FILE *f;
	f = fopen(this->fname.c_str(), "r");
	uint64_t i = 0;

	T *ptr = &data[i];
	while(fetch(ptr,d,f) > 0){
		i+=(d);
		ptr = &data[i];
	}

	fclose(f);
}

template<class T>
void File<T>::read_scanf_t(){
	//std::cout << "Read scanf transpose..." << std::endl;
	FILE *f;
	f = fopen(this->fname.c_str(), "r");
	uint64_t i = 0;

	T *buffer = (T*)malloc(sizeof(T) * this->d);
	T *ptr = &(data[i]);
	while(this->fetch(buffer,this->d,f) > 0){
		for(uint64_t j = 0; j < this->d; j++){
			ptr[ j * this->n + i ] = buffer[j];
		}
		i++;
	}

	fclose(f);
}


template<class T>
void File<T>::load(){
	this->data = (T*)malloc(sizeof(T) * (this->n) * (this->d));
	//this->count();
	this->line_specifier();

	Time<msecs> t;
	t.start();
	if (!this->transpose){
		this->read_scanf();
	}else{
		this->read_scanf_t();
	}
	t.lap("Read elapsed time (ms)!!!");
}

template<class T>
void File<T>::load(T *& data){
	if(data == NULL){
		std::cout << "Load <data> pointer NULL!!!" << std::endl;
		exit(1);
	}
	//this->count();
	this->data=data;
	this->line_specifier();

	Time<msecs> t;
	//t.start();
	if (!this->transpose){
		this->read_scanf();
	}else{
		this->read_scanf_t();
	}
	//t.lap("Read elapsed time (ms)!!!");
}


template<class T>
void File<T>::sample(uint64_t limit){
	if(!this->transpose){
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
