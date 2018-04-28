#ifndef FILE_H
#define FILE_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <cstring>

#include "../time/Time.h"
#include "randdataset-1.1.0/src/randdataset.h"

template<class T>
class File{
	public:
		File(){}
		File(std::string fname,bool gpu, uint64_t n, uint64_t d){
			this->fname = fname;
			this->delimiter=',';
			this->transpose=false;
			this->n = n;
			this->d = d;
			this->gpu=gpu;
			this->packed=false;
		}

		File(std::string fname,bool gpu){
			this->fname = fname;
			this->delimiter=',';
			this->transpose=false;
			this->count();
			this->gpu=gpu;
			this->packed=false;
		}
		File(std::string fname,char delimiter,bool transpose,bool gpu){
			this->fname=fname;
			this->delimiter=delimiter;
			this->transpose=transpose;
			this->count();
			this->gpu=gpu;
			this->packed=false;
		}

		~File(){
			//if(this->data!=NULL && !this->gpu){ free(data); this->data = NULL; }
		}

		void load();
		void load(T *&data);
		void gen(T *&data, uint8_t type);

		void store(std::string fname, T *data);
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
		void write_printf_t();
		int fetch(T *&p, uint64_t d, FILE *f);
		void flush(T *&p, uint64_t d, FILE *f);


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
		bool packed;
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
			break;
		case 3:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2]);
			break;
		case 4:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3]);
			break;
		case 5:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4]);
			break;
		case 6:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5]);
			break;
		case 7:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6]);
		case 8:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7]);
			break;
		case 9:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8]);
			break;
		case 10:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9]);
			break;
		case 11:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10]);
			break;
		case 12:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11]);
			break;
		case 13:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12]);
			break;
		case 14:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13]);
			break;
		case 15:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14]);
			break;
		case 16:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15]);
			break;
		case 17:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16]);
			break;
		case 18:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17]);
			break;
		case 19:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18]);
			break;
		case 20:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19]);
			break;
		case 21:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19],&p[20]);
			break;
		case 22:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19],&p[20],&p[21]);
			break;
		case 24:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19],&p[20],&p[21],&p[22],&p[23]);
			break;
		case 25:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19],&p[20],&p[21],&p[22],&p[23],&p[24]);
			break;
		case 26:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19],&p[20],&p[21],&p[22],&p[23],&p[24],&p[25]);
			break;
		case 27:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19],&p[20],&p[21],&p[22],&p[23],&p[24],&p[25],&p[26]);
			break;
		case 28:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19],&p[20],&p[21],&p[22],&p[23],&p[24],&p[25],&p[26],&p[27]);
			break;
		case 29:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19],&p[20],&p[21],&p[22],&p[23],&p[24],&p[25],&p[26],&p[27],&p[28]);
			break;
		case 30:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19],&p[20],&p[21],&p[22],&p[23],&p[24],&p[25],&p[26],&p[27],&p[28],&p[29]);
			break;
		case 31:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19],&p[20],&p[21],&p[22],&p[23],&p[24],&p[25],&p[26],&p[27],&p[28],&p[29],&p[30]);
			break;
		case 32:
			count = fscanf(f,this->fetch_row,&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8],&p[9],&p[10],&p[11],&p[12],&p[13],&p[14],&p[15],&p[16],&p[17],&p[18],&p[19],&p[20],&p[21],&p[22],&p[23],&p[24],&p[25],&p[26],&p[27],&p[28],&p[29],&p[30],&p[31]);
			break;
		default:
			count = 0;
			break;
	}

	return count;
}

template<class T>
void File<T>::read_scanf(){
	FILE *f=NULL;
	f = fopen(this->fname.c_str(), "r");
	uint64_t i = 0;
	if (f == NULL) {
		std::cout << "Error opening file!!!!" << std::endl;
		exit(1);
	}

	float progress = 0.0;
	uint64_t step = 1024;
	uint64_t ii = 0;

	T *ptr = &data[i];
	while(fetch(ptr,d,f) > 0 && ii < this->n){
		i+=(d);
		ptr = &data[i];

		if((ii & (step - 1)) == 0){
			std::cout << "Progress: [" << int(progress * 100.0) << "] %\r";
			std::cout.flush();
			progress += ((float)step)/this->n; // for demonstration only
		}
		ii++;
	}

	fclose(f);
}

template<class T>
void File<T>::read_scanf_t(){
	//std::cout << "Read scanf transpose..." << std::endl;
	FILE *f = NULL;
	f = fopen(this->fname.c_str(), "r");
	if (f == NULL) {
		std::cout << "Error opening file!!!!" << std::endl;
		exit(1);
	}
	uint64_t i = 0;

	float progress = 0.0;
	uint64_t step = 1024;

	T *buffer = (T*)malloc(sizeof(T) * this->d);
	T *ptr = &(data[i]);
	while(this->fetch(buffer,this->d,f) > 0 && i < this->n){
		//std::cout << "read\n";
		for(uint64_t j = 0; j < this->d; j++){
			ptr[ j * this->n + i ] = buffer[j];
		}
		i++;

		if((i & (step - 1)) == 0){
			std::cout << "Progress: [" << int(progress * 100.0) << "] %\r";
			std::cout.flush();
			progress += ((float)step)/this->n; // for demonstration only
		}
	}

	fclose(f);
	free(buffer);
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
//		this->n = (((this->n - 1)/1024) + 1)*1024;
//		std::cout << "N:" << n << std::endl;
//		data = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->n) * (this->d)));
		data = static_cast<T*>(aligned_alloc(1024, sizeof(T) * (this->n) * (this->d)));
	}
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
void File<T>::gen(T *&data, uint8_t type){
	if(data == NULL){
		data = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->n) * (this->d)));
	}
	this->data=data;

	if(type < 0 || type > 2){
		std::cout << "Type should be 0(correlated),1(independent),2(anticorrelated)!!!";
		exit(1);
	}

	if( type == 0 ){
		generate_corr_inmem(this->data,this->n,this->d,this->transpose);
	}else if ( type == 1 ){
		generate_indep_inmem(this->data,this->n,this->d,this->transpose);
	}else if ( type == 2 ){
		generate_anti_inmem(this->data,this->n,this->d,this->transpose);
	}
}

template<class T>
void File<T>::flush(T *&p, uint64_t d, FILE *f){
	int count = 0;
	switch(d){
		case 2:
			count = fprintf(f,this->fetch_row,p[0],p[1]);
			break;
		case 4:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3]);
			break;
		case 6:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5]);
			break;
		case 8:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
			break;
		case 10:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]);
			break;
		case 12:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11]);
			break;
		case 14:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13]);
			break;
		case 16:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);
			break;
		case 18:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17]);
			break;
		case 20:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],p[19]);
			break;
		case 22:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],p[19],p[20],p[21]);
			break;
		case 24:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],p[19],p[20],p[21],p[22],p[23]);
			break;
		case 26:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],p[19],p[20],p[21],p[22],p[23],p[24],p[25]);
			break;
		case 28:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],p[19],p[20],p[21],p[22],p[23],p[24],p[25],p[26],p[27]);
			break;
		case 30:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],p[19],p[20],p[21],p[22],p[23],p[24],p[25],p[26],p[27],p[28],p[29]);
			break;
		case 32:
			count = fprintf(f,this->fetch_row,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],p[19],p[20],p[21],p[22],p[23],p[24],p[25],p[26],p[27],p[28],p[29],p[30],p[31]);
			break;
		default:
			count = 0;
			break;
	}
}

template<class T>
void File<T>::write_printf_t(){
	FILE *f;
	f = fopen(this->fname.c_str(), "w");

	T *buffer = (T*)malloc(sizeof(T) * this->d);
	for(uint64_t i = 0; i < this->n; i++){
		for(uint64_t m = 0; m < this->d; m++){
			buffer[m] = this->data[m*this->n + i];
		}
		this->flush(buffer,this->d,f);
		fprintf(f,"\n");
	}

	fclose(f);
	free(buffer);
}

template<class T>
void File<T>::store(std::string fname, T * data){
	this->data = data;
	this->fname = fname;
	if (!this->transpose){

	}else{
		this->write_printf_t();
	}
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
