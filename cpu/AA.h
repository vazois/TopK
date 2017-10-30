#ifndef AA_H
#define AA_H

#include<vector>
#include<algorithm>

#include "../input/Input.h"



template<class T>
struct pred{
	pred(){

	}
	pred(uint64_t t, T a){
		tid = t;
		attr = a;
	}
	uint64_t tid;
	T attr;
};

template<class T>
class AA{
public:
	AA(Input<T>* input);
	~AA();

	void init();

private:
	std::vector<std::vector<pred<T>>> lists;
	Input<T>* input;
	static bool cmp_max_pred(const pred<T> &a, const pred<T> &b){ return a.attr > b.attr; };
};

template<class T>
AA<T>::AA(Input<T>* input){
	this->input = input;
	//this->lists = NULL;
}

template<class T>
AA<T>::~AA(){
//	if(this->lists != NULL){
//		delete this->lists;
//	}
}

template<class T>
void AA<T>::init(){
	Time<msecs> t;
	//this->lists = new std::vector<pred<T>>[2];
	t.start();
	this->lists.resize(this->input->get_d());
	for(int i =0;i<this->input->get_d();i++){
		this->lists[i].resize(this->input->get_n());
	}

	for(uint64_t j=0;j<this->input->get_n();j++){
		//std::cout << j << ": ";
		for(int i =0;i<this->input->get_d();i++){
			uint64_t tid = j;
			T attr = (this->input->get_dt())[j*this->input->get_d() + i];
			//std::cout << attr <<" , ";
			this->lists[i].push_back(pred<T>(tid,attr));
		}
		//std::cout << std::endl;
	}
	t.lap("Elapsed time to construct lists (ms)!!!");

	t.start();
	for(int i =0;i<this->input->get_d();i++){
		std::sort(this->lists[i].begin(),this->lists[i].end(),cmp_max_pred);
	}
	t.lap("Elapsed time to sort lists (ms)!!!");
}

#endif
