#ifndef CBA_H
#define CBA_H

#include "AA.h"
#include "../input/Input.h"
#include "reorder_attr_cpu.h"

#include <list>
#include <queue>


template<class T>
struct cpred{
	cpred(){ tid = 0; curr_attr = 0; total = 0; }
	cpred(uint64_t t, T a){ tid = t; curr_attr = a; total = a;}
	bool compare(T threshold, uint64_t r){ return (total + curr_attr * r) < threshold; }
	uint64_t tid;
	T curr_attr;
	T total;
};

template<class T>
static bool cmp_max_cpred(const cpred<T> &a, const cpred<T> &b){ return a.total > b.total; };

template<class T>
class CBA : public AA<T>{
	public:
		CBA(Input<T>* input) : AA<T>(input){ this->algo = "CBA"; };
		~CBA(){ };

		void init();
		void findTopK(uint64_t k);

	protected:
		std::list<cpred<T>> tupples;

	private:
		void check_order();
		T find(uint64_t k);
		T findAndPrune(uint64_t k,uint64_t r);
};

template<class T>
void CBA<T>::init(){
	std::cout << this->algo <<" initialize: " << "(" << this->n << "," << this->d << ")"<< std::endl;

	this->t.start();
	switch(this->d){
		case 4:
			reorder_attr_4(this->cdata,this->n);
			break;
		case 6:
			reorder_attr_6(this->cdata,this->n);
			break;
		case 8:
			reorder_attr_8(this->cdata,this->n);
			break;
		case 10:
			reorder_attr_10(this->cdata,this->n);
			break;
		case 16:
			reorder_attr_16(this->cdata,this->n);
			break;
		default:
			break;
	}
	this->tt_init = this->t.lap();
	this->check_order();//TODO: Comment

	for(uint64_t i = 0; i < this->n; i++){ this->tupples.push_back(cpred<T>(i,this->cdata[i])); }
	//this->sort(cmp_max_cpred<T>);
}

template<class T>
void CBA<T>::check_order(){
	std::string passed = "(PASSED)";
	for(uint64_t i = 0; i < this->n; i++){
		bool ordered = true;
		for(uint64_t j = 0; j < (this->d - 1); j++){ ordered &=(this->cdata[j * this->n + i] >= this->cdata[(j+1) * this->n + i]); }

		if(!ordered){
			passed = "(FAILED)";
			std::cout << "i: <" << i << "> ";
			for(uint64_t j = 0; j < this->d; j++) std::cout << this->cdata[j * this->n + i] << " ";
			std::cout << std::endl;
			std::cout << "check_order: " << passed << std::endl;
			exit(1);
		}
		//if(i < 10){ for(uint64_t j = 0; j < this->d; j++){ std::cout << this->cdata[j * this->n + i] << " "; } std::cout << std::endl;}
	}
}

template<class T>
T CBA<T>::findAndPrune(uint64_t k, uint64_t r){
	typename std::list< cpred<T> >::iterator it = this->tupples.begin();
	uint64_t i = 0;
	std::priority_queue<T, std::vector<T>, std::greater<T>> q;

	//this->t.start();
	while( it != this->tupples.end() ){
		if(q.size() < k){
			q.push(it->total);
		}else if(q.top()<it->total){
			q.pop();
			q.push(it->total);
		}
		//if(q.size() == k){ q.pop(); }
		it++;
	}
	//this->t.lap("<find k - priority_queue>");
	T threshold = q.top();
//	std::cout << "pq threshold: " << threshold << std::endl;
//	q.pop();
//	std::cout << "B: " << q.top() << std::endl;

/*Sort based q finder*/
//	this->t.start();
//	this->tupples.sort(cmp_max_cpred<T>);
//	this->t.lap("<sort>");
//
//	it = this->tupples.begin();
//	this->t.start();
//	while( it != this->tupples.end() && i < k){
//		//std::cout << "<< "<<i<<"," << it->total << std::endl;
//		i++; it++;
//	}
//	this->t.lap("<find k>");
//	T threshold = (it--)->total;
//	std::cout << "threshold: " << threshold << std::endl;

	//this->t.start();
	it=this->tupples.begin();
	uint64_t mult =this->d-(r+1);
	while(it != this->tupples.end()){
		if(it->compare(threshold,mult)){
			it = this->tupples.erase(it);
		}else{
			T next_attr = this->cdata[(r+1) * this->n + it->tid];
			it->total += next_attr;
			it->curr_attr = next_attr;
			this->eval_count++;
			it++;
		}

	}
	//this->t.lap("<prune>");
	std::cout << "tupples: " << this->tupples.size() << std::endl;
}

template<class T>
T CBA<T>::find(uint64_t k){
	typename std::list< cpred<T> >::iterator it;
	this->tupples.sort(cmp_max_cpred<T>);
}

template<class T>
void CBA<T>::findTopK(uint64_t k){

	this->t.start();
	for(uint64_t j = 0; j < this->d; j++){//d-1
		findAndPrune(k,j);
		//this->t.lap("<>");
	//	findAndPrune(k,1);
	//	findAndPrune(k,2);
	}
	//findAndPrune(k,3);
	//this->tupples.sort(cmp_max_cpred<T>);//sort when d-1
	this->tt_processing = this->t.lap("<>");

	uint64_t i = 0;
	typename std::list< cpred<T> >::iterator it;
	for (it=tupples.begin(); it != tupples.end(); ++it){
		//std::cout << it->total << "," << it->tid << std::endl;
		this->res.push_back(tuple<T>(it->tid,it->total));
		i++;
		if(i > k-1) break;
	}

	std::cout << "list_size: " << tupples.size() << std::endl;

}

#endif
