#ifndef ONION_H
#define ONION_H

#include "AA.h"

template<class T, class Z>
struct onion_tuple
{
	Z id;
	T *data;
	onion_tuple(Z id, T *data) : id(id) , data(data) {}
};

template<class T,class Z>
static bool onion_convex_hull_sort(const onion_tuple<T,Z> &a, const onion_tuple<T,Z> &b){ return a.data[0] > b.data[0]; };

template<class T,class Z>
class Onion : public AA<T,Z>{
	public:
	Onion(uint64_t n,uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "Onion";

		}

		~Onion()
		{

		}

		void init();
		void findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);

	private:

};

template<class T, class Z>
void Onion<T,Z>::init(){
	normalize<T,Z>(this->cdata, this->n, this->d);

	std::vector<onion_tuple<T,Z>> objects;
	std::vector<onion_tuple<T,Z>> ch;
	this->t.start();
	for(uint64_t i = 0; i < this->n; i++) objects.push_back(onion_tuple<T,Z>(i,&this->cdata[i*this->d]));
	__gnu_parallel::sort(objects.begin(),objects.end(),onion_convex_hull_sort<T,Z>);

	for(int i = 0; i < 10; i++)
	{
		std::cout <<  i <<" : ";
		for(int m = 0; m < this->d; m++)
		{
			std::cout << objects[i].data[m] << " | ";
		}
		std::cout << std::endl;
	}
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void Onion<T,Z>::findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr){
	std::cout << this->algo << " find top-" << k << " (" << (int)qq << "D) ...";
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(STATS_EFF) this->candidate_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;

	T threshold = !q.empty() ? q.top().score : 1313;
	while(!q.empty()){
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
