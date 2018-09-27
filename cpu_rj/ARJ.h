#ifndef ARJ_H
#define ARJ_H

#include "../tools/RJTools.h"
#include "../tools/HashTable.h"

template<class Z, class T>
class AARankJoin{
	public:
		AARankJoin(RankJoinInstance<Z,T> *rj_inst)
		{
			this->rj_inst = rj_inst;
			this->reset_metrics();
			this->reset_aux_struct();
		};

		~AARankJoin()
		{

		};

		void join();

		void reset_aux_struct();
		void merge_qs();
		void merge_metrics();
		void reset_metrics();
		void benchmark();

	protected:
		void set_algo(std::string algo){ this->algo = algo; }

		std::string algo;
		RankJoinInstance<Z,T> *rj_inst = NULL;
		std::unordered_multimap<Z,T> htR;
		std::unordered_multimap<Z,T> htS;

		//Ranking Structures//
		std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>> q[THREADS];

		Time<msecs> t;
		double t_init;
		double t_join;
		uint64_t tuple_count;

		double tt_init[THREADS];
		double tt_join[THREADS];
		Time<msecs> tt[THREADS];
		uint64_t ttuple_count[THREADS];
};

template<class Z, class T>
void AARankJoin<Z,T>::reset_metrics(){
	this->t_init = 0;
	this->t_join = 0;
	this->tuple_count = 0;

	for(uint32_t i = 0; i < THREADS; i++){
		this->tt_init[i] = 0;
		this->tt_join[i] = 0;
		this->ttuple_count[i] = 0;
	}
}

template<class Z, class T>
void AARankJoin<Z,T>::reset_aux_struct(){
	for(uint32_t i = 0; i < THREADS; i++) this->q[i] = std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>>();
	this->htR.clear();
	this->htS.clear();
}

template<class Z, class T>
void AARankJoin<Z,T>::merge_qs(){
	Z k = this->rj_inst->getK();
	for(uint32_t i = 1; i < THREADS; i++){
		while(!this->q[i].empty()){
			if(this->q[0].size() < k){
				this->q[0].push(this->q[i].top());
			}else if(this->q[0].top().score < this->q[i].top().score){
				this->q[0].pop();
				this->q[0].push(this->q[i].top());
			}
			this->q[i].pop();
		}
	}
}

template<class Z, class T>
void AARankJoin<Z,T>::merge_metrics()
{
	this->t_init = this->tt_init[0];
	this->t_join = this->tt_join[0];
	for(uint32_t i = 0; i < THREADS; i++){
		this->t_init = std::max(this->t_init,this->tt_init[i]);
		this->t_join = std::max(this->t_join,this->tt_join[i]);
		this->tuple_count+=this->ttuple_count[i];
	}
}

template<class Z, class T>
void AARankJoin<Z,T>::benchmark(){
	std::cout << "<<< " << this->algo << " >>>" << std::endl;
	std::cout << "tuple_count: " << this->tuple_count << std::endl;
	std::cout << "join elapsed(ms): " << this->t_join << std::endl;
	if(this->q[0].size() > 0) std::cout << "threshold (" << this->q[0].size() <<"): " << std::fixed << std::setprecision(4) << this->q[0].top().score << std::endl;
	std::cout << "----------------------------" << std::endl;
}

#endif
