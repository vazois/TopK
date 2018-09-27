#ifndef HJR_H
#define HJR_H

#include "ARJ.h"

template<class Z, class T>
struct npo_args_t{
	uint32_t tid = 0;
	pthread_barrier_t *barrier;
	TABLE<Z,T> *R = NULL;
	TABLE<Z,T> *S = NULL;
	uint64_t sR = 0,eR = 0;
	uint64_t sS = 0,eS = 0;
	S_HashTable<Z,T> *htR;
	std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>> *q;
	Z k;

	///////////
	//Metrics//
	double *tt_init;
	double *tt_join;
	Time<msecs> *tt;
	uint64_t *ttuple_count;
};

template<class Z, class T>
class HJR : public AARankJoin<Z,T>{
	public:
		HJR(RankJoinInstance<Z,T> *rj_inst) : AARankJoin<Z,T>(rj_inst){ };
		~HJR(){};

		void snop_hash_join();
		void st_nop_hash_rank_join();
		void mt_nop_hash_rank_join();

	private:
		static void* mt_nop_thread(void *args);

		void pshift(Z *arr, Z arr_n){
			for(uint32_t i = arr_n-1; i > 0; i--){ arr[i] = arr[i-1]; }
			arr[0] = 0;
		}

		void psum(Z *arr, Z arr_n){
			for(uint32_t i = 1; i < arr_n+1; i++){
				Z tmp = arr[0] + arr[i];
				arr[i] = arr[0];
				arr[0] = tmp;
			}
			arr[0] = 0;
		}
};

template<class Z, class T>
void HJR<Z,T>::snop_hash_join(){
	this->set_algo("single-thread no partition hash join");
	this->reset_metrics();
	this->reset_aux_struct();

	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();

	//Build phase
	this->t.start();
	for(uint64_t i = 0; i < R->n; i++){
		Z primary_key = R->keys[i];
		T score = 0;
		for(uint8_t j = 0; j < R->d; j++){ score+= R->scores[j*R->n + i]; }
		this->htR.emplace(primary_key,score);
	}

	//Probe phase
	for(uint64_t i =0; i< S->n; i++){
		Z id = S->ids[i];
		Z foreign_key = S->keys[i];
		auto range = this->htR.equal_range(foreign_key);
		if( range.first != range.second ){ // If probe match
			T score = 0;
			for(uint8_t j = 0; j < S->d; j++){ score+= S->scores[j*S->n + i]; }
			//TODO: Check if can score higher than threshold, break otherwise ?
			for(auto it = range.first; it != range.second; ++it){
				T combined_score = score + it->second;
				//std::cout << key << " = combined: " << score << "," << it->second << "," << combined_score << std::endl;
				this->tuple_count++;
				if(this->q[0].size() < k){
					this->q[0].push(_tuple<Z,T>(id,combined_score));
				}else if(this->q[0].top().score < combined_score){
					this->q[0].pop();
					this->q[0].push(_tuple<Z,T>(id,combined_score));
				}
			}
		}
	}
	this->t_join = this->t.lap();
}

template<class Z, class T>
void HJR<Z,T>::st_nop_hash_rank_join(){
	this->set_algo("st_nop_hash_rank_join");
	this->reset_metrics();
	this->reset_aux_struct();

	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();

	S_HashTable<Z,T> htR;
	htR.initialize(((R->n - 1) / S_HASHT_BUCKET_SIZE) + 1);

	this->t.start();
	htR.build_st(R);
	this->tuple_count = htR.probe_st(S,&this->q[0],k);
	this->t_join += this->t.lap();
}

template<class Z, class T>
void* HJR<Z,T>::mt_nop_thread(void *args)
{
	npo_args_t<Z,T> *a = (npo_args_t<Z,T>*) args;

//	std::string msg = "thread " + std::to_string(a->tid) + " says hello!\n";
//	std::cout << msg;
//	pthread_barrier_wait(a->barrier);
//	msg = "thread " + std::to_string(a->tid) + " passed the barrier!\n";
//	std::cout << msg;
//	a->q[0].push(_tuple<Z,T>(a->tid,13.5));

	uint32_t tid = a->tid;
	a->tt[tid].start();
	(*(a->htR)).build_mt(a->R,a->sR,a->eR);
	pthread_barrier_wait(a->barrier);
	//if(tid == 0){  a->ttuple_count[tid] = (*(a->htR)).probe_st(a->S,a->q,a->k); }
	//uint64_t count = (*(a->htR)).probe_mt(a->S,a->sS,a->eS,a->q,a->k);
	//std::string msg =  std::to_string(a->sS) + "," + std::to_string(a->eS) + "," + std::to_string(a->tid) + "," + std::to_string(count) + "\n";
	//std::cout << msg;
	a->ttuple_count[tid] = (*(a->htR)).probe_mt(a->S,a->sS,a->eS,a->q,a->k);
	a->tt_join[tid] = a->tt[tid].lap();
}

template<class Z, class T>
void HJR<Z,T>::mt_nop_hash_rank_join(){
	this->set_algo("mt_nop_hash_rank_join");
	this->reset_metrics();
	this->reset_aux_struct();

	TABLE<Z,T> *R = this->rj_inst->getR();
	TABLE<Z,T> *S = this->rj_inst->getS();
	Z k = this->rj_inst->getK();
	S_HashTable<Z,T> htR;
	htR.initialize(((R->n - 1) / S_HASHT_BUCKET_SIZE) + 1);

	pthread_t threads[THREADS];
	pthread_barrier_t barrier;
	npo_args_t<Z,T> args[THREADS];
	int  error;
	void *ret;

	pthread_barrier_init(&barrier, NULL, THREADS);
	for(uint64_t i = 0; i < THREADS; i++)
	{
		args[i].tid = i;
		args[i].sR = (i*R->n)/THREADS;
		args[i].eR = ((i+1)*R->n)/THREADS;
		args[i].sS = (i*S->n)/THREADS;
		args[i].eS = ((i+1)*S->n)/THREADS;
		args[i].R = R;
		args[i].S = S;
		args[i].barrier = &barrier;
		args[i].htR = &htR;
		args[i].q = &this->q[i];
		args[i].k = k;

		args[i].tt_init = this->tt_init;
		args[i].tt_join = this->tt_join;
		args[i].tt = this->tt;
		args[i].ttuple_count = this->ttuple_count;

		///std::cout << i << "," << args[i].sS << "," << args[i].eS << std::endl;

 		error = pthread_create(&threads[i], NULL, HJR<Z,T>::mt_nop_thread,(void*)&args[i]);
 		if( error != 0 ){
 			perror("Error creating thread");
 			exit(1);
 		}
	}

	for(uint32_t i = 0; i < THREADS; i++)
	{
		error = pthread_join(threads[i],&ret);
		if( error != 0 ){
			perror("Error joining thread");
			exit(1);
		}
	}
	pthread_barrier_destroy(&barrier);
	this->merge_metrics();
	this->merge_qs();
}




#endif
