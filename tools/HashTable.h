#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include "../tools/Lock.h"
#include "../tools/RJTools.h"
#include <cstring>

#define S_HASHT_BUCKET_SIZE 2
#define SIMD_HASH_BUCKET_SIZE 4

template<class Z, class T>
struct pair_t
{
	Z key = 0;
	T value = 0;
};

template<class Z, class T>
struct bucket_t{
	lock_t lock = 0;/*1B*/
	uint8_t len = 0;/*1B*/
	pair_t<Z,T> pairs[S_HASHT_BUCKET_SIZE];/*8B*BUCKET_SIZE*//*32B*/
	bucket_t *next = NULL;
};

template<class Z, class T>
struct simd_bucket_t{
	lock_t lock = 0;/*1B*/
	uint8_t len = 0;/*1B*/
	Z keys[SIMD_HASH_BUCKET_SIZE];/*4B *SIMD_BUCKET_SIZE*/ /*8B *SIMD_BUCKET_SIZE*/
	T values[SIMD_HASH_BUCKET_SIZE];/*4B *SIMD_BUCKET_SIZE*/ /*8B *SIMD_BUCKET_SIZE*/
	struct simd_bucket_t *next;
};

template<class Z, class T>
class S_HashTable{
	public:
		S_HashTable(){ }
		~S_HashTable(){
			if(this->buckets != NULL) free(this->buckets);
		}

		void alloc(Z num_buckets);
		void build_st(TABLE<Z,T> *rel);
		uint64_t probe_st(TABLE<Z,T> *rel, std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>> *q,Z k);

		void build_mt(TABLE<Z,T> *rel, Z sRel, Z eRel);
		uint64_t probe_mt(TABLE<Z,T> *rel, Z sRel, Z eRel, std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>> *q,Z k);

	private:
		bucket_t<Z,T> *buckets = NULL;
		Z num_buckets = 0;
		Z mask = 0;
		Z bits = 0;

		inline Z __hash(Z key)__attribute__((always_inline)){ return (key & this->mask); }
		/*
		 * Bit Twiddling hacks (http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2)
		 */
		inline Z next_power_2(Z v)__attribute__((always_inline));
};

template<class Z, class T>
inline Z S_HashTable<Z,T>::next_power_2(Z v)
{
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		if( std::is_same<Z,uint64_t>::value ) v |= v >> 32;
		v++;
		return v;
}

template<class Z, class T>
void S_HashTable<Z,T>::alloc(Z num_buckets){
	this->num_buckets = this->next_power_2(num_buckets);
	//std::cout << "num_buckets: " <<num_buckets << "," << this->num_buckets << std::endl;
	this->buckets =  static_cast<bucket_t<Z,T>*>(aligned_alloc(CACHE_LINE_SIZE,sizeof(bucket_t<Z,T>)*this->num_buckets));
	std::memset(this->buckets,0,sizeof(bucket_t<Z,T>)*this->num_buckets);
	this->mask = (this->num_buckets - 1) >> this->bits;
}

template<class Z, class T>
void S_HashTable<Z,T>::build_st(TABLE<Z,T> *rel){

	for(uint64_t i = 0; i < rel->n; i++)
	{
		T score = 0;//Calculate score//
		for(uint8_t m =0; m < rel->d; m++) score+= rel->scores[m*rel->n + i];

		bucket_t<Z,T> *curr;
		Z key = rel->keys[i];
		Z idx = this->__hash(key);
		curr = this->buckets + idx;//Find bucket
		while(curr->len == S_HASHT_BUCKET_SIZE){//If bucket is full
			if(curr->next == NULL){//Check if next bucket has been initialized
				bucket_t<Z,T> *b = (bucket_t<Z,T>*) calloc(1, sizeof(bucket_t<Z,T>));//Create new bucket
				curr->next = b;//set it to next
			}
			curr = curr->next;//traverse buckets until space is found
		}

		curr->pairs[curr->len].key = key;
		curr->pairs[curr->len].value = score;
		curr->len++;
	}
}

template<class Z, class T>
uint64_t S_HashTable<Z,T>::probe_st(TABLE<Z,T> *rel, std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>> *q, Z k){
	uint64_t count = 0;
	for(uint64_t i = 0; i < rel->n; i++){
		T score = 0;//Calculate score//
		for(uint8_t m =0; m < rel->d; m++) score+= rel->scores[m*rel->n + i];

		bucket_t<Z,T> *curr;
		Z key = rel->keys[i];
		Z idx = this->__hash(key);
		curr = this->buckets + idx;
		do{
			for(uint8_t j = 0; j < curr->len; j++)
			{
				if(curr->pairs[j].key == key){
					T combined_score = score + curr->pairs[j].value;
					if(q[0].size() < k){
						q[0].push(_tuple<Z,T>(i,combined_score));
					}else if(q[0].top().score < combined_score){
						q[0].pop();
						q[0].push(_tuple<Z,T>(i,combined_score));
					}
					count++;
				}
			}
			curr = curr->next;
		}while(curr);
	}
	return count;
}

template<class Z, class T>
void S_HashTable<Z,T>::build_mt(TABLE<Z,T> *rel, Z sRel, Z eRel)
{
	for(uint64_t i = sRel; i < eRel; i++)
	{
		bucket_t<Z,T> *curr, *head;
		Z key = rel->keys[i];
		Z idx = this->__hash(key);

		T score = 0;
		for(uint8_t m =0; m < rel->d; m++) score+= rel->scores[m*rel->n + i];
		head = this->buckets + idx;
		curr = head;
		__lock(&head->lock);
		while(curr->len == S_HASHT_BUCKET_SIZE){//If bucket is full
			if(curr->next == NULL){//Check if next bucket has been initialized
				bucket_t<Z,T> *b = (bucket_t<Z,T>*) calloc(1, sizeof(bucket_t<Z,T>));//Create new bucket
				curr->next = b;//set it to next
			}
			curr = curr->next;//traverse buckets until space is found
		}

		curr->pairs[curr->len].key = key;
		curr->pairs[curr->len].value = score;
		curr->len++;
		__unlock(&head->lock);
	}
}

template<class Z, class T>
uint64_t S_HashTable<Z,T>::probe_mt(TABLE<Z,T> *rel, Z sRel, Z eRel, std::priority_queue<T, std::vector<_tuple<Z,T>>, pq_descending<Z,T>> *q,Z k)
{
	uint64_t count = 0;
	for(uint64_t i = sRel; i < eRel; i++)
	{
		T score = 0;
		for(uint8_t m =0; m < rel->d; m++) score+= rel->scores[m*rel->n + i];

		bucket_t<Z,T> *curr;
		Z key = rel->keys[i];
		Z idx = this->__hash(key);
		curr = this->buckets + idx;

		do{
			for(uint8_t j = 0; j < curr->len; j++)
			{
				if(curr->pairs[j].key == key){
					T combined_score = score + curr->pairs[j].value;
					if(q[0].size() < k){
						q[0].push(_tuple<Z,T>(i,combined_score));
					}else if(q[0].top().score < combined_score){
						q[0].pop();
						q[0].push(_tuple<Z,T>(i,combined_score));
					}
					count++;
				}
			}
			curr = curr->next;
		}while(curr);
	}
	return count;
}

template<class Z, class T>
class PB_HashTable{
	public:
		PB_HashTable(){}
		~PB_HashTable(){
			if(this->buckets != NULL) free(this->buckets);
		}

		void initialize(Z num_buckets);
	private:
		bucket_t<Z,T> *buckets = NULL;
		Z num_buckets = 0;
		Z mask = 0;
		Z bits = 0;

		inline Z __hash(Z key)__attribute__((always_inline)){ return (key & this->mask); }
};

template<class Z, class T>
void PB_HashTable<Z,T>::initialize(Z num_buckets)
{

}

#endif
