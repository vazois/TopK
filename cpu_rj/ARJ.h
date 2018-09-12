#ifndef ARJ_H
#define ARJ_H

#include<limits>
#include <stdio.h>
#include <cstdint>
#include <stdio.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <iostream>

#include <chrono>
#include <random>

#define MX_INT 1024*128

template<class Z, class T>
struct TABLE{
	Z c;//cardinality
	Z d;//dimensionality
	Z *ids;//tuple ids
	T *scores;//tuple scores
};

template<class Z, class T>
class GenData{
	public:
		GenData()
		: def_gen(std::chrono::system_clock::now().time_since_epoch().count()), int_d(1,MX_INT), real_d(0,1)
		{
		};
		~GenData(){
			
		};

		void test(){
			for(uint32_t i = 0;i < 5; i++){
				std::cout << "def_gen_int_uniform: " << (uint32_t)this->int_d((this->def_gen)) << std::endl;
				std::cout << "def_gen_real_uniform: " << (float)this->real_d(def_gen) << std::endl;
			}
		}
	private:
		unsigned seed = 0;
		std::uniform_int_distribution<Z> int_d;
		std::uniform_real_distribution<T> real_d;
		std::default_random_engine def_gen;
		
};

#endif
