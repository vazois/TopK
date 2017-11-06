#ifndef FA_H
#define FA_H

#include "AA.h"

template<class T>
class FA : public AA<T>{
	public:
		using AA<T>::AA;
		void findTopK();
};


template<class T>
void FA<T>::findTopK(){
	std::cout << "Computing TopK with TA Algorithm!!!"<< std::endl;

}

#endif

