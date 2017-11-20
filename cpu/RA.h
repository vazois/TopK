#ifndef RA_H
#define RA_H

#include "cpu/FA.h"

template<class T>
class RA : public RA<T>{
	public:
		RA(Input<T>* input) : FA<T>(input){ this->algo = "FA"; };

		void findTopK(uint64_t k);
	protected:

};


template<class T>
void RA<T>::findTopK(uint64_t k){

}


#endif
