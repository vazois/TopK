#ifndef HJR_H
#define HJR_H

#include "ARJ.h"

template<class Z, class T>
class HJR : public AARankJoin<Z,T>{
	public:
		HJR(RankJoinInstance<Z,T> *rj_inst) : AARankJoin<Z,T>(rj_inst){ };
		~HJR(){};

		void join();
};


template<class Z, class T>
void HJR<Z,T>::join(){

}

#endif
