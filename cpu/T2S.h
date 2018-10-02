#ifndef T2S_H
#define T2S_H

#include "AA.h"

template<class T,class Z>
struct t2s_pair{
	Z id;
	T score;
};

template<class T, class Z>
class T2S : public AA<T,Z>{
	public:
		T2S(uint64_t n, uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "T2S";
		};

		~T2S()
		{

		}

		void init();

	private:

};

template<class T, class Z>
void T2S<T,Z>::init()
{
	normalize_transpose<T,Z>(this->cdata, this->n, this->d);
}

#endif
