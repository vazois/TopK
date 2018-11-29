#ifndef TOOLS_H
#define TOOLS_H

#include <cstdint>
#include <mmintrin.h>//MMX
#include <xmmintrin.h>//SSE
#include <emmintrin.h>//SSE2
#include <immintrin.h>//AVX and AVX2 // AVX-512
#include <omp.h>
#include <queue>

template<class T, class Z>
struct gpta_pair{
	Z id;
	T score;
};

template<class Z>
struct gpta_pos{
	Z id;
	Z pos;
};

template<class T,class Z>
struct gvta_block
{
	gvta_block() : data(NULL), tvector(NULL), num_tuples(0){}
	T *data;
	T *tvector;
	Z num_tuples;
};

template<class T,class Z>
struct tuple_{
	tuple_(){ tid = 0; score = 0; }
	tuple_(Z t, T s){ tid = t; score = s; }
	Z tid;
	T score;
};

template<class T,class Z>
class Desc{
	public:
		Desc(){};

		bool operator() (const tuple_<T,Z>& lhs, const tuple_<T,Z>& rhs) const{
			return (lhs.score>rhs.score);
		}
};

template<class T,class Z>
static bool cmp_gpta_pair_asc(const gpta_pair<T,Z> &a, const gpta_pair<T,Z> &b){ return a.score < b.score; };

template<class T,class Z>
static bool cmp_gpta_pair_desc(const gpta_pair<T,Z> &a, const gpta_pair<T,Z> &b){ return a.score > b.score; };

template<class Z>
static bool cmp_gpta_pos_asc(const gpta_pos<Z> &a, const gpta_pos<Z> &b){ return a.pos < b.pos; };

template<class T, class Z>
void pnth_element(gpta_pair<T,Z> *tscore, uint64_t n, uint64_t k, bool ascending);

template<class T, class Z>
void psort(gpta_pair<T,Z> *tpairs,uint64_t n, bool ascending);

template<class Z>
void ppsort(gpta_pos<Z> *tpos, uint64_t n);

template <class T>
void normalize_transpose(T *&cdata, uint64_t n, uint64_t d);

template<class T, class Z>
T findTopKtpac(T *cdata,Z n, Z d, uint64_t k,uint8_t qq, T *weights, uint32_t *attr);

template<class T, class Z>
class VAGG{
	public:
		VAGG<T,Z>(T *cdata, uint64_t n, uint64_t d)
		{
			this->cdata = cdata;
			this->n = n;
			this->d = d;
		};

		~VAGG<T,Z>(){

		};

		T findTopKtpac(uint64_t k,uint8_t qq, T *weights, uint32_t *attr);
		T findTopKgvta(gvta_block<T,Z> *blocks, Z n, Z d, uint64_t k,uint8_t qq, T *weights, uint32_t *attr, uint64_t nb);
	private:
		T *cdata = NULL;
		uint64_t n, d;
};

template<class T, class Z>
class GVAGG{
	public:
		GVAGG<T,Z>(gvta_block<T,Z> *blocks, uint64_t n, uint64_t d, uint64_t bsize, uint64_t pnum, uint64_t nb)
		{
			this->blocks = blocks;
			this->n = n;
			this->d = d;
			this->bsize = bsize;
			this->pnum = pnum;
			this->nb = nb;
		};

		T findTopKgvta(uint64_t k, uint8_t qq, T *weights, uint32_t *attr);
	private:
		gvta_block<T,Z> *blocks;
		uint64_t n, d;
		uint64_t bsize, pnum, nb;
};


#endif
