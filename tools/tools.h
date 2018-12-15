#ifndef TOOLS_H
#define TOOLS_H

#include <cstdint>
#include <mmintrin.h>//MMX
#include <xmmintrin.h>//SSE
#include <emmintrin.h>//SSE2
#include <immintrin.h>//AVX and AVX2 // AVX-512
#include <omp.h>
#include <queue>
#include <iostream>

#define _CMP_EQ_OQ    0x00 /* Equal (ordered, non-signaling)  */
#define _CMP_LT_OS    0x01 /* Less-than (ordered, signaling)  */
#define _CMP_LE_OS    0x02 /* Less-than-or-equal (ordered, signaling)  */
#define _CMP_UNORD_Q  0x03 /* Unordered (non-signaling)  */
#define _CMP_NEQ_UQ   0x04 /* Not-equal (unordered, non-signaling)  */
#define _CMP_NLT_US   0x05 /* Not-less-than (unordered, signaling)  */
#define _CMP_NLE_US   0x06 /* Not-less-than-or-equal (unordered, signaling)  */
#define _CMP_ORD_Q    0x07 /* Ordered (nonsignaling)   */
#define _CMP_EQ_UQ    0x08 /* Equal (unordered, non-signaling)  */
#define _CMP_NGE_US   0x09 /* Not-greater-than-or-equal (unord, signaling)  */
#define _CMP_NGT_US   0x0a /* Not-greater-than (unordered, signaling)  */
#define _CMP_FALSE_OQ 0x0b /* False (ordered, non-signaling)  */
#define _CMP_NEQ_OQ   0x0c /* Not-equal (ordered, non-signaling)  */
#define _CMP_GE_OS    0x0d /* Greater-than-or-equal (ordered, signaling)  */
#define _CMP_GT_OS    0x0e /* Greater-than (ordered, signaling)  */
#define _CMP_TRUE_UQ  0x0f /* True (unordered, non-signaling)  */
#define _CMP_EQ_OS    0x10 /* Equal (ordered, signaling)  */
#define _CMP_LT_OQ    0x11 /* Less-than (ordered, non-signaling)  */
#define _CMP_LE_OQ    0x12 /* Less-than-or-equal (ordered, non-signaling)  */
#define _CMP_UNORD_S  0x13 /* Unordered (signaling)  */
#define _CMP_NEQ_US   0x14 /* Not-equal (unordered, signaling)  */
#define _CMP_NLT_UQ   0x15 /* Not-less-than (unordered, non-signaling)  */
#define _CMP_NLE_UQ   0x16 /* Not-less-than-or-equal (unord, non-signaling)  */
#define _CMP_ORD_S    0x17 /* Ordered (signaling)  */
#define _CMP_EQ_US    0x18 /* Equal (unordered, signaling)  */
#define _CMP_NGE_UQ   0x19 /* Not-greater-than-or-equal (unord, non-sign)  */
#define _CMP_NGT_UQ   0x1a /* Not-greater-than (unordered, non-signaling)  */
#define _CMP_FALSE_OS 0x1b /* False (ordered, signaling)  */
#define _CMP_NEQ_OS   0x1c /* Not-equal (ordered, signaling)  */
#define _CMP_GE_OQ    0x1d /* Greater-than-or-equal (ordered, non-signaling)  */
#define _CMP_GT_OQ    0x1e /* Greater-than (ordered, non-signaling)  */
#define _CMP_TRUE_US  0x1f /* True (unordered, signaling)  */

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
		T findTopKgvta2(uint64_t k, uint8_t qq, T *weights, uint32_t *attr);
	private:
		gvta_block<T,Z> *blocks;
		uint64_t n, d;
		uint64_t bsize, pnum, nb;
};


#endif
