#ifndef TOOLS_H
#define TOOLS_H

#include <cstdint>

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
static bool cmp_gpta_pair_asc(const gpta_pair<T,Z> &a, const gpta_pair<T,Z> &b){ return a.score < b.score; };

template<class T,class Z>
static bool cmp_gpta_pair_desc(const gpta_pair<T,Z> &a, const gpta_pair<T,Z> &b){ return a.score > b.score; };

template<class Z>
static bool cmp_gpta_pos_asc(const gpta_pos<Z> &a, const gpta_pos<Z> &b){ return a.pos < b.pos; };


template<class T, class Z>
void psort(gpta_pair<T,Z> *tpairs,uint64_t n, bool ascending);

template<class Z>
void ppsort(gpta_pos<Z> *tpos, uint64_t n);

template <class T>
void normalize_transpose(T *&cdata, uint64_t n, uint64_t d);

#endif
