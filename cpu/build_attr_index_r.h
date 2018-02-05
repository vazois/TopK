#ifndef BUILD_ATTR_INDEX_R_H
#define BUILD_ATTR_INDEX_R_H

template<class Z>
struct O{
	O(){ tt = 0; ii = 0;}
	uint8_t  ii;
	Z tt;
};

template<class Z>
static inline bool cmp_pos_oo(const O<Z> &a, const O<Z> &b){ return a.tt < b.tt; };


template<class Z>
void build_attr_index_r(uint64_t *II, Z *TT, uint64_t n,uint64_t d){
	O<Z> *oo = (O<Z>*)malloc(sizeof(O<Z>)*d);
	for(uint64_t i = 0;i < n;i++){
		for(uint8_t m = 0; m < d;m++){
			oo[m].ii = m; oo[m].tt = TT[i*d + m];
		}
		std::sort(oo,oo + d,cmp_pos_oo<Z>);
		II[i] = 0;
		for(uint8_t m = 0; m < d;m++){
			II[i] = II[i] | (oo[m].ii << (m*4));
		}
	}
	free(oo);
}

template<class Z>
void build_attr_index_r_4(uint64_t *II, Z *TT, uint64_t n,uint64_t d){
	O<Z> oo[4];
	for(uint64_t i = 0;i < n;i++){
		oo[0].ii = 0; oo[0].tt = TT[i*d];
		oo[1].ii = 1; oo[1].tt = TT[i*d + 1];
		oo[2].ii = 2; oo[2].tt = TT[i*d + 2];
		oo[3].ii = 3; oo[3].tt = TT[i*d + 3];

		std::sort(oo,oo + d,cmp_pos_oo<Z>);

		II[i] = 0;
		II[i] = II[i] | (oo[0].ii);
		II[i] = II[i] | (oo[1].ii << 4);
		II[i] = II[i] | (oo[2].ii << 8);
		II[i] = II[i] | (oo[3].ii << 12);
 	}
}

template<class Z>
void build_attr_index_r_6(uint64_t *II, Z *TT, uint64_t n,uint64_t d){
	O<Z> oo[6];
	for(uint64_t i = 0;i < n;i++){
		oo[0].ii = 0; oo[0].tt = TT[i*d];
		oo[1].ii = 1; oo[1].tt = TT[i*d + 1];
		oo[2].ii = 2; oo[2].tt = TT[i*d + 2];
		oo[3].ii = 3; oo[3].tt = TT[i*d + 3];
		oo[4].ii = 4; oo[4].tt = TT[i*d + 4];
		oo[5].ii = 5; oo[5].tt = TT[i*d + 5];

		std::sort(oo,oo + d,cmp_pos_oo<Z>);

		II[i] = 0;
		II[i] = II[i] | (oo[0].ii);
		II[i] = II[i] | (oo[1].ii << 4);
		II[i] = II[i] | (oo[2].ii << 8);
		II[i] = II[i] | (oo[3].ii << 12);
		II[i] = II[i] | (oo[4].ii << 16);
		II[i] = II[i] | (oo[5].ii << 20);
 	}
}

template<class Z>
void build_attr_index_r_8(uint64_t *II, Z *TT, uint64_t n,uint64_t d){
	O<Z> oo[8];
	for(uint64_t i = 0;i < n;i++){
		oo[0].ii = 0; oo[0].tt = TT[i*d];
		oo[1].ii = 1; oo[1].tt = TT[i*d + 1];
		oo[2].ii = 2; oo[2].tt = TT[i*d + 2];
		oo[3].ii = 3; oo[3].tt = TT[i*d + 3];
		oo[4].ii = 4; oo[4].tt = TT[i*d + 4];
		oo[5].ii = 5; oo[5].tt = TT[i*d + 5];
		oo[6].ii = 6; oo[6].tt = TT[i*d + 6];
		oo[7].ii = 7; oo[7].tt = TT[i*d + 7];

		std::sort(oo,oo + d,cmp_pos_oo<Z>);

		II[i] = 0;
		II[i] = II[i] | (oo[0].ii);
		II[i] = II[i] | (oo[1].ii << 4);
		II[i] = II[i] | (oo[2].ii << 8);
		II[i] = II[i] | (oo[3].ii << 12);
		II[i] = II[i] | (oo[4].ii << 16);
		II[i] = II[i] | (oo[5].ii << 20);
		II[i] = II[i] | (oo[6].ii << 24);
		II[i] = II[i] | (oo[7].ii << 28);
 	}
}

#endif
