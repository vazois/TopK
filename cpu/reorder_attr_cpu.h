#ifndef REORDER_ATTR_CPU_H
#define REORDER_ATTR_CPU_H

//#define MAX(x,y) (x > y ? x : y)
//#define MIN(x,y) (x < y ? x : y)

template<class T> inline T MAX(T x, T y){ return(x > y ? x : y); }
template<class T> inline T MIN(T x, T y){ return(x < y ? x : y); }

#define DEBUG_TUPLE 0

template<class T>
inline void sort_4_attr(T *&data,uint64_t n){
	T a0 = data[0],a1 = data[n],a2 = data[2*n],a3 = data[3*n];
	T t0;

	t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;
	t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;

		//2
	t0 = MAX(a0,a2); a2 = MIN(a0,a2); a0 = t0;
	t0 = MAX(a1,a3); a3 = MIN(a1,a3); a1 = t0;

		//3
	t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;
	data[0] = a0; data[n] = a1; data[2*n] = a2; data[3*n] = a3;
}

template<class T>
inline void sort_6_attr(T *&data,uint64_t n){
	T a0 = data[0], a1 = data[n], a2 = data[2*n], a3 = data[3*n],a4 = data[4*n], a5 = data[5*n];
	T t0;

	t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;
	t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;
	t0 = MAX(a4,a5); a5 = MIN(a4,a5); a4 = t0;

	t0 = MAX(a0,a2); a2 = MIN(a0,a2); a0 = t0;
	t0 = MAX(a1,a4); a4 = MIN(a1,a4); a1 = t0;
	t0 = MAX(a3,a5); a5 = MIN(a3,a5); a3 = t0;

	t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;
	t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;
	t0 = MAX(a4,a5); a5 = MIN(a4,a5); a4 = t0;

	t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;
	t0 = MAX(a3,a4); a4 = MIN(a3,a4); a3 = t0;
	t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;

	data[0] = a0; data[n] = a1; data[2*n] = a2; data[3*n] = a3; data[4*n] = a4; data[5*n] = a5;
}

template<class T>
inline void sort_8_attr(T *&data,uint64_t n){
	T a0 = data[0], a1 = data[n], a2 = data[2*n], a3 = data[3*n];
	T a4 = data[4*n], a5 = data[5*n], a6 = data[6*n], a7 = data[7*n];
	T t0;

	t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;
	t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;
	t0 = MAX(a4,a5); a5 = MIN(a4,a5); a4 = t0;
	t0 = MAX(a6,a7); a7 = MIN(a6,a7); a6 = t0;

	t0 = MAX(a0,a2); a2 = MIN(a0,a2); a0 = t0;
	t0 = MAX(a1,a3); a3 = MIN(a1,a3); a1 = t0;
	t0 = MAX(a4,a6); a6 = MIN(a4,a6); a4 = t0;
	t0 = MAX(a5,a7); a7 = MIN(a5,a7); a5 = t0;

	t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;
	t0 = MAX(a5,a6); a6 = MIN(a5,a6); a5 = t0;

	t0 = MAX(a0,a4); a4 = MIN(a0,a4); a0 = t0;
	t0 = MAX(a1,a5); a5 = MIN(a1,a5); a1 = t0;
	t0 = MAX(a2,a6); a6 = MIN(a2,a6); a2 = t0;
	t0 = MAX(a3,a7); a7 = MIN(a3,a7); a3 = t0;

	t0 = MAX(a2,a4); a4 = MIN(a2,a4); a2 = t0;
	t0 = MAX(a3,a5); a5 = MIN(a3,a5); a3 = t0;

	t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;
	t0 = MAX(a3,a4); a4 = MIN(a3,a4); a3 = t0;
	t0 = MAX(a5,a6); a6 = MIN(a5,a6); a5 = t0;

	data[0] = a0; data[n] = a1; data[2*n] = a2; data[3*n] = a3;
	data[4*n] = a4; data[5*n] = a5; data[6*n] = a6; data[7*n] = a7;
}

template<class T>
inline void sort_10_attr(T *&data,uint64_t n){
	T a0 = data[0], a1 = data[n], a2 = data[2*n], a3 = data[3*n];
	T a4 = data[4*n], a5 = data[5*n], a6 = data[6*n], a7 = data[7*n];
	T a8 = data[8*n], a9 = data[9*n];
	T t0;

	t0 = MAX(a0,a5); a5 = MIN(a0,a5); a0 = t0;
	t0 = MAX(a1,a6); a6 = MIN(a1,a6); a1 = t0;
	t0 = MAX(a2,a7); a7 = MIN(a2,a7); a2 = t0;
	t0 = MAX(a3,a8); a8 = MIN(a3,a8); a3 = t0;
	t0 = MAX(a4,a9); a9 = MIN(a4,a9); a4 = t0;

	t0 = MAX(a0,a3); a3 = MIN(a0,a3); a0 = t0;
	t0 = MAX(a1,a4); a4 = MIN(a1,a4); a1 = t0;
	t0 = MAX(a5,a8); a8 = MIN(a5,a8); a5 = t0;
	t0 = MAX(a6,a9); a9 = MIN(a6,a9); a6 = t0;

	t0 = MAX(a0,a2); a2 = MIN(a0,a2); a0 = t0;
	t0 = MAX(a3,a6); a6 = MIN(a3,a6); a3 = t0;
	t0 = MAX(a7,a9); a9 = MIN(a7,a9); a7 = t0;

	t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;
	t0 = MAX(a2,a4); a4 = MIN(a2,a4); a2 = t0;
	t0 = MAX(a5,a7); a7 = MIN(a5,a7); a5 = t0;
	t0 = MAX(a8,a9); a9 = MIN(a8,a9); a8 = t0;

	t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;
	t0 = MAX(a3,a5); a5 = MIN(a3,a5); a3 = t0;
	t0 = MAX(a4,a6); a6 = MIN(a4,a6); a4 = t0;
	t0 = MAX(a7,a8); a8 = MIN(a7,a8); a7 = t0;

	t0 = MAX(a1,a3); a3 = MIN(a1,a3); a1 = t0;
	t0 = MAX(a4,a7); a7 = MIN(a4,a7); a4 = t0;
	t0 = MAX(a2,a5); a5 = MIN(a2,a5); a2 = t0;
	t0 = MAX(a6,a8); a8 = MIN(a6,a8); a6 = t0;

	t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;
	t0 = MAX(a4,a5); a5 = MIN(a4,a5); a4 = t0;
	t0 = MAX(a6,a7); a7 = MIN(a6,a7); a6 = t0;

	t0 = MAX(a3,a4); a4 = MIN(a3,a4); a3 = t0;
	t0 = MAX(a5,a6); a6 = MIN(a5,a6); a5 = t0;

	data[0] = a0; data[n] = a1; data[2*n] = a2; data[3*n] = a3;
	data[4*n] = a4; data[5*n] = a5; data[6*n] = a6; data[7*n] = a7;
	data[8*n] = a8; data[9*n] = a9;
}

template<class T>
inline void sort_12_attr(T *&data,uint64_t n){
	T a0 = data[0], a1 = data[n], a2 = data[2*n], a3 = data[3*n], a4 = data[4*n], a5 = data[5*n];
	T a6 = data[6*n], a7 = data[7*n], a8 = data[8*n], a9 = data[9*n], a10 = data[10*n], a11 = data[11*n];
	T t0;

	//1
	t0 = MAX(a0,a6); a6 = MIN(a0,a6); a0 = t0;
	t0 = MAX(a1,a7); a7 = MIN(a1,a7); a1 = t0;
	t0 = MAX(a2,a8); a8 = MIN(a2,a8); a2 = t0;
	t0 = MAX(a3,a9); a9 = MIN(a3,a9); a3 = t0;
	t0 = MAX(a4,a10); a10 = MIN(a4,a10); a4 = t0;
	t0 = MAX(a5,a11); a11 = MIN(a5,a11); a5 = t0;

	//2
	t0 = MAX(a0,a3); a3 = MIN(a0,a3); a0 = t0;
	t0 = MAX(a1,a4); a4 = MIN(a1,a4); a1 = t0;
	t0 = MAX(a2,a5); a5 = MIN(a2,a5); a2 = t0;
	t0 = MAX(a6,a9); a9 = MIN(a6,a9); a6 = t0;
	t0 = MAX(a7,a10); a10 = MIN(a7,a10); a7 = t0;
	t0 = MAX(a8,a11); a11 = MIN(a8,a11); a8 = t0;

	//3
	t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;
	t0 = MAX(a3,a4); a4 = MIN(a3,a4); a3 = t0;
	t0 = MAX(a5,a8); a8 = MIN(a5,a8); a5 = t0;
	t0 = MAX(a6,a7); a7 = MIN(a6,a7); a6 = t0;
	t0 = MAX(a10,a11); a11 = MIN(a10,a11); a10 = t0;

	//4
	t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;
	t0 = MAX(a3,a6); a6 = MIN(a3,a6); a3 = t0;
	t0 = MAX(a4,a5); a5 = MIN(a4,a5); a4 = t0;
	t0 = MAX(a7,a8); a8 = MIN(a7,a8); a7 = t0;
	t0 = MAX(a9,a10); a10 = MIN(a9,a10); a9 = t0;

	//5
	t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;
	t0 = MAX(a2,a9); a9 = MIN(a2,a9); a2 = t0;
	t0 = MAX(a3,a4); a4 = MIN(a3,a4); a3 = t0;
	t0 = MAX(a5,a8); a8 = MIN(a5,a8); a5 = t0;
	t0 = MAX(a6,a7); a7 = MIN(a6,a7); a6 = t0;
	t0 = MAX(a10,a11); a11 = MIN(a10,a11); a10 = t0;

	//6
	t0 = MAX(a1,a3); a3 = MIN(a1,a3); a1 = t0;
	t0 = MAX(a2,a6); a6 = MIN(a2,a6); a2 = t0;
	t0 = MAX(a4,a7); a7 = MIN(a4,a7); a4 = t0;
	t0 = MAX(a5,a9); a9 = MIN(a5,a9); a5 = t0;
	t0 = MAX(a8,a10); a10 = MIN(a8,a10); a8 = t0;

	//7
	t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;
	t0 = MAX(a4,a6); a6 = MIN(a4,a6); a4 = t0;
	t0 = MAX(a5,a7); a7 = MIN(a5,a7); a5 = t0;
	t0 = MAX(a8,a9); a9 = MIN(a8,a9); a8 = t0;

	//8
	t0 = MAX(a3,a4); a4 = MIN(a3,a4); a3 = t0;
	t0 = MAX(a5,a6); a6 = MIN(a5,a6); a5 = t0;
	t0 = MAX(a7,a8); a8 = MIN(a7,a8); a7 = t0;

	data[0] = a0; data[n] = a1; data[2*n] = a2; data[3*n] = a3; data[4*n] = a4; data[5*n] = a5;
	data[6*n] = a6; data[7*n] = a7; data[8*n] = a8; data[9*n] = a9; data[10*n] = a10; data[11*n] = a11;
}

template<class T>
inline void sort_14_attr(T *&data,uint64_t n){
	T a0 = data[0], a1 = data[n], a2 = data[2*n], a3 = data[3*n], a4 = data[4*n], a5 = data[5*n];
	T a6 = data[6*n], a7 = data[7*n], a8 = data[8*n], a9 = data[9*n], a10 = data[10*n], a11 = data[11*n];
	T a12 = data[12*n], a13 = data[13*n];
	T t0;




	data[0] = a0; data[n] = a1; data[2*n] = a2; data[3*n] = a3; data[4*n] = a4; data[5*n] = a5;
	data[6*n] = a6; data[7*n] = a7; data[8*n] = a8; data[9*n] = a9; data[10*n] = a10; data[11*n] = a11;
	data[12*n] = a12; data[13*n] = a13;
}

template<class T>
inline void sort_16_attr(T *&data,uint64_t n){
	T a0 = data[0], a1 = data[n], a2 = data[2*n], a3 = data[3*n], a4 = data[4*n], a5 = data[5*n];
	T a6 = data[6*n], a7 = data[7*n], a8 = data[8*n], a9 = data[9*n], a10 = data[10*n], a11 = data[11*n];
	T a12 = data[12*n], a13 = data[13*n], T a14 = data[14*n], a15 = data[15*n];
	T t0;




	data[0] = a0; data[n] = a1; data[2*n] = a2; data[3*n] = a3; data[4*n] = a4; data[5*n] = a5;
	data[6*n] = a6; data[7*n] = a7; data[8*n] = a8; data[9*n] = a9; data[10*n] = a10; data[11*n] = a11;
	data[12*n] = a12; data[13*n] = a13; data[14*n] = a14; data[15*n] = a15;
}

template<class T>
void reorder_attr_4(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T *offset = &data[i];
		sort_4_attr<T>(offset,n);
	}
}

template<class T>
void reorder_attr_6(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T *offset = &data[i];
		sort_6_attr<T>(offset,n);
	}
}

template<class T>
void reorder_attr_8(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T *offset = &data[i];
		sort_8_attr<T>(offset,n);
	}
}

template<class T>
void reorder_attr_10(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T *offset = &data[i];
		sort_10_attr<T>(offset,n);
	}
}

template<class T>
void reorder_attr_12(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T *offset = &data[i];
		sort_12_attr<T>(offset,n);
	}
}

template<class T>
void reorder_attr_14(T *&data,uint64_t n){

}

template<class T>
void reorder_attr_16(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T *offset = &data[i];
		sort_8_attr<T>(offset,n);
		offset = &data[i+8*n];
		sort_8_attr<T>(offset,n);
	}
}

#endif
