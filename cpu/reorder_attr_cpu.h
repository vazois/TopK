#ifndef REORDER_ATTR_CPU_H
#define REORDER_ATTR_CPU_H

//#define MAX(x,y) (x > y ? x : y)
//#define MIN(x,y) (x < y ? x : y)

template<class T> inline T MAX(T x, T y){ return(x > y ? x : y); }
template<class T> inline T MIN(T x, T y){ return(x < y ? x : y); }

#define DEBUG_TUPLE 0

template<class T>
void reorder_attr_4(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T a0 = data[i],a1 = data[i+n],a2 = data[i+2*n],a3 = data[i+3*n];
		T t0;

		t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;
		t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;

		//2
		t0 = MAX(a0,a2); a2 = MIN(a0,a2); a0 = t0;
		t0 = MAX(a1,a3); a3 = MIN(a1,a3); a1 = t0;

		//3
		t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;

		data[i] = a0; data[i+n] = a1; data[i+2*n] = a2; data[i+3*n] = a3;
	}
}

template<class T>
void reorder_attr_6(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T a0 = data[i], a1 = data[i+n], a2 = data[i+2*n], a3 = data[i+3*n],a4 = data[i+4*n], a5 = data[i+5*n];
		T t0;

		//if( i == 0 ) std::cout << "<sort 0>: " << a0 << " " << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5 << std::endl;
		//1
		t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;//0.537701 0.178567 0.76307 0.791738 0.840474 0.970125
		t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;
		t0 = MAX(a4,a5); a5 = MIN(a4,a5); a4 = t0;

		//if( i == 0 ) std::cout << "<sort 1>: " << a0 << " " << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5 << std::endl;
		//2
		t0 = MAX(a0,a2); a2 = MIN(a0,a2); a0 = t0;//0.537701 0.178567 0.791738 0.76307 0.970125 0.840474
		t0 = MAX(a1,a4); a4 = MIN(a1,a4); a1 = t0;
		t0 = MAX(a3,a5); a5 = MIN(a3,a5); a3 = t0;

		//if( i == 0 ) std::cout << "<sort 2>: " << a0 << " " << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5 << std::endl;
		//3
		t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;//0.791738 0.970125 0.537701 0.840474 0.178567 0.76307
		t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;
		t0 = MAX(a4,a5); a5 = MIN(a4,a5); a4 = t0;

		//if( i == 0 ) std::cout << "<sort 3>: " << a0 << " " << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5 << std::endl;
		//4
		t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;//0.970125 0.791738 0.840474 0.537701 0.76307 0.178567
		t0 = MAX(a3,a4); a4 = MIN(a3,a4); a3 = t0;
		t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;

		//if( i == 0 ) std::cout << "<sort 4>: " << a0 << " " << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5 << std::endl;
		//0.970125 0.840474 0.791738 0.76307 0.537701 0.178567
		data[i] = a0; data[i+n] = a1; data[i+2*n] = a2; data[i+3*n] = a3; data[i+4*n] = a4; data[i+5*n] = a5;
	}
}

template<class T>
void reorder_attr_8(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T a0 = data[i], a1 = data[i+n], a2 = data[i+2*n], a3 = data[i+3*n];
		T a4 = data[i+4*n], a5 = data[i+5*n], a6 = data[i+6*n], a7 = data[i+7*n];
		T t0;

		//1
		//if( i == DEBUG_TUPLE ) std::cout << "<sort 1>: " << a0 << " " << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5 << " " << a6 << " " << a7 << std::endl;
		t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;
		t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;
		t0 = MAX(a4,a5); a5 = MIN(a4,a5); a4 = t0;
		t0 = MAX(a6,a7); a7 = MIN(a6,a7); a6 = t0;

		//2
		//if( i == DEBUG_TUPLE ) std::cout << "<sort 2>: " << a0 << " " << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5 << " " << a6 << " " << a7 << std::endl;
		t0 = MAX(a0,a2); a2 = MIN(a0,a2); a0 = t0;
		t0 = MAX(a1,a3); a3 = MIN(a1,a3); a1 = t0;
		t0 = MAX(a4,a6); a6 = MIN(a4,a6); a4 = t0;
		t0 = MAX(a5,a7); a7 = MIN(a5,a7); a5 = t0;

		//3
		//if( i == DEBUG_TUPLE ) std::cout << "<sort 3>: " << a0 << " " << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5 << " " << a6 << " " << a7 << std::endl;
		t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;
		t0 = MAX(a5,a6); a6 = MIN(a5,a6); a5 = t0;

		//4
		//if( i == DEBUG_TUPLE ) std::cout << "<sort 4>: " << a0 << " " << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5 << " " << a6 << " " << a7 << std::endl;
		t0 = MAX(a0,a4); a4 = MIN(a0,a4); a0 = t0;
		t0 = MAX(a1,a5); a5 = MIN(a1,a5); a1 = t0;
		t0 = MAX(a2,a6); a6 = MIN(a2,a6); a2 = t0;
		t0 = MAX(a3,a7); a7 = MIN(a3,a7); a3 = t0;

		//5
		//if( i == DEBUG_TUPLE ) std::cout << "<sort 4>: " << a0 << " " << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5 << " " << a6 << " " << a7 << std::endl;
		t0 = MAX(a2,a4); a4 = MIN(a2,a4); a2 = t0;
		t0 = MAX(a3,a5); a5 = MIN(a3,a5); a3 = t0;

		//6
		//if( i == DEBUG_TUPLE ) std::cout << "<sort 5>: " << a0 << " " << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5 << " " << a6 << " " << a7 << std::endl;
		t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;
		t0 = MAX(a3,a4); a4 = MIN(a3,a4); a3 = t0;
		t0 = MAX(a5,a6); a6 = MIN(a5,a6); a5 = t0;

		if( i == DEBUG_TUPLE )
			std::cout << "<sort 6>: " << a0 << " " << a1 << " " << a2 << " " << a3
			<< " " << a4 << " " << a5 << " " << a6 << " " << a7 << std::endl;
		data[i] = a0; data[i+n] = a1; data[i+2*n] = a2; data[i+3*n] = a3;
		data[i+4*n] = a4; data[i+5*n] = a5; data[i+6*n] = a6; data[i+7*n] = a7;
	}
}

template<class T>
void reorder_attr_10(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T a0 = data[i], a1 = data[i+n], a2 = data[i+2*n], a3 = data[i+3*n];
		T a4 = data[i+4*n], a5 = data[i+5*n], a6 = data[i+6*n], a7 = data[i+7*n];
		T a8 = data[i+8*n], a9 = data[i+9*n];
		T t0;

		//1
//		if( i == DEBUG_TUPLE )
//			std::cout << "<sort 1>: " << a0 << " " << a1 << " " << a2 << " " << a3
//				<< " " << a4 << " " << a5 << " " << a6 << " " << a7 << " " << a8 << " " << a9 << std::endl;
		t0 = MAX(a0,a5); a5 = MIN(a0,a5); a0 = t0;
		t0 = MAX(a1,a6); a6 = MIN(a1,a6); a1 = t0;
		t0 = MAX(a2,a7); a7 = MIN(a2,a7); a2 = t0;
		t0 = MAX(a3,a8); a8 = MIN(a3,a8); a3 = t0;
		t0 = MAX(a4,a9); a9 = MIN(a4,a9); a4 = t0;

		//2
//		if( i == DEBUG_TUPLE )
//			std::cout << "<sort 2>: " << a0 << " " << a1 << " " << a2 << " " << a3
//				<< " " << a4 << " " << a5 << " " << a6 << " " << a7 << " " << a8 << " " << a9 << std::endl;
		t0 = MAX(a0,a3); a3 = MIN(a0,a3); a0 = t0;
		t0 = MAX(a1,a4); a4 = MIN(a1,a4); a1 = t0;
		t0 = MAX(a5,a8); a8 = MIN(a5,a8); a5 = t0;
		t0 = MAX(a6,a9); a9 = MIN(a6,a9); a6 = t0;

		//3
//		if( i == DEBUG_TUPLE )
//			std::cout << "<sort 3>: " << a0 << " " << a1 << " " << a2 << " " << a3
//				<< " " << a4 << " " << a5 << " " << a6 << " " << a7 << " " << a8 << " " << a9 << std::endl;
		t0 = MAX(a0,a2); a2 = MIN(a0,a2); a0 = t0;
		t0 = MAX(a3,a6); a6 = MIN(a3,a6); a3 = t0;
		t0 = MAX(a7,a9); a9 = MIN(a7,a9); a7 = t0;

		//4
//		if( i == DEBUG_TUPLE )
//			std::cout << "<sort 4>: " << a0 << " " << a1 << " " << a2 << " " << a3
//				<< " " << a4 << " " << a5 << " " << a6 << " " << a7 << " " << a8 << " " << a9 << std::endl;
		t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;
		t0 = MAX(a2,a4); a4 = MIN(a2,a4); a2 = t0;
		t0 = MAX(a5,a7); a7 = MIN(a5,a7); a5 = t0;
		t0 = MAX(a8,a9); a9 = MIN(a8,a9); a8 = t0;

		//5
//		if( i == DEBUG_TUPLE )
//			std::cout << "<sort 5>: " << a0 << " " << a1 << " " << a2 << " " << a3
//				<< " " << a4 << " " << a5 << " " << a6 << " " << a7 << " " << a8 << " " << a9 << std::endl;
		t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;
		t0 = MAX(a3,a5); a5 = MIN(a3,a5); a3 = t0;
		t0 = MAX(a4,a6); a6 = MIN(a4,a6); a4 = t0;
		t0 = MAX(a7,a8); a8 = MIN(a7,a8); a7 = t0;

		//6
//		if( i == DEBUG_TUPLE )
//			std::cout << "<sort 6>: " << a0 << " " << a1 << " " << a2 << " " << a3
//				<< " " << a4 << " " << a5 << " " << a6 << " " << a7 << " " << a8 << " " << a9 << std::endl;
		t0 = MAX(a1,a3); a3 = MIN(a1,a3); a1 = t0;
		t0 = MAX(a4,a7); a7 = MIN(a4,a7); a4 = t0;
		t0 = MAX(a2,a5); a5 = MIN(a2,a5); a2 = t0;
		t0 = MAX(a6,a8); a8 = MIN(a6,a8); a6 = t0;

		//7
//		if( i == DEBUG_TUPLE )
//			std::cout << "<sort 7>: " << a0 << " " << a1 << " " << a2 << " " << a3
//				<< " " << a4 << " " << a5 << " " << a6 << " " << a7 << " " << a8 << " " << a9 << std::endl;
		t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;
		t0 = MAX(a4,a5); a5 = MIN(a4,a5); a4 = t0;
		t0 = MAX(a6,a7); a7 = MIN(a6,a7); a6 = t0;

		//8
		t0 = MAX(a3,a4); a4 = MIN(a3,a4); a3 = t0;
		t0 = MAX(a5,a6); a6 = MIN(a5,a6); a5 = t0;

		data[i] = a0; data[i+n] = a1; data[i+2*n] = a2; data[i+3*n] = a3;
		data[i+4*n] = a4; data[i+5*n] = a5; data[i+6*n] = a6; data[i+7*n] = a7;
		data[i+8*n] = a8; data[i+9*n] = a9;
	}
}

template<class T>
void reorder_attr_12(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T a0 = data[i], a1 = data[i+n], a2 = data[i+2*n], a3 = data[i+3*n];
		T a4 = data[i+4*n], a5 = data[i+5*n], a6 = data[i+6*n], a7 = data[i+7*n];
		T a8 = data[i+8*n], a9 = data[i+9*n], a10 = data[i+10*n], a11 = data[i+11*n];
		T t0;


	}
}

template<class T>
void reorder_attr_14(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T a0 = data[i], a1 = data[i+n], a2 = data[i+2*n], a3 = data[i+3*n];
		T a4 = data[i+4*n], a5 = data[i+5*n], a6 = data[i+6*n], a7 = data[i+7*n];
		T a8 = data[i+8*n], a9 = data[i+9*n], a10 = data[i+10*n], a11 = data[i+11*n];
		T a12 = data[i+12*n], a13 = data[i+13*n];
		T t0;


		//1
		t0 = MAX(a0,a5); a5 = MIN(a0,a5); a0 = t0;
	}
}

template<class T>
void reorder_attr_16(T *&data,uint64_t n){
	for(uint64_t i = 0;i < n;i++){
		T a0 = data[i], a1 = data[i+n], a2 = data[i+2*n], a3 = data[i+3*n];
		T a4 = data[i+4*n], a5 = data[i+5*n], a6 = data[i+6*n], a7 = data[i+7*n];
		T a8 = data[i+8*n], a9 = data[i+9*n], a10 = data[i+10*n], a11 = data[i+11*n];
		T a12 = data[i+12*n], a13 = data[i+13*n], a14 = data[i+14*n], a15 = data[i+15*n];
		T t0;

		//1
		t0 = MAX(a0,a1); a1 = MIN(a0,a1); a0 = t0;
		t0 = MAX(a2,a3); a3 = MIN(a2,a3); a2 = t0;
		t0 = MAX(a4,a5); a5 = MIN(a4,a5); a4 = t0;
		t0 = MAX(a6,a7); a7 = MIN(a6,a7); a6 = t0;
		t0 = MAX(a8,a9); a9 = MIN(a8,a9); a8 = t0;
		t0 = MAX(a10,a11); a11 = MIN(a10,a11); a10 = t0;
		t0 = MAX(a12,a13); a13 = MIN(a12,a13); a12 = t0;
		t0 = MAX(a14,a15); a15 = MIN(a14,a15); a14 = t0;

		//2
		t0 = MAX(a0,a2); a2 = MIN(a0,a2); a0 = t0;
		t0 = MAX(a1,a3); a3 = MIN(a1,a3); a1 = t0;
		t0 = MAX(a4,a6); a6 = MIN(a4,a6); a4 = t0;
		t0 = MAX(a5,a7); a7 = MIN(a5,a7); a5 = t0;
		t0 = MAX(a8,a10); a10 = MIN(a8,a10); a8 = t0;
		t0 = MAX(a9,a11); a11 = MIN(a9,a11); a9 = t0;
		t0 = MAX(a12,a14); a14 = MIN(a12,a14); a12 = t0;
		t0 = MAX(a13,a15); a15 = MIN(a13,a15); a13 = t0;

		//3
		t0 = MAX(a0,a4); a4 = MIN(a0,a4); a0 = t0;
		t0 = MAX(a1,a5); a5 = MIN(a1,a5); a1 = t0;
		t0 = MAX(a2,a6); a6 = MIN(a2,a6); a2 = t0;
		t0 = MAX(a3,a7); a7 = MIN(a3,a7); a3 = t0;
		t0 = MAX(a8,a12); a12 = MIN(a8,a12); a8 = t0;
		t0 = MAX(a9,a13); a13 = MIN(a9,a13); a9 = t0;
		t0 = MAX(a10,a14); a14 = MIN(a10,a14); a10 = t0;
		t0 = MAX(a11,a15); a15 = MIN(a11,a15); a11 = t0;

		//4
		t0 = MAX(a0,a8); a8 = MIN(a0,a8); a0 = t0;
		t0 = MAX(a1,a9); a9 = MIN(a1,a9); a1 = t0;
		t0 = MAX(a2,a10); a10 = MIN(a2,a10); a2 = t0;
		t0 = MAX(a3,a11); a11 = MIN(a3,a11); a3 = t0;
		t0 = MAX(a4,a12); a12 = MIN(a4,a12); a4 = t0;
		t0 = MAX(a5,a13); a13 = MIN(a5,a13); a5 = t0;
		t0 = MAX(a6,a14); a14 = MIN(a6,a14); a6 = t0;
		t0 = MAX(a7,a15); a15 = MIN(a7,a15); a7 = t0;

		//5
		t0 = MAX(a4,a8); a8 = MIN(a4,a8); a4 = t0;
		t0 = MAX(a5,a9); a9 = MIN(a5,a9); a5 = t0;
		t0 = MAX(a6,a10); a10 = MIN(a6,a10); a6 = t0;
		t0 = MAX(a7,a11); a11 = MIN(a7,a11); a7 = t0;

		//6
		t0 = MAX(a2,a8); a8 = MIN(a2,a8); a2 = t0;
		t0 = MAX(a3,a9); a9 = MIN(a3,a9); a3 = t0;
		t0 = MAX(a6,a12); a12 = MIN(a6,a12); a6 = t0;
		t0 = MAX(a7,a13); a13 = MIN(a7,a13); a7 = t0;

		//7
		t0 = MAX(a2,a4); a4 = MIN(a2,a4); a2 = t0;
		t0 = MAX(a3,a5); a5 = MIN(a3,a5); a3 = t0;
		t0 = MAX(a6,a8); a8 = MIN(a6,a8); a6 = t0;
		t0 = MAX(a7,a9); a9 = MIN(a7,a9); a7 = t0;
		t0 = MAX(a10,a12); a12 = MIN(a10,a12); a10 = t0;
		t0 = MAX(a11,a13); a13 = MIN(a11,a13); a11 = t0;

		//8
		t0 = MAX(a1,a8); a8 = MIN(a1,a8); a1 = t0;
		t0 = MAX(a3,a10); a10 = MIN(a3,a10); a3 = t0;
		t0 = MAX(a5,a12); a12 = MIN(a5,a12); a5 = t0;
		t0 = MAX(a7,a14); a14 = MIN(a7,a14); a7 = t0;

		//9
		t0 = MAX(a1,a4); a4 = MIN(a1,a4); a1 = t0;
		t0 = MAX(a5,a8); a8 = MIN(a5,a8); a5 = t0;
		t0 = MAX(a9,a12); a12 = MIN(a9,a12); a9 = t0;
		t0 = MAX(a3,a6); a6 = MIN(a3,a6); a3 = t0;
		t0 = MAX(a7,a10); a10 = MIN(a7,a10); a7 = t0;
		t0 = MAX(a11,a4); a4 = MIN(a11,a4); a11 = t0;

		//10
		t0 = MAX(a1,a2); a2 = MIN(a1,a2); a1 = t0;
		t0 = MAX(a3,a4); a4 = MIN(a3,a4); a3 = t0;
		t0 = MAX(a5,a6); a6 = MIN(a5,a6); a5 = t0;
		t0 = MAX(a7,a8); a8 = MIN(a7,a8); a7 = t0;
		t0 = MAX(a9,a10); a10 = MIN(a9,a10); a9 = t0;
		t0 = MAX(a11,a12); a12 = MIN(a11,a12); a11 = t0;
		t0 = MAX(a13,a14); a14 = MIN(a13,a14); a13 = t0;


		data[i] = a0; data[i+n] = a1; data[i+2*n] = a2; data[i+3*n] = a3;
		data[i+4*n] = a4; data[i+5*n] = a5; data[i+6*n] = a6; data[i+7*n] = a7;
		data[i+8*n] = a8; data[i+9*n] = a9; data[i+10*n] = a10; data[i+11*n] = a11;
		data[i+12*n] = a12; data[i+13*n] = a13; data[i+14*n] = a14; data[i+15*n] = a15;

	}
}

#endif
