#ifndef REORDER_ATTR_H
#define REORDER_ATTR_H

/*
 * Implement Sorting Networks
 * Number of threads equal to number of tupples
 */

template<class T> inline __device__ T _MAX_(T x, T y){ return x > y ? x : y; }
template<class T> inline __device__ T _MIN_(T x, T y){ return x < y ? x : y; }

template<class T,uint32_t block>
__global__ void reorder_max_2_full(T *gdata, uint64_t n, uint64_t d){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	if ( offset < n ){
		T a0,a1;
		T t0;

		a0 = gdata[offset]; a1 = gdata[offset + n];
		t0 = max(a0,a1); a1 = min(a0,a1); a0 = t0;//1
		gdata[offset] = a0; gdata[offset+n] = a1;
	}
}

template<class T,uint32_t block>
__global__ void reorder_max_4_full(T *gdata, uint64_t n, uint64_t d){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	if ( offset < n ){
		T a0,a1,a2,a3;
		T t0;

		a0 = gdata[offset]; a1 = gdata[offset + n]; a2 = gdata[offset + 2*n]; a3 = gdata[offset + 3*n];

		t0 = max(a0,a1); a1 = min(a0,a1); a0 = t0;//1
		t0 = max(a2,a3); a3 = min(a2,a3); a2 = t0;

		t0 = max(a0,a2); a2 = min(a0,a2); a0 = t0;//2
		t0 = max(a1,a3); a3 = min(a1,a3); a1 = t0;

		t0 = max(a1,a2); a2 = min(a1,a2); a1 = t0;//3

		gdata[offset] = a0; gdata[offset+n] = a1; gdata[offset+2*n] = a2; gdata[offset+3*n] = a3;
	}
}

template<class T,uint32_t block>
__global__ void reorder_max_6_full(T *gdata, uint64_t n, uint64_t d){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	if ( offset < n ){
		T a0 = gdata[offset], a1 = gdata[offset+n], a2 = gdata[offset+2*n], a3 = gdata[offset+3*n],a4 = gdata[offset+4*n], a5 = gdata[offset+5*n];
		T t0;

		t0 = max(a0,a1); a1 = min(a0,a1); a0 = t0;
		t0 = max(a2,a3); a3 = min(a2,a3); a2 = t0;
		t0 = max(a4,a5); a5 = min(a4,a5); a4 = t0;

		t0 = max(a0,a2); a2 = min(a0,a2); a0 = t0;
		t0 = max(a1,a4); a4 = min(a1,a4); a1 = t0;
		t0 = max(a3,a5); a5 = min(a3,a5); a3 = t0;

		t0 = max(a0,a1); a1 = min(a0,a1); a0 = t0;
		t0 = max(a2,a3); a3 = min(a2,a3); a2 = t0;
		t0 = max(a4,a5); a5 = min(a4,a5); a4 = t0;

		t0 = max(a1,a2); a2 = min(a1,a2); a1 = t0;
		t0 = max(a3,a4); a4 = min(a3,a4); a3 = t0;
		t0 = max(a2,a3); a3 = min(a2,a3); a2 = t0;

		gdata[offset] = a0; gdata[offset+n] = a1; gdata[offset+2*n] = a2; gdata[offset+3*n] = a3; gdata[offset+4*n] = a4; gdata[offset+5*n] = a5;
	}
}

template<class T,uint32_t block>
__global__ void reorder_max_8_full(T *gdata, uint64_t n, uint64_t d){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	if ( offset < n ){
		T a0 = gdata[offset], a1 = gdata[offset+n], a2 = gdata[offset+2*n], a3 = gdata[offset+3*n];
		T a4 = gdata[offset+4*n], a5 = gdata[offset+5*n], a6 = gdata[offset+6*n], a7 = gdata[offset+7*n];
		T t0;

		t0 = max(a0,a1); a1 = min(a0,a1); a0 = t0;
		t0 = max(a2,a3); a3 = min(a2,a3); a2 = t0;
		t0 = max(a4,a5); a5 = min(a4,a5); a4 = t0;
		t0 = max(a6,a7); a7 = min(a6,a7); a6 = t0;

		t0 = max(a0,a2); a2 = min(a0,a2); a0 = t0;
		t0 = max(a1,a3); a3 = min(a1,a3); a1 = t0;
		t0 = max(a4,a6); a6 = min(a4,a6); a4 = t0;
		t0 = max(a5,a7); a7 = min(a5,a7); a5 = t0;

		t0 = max(a1,a2); a2 = min(a1,a2); a1 = t0;
		t0 = max(a5,a6); a6 = min(a5,a6); a5 = t0;

		t0 = max(a0,a4); a4 = min(a0,a4); a0 = t0;
		t0 = max(a1,a5); a5 = min(a1,a5); a1 = t0;
		t0 = max(a2,a6); a6 = min(a2,a6); a2 = t0;
		t0 = max(a3,a7); a7 = min(a3,a7); a3 = t0;

		t0 = max(a2,a4); a4 = min(a2,a4); a2 = t0;
		t0 = max(a3,a5); a5 = min(a3,a5); a3 = t0;

		t0 = max(a1,a2); a2 = min(a1,a2); a1 = t0;
		t0 = max(a3,a4); a4 = min(a3,a4); a3 = t0;
		t0 = max(a5,a6); a6 = min(a5,a6); a5 = t0;

		gdata[offset] = a0; gdata[offset+n] = a1; gdata[offset+2*n] = a2; gdata[offset+3*n] = a3;
		gdata[offset+4*n] = a4; gdata[offset+5*n] = a5; gdata[offset+6*n] = a6; gdata[offset+7*n] = a7;
	}
}

template<class T,uint32_t block>
__global__ void reorder_max_10_full(T *gdata, uint64_t n, uint64_t d){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	if ( offset < n ){
		T a0 = gdata[offset], a1 = gdata[offset+n], a2 = gdata[offset+2*n], a3 = gdata[offset+3*n];
		T a4 = gdata[offset+4*n], a5 = gdata[offset+5*n], a6 = gdata[offset+6*n], a7 = gdata[offset+7*n];
		T a8 = gdata[offset+8*n], a9 = gdata[offset+9*n];
		T t0;

		t0 = max(a0,a5); a5 = min(a0,a5); a0 = t0;
		t0 = max(a1,a6); a6 = min(a1,a6); a1 = t0;
		t0 = max(a2,a7); a7 = min(a2,a7); a2 = t0;
		t0 = max(a3,a8); a8 = min(a3,a8); a3 = t0;
		t0 = max(a4,a9); a9 = min(a4,a9); a4 = t0;

		t0 = max(a0,a3); a3 = min(a0,a3); a0 = t0;
		t0 = max(a1,a4); a4 = min(a1,a4); a1 = t0;
		t0 = max(a5,a8); a8 = min(a5,a8); a5 = t0;
		t0 = max(a6,a9); a9 = min(a6,a9); a6 = t0;

		t0 = max(a0,a2); a2 = min(a0,a2); a0 = t0;
		t0 = max(a3,a6); a6 = min(a3,a6); a3 = t0;
		t0 = max(a7,a9); a9 = min(a7,a9); a7 = t0;

		t0 = max(a0,a1); a1 = min(a0,a1); a0 = t0;
		t0 = max(a2,a4); a4 = min(a2,a4); a2 = t0;
		t0 = max(a5,a7); a7 = min(a5,a7); a5 = t0;
		t0 = max(a8,a9); a9 = min(a8,a9); a8 = t0;

		t0 = max(a1,a2); a2 = min(a1,a2); a1 = t0;
		t0 = max(a3,a5); a5 = min(a3,a5); a3 = t0;
		t0 = max(a4,a6); a6 = min(a4,a6); a4 = t0;
		t0 = max(a7,a8); a8 = min(a7,a8); a7 = t0;

		t0 = max(a1,a3); a3 = min(a1,a3); a1 = t0;
		t0 = max(a4,a7); a7 = min(a4,a7); a4 = t0;
		t0 = max(a2,a5); a5 = min(a2,a5); a2 = t0;
		t0 = max(a6,a8); a8 = min(a6,a8); a6 = t0;

		t0 = max(a2,a3); a3 = min(a2,a3); a2 = t0;
		t0 = max(a4,a5); a5 = min(a4,a5); a4 = t0;
		t0 = max(a6,a7); a7 = min(a6,a7); a6 = t0;

		t0 = max(a3,a4); a4 = min(a3,a4); a3 = t0;
		t0 = max(a5,a6); a6 = min(a5,a6); a5 = t0;

		gdata[offset] = a0; gdata[offset+n] = a1; gdata[offset+2*n] = a2; gdata[offset+3*n] = a3;
		gdata[offset+4*n] = a4; gdata[offset+5*n] = a5; gdata[offset+6*n] = a6; gdata[offset+7*n] = a7;
		gdata[offset+8*n] = a8; gdata[offset+9*n] = a9;
	}
}

template<class T,uint32_t block>
__global__ void reorder_max_12_full(T *gdata, uint64_t n, uint64_t d){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	if ( offset < n ){
		T a0 = gdata[offset], a1 = gdata[offset+n], a2 = gdata[offset+2*n], a3 = gdata[offset+3*n], a4 = gdata[offset+4*n], a5 = gdata[offset+5*n];
		T a6 = gdata[offset+6*n], a7 = gdata[offset+7*n], a8 = gdata[offset+8*n], a9 = gdata[offset+9*n], a10 = gdata[offset+10*n], a11 = gdata[offset+11*n];
		T t0;

		//1
		t0 = max(a0,a6); a6 = min(a0,a6); a0 = t0;
		t0 = max(a1,a7); a7 = min(a1,a7); a1 = t0;
		t0 = max(a2,a8); a8 = min(a2,a8); a2 = t0;
		t0 = max(a3,a9); a9 = min(a3,a9); a3 = t0;
		t0 = max(a4,a10); a10 = min(a4,a10); a4 = t0;
		t0 = max(a5,a11); a11 = min(a5,a11); a5 = t0;

		//2
		t0 = max(a0,a3); a3 = min(a0,a3); a0 = t0;
		t0 = max(a1,a4); a4 = min(a1,a4); a1 = t0;
		t0 = max(a2,a5); a5 = min(a2,a5); a2 = t0;
		t0 = max(a6,a9); a9 = min(a6,a9); a6 = t0;
		t0 = max(a7,a10); a10 = min(a7,a10); a7 = t0;
		t0 = max(a8,a11); a11 = min(a8,a11); a8 = t0;

		//3
		t0 = max(a0,a1); a1 = min(a0,a1); a0 = t0;
		t0 = max(a3,a4); a4 = min(a3,a4); a3 = t0;
		t0 = max(a5,a8); a8 = min(a5,a8); a5 = t0;
		t0 = max(a6,a7); a7 = min(a6,a7); a6 = t0;
		t0 = max(a10,a11); a11 = min(a10,a11); a10 = t0;

		//4
		t0 = max(a1,a2); a2 = min(a1,a2); a1 = t0;
		t0 = max(a3,a6); a6 = min(a3,a6); a3 = t0;
		t0 = max(a4,a5); a5 = min(a4,a5); a4 = t0;
		t0 = max(a7,a8); a8 = min(a7,a8); a7 = t0;
		t0 = max(a9,a10); a10 = min(a9,a10); a9 = t0;

		//5
		t0 = max(a0,a1); a1 = min(a0,a1); a0 = t0;
		t0 = max(a2,a9); a9 = min(a2,a9); a2 = t0;
		t0 = max(a3,a4); a4 = min(a3,a4); a3 = t0;
		t0 = max(a5,a8); a8 = min(a5,a8); a5 = t0;
		t0 = max(a6,a7); a7 = min(a6,a7); a6 = t0;
		t0 = max(a10,a11); a11 = min(a10,a11); a10 = t0;

		//6
		t0 = max(a1,a3); a3 = min(a1,a3); a1 = t0;
		t0 = max(a2,a6); a6 = min(a2,a6); a2 = t0;
		t0 = max(a4,a7); a7 = min(a4,a7); a4 = t0;
		t0 = max(a5,a9); a9 = min(a5,a9); a5 = t0;
		t0 = max(a8,a10); a10 = min(a8,a10); a8 = t0;

		//7
		t0 = max(a2,a3); a3 = min(a2,a3); a2 = t0;
		t0 = max(a4,a6); a6 = min(a4,a6); a4 = t0;
		t0 = max(a5,a7); a7 = min(a5,a7); a5 = t0;
		t0 = max(a8,a9); a9 = min(a8,a9); a8 = t0;

		//8
		t0 = max(a3,a4); a4 = min(a3,a4); a3 = t0;
		t0 = max(a5,a6); a6 = min(a5,a6); a5 = t0;
		t0 = max(a7,a8); a8 = min(a7,a8); a7 = t0;

		gdata[offset] = a0; gdata[offset+n] = a1; gdata[offset+2*n] = a2; gdata[offset+3*n] = a3; gdata[offset+4*n] = a4; gdata[offset+5*n] = a5;
		gdata[offset+6*n] = a6; gdata[offset+7*n] = a7; gdata[offset+8*n] = a8; gdata[offset+9*n] = a9; gdata[offset+10*n] = a10; gdata[offset+11*n] = a11;
	}
}
#endif
