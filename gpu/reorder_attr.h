#ifndef REORDER_ATTR_H
#define REORDER_ATTR_H

/*
 * Implement Sorting Networks
 * Number of threads equal to number of tupples
 */

template<class T,uint32_t block>
__global__ void reorder_max_4_full(T *gdata, uint64_t n, uint64_t d){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	if ( offset < n ){
		T a0,a1,a2,a3;
		T t0;

		a0 = gdata[offset];//0.933451
		a1 = gdata[offset + n];//0.359598
		a2 = gdata[offset + 2*n];//0.198643
		a3 = gdata[offset + 3*n];//0.19599

		//1
		t0 = max(a0,a1);//0.933451
		a1 = min(a0,a1);//0.359598
		a0 = t0;
		t0 = max(a2,a3);//0.198643
		a3 = min(a2,a3);//0.19599
		a2 = t0;

		//2
		t0 = max(a0,a2);//0.933451
		a2 = min(a0,a2);//0.198643
		a0 = t0;
		t0 = max(a1,a3);//0.359598
		a3 = min(a1,a3);//0.19599
		a1 = t0;

		//3
		t0 = max(a1,a2);//0.359598
		a2 = min(a1,a2);//0.198643
		a1 = t0;

		gdata[offset] = a0;
		gdata[offset+n] = a1;
		gdata[offset+2*n] = a2;
		gdata[offset+3*n] = a3;

		//if(blockIdx.x == 0 && threadIdx.x ==0) printf("<%f,%f,%f,%f>\n",a0,a1,a2,a3);
	}
}

template<class T,uint32_t block>
__global__ void reorder_max_4_full_shared(T *gdata, uint64_t n, uint64_t d){
	__shared__ T attr[block][4];
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	if ( offset < n ){
		T a0,a1,a2,a3;
		attr[threadIdx.x][0] = gdata[offset];//0.933451
		attr[threadIdx.x][1] = gdata[offset + n];//0.359598
		attr[threadIdx.x][2] = gdata[offset + 2*n];//0.198643
		attr[threadIdx.x][3] = gdata[offset + 3*n];//0.19599

		a0 = max(attr[threadIdx.x][0],attr[threadIdx.x][1]);//0.933451
		a1 = min(attr[threadIdx.x][0],attr[threadIdx.x][1]);//0.359598
		a2 = max(attr[threadIdx.x][2],attr[threadIdx.x][3]);//0.198643
		a3 = min(attr[threadIdx.x][2],attr[threadIdx.x][3]);//0.19599

		attr[threadIdx.x][0] = a0;//0.933451
		attr[threadIdx.x][1] = a1;//0.359598
		attr[threadIdx.x][2] = a2;//0.198643
		attr[threadIdx.x][3] = a3;//0.19599

		a0 = max(attr[threadIdx.x][0],attr[threadIdx.x][2]);//0.933451
		a1 = max(attr[threadIdx.x][1],attr[threadIdx.x][3]);//0.359598
		a2 = min(attr[threadIdx.x][0],attr[threadIdx.x][2]);//0.198643
		a3 = min(attr[threadIdx.x][1],attr[threadIdx.x][3]);//0.19599

		attr[threadIdx.x][0] = a0;//0.933451
		attr[threadIdx.x][1] = a1;//0.359598
		attr[threadIdx.x][2] = a2;//0.198643
		attr[threadIdx.x][3] = a3;//0.19599

		a1 = max(attr[threadIdx.x][1],attr[threadIdx.x][2]);//0.359598
		a2 = min(attr[threadIdx.x][1],attr[threadIdx.x][2]);//0.198643
		//a3 = min(attr[threadIdx.x][1],attr[threadIdx.x][3]);

		//attr[threadIdx.x][0] = a0;//0.933451
		attr[threadIdx.x][1] = a1;//0.359598
		attr[threadIdx.x][2] = a2;//0.198643
		//attr[threadIdx.x][3] = a3;//0.19599

		gdata[offset] = attr[threadIdx.x][0];
		gdata[offset+n] = attr[threadIdx.x][1];
		gdata[offset+2*n] = attr[threadIdx.x][2];
		gdata[offset+3*n] = attr[threadIdx.x][3];

		//if(blockIdx.x == 0 && threadIdx.x ==0) printf("<%f,%f,%f,%f>\n",a0,a1,a2,a3);
	}
}

template<class T,uint32_t block>
__global__ void reorder_max_8_full(T *gdata, uint64_t n, uint64_t d){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	if ( offset < n ){
		T a0,a1,a2,a3,a4,a5,a6,a7;
		T t0;

		a0 = gdata[offset];
		a1 = gdata[offset + n];
		a2 = gdata[offset + 2*n];
		a3 = gdata[offset + 3*n];
		a4 = gdata[offset];
		a5 = gdata[offset + n];
		a6 = gdata[offset + 2*n];
		a7 = gdata[offset + 3*n];

		t0 = max(a0,a1); a1 = min(a0,a1); a0=t0;
		t0 = max(a2,a3); a3 = min(a2,a3); a2=t0;
		t0 = max(a4,a5); a5 = min(a4,a5); a4=t0;
		t0 = max(a6,a7); a7 = min(a6,a7); a6=t0;

	}
}

#endif
