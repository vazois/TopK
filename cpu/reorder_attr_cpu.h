#ifndef REORDER_ATTR_CPU_H
#define REORDER_ATTR_CPU_H

#define MAX(x,y) (x > y ? x : y)
#define MIN(x,y) (x < y ? x : y)

template<class T>
void reorder_attr_4(T *&data,uint64_t n){

	for(uint64_t i = 0;i < n;i++){
		T a0 = data[i];
		T a1 = data[i+n];
		T a2 = data[i+2*n];
		T a3 = data[i+3*n];
		T t0;

		t0 = MAX(a0,a1);//0.933451
		a1 = MIN(a0,a1);//0.359598
		a0 = t0;
		t0 = MAX(a2,a3);//0.198643
		a3 = MIN(a2,a3);//0.19599
		a2 = t0;

		//2
		t0 = MAX(a0,a2);//0.933451
		a2 = MIN(a0,a2);//0.198643
		a0 = t0;
		t0 = MAX(a1,a3);//0.359598
		a3 = MIN(a1,a3);//0.19599
		a1 = t0;

		//3
		t0 = MAX(a1,a2);//0.359598
		a2 = MIN(a1,a2);//0.198643
		a1 = t0;

		data[i] = a0;
		data[i+n] = a1;
		data[i+2*n] = a2;
		data[i+3*n] = a3;
	}
}

#endif
