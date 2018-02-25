#ifndef GSA_H
#define GSA_H

#include "../cpu/AA.h"
#include <cmath>

template<class T, class Z>
struct gsa_pair{
	Z id;
	T score;
};

template<class T, class Z>
class GSA : public AA<T,Z>{
	public:
		GSA(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "GSA";
			this->tuples = NULL;
			this->bin_count = 256;
			this->bins = (T*)malloc(sizeof(T)*this->bin_count);
			this->attr_map = (uint8_t*)malloc(sizeof(uint8_t)*this->n*this->d);
			this->mx = (uint8_t*)malloc(sizeof(uint8_t)*this->n);

			this->query = (uint8_t*) malloc(sizeof(uint8_t)*this->d);
			this->query_len = this->d;
			for(uint8_t m = 0; m < this->d; m++) this->query[m] = m;
		}

		~GSA(){
			if(this->bins!=NULL) free(this->bins);
			if(this->attr_map!=NULL) free(this->attr_map);
			if(this->mx!=NULL) free(this->mx);
			if(this->tuples!=NULL) free(this->tuples);
		}

		void init();
		void findTopK(uint64_t k);
		void findTopK2(uint64_t k);

	private:
		gsa_pair<T,Z>* partition(
				gsa_pair<T,Z> *first,
				gsa_pair<T,Z> *last,
				const float *binss,
				T threshold,
				uint8_t remainder
			);

		gsa_pair<T,Z> *tuples;
		uint32_t bin_count;
		T *bins;

		uint8_t *attr_map;
		uint8_t *mx;

		uint8_t *query;
		uint8_t query_len;

};


template<class T, class Z>
void GSA<T,Z>::init(){
	this->t.start();

	for(uint32_t i = 0; i< this->bin_count; i++){ this->bins[i] = ((float)(i+1))/this->bin_count; }
//	for(uint32_t i = 0; i< this->bin_count; i++){ std::cout << std::fixed << std::setprecision(4) << this->bins[i] <<" "; }
//	std::cout << std::endl;
	for(uint64_t i = 0; i < this->n; i++){ this->mx[i] = 0; }

	for(uint8_t m = 0; m < this->d; m++){
		for(uint64_t i = 0; i < this->n; i++){
			T aa = this->cdata[m * this->n + i];
			uint32_t p = 0;
			while(p < this->bin_count){
				p++;
				if(aa < this->bins[p]) break;
			}
			this->attr_map[m * this->n + i] = p-1;
			this->mx[i] = this->mx[i] > p-1 ? this->mx[i] : p-1;
		}
	}

//	for(uint64_t i = 0; i < 10; i++){
//		for(uint8_t m = 0; m < this->d-1; m++){
//			std::cout << std::fixed << std::setprecision(4) << this->cdata[m * this->n + i] << " ";
//		}
//		std::cout << " | ";
//		for(uint8_t m = 0; m < this->d-1; m++){
//			std::cout << std::setfill('0') << std::setw(3) << (int)this->attr_map[m * this->n + i] << " || ";
//		}
//		std::cout << "| "<<std::setfill('0') << std::setw(3) << (int)this->mx[i];
//		std::cout << std::endl;
//	}

	this->tt_init = this->t.lap();
}

template<class T, class Z>
void GSA<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	const float binss[256] = {
		0.0039,0.0078,0.0117,0.0156,0.0195,0.0234,0.0273,0.0312,0.0352,0.0391,0.0430,0.0469,0.0508,0.0547,0.0586,0.0625,0.0664,0.0703,
		0.0742,0.0781,0.0820,0.0859,0.0898,0.0938,0.0977,0.1016,0.1055,0.1094,0.1133,0.1172,0.1211,0.1250,0.1289,0.1328,0.1367,0.1406,
		0.1445,0.1484,0.1523,0.1562,0.1602,0.1641,0.1680,0.1719,0.1758,0.1797,0.1836,0.1875,0.1914,0.1953,0.1992,0.2031,0.2070,0.2109,
		0.2148,0.2188,0.2227,0.2266,0.2305,0.2344,0.2383,0.2422,0.2461,0.2500,0.2539,0.2578,0.2617,0.2656,0.2695,0.2734,0.2773,0.2812,
		0.2852,0.2891,0.2930,0.2969,0.3008,0.3047,0.3086,0.3125,0.3164,0.3203,0.3242,0.3281,0.3320,0.3359,0.3398,0.3438,0.3477,0.3516,
		0.3555,0.3594,0.3633,0.3672,0.3711,0.3750,0.3789,0.3828,0.3867,0.3906,0.3945,0.3984,0.4023,0.4062,0.4102,0.4141,0.4180,0.4219,
		0.4258,0.4297,0.4336,0.4375,0.4414,0.4453,0.4492,0.4531,0.4570,0.4609,0.4648,0.4688,0.4727,0.4766,0.4805,0.4844,0.4883,0.4922,
		0.4961,0.5000,0.5039,0.5078,0.5117,0.5156,0.5195,0.5234,0.5273,0.5312,0.5352,0.5391,0.5430,0.5469,0.5508,0.5547,0.5586,0.5625,
		0.5664,0.5703,0.5742,0.5781,0.5820,0.5859,0.5898,0.5938,0.5977,0.6016,0.6055,0.6094,0.6133,0.6172,0.6211,0.6250,0.6289,0.6328,
		0.6367,0.6406,0.6445,0.6484,0.6523,0.6562,0.6602,0.6641,0.6680,0.6719,0.6758,0.6797,0.6836,0.6875,0.6914,0.6953,0.6992,0.7031,
		0.7070,0.7109,0.7148,0.7188,0.7227,0.7266,0.7305,0.7344,0.7383,0.7422,0.7461,0.7500,0.7539,0.7578,0.7617,0.7656,0.7695,0.7734,
		0.7773,0.7812,0.7852,0.7891,0.7930,0.7969,0.8008,0.8047,0.8086,0.8125,0.8164,0.8203,0.8242,0.8281,0.8320,0.8359,0.8398,0.8438,
		0.8477,0.8516,0.8555,0.8594,0.8633,0.8672,0.8711,0.8750,0.8789,0.8828,0.8867,0.8906,0.8945,0.8984,0.9023,0.9062,0.9102,0.9141,
		0.9180,0.9219,0.9258,0.9297,0.9336,0.9375,0.9414,0.9453,0.9492,0.9531,0.9570,0.9609,0.9648,0.9688,0.9727,0.9766,0.9805,0.9844,
		0.9883,0.9922,0.9961,1.0000
	};

	__builtin_prefetch(&binss[0],0,3);
	__builtin_prefetch(&binss[64],0,3);
	__builtin_prefetch(&binss[128],0,3);
	__builtin_prefetch(&binss[192],0,3);

	this->tuples = (gsa_pair<T,Z>*)malloc(sizeof(gsa_pair<T,Z>)*this->n);
	this->t.start();
	std::priority_queue<T, std::vector<T>, std::greater<T>> q0;
	std::priority_queue<T, std::vector<T>, std::greater<T>> q1;
	uint32_t count = 0;
	T _acc[4];
	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
	for(uint64_t i = 0; i < this->n; i+=4){
		__m128 acc = _mm_setzero_ps();
		for(uint8_t m = 0; m < this->d; m++){
			uint64_t offset0 = m * this->n + i;
			uint64_t offset1= m * this->n + i + 1;
			uint64_t offset2 = m * this->n + i + 2;
			uint64_t offset3 = m * this->n + i + 3;

			uint8_t o00 = this->attr_map[offset0];
			uint8_t o01 = this->attr_map[offset1];
			uint8_t o02 = this->attr_map[offset2];
			uint8_t o03 = this->attr_map[offset3];

			__m128 add = _mm_set_ps(binss[o00],binss[o01],binss[o02],binss[o03]);
			acc = _mm_add_ps(acc,add);
		}

		_mm_store_ps(_acc,acc);
		if(q.size() + 4< k){
			q.push(tuple<T,Z>(i,_acc[0]));
			q.push(tuple<T,Z>(i,_acc[1]));
			q.push(tuple<T,Z>(i,_acc[2]));
			q.push(tuple<T,Z>(i,_acc[3]));
		}else{
			if(q.top().score < _acc[0]){ q.pop(); q.push(tuple<T,Z>(i,_acc[0])); }
			if(q.top().score < _acc[1]){ q.pop(); q.push(tuple<T,Z>(i,_acc[1])); }
			if(q.top().score < _acc[2]){ q.pop(); q.push(tuple<T,Z>(i,_acc[2])); }
			if(q.top().score < _acc[3]){ q.pop(); q.push(tuple<T,Z>(i,_acc[3])); }
		}
	}

	this->tt_processing = this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

template<class T, class Z>
gsa_pair<T,Z>* GSA<T,Z>::partition(
		gsa_pair<T,Z> *first,
		gsa_pair<T,Z> *last,
		const float *binss,
		T threshold,
		uint8_t remainder
		){

	while(first < last)
	{
		while(first->score + remainder * binss[this->mx[first->id]] >= threshold){
			++first;
			if(first == last) return first;
		}

		do{
			--last;
			if(first == last) return first;
		}while(last->score + remainder * binss[this->mx[last->id]] < threshold);
		std::swap(*first,*last);
		++first;
	}
	return first;
}

template<class T, class Z>
void GSA<T,Z>::findTopK2(uint64_t k){
	std::cout << this->algo << " find topK ...";
	const float binss[256] = {
		0.0039,0.0078,0.0117,0.0156,0.0195,0.0234,0.0273,0.0312,0.0352,0.0391,0.0430,0.0469,0.0508,0.0547,0.0586,0.0625,0.0664,0.0703,
		0.0742,0.0781,0.0820,0.0859,0.0898,0.0938,0.0977,0.1016,0.1055,0.1094,0.1133,0.1172,0.1211,0.1250,0.1289,0.1328,0.1367,0.1406,
		0.1445,0.1484,0.1523,0.1562,0.1602,0.1641,0.1680,0.1719,0.1758,0.1797,0.1836,0.1875,0.1914,0.1953,0.1992,0.2031,0.2070,0.2109,
		0.2148,0.2188,0.2227,0.2266,0.2305,0.2344,0.2383,0.2422,0.2461,0.2500,0.2539,0.2578,0.2617,0.2656,0.2695,0.2734,0.2773,0.2812,
		0.2852,0.2891,0.2930,0.2969,0.3008,0.3047,0.3086,0.3125,0.3164,0.3203,0.3242,0.3281,0.3320,0.3359,0.3398,0.3438,0.3477,0.3516,
		0.3555,0.3594,0.3633,0.3672,0.3711,0.3750,0.3789,0.3828,0.3867,0.3906,0.3945,0.3984,0.4023,0.4062,0.4102,0.4141,0.4180,0.4219,
		0.4258,0.4297,0.4336,0.4375,0.4414,0.4453,0.4492,0.4531,0.4570,0.4609,0.4648,0.4688,0.4727,0.4766,0.4805,0.4844,0.4883,0.4922,
		0.4961,0.5000,0.5039,0.5078,0.5117,0.5156,0.5195,0.5234,0.5273,0.5312,0.5352,0.5391,0.5430,0.5469,0.5508,0.5547,0.5586,0.5625,
		0.5664,0.5703,0.5742,0.5781,0.5820,0.5859,0.5898,0.5938,0.5977,0.6016,0.6055,0.6094,0.6133,0.6172,0.6211,0.6250,0.6289,0.6328,
		0.6367,0.6406,0.6445,0.6484,0.6523,0.6562,0.6602,0.6641,0.6680,0.6719,0.6758,0.6797,0.6836,0.6875,0.6914,0.6953,0.6992,0.7031,
		0.7070,0.7109,0.7148,0.7188,0.7227,0.7266,0.7305,0.7344,0.7383,0.7422,0.7461,0.7500,0.7539,0.7578,0.7617,0.7656,0.7695,0.7734,
		0.7773,0.7812,0.7852,0.7891,0.7930,0.7969,0.8008,0.8047,0.8086,0.8125,0.8164,0.8203,0.8242,0.8281,0.8320,0.8359,0.8398,0.8438,
		0.8477,0.8516,0.8555,0.8594,0.8633,0.8672,0.8711,0.8750,0.8789,0.8828,0.8867,0.8906,0.8945,0.8984,0.9023,0.9062,0.9102,0.9141,
		0.9180,0.9219,0.9258,0.9297,0.9336,0.9375,0.9414,0.9453,0.9492,0.9531,0.9570,0.9609,0.9648,0.9688,0.9727,0.9766,0.9805,0.9844,
		0.9883,0.9922,0.9961,1.0000
	};

	__builtin_prefetch(&binss[0],0,3);
	__builtin_prefetch(&binss[64],0,3);
	__builtin_prefetch(&binss[128],0,3);
	__builtin_prefetch(&binss[192],0,3);

	this->tuples = (gsa_pair<T,Z>*)malloc(sizeof(gsa_pair<T,Z>)*this->n);
	this->t.start();

	for(uint64_t i = 0; i < this->n; i++){
		this->tuples[i].id = i;
		this->tuples[i].score = binss[this->attr_map[i]];
	}

	gsa_pair<T,Z> *first = this->tuples;
	gsa_pair<T,Z> *last = this->tuples + this->n;
	for(uint8_t m = 0; m < this->d; m++){
		//Find Threshold//
		std::priority_queue<T, std::vector<T>, std::greater<T>> q;
//		std::cout << first << "," << last << std::endl;
		first = this->tuples;
		while(first < last){
			if(q.size() < k){
				q.push(first->score);
			}else if(q.top() < first->score){
				q.pop();
				q.push(first->score);
			}
			first++;
		}
		T threshold = q.top();
		//std::cout << "t: " << threshold << std::endl;

		//Partition data//
		first = this->tuples;
		uint32_t remainder = this->d-(m+1);
		last = this->partition(first,last,binss,threshold,remainder);

		uint64_t size=0;
		if(m < this->d-1){
			first = this->tuples;
			while(first < last){
				first->score+= binss[this->attr_map[first->id]];
				first++;
				size++;
			}
			std::cout << (int)m << " = " << size << std::endl;
		}

	}
	this->tt_processing = this->t.lap();

	T threshold = 1313;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}



#endif
