#ifndef SLA_H
#define SLA_H

/*
 * Dual Resolution Layer Aggregation
 */

#include "../cpu/AA.h"
#include "../skyline/hybrid/hybrid.h"

#define ROW_WISE true

template<class T, class Z>
struct sla_pair{
	Z id;
	T sum;
	T max;
};

template<class T, class Z, int8_t D>
struct sla_tuple{
	Z id;
	T tuple[D];
};

template<class T,class Z>
static bool cmp_dla_pair(const sla_pair<T,Z> &a, const sla_pair<T,Z> &b){
	if(a.max == b.max){
		return a.sum > b.sum;
	}else{
		return a.max > b.max;
	}
};

template<class T, class Z>
class SLA : public AA<T,Z>{
	public:
		SLA(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "SLA";
		};

		~SLA(){

		};

		void init();
		void findTopK(uint64_t k, uint8_t qq);

	private:
		uint64_t partition_table(uint64_t first, uint64_t last, std::unordered_set<uint32_t> layer_set, T **cdata, Z *offset);

};

template<class T, class Z>
uint64_t SLA<T,Z>::partition_table(uint64_t first, uint64_t last, std::unordered_set<uint32_t> layer_set, T **cdata, Z *offset){
//	if(layer_set.find(375273) != layer_set.end()){
//		std::cout << "375273 exists" << std::endl;
//	}
//	bool stop = false;
	while(first < last){
		while(layer_set.find(first) == layer_set.end()){//Find a skyline point
			++first;
			if(first == last) return first;
		}
//
//		if(!stop){
//			std::cout << "pfirst: " << first << std::endl;
//		}

		do{//Find a non-skyline point
			--last;
			if(first == last) return first;
		}while(layer_set.find(last) != layer_set.end());
		//std::swap(*first,*last);
//		if(!stop){
//			std::cout << "plast: " << last<< std::endl;
//			stop=true;
//		}
		offset[first] = offset[last];
		memcpy(cdata[first],cdata[last],sizeof(T)*this->d);
		++first;
	}
	return first;
}

template<class T, class Z>
void SLA<T,Z>::init(){
	uint32_t THREADS_ = 16;
	const uint32_t ALPHA = 1024;
	const uint32_t QSIZE = 8;

	//Copy data to compute skyline//
	T **cdata = static_cast<T**>(aligned_alloc(32, sizeof(T*) * (this->n)));
	for(uint64_t i = 0; i < this->n; i++) cdata[i] = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->d)));
	for(uint64_t i = 0; i < this->n; i++){
		for(uint8_t m = 0; m < this->d; m++){
			//cdata[i][m] = this->cdata[m*this->n + i];
			cdata[i][m] = (1.0f - this->cdata[m*this->n + i]);
		}
	}

	this->t.start();
	std::vector<std::vector<Z>> layers;
	Z *offset = (Z*)malloc(sizeof(Z)*this->n);
	for(uint64_t i = 0; i < this->n; i++) offset[i]=i;
	uint64_t first = 0;
	uint64_t last = this->n;

	uint64_t ii = 0;
	while(last > 100)
	{
		SkylineI* skyline = new Hybrid( THREADS_, (uint32_t)(last), (uint32_t)(this->d), ALPHA, QSIZE );
		skyline->Init( cdata );
		std::vector<uint32_t> layer = skyline->Execute();
		delete skyline;
		std::cout << std::dec << std::setfill('0') << std::setw(10);
		std::cout << last << " - ";
		std::cout << std::dec << std::setfill('0') << std::setw(10);
		std::cout << layer.size() << " = ";

		std::unordered_set<uint32_t> layer_set;
		for(uint64_t i = 0; i <layer.size(); i++){
			layer_set.insert(layer[i]);
			layer[i] = offset[layer[i]];//Real tuple id
		}

		last = this->partition_table(first, last, layer_set, cdata, offset);
		std::cout << std::dec << std::setfill('0') << std::setw(10);
		std::cout << last << std::endl;
		layers.push_back(layer);
	}

	std::vector<uint32_t> layer;
	for(uint64_t i = 0; i < last; i++){
		//std::cout << "t: " << offset[i] << std::endl;
		layer.push_back(offset[i]);
	}
	layers.push_back(layer);

//    uint64_t jj = 0;
//    for(uint64_t i = 0; i < size; i++){
//    	if(layer_set.find(i)==layer_set.end()){
//    		for(uint8_t m = 0; m < this->d; m++) cdata[jj][m] = cdata[i][m];
//    		memcpy(cdata[jj],cdata[i],sizeof(T)*this->d);
//    		offset[jj]=i;
//    		jj++;
//    	}
//    }
//    size = size - layer.size();
//    std::cout << "Remaining tuples: " << size << std::endl;

//	for(uint64_t i = 0; i < (25<layer.size() ? 25 : layer.size()); i++){
//		std::cout << "<";
//		std::cout << std::dec << std::setfill('0') << std::setw(10);
//		std::cout << layer[i] << ">";
//		std::cout << std::fixed << std::setprecision(4);
//		std::cout << " || ";
//		for(uint8_t m = 0; m < this->d; m++){ std::cout << this->cdata[m * this->n + layer[i]] << " "; }
//		std::cout << " || ";
//		std::cout << std::endl;
//	}
//    layers.push_back(layer);


	this->tt_init = this->t.lap();
	for(uint64_t i = 0; i < this->n; i++) free(cdata[i]);
	free(cdata);
	free(offset);
}
#endif
