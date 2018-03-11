#ifndef PSLA_H
#define PSLA_H

/*
 * Parallel Skyline Layered Aggregation
 */

#include "../cpu/AA.h"
#include "../skyline/hybrid/hybrid.h"

#define SLA_THREADS 16
#define SLA_ALPHA 1024
#define SLA_QSIZE 8

template<class T, class Z>
class PSLA : public AA<T,Z>{
	public:
		PSLA(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "SLA";
		};

		~PSLA(){

		};

		void init();
		void findTopK(uint64_t k, uint8_t qq);

	private:
		std::vector<std::vector<Z>> layers;
		uint64_t partition_table(uint64_t first, uint64_t last, std::unordered_set<uint32_t> layer_set, T **cdata, Z *offset);
};

template<class T, class Z>
uint64_t PSLA<T,Z>::partition_table(uint64_t first, uint64_t last, std::unordered_set<uint32_t> layer_set, T **cdata, Z *offset){
	while(first < last){
		while(layer_set.find(first) == layer_set.end()){//Find a skyline point
			++first;
			if(first == last) return first;
		}

		do{//Find a non-skyline point
			--last;
			if(first == last) return first;
		}while(layer_set.find(last) != layer_set.end());
		offset[first] = offset[last];
		memcpy(cdata[first],cdata[last],sizeof(T)*this->d);
		++first;
	}
	return first;
}

template<class T, class Z>
void PSLA<T,Z>::init(){
	/////////////////////////////////
	//Copy data to compute skyline//
	////////////////////////////////
	T **cdata = static_cast<T**>(aligned_alloc(32, sizeof(T*) * (this->n)));
	for(uint64_t i = 0; i < this->n; i++) cdata[i] = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->d)));
	for(uint64_t i = 0; i < this->n; i++){
		for(uint8_t m = 0; m < this->d; m++){
			//cdata[i][m] = this->cdata[m*this->n + i];
			cdata[i][m] = (1.0f - this->cdata[m*this->n + i]);//Calculate maximum skyline
		}
	}

	///////////////////////////
	//Compute Skyline Layers//
	//////////////////////////
	this->t.start();
	Z *offset = (Z*)malloc(sizeof(Z)*this->n);
	for(uint64_t i = 0; i < this->n; i++) offset[i]=i;
	uint64_t first = 0;
	uint64_t last = this->n;

	uint64_t ii = 0;
	while(last > 100)
	{
		SkylineI* skyline = new Hybrid( SLA_THREADS, (uint32_t)(last), (uint32_t)(this->d), SLA_ALPHA, SLA_QSIZE );
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
		this->layers.push_back(layer);
	}

	std::vector<uint32_t> layer;
	for(uint64_t i = 0; i < last; i++) layer.push_back(offset[i]);
	this->layers.push_back(layer);
	std::cout << "Layer count: " << this->layers.size() << std::endl;
	for(uint64_t i = 0; i < this->n; i++) free(cdata[i]);
	free(cdata);
	free(offset);

	//DEBUG//
	std::unordered_set<uint64_t> st;
	for(uint32_t i = 0; i < this->layers.size();i++){
		std::cout << "Layer (" << i << ")" << " = " << this->layers[i].size() << std::endl;
		for(uint32_t j = 0; j < this->layers[i].size();j++){ st.insert(this->layers[i][j]); }
	}
	std::cout << "set values: <" << st.size() << " ? " << this->n << ">" << std::endl;

	//////////////////////
	//Reorder base table//
	/////////////////////
//	omp_set_num_threads(ITHREADS);
//	T *rdata = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->n) * (this->d)));
//	uint64_t jj = 0;
//	for(uint32_t i = 0; i < this->layers.size();i++){
//		for(uint32_t j = 0; j < this->layers[i].size();j++){
//			Z id = this->layers[i][j];
//			for(uint8_t m = 0; m < this->d; m++){ rdata[m*this->n + (jj+j)] = this->cdata[m*this->n + id]; }
//		}
//		jj+=this->layers[i].size();
//	}
//	free(this->cdata); this->cdata = rdata;


	this->tt_init = this->t.lap();
}

template<class T, class Z>
void PSLA<T,Z>::findTopK(uint64_t k, uint8_t qq){
	this->t.start();

	this->tt_processing=this->t.lap();
}



#endif
