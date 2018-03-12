#ifndef SLA_H
#define SLA_H

/*
 * Skyline Layered Aggregation
 */

#include "../skyline/hybrid/hybrid.h"
#include "../cpu/AA.h"

#define SLA_THREADS 16
#define SLA_ALPHA 1024
#define SLA_QSIZE 8

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
		std::vector<std::vector<Z>> layers;
		std::vector<std::unordered_map<Z,std::vector<Z>>> parents;

		T** sky_data(T **cdata);
		void build_layers(T **cdata);
		void create_graph(T **cdata);
		uint64_t partition_table(uint64_t first, uint64_t last, std::unordered_set<uint32_t> layer_set, T **cdata, Z *offset);

		inline bool DT(T *p, T *q);
};

template<class T, class Z>
T** SLA<T,Z>::sky_data(T **cdata){
	if(cdata == NULL){
		cdata = static_cast<T**>(aligned_alloc(32, sizeof(T*) * (this->n)));
		for(uint64_t i = 0; i < this->n; i++) cdata[i] = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->d)));
	}
	for(uint64_t i = 0; i < this->n; i++){
		for(uint8_t m = 0; m < this->d; m++){
			//cdata[i][m] = this->cdata[m*this->n + i];
			cdata[i][m] = (1.0f - this->cdata[m*this->n + i]);//Calculate maximum skyline
		}
	}
	return cdata;
}

template<class T, class Z>
void SLA<T,Z>::build_layers(T **cdata){
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

	if( last > 0 ){
		std::vector<uint32_t> layer;
		for(uint64_t i = 0; i < last; i++) layer.push_back(offset[i]);
		this->layers.push_back(layer);
	}
	std::cout << "Layer count: " << this->layers.size() << std::endl;
	free(offset);

	//DEBUG//
	std::unordered_set<uint64_t> st;
	for(uint32_t i = 0; i < this->layers.size();i++){
		std::cout << "Layer (" << i << ")" << " = " << this->layers[i].size() << std::endl;
		for(uint32_t j = 0; j < this->layers[i].size();j++){ st.insert(this->layers[i][j]); }
	}
	std::cout << "set values: <" << st.size() << " ? " << this->n << ">" << std::endl;
}

template<class T, class Z>
inline bool SLA<T,Z>::DT(T *p, T *q){
	uint32_t dt = 0xFF;
	uint32_t mask = 0xFF;

	__m128 p4_00,q4_00;
	__m128 p4_01,q4_01;
	__m128 gt4_00, gt4_01;
	__m256 p8_00,q8_00;
	__m256 p8_01,q8_01;
	__m256 gt8_00,gt8_01;

	switch(this->d){
		case 4:
			mask =0xF;
			p4_00 = _mm_load_ps(p);
			q4_00 = _mm_load_ps(q);
			gt4_00 = _mm_cmp_ps(p4_00,q4_00,14);
			dt = dt & _mm_movemask_ps(gt4_00);
			break;
		case 8:
			p8_00 = _mm256_load_ps(p);
			q8_00 = _mm256_load_ps(q);
			gt8_00 = _mm256_cmp_ps(p8_00,q8_00,14);
			dt = dt & _mm256_movemask_ps(gt8_00);
			break;
		case 12:
			p8_00 = _mm256_load_ps(p);
			q8_00 = _mm256_load_ps(q);
			p4_00 = _mm_load_ps(&p[8]);
			q4_00 = _mm_load_ps(&q[8]);

			gt4_00 = _mm_cmp_ps(p4_00,q4_00,14);
			gt8_00 = _mm256_set_m128(gt4_00,gt4_00);
			gt8_01 = _mm256_cmp_ps(p8_00,q8_00,14);

			gt8_00 = _mm256_and_ps(gt8_00,gt8_01);
			dt = dt & _mm256_movemask_ps(gt8_00);
			break;
		case 16:
			p8_00 = _mm256_load_ps(p);
			q8_00 = _mm256_load_ps(q);
			p8_01 = _mm256_load_ps(&p[8]);
			q8_01 = _mm256_load_ps(&q[8]);

			gt8_00 = _mm256_cmp_ps(p8_00,q8_00,14);
			gt8_01 = _mm256_cmp_ps(p8_01,q8_01,14);
			gt8_00 = _mm256_and_ps(gt8_00,gt8_01);

			dt = dt & _mm256_movemask_ps(gt8_00);
			break;
		default:
			break;
	};

	return (mask == dt);
}

template<class T, class Z>
void SLA<T,Z>::create_graph(T **cdata){
	parents.resize(this->layers.size());
	for(uint64_t i = this->layers.size()-1; i > 0;i--){
		std::unordered_map<Z,std::vector<Z>> pp = parents[i];
		for(uint64_t j = 0; j < this->layers[i].size();j++){
			Z q = this->layers[i][j];
			pp.emplace(q,std::vector<Z>());
		}
	}

	for(uint64_t i = this->layers.size()-1; i > 0;i--){
		std::unordered_map<Z,std::vector<Z>> pp = parents[i];
		for(uint64_t j = 0; j < this->layers[i].size();j++){
			Z q00 = this->layers[i][j];
//			Z q01 = this->layers[i][j];
//			Z q02 = this->layers[i][j];
//			Z q03 = this->layers[i][j];

			for(uint64_t m = 0; m < this->layers[i-1].size();m++){
				Z p = this->layers[i-1][m];
				if(this->DT(cdata[p],cdata[q00])==0x1){ pp[q00].push_back(p); }
//				if(this->DT(cdata[p],cdata[q01])==0x1){ pp[q01].push_back(p); }
//				if(this->DT(cdata[p],cdata[q02])==0x1){ pp[q02].push_back(p); }
//				if(this->DT(cdata[p],cdata[q03])==0x1){ pp[q03].push_back(p); }
			}
		}
	}


//	parents.resize(this->layers.size());
//	//for(uint64_t i = this->layers.size()-1; i >= this->layers.size()-2;i--){//From last to first layer
//	for(uint64_t i = this->layers.size()-1; i > 0;i--){//From last to first layer
//		//std::vector<std::unordered_map<Z,std::vector<Z>> pp;
//		std::unordered_map<Z,std::vector<Z>> pp = parents[i];
//		for(uint64_t j = 0; j < this->layers[i].size();j++){
//			Z q = this->layers[i][j];
//			pp.emplace(q,std::vector<Z>());
//		}
//
//		#pragma omp parallel for schedule(static) num_threads(ITHREADS)
//		for(uint64_t j = 0; j < this->layers[i].size();j++){
//			Z q = this->layers[i][j];
//			for(uint64_t m = 0; m < this->layers[i-1].size();m++){
//				Z p = this->layers[i-1][m];
//				if(this->DT(cdata[p],cdata[q])==0x1){
//					pp[q].push_back(p);
//				}
//			}
//		}
//	}
}

template<class T, class Z>
uint64_t SLA<T,Z>::partition_table(uint64_t first, uint64_t last, std::unordered_set<uint32_t> layer_set, T **cdata, Z *offset){
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
mak
template<class T, class Z>
void SLA<T,Z>::init(){
	///////////////////////////////////////
	//Copy data to compute skyline layers//
	//////////////////////////////////////
	T **cdata = NULL;
	cdata = this->sky_data(cdata);

	///////////////////////////
	//Compute Skyline Layers//
	//////////////////////////
	this->t.start();
	this->build_layers(cdata);

	cdata = this->sky_data(cdata);
	//this->create_graph(cdata);

	for(uint64_t i = 0; i < this->n; i++) free(cdata[i]);
	free(cdata);

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
//

	this->tt_init = this->t.lap();
}

template<class T, class Z>
void SLA<T,Z>::findTopK(uint64_t k, uint8_t qq){
	std::cout << this->algo << " find topK (" << (int)qq << "D) ...";
	this->t.start();

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;
	for(uint64_t i = 0; i < this->layers.size(); i++){
		for(uint64_t j = 0; j < this->layers[i].size(); j++){
			Z id = this->layers[i][j];
			T score00 = 0;

			for(uint8_t m = 0; m < this->d; m++){
				score00+= this->cdata[m*this->n + id];
			}

			if(q.size() < k){//insert if empty space in queue
				q.push(tuple_<T,Z>(id,score00));
			}else if(q.top().score<score00){//delete smallest element if current score is bigger
				q.pop();
				q.push(tuple_<T,Z>(id,score00));
			}
		}
	}
	this->tt_processing=this->t.lap();

	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
