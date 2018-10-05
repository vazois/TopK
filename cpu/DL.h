#ifndef DL_H
#define DL_H

#include "AA.h"

#define DL_THREADS 16

template<class T,class Z>
class DL : public AA<T,Z>
{
	public:
		DL(uint64_t n,uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "DL";
		}

		~DL()
		{

		}

		void init();
		void findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);

	private:
		std::vector<std::vector<Z>> layers;
		uint64_t layer_num;

		T **sky_data(T **cdata);
		void create_layers(T **cdata);
		uint64_t partition_table(uint64_t first, uint64_t last, std::unordered_set<uint64_t> layer_set, T **cdata, Z *offset);

};

template<class T, class Z>
T** DL<T,Z>::sky_data(T **cdata)
{
	if(cdata == NULL){
		cdata = static_cast<T**>(aligned_alloc(32, sizeof(T*) * (this->n)));
		for(uint64_t i = 0; i < this->n; i++) cdata[i] = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->d)));
	}
	for(uint64_t i = 0; i < this->n; i++){
		for(uint8_t m = 0; m < this->d; m++){
			cdata[i][m] = (1.0f - this->cdata[i*this->d + m]);//Calculate maximum skyline
		}
	}
	return cdata;
}

template<class T, class Z>
void DL<T,Z>::create_layers(T **cdata)
{
	Z *offset = (Z*)malloc(sizeof(Z)*this->n);
	for(uint64_t i = 0; i < this->n; i++) offset[i]=i;
	uint64_t first = 0;
	uint64_t last = this->n;

	while(last > 100)
	{
		SkylineI* skyline = new Hybrid( HLI_THREADS, (uint64_t)(last), (uint64_t)(this->d), SLA_ALPHA, SLA_QSIZE );
		skyline->Init( cdata );
		std::vector<uint64_t> layer = skyline->Execute();//Find ids of tuples that belong to the skyline
		delete skyline;

		std::unordered_set<uint64_t> layer_set;
		for(uint64_t i = 0; i <layer.size(); i++){
			layer_set.insert(layer[i]);
			layer[i] = offset[layer[i]];//Real tuple id
		}

		last = this->partition_table(first, last, layer_set, cdata, offset);
		this->layers.push_back(layer);
	}

	if( last > 0 ){
		std::vector<uint64_t> layer;
		for(uint64_t i = 0; i < last; i++) layer.push_back(offset[i]);
		this->layers.push_back(layer);
	}
	this->layer_num = this->layers.size();
	std::cout << "Layer count: " << this->layer_num << std::endl;
	free(offset);
}


template<class T, class Z>
uint64_t DL<T,Z>::partition_table(uint64_t first, uint64_t last, std::unordered_set<uint64_t> layer_set, T **cdata, Z *offset){
	while(first < last){
		while(layer_set.find(first) == layer_set.end()){//Find a skyline point
			++first;
			if(first == last) return first;
		}

		do{//Find a non-skyline point
			--last;
			if(first == last) return first;
		}while(layer_set.find(last) != layer_set.end());
		offset[first] = offset[last];// Copy real-id of non-skyline point to the beginning of the array
		memcpy(cdata[first],cdata[last],sizeof(T)*this->d);// Copy non-skyline point to beginning of the array
		++first;
	}
	return first;
}

template<class T, class Z>
void DL<T,Z>::init()
{
	normalize_transpose<T,Z>(this->cdata, this->n, this->d);
	T **cdata = NULL;
	cdata = this->sky_data(cdata);

	this->create_layers(cdata);//Assign tuples to different layers
	for(uint64_t i = 0; i < this->n; i++) free(cdata[i]);
	free(cdata);

	this->t.start();


	this->tt_init = this->t.lap();
}

template<class T,class Z>
void DL<T,Z>::findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr)
{
	std::cout << this->algo << " find top-" << k << " (" << (int)qq << "D) ...";
	std::vector<std::unordered_set<Z>> eset_vec;
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	std::priority_queue<T, std::vector<tuple_<T,Z>>, PQComparison<T,Z>> q;

	this->t.start();


	this->tt_processing += this->t.lap();

	T threshold = q.size() > 0 ? q.top().score : 1313;
	while(!q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		this->res.push_back(q.top());
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;
}

#endif
