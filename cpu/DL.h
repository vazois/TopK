#ifndef DL_H
#define DL_H

#include "AA.h"

#define DL_THREADS 16

template<class T, class Z>
struct DL_BLOCK{
	Z offset;
	Z size;
	T *data;
};

template<class T,class Z>
class PQMin{
	public:
		PQMin(){};

		bool operator() (const tuple_<T,Z>& lhs, const tuple_<T,Z>& rhs) const{
			return (lhs.score>rhs.score);
		}
};

template<class T,class Z>
class PQMax{
	public:
		PQMax(){};

		bool operator() (const tuple_<T,Z>& lhs, const tuple_<T,Z>& rhs) const{
			return (lhs.score<rhs.score);
		}
};

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
			//if(this->fwd != NULL) free(this->fwd);
			if(this->fwd != NULL) delete[] this->fwd;
			if(this->bwd != NULL) delete[] this->bwd;
			if(this->layer_blocks != NULL){
				for(uint64_t i = 0; i < this->layers.size(); i++) free(this->layer_blocks[i].data);
				free(this->layer_blocks);
			}
		}

		void init();
		void findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr);

	private:
		std::vector<std::vector<Z>> layers;
		uint64_t layer_num;
//		std::vector<std::unordered_map<Z,std::vector<Z>>> ss;
		std::unordered_map<Z,std::vector<Z>> *fwd = NULL;
		std::unordered_map<Z,std::vector<Z>> *bwd = NULL;
		std::unordered_map<Z,Z> layer_map;
		DL_BLOCK<T,Z> *layer_blocks = NULL;

		T **sky_data(T **cdata);
		void create_layers(T **cdata);
		uint64_t partition_table(uint64_t first, uint64_t last, std::unordered_set<uint64_t> layer_set, T **cdata, Z *offset);

		void make_blocks();
		void build_edges();
		bool DT(T *p, T *q);
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

		last = this->partition_table(first, last, layer_set, cdata, offset);// Push dominated tables at the bottom//
		this->layers.push_back(layer);
		//std::cout << "layer info: (" << this->layers.size() - 1<< ") = " << layer.size() << std::endl;
	}

	if( last > 0 ){
		std::vector<uint64_t> layer;
		for(uint64_t i = 0; i < last; i++) layer.push_back(offset[i]);
		this->layers.push_back(layer);
		//std::cout << "layer info: (" << this->layers.size() - 1<< ") = " << layer.size() << std::endl;
	}
	this->layer_num = this->layers.size();
	//std::cout << "Layer count: " << this->layer_num << std::endl;
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
void DL<T,Z>::make_blocks()
{
	for(uint64_t i = 0; i < this->layers.size(); i++)
	{
		uint64_t size = 0;
		for(uint64_t j = 0; j < this->layers[i].size(); j++){
			Z id = this->layers[i][j];
			std::memcpy(&this->layer_blocks[i].data[size*this->d],&this->cdata[id*this->d],sizeof(T)*this->d);
			size++;
		}
	}
}

template<class T, class Z>
void DL<T,Z>::build_edges()
{
	uint64_t offset = 0;
	for(uint64_t i = 0; i < layer_num - 1; i++)
	{
		//std::cout << "layer (" << i << ")" << std::endl;
		//Forward edges
		for(uint64_t j = 0; j < this->layer_blocks[i].size; j++){
			layer_map.emplace(offset+j,i);//id, layer
			this->fwd[i].emplace(offset+j,std::vector<Z>());//initialize forward edge for layer i
		}
		this->layer_blocks[i].offset = offset;// increase offset to keep track of id
		offset += this->layer_blocks[i].size;
		//Backward edges
		for(uint64_t j = 0; j < this->layer_blocks[i+1].size; j++) this->bwd[i+1].emplace(offset+j,std::vector<Z>());//initialize backward edges for next layer
		this->layer_blocks[i+1].offset = offset;

		for(auto it = this->fwd[i].begin(); it != this->fwd[i].end(); ++it)
		{
			Z offset01 = it->first - this->layer_blocks[i].offset;
			T *p = &this->layer_blocks[i].data[offset01 * this->d];//for everyone in layer i
			for(auto it2 = this->bwd[i+1].begin(); it2 != this->bwd[i+1].end(); ++it2)
			{
				Z offset02 = it2->first - this->layer_blocks[i+1].offset;
				T *q = &this->layer_blocks[i+1].data[offset02 * this->d];// for everyone in layer i+1
				if(DT(p,q))//insert fwd and bwd edge if p dominates q
				{
					it->second.push_back(it2->first);
					it2->second.push_back(it->first);
				}
			}
		}
	}
}

template<class T, class Z>
bool DL<T,Z>::DT(T *p, T *q)
{
	//pb = ( p[i] < q[i] ) | pb;//At least one dimension better
	//qb = ( q[i] < p[i] ) | qb;//No dimensions better
	bool g0;
	bool g1;
	switch(this->d)
	{
		case 8:
			g0 = (p[0] > q[0]) | (p[1] > q[1]) | (p[2] > q[2]) | (p[3] > q[3]) | (p[4] > q[4]) | (p[5] > q[5]) | (p[6] > q[6]) | (p[7] > q[7]);
			g1 = (p[0] < q[0]) | (p[1] < q[1]) | (p[2] < q[2]) | (p[3] < q[3]) | (p[4] < q[4]) | (p[5] < q[5]) | (p[6] < q[6]) | (p[7] < q[7]);
			return (~g1 & g0);
		default:
			perror("Dimension size not supported!!!");//TODO
			exit(1);
	}
}

template<class T, class Z>
void DL<T,Z>::init()
{
	normalize_transpose<T,Z>(this->cdata, this->n, this->d);
	T **cdata = NULL;
	cdata = this->sky_data(cdata);

	std::cout << "START_INIT\n";
	this->t.start();
	this->create_layers(cdata);//Assign tuples to different layers
	for(uint64_t i = 0; i < this->n; i++) free(cdata[i]);
	this->tt_init += this->t.lap();
	free(cdata);

	//reordered data into blocks
	this->layer_blocks = (DL_BLOCK<T,Z>*)malloc(sizeof(DL_BLOCK<T,Z>)*this->layers.size());// Reorder skyline in blocks
	for(uint64_t i = 0; i < this->layers.size(); i++)
	{
		this->layer_blocks[i].size = this->layers[i].size();
		this->layer_blocks[i].data = (T*)malloc(sizeof(T)*this->layers[i].size()*this->d);
	}

	//initialize edges between layers
	this->fwd = new std::unordered_map<Z,std::vector<Z>>[this->layers.size()];
	this->bwd = new std::unordered_map<Z,std::vector<Z>>[this->layers.size()];


	this->t.start();
	std::cout << "MAKE BLOCKS\n";
	this->make_blocks();
	std::cout << "FINISH BLOCKS\n";
	this->layers.clear();
	free(this->cdata);
	this->cdata = NULL;
	std::cout << "MAKE EDGES\n";
	this->build_edges();
	std::cout << "FINISH EDGES\n";
	this->tt_init = this->t.lap();
}

template<class T,class Z>
void DL<T,Z>::findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr)
{
	std::cout << "START TOPK\n";
	std::cout << this->algo << " find top-" << k << " (" << (int)qq << "D) ...";
	std::vector<std::unordered_set<Z>> eset_vec;
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;

	pqueue<T,Z,pqueue_desc<T,Z>> q(0);
	pqueue<T,Z,pqueue_desc<T,Z>> cl(k);
	std::unordered_set<Z> rs;
	this->t.start();

	//////////////////////////
	//First layer evaluation//
	for(uint64_t j = 0; j < this->layer_blocks[0].size; j++)
	{
		T score = 0;
		for(uint8_t mm = 0; mm < qq; mm++){
			uint8_t aa = attr[mm];
			score += weights[aa] * this->layer_blocks[0].data[j*this->d + aa];
		}
		if(STATS_EFF) this->tuple_count++;
		qpair<T,Z> p;
		p.id = j;
		p.score = score;
		if(cl.size() < k)
		{
			cl.push(p);
		}else if(cl.top().score<p.score){
			cl.pop();
			cl.push(p);
		}

		//std::cout << p.id << "," << p.score << std::endl;
	}

	////////////////////////////
	//Evaluate rest of layers///
	while(q.size() < k)
	{
		qpair<T,Z> p;
		p.id=cl.top().id;
		p.score=cl.top().score;
		cl.pop_back();
		q.push(p);// Get first object from cl queue

		//std::cout << p.id << "," << p.score << std::endl;
		rs.insert(p.id);// object from cl now in rs
		for(auto it=rs.begin(); it!= rs.end(); ++it)// For every object in rs
		{
			auto plm = layer_map.find(*it);//find which layer i belong to//
			if(plm == layer_map.end()) continue;
			auto clist = this->fwd[plm->second].find(plm->first);
			if(clist == this->fwd[plm->second].end()) continue;
			for(auto child = clist->second.begin(); child!= clist->second.end(); ++child)
			{
				Z child_id = *child;
				if(rs.find(child_id) != rs.end()) continue;
				auto clm = layer_map.find(child_id);
				if(clm == layer_map.end()) continue;
				auto plist = this->bwd[clm->second].find(clm->first);
				if(plist == this->bwd[clm->second].end()) continue;
				if(plist->second.size() > k) break;

				bool parents  = true;
				for(auto parent = plist->second.begin(); parent != plist->second.end(); ++parent)
				{
					parents &= (rs.find(*parent) != rs.end());
				}

				if(parents)
				{
					Z offset = child_id - this->layer_blocks[clm->second].offset;//local offset in DL_BLOCK
					T score = 0;
					for(uint8_t mm = 0; mm < qq; mm++){
						uint8_t aa = attr[mm];
						score += weights[aa] * this->layer_blocks[clm->second].data[offset*this->d + aa];
					}

					if(STATS_EFF) this->tuple_count++;

					qpair<T,Z> pp;
					pp.id = child_id;
					pp.score = score;
					if(cl.size() < k)
					{
						cl.push(pp);
					}else if(cl.top().score<pp.score){
						cl.pop();
						cl.push(pp);
					}
				}
			}

			//std::cout << layer_num << "," << l << std::endl;
//			if(l >= this->layer_num) continue;
//			for(uint64_t j = 0; j < this->layer_blocks[l].size; j++)
//			{
//				Z id = this->layer_blocks[l].offset + j;
//				//if(rs.find(id) != rs.end()) continue;
//				auto it = this->bwd[l].find(id);
////				if( it->second.size() < q.size() )
////				{
//					bool parents  = true;
//					for(uint64_t jj = 0; jj < it->second.size(); jj++)
//					{
//						parents &= (rs.find(it->second[jj]) != rs.end());//Check if all parents of child are in rs
//					}
//
//					if(parents)// If all parents of child in rs
//					{
//						T score = 0;
//						for(uint8_t mm = 0; mm < qq; mm++){
//							uint8_t aa = attr[mm];
//							score += weights[aa] * this->layer_blocks[l].data[j*this->d + aa];
//						}
//
//						qpair<T,Z> pp;
//						pp.id = id;
//						pp.score = score;
//						if(cl.size() < k)
//						{
//							cl.push(pp);
//						}else if(cl.top().score<pp.score){
//							cl.pop();
//							cl.push(pp);
//						}
//					}
//				//}
//			}

		}
	}
	this->tt_processing += this->t.lap();

	T threshold = q.size() > 0 ? q.top().score : 1313;
	while(!q.empty()){
		//std::cout << this->algo <<" : " << q.top().tid << "," << q.top().score << std::endl;
		tuple_<T,Z> t;
		t.tid=q.top().id;
		t.score=q.top().score;
		this->res.push_back(t);
		q.pop();
	}
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;

	std::cout << "FINISH TOPK\n";
}

#endif
