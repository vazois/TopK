#ifndef SLA_H
#define SLA_H

/*
 * Dual Resolution Layer Aggregation
 */

#include "../cpu/AA.h"

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



};

template<class M, class N, int8_t D>
inline bool DT_r(M *cdata, N p, N q){
	uint64_t qoffset = q*D;
	uint64_t poffset = p*D;

	__m256 qq0,qq1,qq2,qq3;
	__m256 pp0,pp1,pp2,pp3;
	__m256 gt;
	uint32_t dom = 0xFF;
	switch(D){
		case 8:
			qq0 = _mm256_load_ps(&cdata[qoffset]);
			pp0 = _mm256_load_ps(&cdata[poffset]);
			gt = _mm256_cmp_ps(pp0,qq0,14);
			dom = dom & _mm256_movemask_ps(gt);
			break;
		case 16:
			qq0 = _mm256_load_ps(&cdata[qoffset]);
			qq1 = _mm256_load_ps(&cdata[qoffset+8]);
			pp0 = _mm256_load_ps(&cdata[poffset]);
			pp1 = _mm256_load_ps(&cdata[poffset+8]);
			gt = _mm256_cmp_ps(pp0,qq0,14); dom = dom & _mm256_movemask_ps(gt);
			gt = _mm256_cmp_ps(pp1,qq1,14); dom = dom & _mm256_movemask_ps(gt);
			break;
		case 24:
			qq0 = _mm256_load_ps(&cdata[qoffset]);
			qq1 = _mm256_load_ps(&cdata[qoffset+8]);
			qq2 = _mm256_load_ps(&cdata[qoffset+16]);
			pp0 = _mm256_load_ps(&cdata[poffset]);
			pp1 = _mm256_load_ps(&cdata[poffset+8]);
			pp2 = _mm256_load_ps(&cdata[poffset+16]);
			gt = _mm256_cmp_ps(pp0,qq0,14); dom = dom & _mm256_movemask_ps(gt);
			gt = _mm256_cmp_ps(pp1,qq1,14); dom = dom & _mm256_movemask_ps(gt);
			gt = _mm256_cmp_ps(pp2,qq2,14); dom = dom & _mm256_movemask_ps(gt);
			break;
		case 32:
			qq0 = _mm256_load_ps(&cdata[qoffset]);
			qq1 = _mm256_load_ps(&cdata[qoffset+8]);
			qq2 = _mm256_load_ps(&cdata[qoffset+16]);
			qq3 = _mm256_load_ps(&cdata[qoffset+24]);
			pp0 = _mm256_load_ps(&cdata[poffset]);
			pp1 = _mm256_load_ps(&cdata[poffset+8]);
			pp2 = _mm256_load_ps(&cdata[poffset+16]);
			pp3 = _mm256_load_ps(&cdata[poffset+24]);
			gt = _mm256_cmp_ps(pp0,qq0,14); dom = dom & _mm256_movemask_ps(gt);
			gt = _mm256_cmp_ps(pp1,qq1,14); dom = dom & _mm256_movemask_ps(gt);
			gt = _mm256_cmp_ps(pp2,qq2,14); dom = dom & _mm256_movemask_ps(gt);
			gt = _mm256_cmp_ps(pp3,qq3,14); dom = dom & _mm256_movemask_ps(gt);
			break;
		default:
			break;
	};

	return (0xFF == dom);
}

template<class M, class N, int8_t D>
inline void pstopf(M *pstop, M &plevel, M *q){
	M mn = 1;
	switch(D){
		case 8:
			mn = std::min(mn,q[0]); mn = std::min(mn,q[1]); mn = std::min(mn,q[2]); mn = std::min(mn,q[3]);
			mn = std::min(mn,q[4]); mn = std::min(mn,q[5]); mn = std::min(mn,q[6]); mn = std::min(mn,q[7]);
			break;
		case 16:
			mn = std::min(mn,q[0]); mn = std::min(mn,q[1]); mn = std::min(mn,q[2]); mn = std::min(mn,q[3]);
			mn = std::min(mn,q[4]); mn = std::min(mn,q[5]); mn = std::min(mn,q[6]); mn = std::min(mn,q[7]);
			mn = std::min(mn,q[8]); mn = std::min(mn,q[9]); mn = std::min(mn,q[10]); mn = std::min(mn,q[11]);
			mn = std::min(mn,q[12]); mn = std::min(mn,q[13]); mn = std::min(mn,q[14]); mn = std::min(mn,q[15]);
			break;
		case 24:
			mn = std::min(mn,q[0]); mn = std::min(mn,q[1]); mn = std::min(mn,q[2]); mn = std::min(mn,q[3]);
			mn = std::min(mn,q[4]); mn = std::min(mn,q[5]); mn = std::min(mn,q[6]); mn = std::min(mn,q[7]);
			mn = std::min(mn,q[8]); mn = std::min(mn,q[9]); mn = std::min(mn,q[10]); mn = std::min(mn,q[11]);
			mn = std::min(mn,q[12]); mn = std::min(mn,q[13]); mn = std::min(mn,q[14]); mn = std::min(mn,q[15]);
			mn = std::min(mn,q[16]); mn = std::min(mn,q[17]); mn = std::min(mn,q[18]); mn = std::min(mn,q[19]);
			mn = std::min(mn,q[20]); mn = std::min(mn,q[21]); mn = std::min(mn,q[22]); mn = std::min(mn,q[23]);
			break;
		case 32:
			mn = std::min(mn,q[0]); mn = std::min(mn,q[1]); mn = std::min(mn,q[2]); mn = std::min(mn,q[3]);
			mn = std::min(mn,q[4]); mn = std::min(mn,q[5]); mn = std::min(mn,q[6]); mn = std::min(mn,q[7]);
			mn = std::min(mn,q[8]); mn = std::min(mn,q[9]); mn = std::min(mn,q[10]); mn = std::min(mn,q[11]);
			mn = std::min(mn,q[12]); mn = std::min(mn,q[13]); mn = std::min(mn,q[14]); mn = std::min(mn,q[15]);
			mn = std::min(mn,q[16]); mn = std::min(mn,q[17]); mn = std::min(mn,q[18]); mn = std::min(mn,q[19]);
			mn = std::min(mn,q[20]); mn = std::min(mn,q[21]); mn = std::min(mn,q[22]); mn = std::min(mn,q[23]);
			mn = std::min(mn,q[24]); mn = std::min(mn,q[25]); mn = std::min(mn,q[26]); mn = std::min(mn,q[27]);
			mn = std::min(mn,q[28]); mn = std::min(mn,q[29]); mn = std::min(mn,q[30]); mn = std::min(mn,q[31]);
			break;
		default:
			break;
	};
	if(mn > plevel){ plevel = mn; memcpy(pstop,q,sizeof(M)*D); }
}

template<class M, class N, int8_t D>
static void skyline(M *cdata, M *level,uint64_t n){
	std::list<N> R;
	for(uint64_t i = 0; i < n; i++) R.push_back(i);

	uint32_t ii = 0;
	std::vector<std::vector<N>> layers;
//	while(R.size() > 0){
		std::cout << "Layer (" << ii << ")" <<std::endl;
		ii++;

		typename std::list<N>::iterator it = R.begin();
		std::vector<N> layer;
		M pstop[D];
		M plevel = 0;

		layer.push_back(*it);
		memcpy(pstop,&cdata[(*it)*D],sizeof(M)*D);
		it = R.erase(it);

		while(it != R.end()){
			N q = *it;//For every q

			if(plevel > level[q]) break;
			bool dom = 0;
			for(uint64_t i = 0;i < layer.size();i++){
				N p = layer[i];
				if(ROW_WISE){
					dom |= DT_r<M,N,D>(cdata,p,q);
				}else{
					dom = 0;//TODO
				}
				if(dom == 0x1) break;
			}
			if(dom == 0x1){//Dominated, then continue to next point
				++it;
			}else{// In skyline, then put it in current layer
				layer.push_back(q);
				pstopf<M,N,D>(pstop,plevel,&cdata[(*it)*D]);
				it = R.erase(it);
			}
		}
		layers.push_back(layer);
//	}
	std::cout << "Layers: " << layers.size() << std::endl;
}

template<class M, class N, int8_t D>
static void skyline2(M *cdata, M *level,uint64_t n){
	N *tuples = static_cast<N*>(aligned_alloc(32, sizeof(N) * n));;
	uint8_t *pruned= static_cast<uint8_t*>(aligned_alloc(32, sizeof(uint8_t) * n));;
	for(uint64_t i = 0; i < n; i++){
		tuples[i]=i;
		pruned[i]=0;
	}

	uint64_t size = n;
	std::vector<std::vector<N>> layers;
	uint8_t ll = 0;
	omp_set_num_threads(32);
	while(size > 0){
		uint32_t step = 4096;
		uint64_t start = 0;
		uint64_t end = (step < size) ? step : size;

		for(uint64_t i = 0; i < size; i+=step){
			uint64_t start_q = i;
			uint64_t end_q = (i+step < size) ? (i+step) : size;

			#pragma omp parallel for schedule(dynamic, 16) default(shared)
			for(uint64_t j = start_q; j < end_q; j++){
				N q=tuples[j];
				bool dom = 0;
				for(uint64_t m = 0; m < j ; m++){
					if(pruned[m] == ll){
						N p=tuples[m];
						dom |= DT_r<M,N,D>(cdata,p,q);
						if(dom == 0x1) break;
					}
				}
				pruned[j]+= dom;
			}
		}

		ll++;
		uint64_t j = 0;
		for(uint64_t i = 0; i < size;i++){
			if(pruned[i] == ll){
				//memcpy(&cdata[j*D],&cdata[i*D],sizeof(M)*D);
				tuples[j] = tuples[i];
				level[j] = level[i];
				pruned[i] = ll;
				j++;
			}
		}
		size = j;
		std::cout << "New size: " << j << std::endl;
	}
}

template<class M, class N, int8_t D>
static void skyline3(M *cdata, M *level,uint64_t n){
	N *tuples = static_cast<N*>(aligned_alloc(32, sizeof(N) * n));;
	uint8_t *pruned= static_cast<uint8_t*>(aligned_alloc(32, sizeof(uint8_t) * n));;
	for(uint64_t i = 0; i < n; i++){
		tuples[i]=i;
		pruned[i]=0;
	}

	uint64_t size = n;
	std::vector<std::vector<N>> layers;
	uint8_t ll = 0;
	omp_set_num_threads(32);
	while(size > 0){
		uint32_t step = 4096;
		uint64_t start = 0;
		uint64_t end = (step < size) ? step : size;

		for(uint64_t i = 0; i < size; i+=step){
			uint64_t start_q = i;
			uint64_t end_q = (i+step < size) ? (i+step) : size;

			#pragma omp parallel for schedule(dynamic, 16) default(shared)
			for(uint64_t j = start_q; j < end_q; j++){
				N q=tuples[j];
				bool dom = 0;
				for(uint64_t m = 0; m < j ; m++){
					if(pruned[m] == ll){
						N p=tuples[m];
						dom |= DT_r<M,N,D>(cdata,p,q);
						if(dom == 0x1) break;
					}
				}
				pruned[j]+= dom;
			}
		}

		ll++;
		uint64_t j = 0;
		for(uint64_t i = 0; i < size;i++){
			if(pruned[i] == ll){
				//memcpy(&cdata[j*D],&cdata[i*D],sizeof(M)*D);
				tuples[j] = tuples[i];
				level[j] = level[i];
				pruned[i] = ll;
				j++;
			}
		}
		size = j;
		std::cout << "New size: " << j << std::endl;
	}
}


template<class T, class Z>
void SLA<T,Z>::init(){
	sla_pair<T,Z> *tuples = (sla_pair<T,Z>*)malloc(sizeof(sla_pair<T,Z>)*this->n);
	float score[16] __attribute__((aligned(32)));
	float mx[16] __attribute__((aligned(32)));

	this->t.start();
	//Find Score to Compute Skyline
	for(uint64_t i = 0; i < this->n; i+=16){
		__m256 score00 = _mm256_setzero_ps();
		__m256 score01 = _mm256_setzero_ps();
		__m256 max00 = _mm256_setzero_ps();
		__m256 max01 = _mm256_setzero_ps();
		for(uint8_t m = 0; m < this->d; m++){
			uint64_t offset00 = m * this->n + i;
			uint64_t offset01 = m * this->n + i + 8;

			__m256 load00 = _mm256_load_ps(&this->cdata[offset00]);
			__m256 load01 = _mm256_load_ps(&this->cdata[offset01]);

			score00 = _mm256_add_ps(score00,load00);
			score01 = _mm256_add_ps(score01,load01);
			max00 = _mm256_max_ps(max00,load00);
			max01 = _mm256_max_ps(max01,load01);
		}
		_mm256_store_ps(&score[0],score00);
		_mm256_store_ps(&score[8],score01);
		_mm256_store_ps(&mx[0],max00);
		_mm256_store_ps(&mx[8],max01);

		tuples[i].id = i; tuples[i].sum = score[0]; tuples[i].max = mx[0];
		tuples[i+1].id = i+1; tuples[i+1].sum = score[1]; tuples[i+1].max = mx[1];
		tuples[i+2].id = i+2; tuples[i+2].sum = score[2]; tuples[i+2].max = mx[2];
		tuples[i+3].id = i+3; tuples[i+3].sum = score[3]; tuples[i+3].max = mx[3];
		tuples[i+4].id = i+4; tuples[i+4].sum = score[4]; tuples[i+4].max = mx[4];
		tuples[i+5].id = i+5; tuples[i+5].sum = score[5]; tuples[i+5].max = mx[5];
		tuples[i+6].id = i+6; tuples[i+6].sum = score[6]; tuples[i+6].max = mx[6];
		tuples[i+7].id = i+7; tuples[i+7].sum = score[7]; tuples[i+7].max = mx[7];
		tuples[i+8].id = i+8; tuples[i+8].sum = score[8]; tuples[i+8].max = mx[8];
		tuples[i+9].id = i+9; tuples[i+9].sum = score[9]; tuples[i+9].max = mx[9];
		tuples[i+10].id = i+10; tuples[i+10].sum = score[10]; tuples[i+10].max = mx[10];
		tuples[i+11].id = i+11; tuples[i+11].sum = score[11]; tuples[i+11].max = mx[11];
		tuples[i+12].id = i+12; tuples[i+12].sum = score[12]; tuples[i+12].max = mx[12];
		tuples[i+13].id = i+13; tuples[i+13].sum = score[13]; tuples[i+13].max = mx[13];
		tuples[i+14].id = i+14; tuples[i+14].sum = score[14]; tuples[i+14].max = mx[14];
		tuples[i+15].id = i+15; tuples[i+15].sum = score[15]; tuples[i+15].max = mx[15];
	}
	///////////////////////////////////////////////////////////////////////
	uint32_t THREADS_ = 32;
	omp_set_num_threads(THREADS_);
	//std::__parallel::sort(&tuples[0],&tuples[0]+this->n, cmp_dla_pair<T,Z>);
	__gnu_parallel::sort(&tuples[0],(&tuples[0])+this->n, cmp_dla_pair<T,Z>);

	//Reorder Data based on score
	T *cdata = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->n) * (this->d)));
	T *level = static_cast<T*>(aligned_alloc(32, sizeof(T) * (this->n)));
	#pragma omp parallel
	{
		uint32_t thread_id = omp_get_thread_num();
		uint64_t start = ((uint64_t)thread_id)*(this->n)/THREADS_;
		uint64_t end = ((uint64_t)(thread_id+1))*(this->n)/THREADS_;
		if(ROW_WISE){
			for(uint64_t i = start; i < end;i++){
				for(uint8_t m = 0; m < this->d; m++){ cdata[i*this->d + m] = this->cdata[m*this->n + tuples[i].id];} //Column Store//
				level[i] = tuples[i].max;
			}
		}else{
			for(uint64_t i = start; i < end;i++){
				for(uint8_t m = 0; m < this->d; m++){ cdata[m*this->n + i] = this->cdata[m*this->n + tuples[i].id];} //Column Store//
				level[i] = tuples[i].max;
			}
		}
	}
	free(this->cdata); this->cdata = cdata;
	///////////////////////////////////////////////////////////////////////

	for(uint64_t i = 0; i < 25; i++){
		std::cout << "<";
		std::cout << std::dec << std::setfill('0') << std::setw(10);
		std::cout << tuples[i].id << ",";
		std::cout << std::fixed << std::setprecision(4);
		std::cout << tuples[i].sum << "," << tuples[i].max << ">";
		std::cout << " || ";
		if(ROW_WISE){
			for(uint8_t m = 0; m < this->d; m++){ std::cout << this->cdata[i * this->d + m] << " "; }
		}else{
			for(uint8_t m = 0; m < this->d; m++){ std::cout << this->cdata[m * this->n + i] << " "; }
		}
		std::cout << " || ";
		std::cout << std::endl;
	}
	free(tuples);

	switch(this->d){
		case 8:
			//skyline<T,Z,8>(this->cdata,level,this->n);
			skyline2<T,Z,8>(this->cdata,level,this->n);
			break;
		case 16:
			skyline<T,Z,16>(this->cdata,level,this->n);
			break;
		case 24:
			skyline<T,Z,24>(this->cdata,level,this->n);
			break;
		default:
			break;
	};
	this->cdata=(T*)malloc(sizeof(T));
	free(level);
	this->tt_init = this->t.lap();
}

#endif
