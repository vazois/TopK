#include "tools.h"
#include <parallel/algorithm>

template<class T, class Z>
void psort(gpta_pair<T,Z> *tpairs,uint64_t n, bool ascending){
	if(ascending){
		__gnu_parallel::sort(&tpairs[0],(&tpairs[0])+n,cmp_gpta_pair_asc<T,Z>);
	}else{
		__gnu_parallel::sort(&tpairs[0],(&tpairs[0])+n,cmp_gpta_pair_desc<T,Z>);
	}
}

template void psort(gpta_pair<float,uint64_t> *tpairs,uint64_t n, bool ascending);
template void psort(gpta_pair<float,uint32_t> *tpairs,uint64_t n, bool ascending);

template<class Z>
void ppsort(gpta_pos<Z> *tpos, uint64_t n){
	__gnu_parallel::sort(&tpos[0],(&tpos[0])+n,cmp_gpta_pos_asc<Z>);
}

template void ppsort(gpta_pos<uint64_t> *tpos,uint64_t n);
template void ppsort(gpta_pos<uint32_t> *tpos,uint64_t n);


template<class T, class Z>
void pnth_element(gpta_pair<T,Z> *tscore, uint64_t n, uint64_t k, bool ascending){
	__gnu_parallel::nth_element(&tscore[0],(&tscore[k]),(&tscore[0])+n,cmp_gpta_pair_asc<T,Z>);
}

template void pnth_element(gpta_pair<float,uint32_t> *tscore, uint64_t n, uint64_t k, bool ascending);
template void pnth_element(gpta_pair<float,uint64_t> *tscore, uint64_t n, uint64_t k, bool ascending);

template <class T>
void normalize_transpose(T *&cdata, uint64_t n, uint64_t d){
	T *mmax = static_cast<T*>(aligned_alloc(32,sizeof(T)*d));
	T *mmin = static_cast<T*>(aligned_alloc(32,sizeof(T)*d));

	//Find min and max for each attribute list
	for(uint64_t m = 0; m < d; m++){
		mmax[m] = 0;
		mmin[m] = std::numeric_limits<T>::max();
		for(uint64_t i = 0; i < n; i++){
			T value = cdata[m*n + i];
			//if (m == 0) std::cout << m << " < " << value << "," << value <<"," << mmax[m] << std::endl;
			mmax[m] = std::max(mmax[m],value);
			mmin[m] = std::min(mmin[m],value);
		}
	}

	//Normalize values
	for(uint64_t m = 0; m < d; m++){
		T _max = mmax[m];
		T _min = mmin[m];
		T _mm = _max - _min;
		//if ( _mm == 0 ){ std::cout << m << " <"<< _max << " - " << _min << std::endl; }
		for(uint64_t i = 0; i < n; i++){
			T value = cdata[m*n+i];
			value = (value - _min)/_mm;
			cdata[m*n + i] = value;
		}
	}
	free(mmax);
	free(mmin);
}

template void normalize_transpose(float *&cdata, uint64_t n, uint64_t d);

template<class T, class Z>
T VAGG<T,Z>::findTopKtpac(uint64_t k,uint8_t qq, T *weights, uint32_t *attr){
	//	if(this->cdata == NULL){ perror("cdata not initialized"); }
	int THREADS = 16;
	omp_set_num_threads(THREADS);
	std::priority_queue<T, std::vector<tuple_<T,Z>>, Desc<T,Z>> q[THREADS];
	#pragma omp parallel
	{
		uint32_t thread_id = omp_get_thread_num();
		float score[16] __attribute__((aligned(32)));
		uint64_t start = ((uint64_t)thread_id)*(n)/THREADS;
		uint64_t end = ((uint64_t)(thread_id+1))*(n)/THREADS;

		for(uint64_t i = start; i < end; i+=16)
		{
			__m256 score00 = _mm256_setzero_ps();
			__m256 score01 = _mm256_setzero_ps();
			for(uint8_t m = 0; m < qq; m++){
				uint64_t offset00 = attr[m] * n + i;
				uint64_t offset01 = attr[m] * n + i + 8;
				T weight = weights[attr[m]];

				__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
				__m256 load00 = _mm256_load_ps(&cdata[offset00]);
				__m256 load01 = _mm256_load_ps(&cdata[offset01]);

				load00 = _mm256_mul_ps(load00,_weight);
				load01 = _mm256_mul_ps(load01,_weight);
				score00 = _mm256_add_ps(score00,load00);
				score01 = _mm256_add_ps(score01,load01);
			}
			_mm256_store_ps(&score[0],score00);
			_mm256_store_ps(&score[8],score01);

			if(q[thread_id].size() < k){
				q[thread_id].push(tuple_<T,Z>(i,score[0]));
				q[thread_id].push(tuple_<T,Z>(i+1,score[1]));
				q[thread_id].push(tuple_<T,Z>(i+2,score[2]));
				q[thread_id].push(tuple_<T,Z>(i+3,score[3]));
				q[thread_id].push(tuple_<T,Z>(i+4,score[4]));
				q[thread_id].push(tuple_<T,Z>(i+5,score[5]));
				q[thread_id].push(tuple_<T,Z>(i+6,score[6]));
				q[thread_id].push(tuple_<T,Z>(i+7,score[7]));
				q[thread_id].push(tuple_<T,Z>(i+8,score[8]));
				q[thread_id].push(tuple_<T,Z>(i+9,score[9]));
				q[thread_id].push(tuple_<T,Z>(i+10,score[10]));
				q[thread_id].push(tuple_<T,Z>(i+11,score[11]));
				q[thread_id].push(tuple_<T,Z>(i+12,score[12]));
				q[thread_id].push(tuple_<T,Z>(i+13,score[13]));
				q[thread_id].push(tuple_<T,Z>(i+14,score[14]));
				q[thread_id].push(tuple_<T,Z>(i+15,score[15]));
			}else{
				if(q[thread_id].top().score < score[0]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i,score[0])); }
				if(q[thread_id].top().score < score[1]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+1,score[1])); }
				if(q[thread_id].top().score < score[2]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+2,score[2])); }
				if(q[thread_id].top().score < score[3]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+3,score[3])); }
				if(q[thread_id].top().score < score[4]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+4,score[4])); }
				if(q[thread_id].top().score < score[5]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+5,score[5])); }
				if(q[thread_id].top().score < score[6]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+6,score[6])); }
				if(q[thread_id].top().score < score[7]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+7,score[7])); }
				if(q[thread_id].top().score < score[8]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+8,score[8])); }
				if(q[thread_id].top().score < score[9]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+9,score[9])); }
				if(q[thread_id].top().score < score[10]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+10,score[10])); }
				if(q[thread_id].top().score < score[11]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+11,score[11])); }
				if(q[thread_id].top().score < score[12]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+12,score[12])); }
				if(q[thread_id].top().score < score[13]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+13,score[13])); }
				if(q[thread_id].top().score < score[14]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+14,score[14])); }
				if(q[thread_id].top().score < score[15]){ q[thread_id].pop(); q[thread_id].push(tuple_<T,Z>(i+15,score[15])); }
			}
		}
	}

	std::priority_queue<T, std::vector<tuple_<T,Z>>, Desc<T,Z>> _q;
	for(uint32_t m = 0; m < THREADS; m++){
		while(!q[m].empty()){
			if(_q.size() < k){
				_q.push(q[m].top());
			}else if(_q.top().score < q[m].top().score){
				_q.pop();
				_q.push(q[m].top());
			}
			q[m].pop();
		}
	}
	while(_q.size() > k){ _q.pop(); }
	T threshold = _q.top().score;
//	std::cout << std::fixed << std::setprecision(4);
//	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	return threshold;
}

template class VAGG<float, uint32_t>;
template class VAGG<float, uint64_t>;

template<class T, class Z>
T GVAGG<T,Z>::findTopKgvta(uint64_t k, uint8_t qq, T *weights, uint32_t *attr)
{
	uint64_t vsize = this->bsize * this->pnum;
	//std::priority_queue<T, std::vector<tuple_<T,Z>>, Desc<T,Z>> q[THREADS];
	std::priority_queue<T, std::vector<tuple_<T,Z>>, Desc<T,Z>> q;
	float score[16] __attribute__((aligned(32)));
	Z i = 0;
	for(uint64_t b = 0; b < this->nb; b++)
	{
		T *data = blocks[b].data;
		T *tvector = blocks[b].tvector;
		for(uint64_t j = 0; j < vsize; j+=8)
		{
			__m256 score00 = _mm256_setzero_ps();
			__m256 score01 = _mm256_setzero_ps();
			for(uint64_t m = 0; m < qq; m++)
			{
				uint32_t a = attr[m];
				T weight = weights[a];
				__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
				__m256 load00 = _mm256_load_ps(&data[vsize * a + j]);

				load00 = _mm256_mul_ps(load00,_weight);
				score00 = _mm256_add_ps(score00,load00);
			}
			_mm256_store_ps(&score[0],score00);

			if(q.size() < k){
				q.push(tuple_<T,Z>(i+j,score[0]));
				q.push(tuple_<T,Z>(i+j+1,score[1]));
				q.push(tuple_<T,Z>(i+j+2,score[2]));
				q.push(tuple_<T,Z>(i+j+3,score[3]));
				q.push(tuple_<T,Z>(i+j+4,score[4]));
				q.push(tuple_<T,Z>(i+j+5,score[5]));
				q.push(tuple_<T,Z>(i+j+6,score[6]));
				q.push(tuple_<T,Z>(i+j+7,score[7]));
			}else{
				if(q.top().score < score[0]){ q.pop(); q.push(tuple_<T,Z>(i+j,score[0])); }
				if(q.top().score < score[1]){ q.pop(); q.push(tuple_<T,Z>(i+j+1,score[1])); }
				if(q.top().score < score[2]){ q.pop(); q.push(tuple_<T,Z>(i+j+2,score[2])); }
				if(q.top().score < score[3]){ q.pop(); q.push(tuple_<T,Z>(i+j+3,score[3])); }
				if(q.top().score < score[4]){ q.pop(); q.push(tuple_<T,Z>(i+j+4,score[4])); }
				if(q.top().score < score[5]){ q.pop(); q.push(tuple_<T,Z>(i+j+5,score[5])); }
				if(q.top().score < score[6]){ q.pop(); q.push(tuple_<T,Z>(i+j+6,score[6])); }
				if(q.top().score < score[7]){ q.pop(); q.push(tuple_<T,Z>(i+j+7,score[7])); }
			}
		}
		i+=vsize;
	}

	while(q.size() > k){ q.pop(); }
	T threshold = q.top().score;
	return threshold;
}

template class GVAGG<float, uint32_t>;
template class GVAGG<float, uint64_t>;


