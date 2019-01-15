#include "tools.h"
#include <parallel/algorithm>

inline __m256 _mm256_sel_ps(__m256 a, __m256 b, __m256 mask)
{//((b^a) & mask)^a : mask = 0x0 select a else select b
	return _mm256_xor_ps(_mm256_and_ps(_mm256_xor_ps(a,b),mask),a);
}

inline void swap(__m256 &a, __m256 &b){
	__m256 tmp;
	tmp = _mm256_sel_ps(a,b,_mm256_cmp_ps(a,b,_CMP_LE_OQ));//MAX
	a = _mm256_sel_ps(a,b,_mm256_cmp_ps(a,b,_CMP_GT_OQ));
	b = tmp;
}

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
	int THREADS = 1;
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

/*
 * Vectorized TA using Priority Queue
 */
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
		for(uint64_t j = 0; j < vsize; j+=16)
		{
			//if(j + 16 < vsize)
			__m256 score00 = _mm256_setzero_ps();
			__m256 score01 = _mm256_setzero_ps();
			//a: Aggregate 16 tuple scores
			for(uint64_t m = 0; m < qq; m++)
			{
				uint8_t a = attr[m];
				uint64_t offset = vsize * a + j;
				T weight = weights[a];

				__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
				__m256 load00 = _mm256_load_ps(data + offset);
				__m256 load01 = _mm256_load_ps(data + offset + 8);

				load00 = _mm256_mul_ps(load00,_weight);
				load01 = _mm256_mul_ps(load01,_weight);
				score00 = _mm256_add_ps(score00,load00);
				score01 = _mm256_add_ps(score01,load01);
			}
			_mm256_store_ps(&score[0],score00);
			_mm256_store_ps(&score[8],score01);

			//b: Push scores to priority queue
			if(q.size() < k){
				q.push(tuple_<T,Z>(i+j,score[0]));
				q.push(tuple_<T,Z>(i+j+1,score[1]));
				q.push(tuple_<T,Z>(i+j+2,score[2]));
				q.push(tuple_<T,Z>(i+j+3,score[3]));
				q.push(tuple_<T,Z>(i+j+4,score[4]));
				q.push(tuple_<T,Z>(i+j+5,score[5]));
				q.push(tuple_<T,Z>(i+j+6,score[6]));
				q.push(tuple_<T,Z>(i+j+7,score[7]));

				q.push(tuple_<T,Z>(i+j+8,score[8]));
				q.push(tuple_<T,Z>(i+j+9,score[9]));
				q.push(tuple_<T,Z>(i+j+10,score[10]));
				q.push(tuple_<T,Z>(i+j+11,score[11]));
				q.push(tuple_<T,Z>(i+j+12,score[12]));
				q.push(tuple_<T,Z>(i+j+13,score[13]));
				q.push(tuple_<T,Z>(i+j+14,score[14]));
				q.push(tuple_<T,Z>(i+j+15,score[15]));
			}else{
				if(q.top().score < score[0]){ q.pop(); q.push(tuple_<T,Z>(i+j,score[0])); }
				if(q.top().score < score[1]){ q.pop(); q.push(tuple_<T,Z>(i+j+1,score[1])); }
				if(q.top().score < score[2]){ q.pop(); q.push(tuple_<T,Z>(i+j+2,score[2])); }
				if(q.top().score < score[3]){ q.pop(); q.push(tuple_<T,Z>(i+j+3,score[3])); }
				if(q.top().score < score[4]){ q.pop(); q.push(tuple_<T,Z>(i+j+4,score[4])); }
				if(q.top().score < score[5]){ q.pop(); q.push(tuple_<T,Z>(i+j+5,score[5])); }
				if(q.top().score < score[6]){ q.pop(); q.push(tuple_<T,Z>(i+j+6,score[6])); }
				if(q.top().score < score[7]){ q.pop(); q.push(tuple_<T,Z>(i+j+7,score[7])); }

				if(q.top().score < score[8]){ q.pop(); q.push(tuple_<T,Z>(i+j,score[8])); }
				if(q.top().score < score[9]){ q.pop(); q.push(tuple_<T,Z>(i+j+1,score[9])); }
				if(q.top().score < score[10]){ q.pop(); q.push(tuple_<T,Z>(i+j+2,score[10])); }
				if(q.top().score < score[11]){ q.pop(); q.push(tuple_<T,Z>(i+j+3,score[11])); }
				if(q.top().score < score[12]){ q.pop(); q.push(tuple_<T,Z>(i+j+4,score[12])); }
				if(q.top().score < score[13]){ q.pop(); q.push(tuple_<T,Z>(i+j+5,score[13])); }
				if(q.top().score < score[14]){ q.pop(); q.push(tuple_<T,Z>(i+j+6,score[14])); }
				if(q.top().score < score[15]){ q.pop(); q.push(tuple_<T,Z>(i+j+7,score[15])); }
			}
		}

		__m256 mx = _mm256_setzero_ps();
		for(uint64_t j = 0; j < this->pnum; j+=8){
			__m256 score00 = _mm256_setzero_ps();
			for(uint64_t m = 0; m < qq; m++){
				uint32_t a = attr[m];
				T weight = weights[a];
				__m256 _weight = _mm256_set_ps(weight,weight,weight,weight,weight,weight,weight,weight);
				__m256 load00 = _mm256_load_ps(&tvector[j]);
				load00 = _mm256_mul_ps(load00,_weight);
				score00 = _mm256_add_ps(score00,load00);
			}
			mx = _mm256_max_ps(mx,score00);
		}
		_mm256_store_ps(&score[0],mx);
		T t0 = std::max(score[0],score[1]);
		T t1 = std::max(score[2],score[3]);
		T t2 = std::max(score[4],score[5]);
		T t3 = std::max(score[6],score[7]);
		t0 = std::max(t0,t1);
		t1 = std::max(t2,t3);
		t0 = std::max(t0,t1);
		if(q.size() >= k && ((q.top().score) >= t0) ){ break; }
		i+=vsize;
	}

	while(q.size() > k){ q.pop(); }
	T threshold = q.top().score;
	return threshold;
}

template class GVAGG<float, uint32_t>;
template class GVAGG<float, uint64_t>;

/*
 * Vectorized TA using Bitonic Sort
 */
template<class T, class Z>
T GVAGG<T,Z>::findTopKgvta2(uint64_t k, uint8_t qq, T *weights, uint32_t *attr)
{
	uint64_t vsize = this->bsize * this->pnum;
	float buffer[64] __attribute__((aligned(64)));
	//std::priority_queue<T, std::vector<tuple_<T,Z>>, Desc<T,Z>> q;
	float *max_scores = static_cast<float*>(aligned_alloc(32,sizeof(float)*k));
	for(uint64_t b = 0; b < this->nb; b++)
	{
		T *data = blocks[b].data;
		T *tvector = blocks[b].tvector;
		for(uint64_t j = 0; j < vsize; j+=64)
		{
			__m256 score00 = _mm256_setzero_ps();//8
			__m256 score01 = _mm256_setzero_ps();//16
			__m256 score02 = _mm256_setzero_ps();//24
			__m256 score03 = _mm256_setzero_ps();//32
			__m256 score04 = _mm256_setzero_ps();//40
			__m256 score05 = _mm256_setzero_ps();//48
			__m256 score06 = _mm256_setzero_ps();//56
			__m256 score07 = _mm256_setzero_ps();//64

			for(uint8_t m = 0; m <qq ; m++)
			{
				uint8_t a = attr[m];
				uint64_t offset = vsize * a + j;
				T weight = weights[a];

				__m256 _weight = _mm256_set1_ps(weight);
				score00 = _mm256_add_ps(score00,_mm256_mul_ps(_mm256_load_ps(data + offset),_weight));
				score01 = _mm256_add_ps(score01,_mm256_mul_ps(_mm256_load_ps(data + offset + 8),_weight));
				score02 = _mm256_add_ps(score02,_mm256_mul_ps(_mm256_load_ps(data + offset + 16),_weight));
				score03 = _mm256_add_ps(score03,_mm256_mul_ps(_mm256_load_ps(data + offset + 24),_weight));
				score04 = _mm256_add_ps(score04,_mm256_mul_ps(_mm256_load_ps(data + offset + 32),_weight));
				score05 = _mm256_add_ps(score05,_mm256_mul_ps(_mm256_load_ps(data + offset + 40),_weight));
				score06 = _mm256_add_ps(score06,_mm256_mul_ps(_mm256_load_ps(data + offset + 48),_weight));
				score07 = _mm256_add_ps(score07,_mm256_mul_ps(_mm256_load_ps(data + offset + 56),_weight));
			}

			//////////////////////
			//simd bitonic sort//
			//1//
			__m256 tmp;
			if(k >= 2)
			{
				//tmp = swap(score00,score01,_CMP_LE_OQ);
				swap(score00,score01); swap(score03,score02); swap(score04,score05); swap(score07,score06);
//				tmp = _mm256_sel_ps(score00,score01,_mm256_cmp_ps(score00,score01,_CMP_LE_OQ));//MAX
//				score01 = _mm256_sel_ps(score00,score01,_mm256_cmp_ps(score00,score01,_CMP_GT_OQ));
//				score00 = tmp;
//				tmp = _mm256_sel_ps(score02,score03,_mm256_cmp_ps(score02,score03,_CMP_GT_OQ));//MIN
//				score03 = _mm256_sel_ps(score02,score03,_mm256_cmp_ps(score02,score03,_CMP_LE_OQ));
//				score02 = tmp;
//				tmp = _mm256_sel_ps(score04,score05,_mm256_cmp_ps(score04,score05,_CMP_LE_OQ));//MAX
//				score05 = _mm256_sel_ps(score04,score05,_mm256_cmp_ps(score04,score05,_CMP_GT_OQ));
//				score04 = tmp;
//				tmp = _mm256_sel_ps(score06,score07,_mm256_cmp_ps(score06,score07,_CMP_GT_OQ));//MIN
//				score07 = _mm256_sel_ps(score06,score07,_mm256_cmp_ps(score06,score07,_CMP_LE_OQ));
//				score06 = tmp;
			}

			//2//
			if(k >= 4){
				swap(score00,score02); swap(score01,score03); swap(score06,score04); swap(score07,score05);
//				tmp = _mm256_sel_ps(score00,score02,_mm256_cmp_ps(score00,score02,_CMP_LE_OQ));//MAX
//				score02 = _mm256_sel_ps(score00,score02,_mm256_cmp_ps(score00,score02,_CMP_GT_OQ));
//				score00 = tmp;
//				tmp = _mm256_sel_ps(score01,score03,_mm256_cmp_ps(score01,score03,_CMP_LE_OQ));//MAX
//				score03 = _mm256_sel_ps(score01,score03,_mm256_cmp_ps(score01,score03,_CMP_GT_OQ));
//				score01 = tmp;
//				tmp = _mm256_sel_ps(score04,score06,_mm256_cmp_ps(score04,score06,_CMP_GT_OQ));//MIN
//				score06 = _mm256_sel_ps(score04,score06,_mm256_cmp_ps(score04,score06,_CMP_LE_OQ));
//				score04 = tmp;
//				tmp = _mm256_sel_ps(score05,score07,_mm256_cmp_ps(score05,score07,_CMP_GT_OQ));//MIN
//				score07 = _mm256_sel_ps(score05,score07,_mm256_cmp_ps(score05,score07,_CMP_LE_OQ));
//				score05 = tmp;

				swap(score00,score01); swap(score02,score03); swap(score05,score04); swap(score07,score06);
//				tmp = _mm256_sel_ps(score00,score01,_mm256_cmp_ps(score00,score01,_CMP_LE_OQ));//MAX
//				score01 = _mm256_sel_ps(score00,score01,_mm256_cmp_ps(score00,score01,_CMP_GT_OQ));
//				score00 = tmp;
//				tmp = _mm256_sel_ps(score02,score03,_mm256_cmp_ps(score02,score03,_CMP_LE_OQ));//MAX
//				score03 = _mm256_sel_ps(score02,score03,_mm256_cmp_ps(score02,score03,_CMP_GT_OQ));
//				score02 = tmp;
//				tmp = _mm256_sel_ps(score04,score05,_mm256_cmp_ps(score04,score05,_CMP_GT_OQ));//MIN
//				score05 = _mm256_sel_ps(score04,score05,_mm256_cmp_ps(score04,score05,_CMP_LE_OQ));
//				score04 = tmp;
//				tmp = _mm256_sel_ps(score06,score07,_mm256_cmp_ps(score06,score07,_CMP_GT_OQ));//MIN
//				score07 = _mm256_sel_ps(score06,score07,_mm256_cmp_ps(score06,score07,_CMP_LE_OQ));
//				score06 = tmp;
			}

			if(k >= 8){
				swap(score04,score00); swap(score05,score01); swap(score06,score02); swap(score07,score03);
//				tmp = _mm256_sel_ps(score00,score04,_mm256_cmp_ps(score00,score04,_CMP_LE_OQ));//MAX
//				score04 = _mm256_sel_ps(score00,score04,_mm256_cmp_ps(score00,score04,_CMP_GT_OQ));
//				score00 = tmp;
//				tmp = _mm256_sel_ps(score01,score05,_mm256_cmp_ps(score01,score05,_CMP_LE_OQ));//MAX
//				score05 = _mm256_sel_ps(score01,score05,_mm256_cmp_ps(score01,score05,_CMP_GT_OQ));
//				score01 = tmp;
//				tmp = _mm256_sel_ps(score02,score06,_mm256_cmp_ps(score02,score06,_CMP_LE_OQ));//MAX
//				score06 = _mm256_sel_ps(score02,score06,_mm256_cmp_ps(score02,score06,_CMP_GT_OQ));
//				score02 = tmp;
//				tmp = _mm256_sel_ps(score03,score07,_mm256_cmp_ps(score03,score07,_CMP_LE_OQ));//MAX
//				score07 = _mm256_sel_ps(score03,score07,_mm256_cmp_ps(score03,score07,_CMP_GT_OQ));
//				score03 = tmp;

				swap(score02,score00); swap(score03,score01); swap(score06,score04); swap(score07,score05);
//				tmp = _mm256_sel_ps(score00,score02,_mm256_cmp_ps(score00,score02,_CMP_LE_OQ));//MAX
//				score02 = _mm256_sel_ps(score00,score02,_mm256_cmp_ps(score00,score02,_CMP_GT_OQ));
//				score00 = tmp;
//				tmp = _mm256_sel_ps(score01,score03,_mm256_cmp_ps(score01,score03,_CMP_LE_OQ));//MAX
//				score03 = _mm256_sel_ps(score01,score03,_mm256_cmp_ps(score01,score03,_CMP_GT_OQ));
//				score01 = tmp;
//				tmp = _mm256_sel_ps(score04,score06,_mm256_cmp_ps(score04,score06,_CMP_LE_OQ));//MAX
//				score06 = _mm256_sel_ps(score04,score06,_mm256_cmp_ps(score04,score06,_CMP_GT_OQ));
//				score04 = tmp;
//				tmp = _mm256_sel_ps(score05,score07,_mm256_cmp_ps(score05,score07,_CMP_LE_OQ));//MAX
//				score07 = _mm256_sel_ps(score05,score07,_mm256_cmp_ps(score05,score07,_CMP_GT_OQ));
//				score05 = tmp;

				swap(score01,score00); swap(score03,score02); swap(score05,score04); swap(score07,score06);
//				tmp = _mm256_sel_ps(score00,score01,_mm256_cmp_ps(score00,score01,_CMP_LE_OQ));//MAX
//				score01 = _mm256_sel_ps(score00,score01,_mm256_cmp_ps(score00,score01,_CMP_GT_OQ));
//				score00 = tmp;
//				tmp = _mm256_sel_ps(score02,score03,_mm256_cmp_ps(score02,score03,_CMP_LE_OQ));//MAX
//				score03 = _mm256_sel_ps(score02,score03,_mm256_cmp_ps(score02,score03,_CMP_GT_OQ));
//				score02 = tmp;
//				tmp = _mm256_sel_ps(score04,score05,_mm256_cmp_ps(score04,score05,_CMP_LE_OQ));//MAX
//				score05 = _mm256_sel_ps(score04,score05,_mm256_cmp_ps(score04,score05,_CMP_GT_OQ));
//				score04 = tmp;
//				tmp = _mm256_sel_ps(score06,score07,_mm256_cmp_ps(score06,score07,_CMP_LE_OQ));//MAX
//				score07 = _mm256_sel_ps(score06,score07,_mm256_cmp_ps(score06,score07,_CMP_GT_OQ));
//				score06 = tmp;
			}
			//__m256i p = _mm256_set_epi32(0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7);//
			//score01 = _mm256_permutevar8x32_ps(score01,p);

			/////////////////////////////////////
			//store to buffer for
			_mm256_store_ps(&buffer[0],score00);
			_mm256_store_ps(&buffer[8],score01);
			_mm256_store_ps(&buffer[16],score02);
			_mm256_store_ps(&buffer[24],score03);
			_mm256_store_ps(&buffer[32],score04);
			_mm256_store_ps(&buffer[40],score05);
			_mm256_store_ps(&buffer[48],score06);
			_mm256_store_ps(&buffer[56],score07);

			std::cout << std::endl << "<SORTED>-------------------------" << std::endl;
			for(uint32_t m = 0; m < 64; m+=8){
				for(uint32_t mm = m; mm < m+8; mm++ ){
					std::cout << buffer[mm] << " | ";
				}
				std::cout << std::endl;
			}
			std::cout << "-------------------------" << std::endl;
			tmp = _mm256_unpackhi_ps(score00,score01);
			score00 = _mm256_unpacklo_ps(score00,score01);
			score01 = tmp;

			tmp = _mm256_unpackhi_ps(score02,score03);
			score02 = _mm256_unpacklo_ps(score02,score03);
			score03 = tmp;

			tmp = _mm256_unpackhi_ps(score04,score05);
			score04 = _mm256_unpacklo_ps(score04,score05);
			score05 = tmp;

			tmp = _mm256_unpackhi_ps(score06,score07);
			score06 = _mm256_unpacklo_ps(score06,score07);
			score07 = tmp;

			//shuffle
//			tmp = _mm256_shuffle_ps(score00,score02, 0x44);
//			score00 = _mm256_shuffle_ps(score00,score02, 0xEE);
//			score01 = tmp;

//			_mm256_store_ps(&buffer[0],score00);
//			_mm256_store_ps(&buffer[8],score01);
//			_mm256_store_ps(&buffer[16],score02);
//			_mm256_store_ps(&buffer[24],score03);
//			_mm256_store_ps(&buffer[32],score04);
//			_mm256_store_ps(&buffer[40],score05);
//			_mm256_store_ps(&buffer[48],score06);
//			_mm256_store_ps(&buffer[56],score07);
//
//			std::cout << std::endl << "<TRANSPOSE I> -------------------------" << std::endl;
//			for(uint32_t m = 0; m < 64; m+=8){
//				for(uint32_t mm = m; mm < m+8; mm++ ){
//					std::cout << buffer[mm] << " | ";
//				}
//				std::cout << std::endl;
//			}
//			std::cout << "-------------------------" << std::endl;

			//score00[0] score00[1] score02[0] score02[1]
			tmp = _mm256_shuffle_ps(score00,score02,_MM_SHUFFLE(1,0,1,0));
			score02 = _mm256_shuffle_ps(score00,score02,_MM_SHUFFLE(3,2,3,2));
			score00 = tmp;

			tmp = _mm256_shuffle_ps(score01,score03,_MM_SHUFFLE(1,0,1,0));
			score03 = _mm256_shuffle_ps(score01,score03,_MM_SHUFFLE(3,2,3,2));
			score01 = tmp;

			tmp = _mm256_shuffle_ps(score04,score06,_MM_SHUFFLE(1,0,1,0));
			score06 = _mm256_shuffle_ps(score04,score06,_MM_SHUFFLE(3,2,3,2));
			score04 = tmp;

			tmp = _mm256_shuffle_ps(score05,score07,_MM_SHUFFLE(1,0,1,0));
			score07 = _mm256_shuffle_ps(score05,score07,_MM_SHUFFLE(3,2,3,2));
			score05 = tmp;

//			_mm256_store_ps(&buffer[0],score00);
//			_mm256_store_ps(&buffer[8],score01);
//			_mm256_store_ps(&buffer[16],score02);
//			_mm256_store_ps(&buffer[24],score03);
//			_mm256_store_ps(&buffer[32],score04);
//			_mm256_store_ps(&buffer[40],score05);
//			_mm256_store_ps(&buffer[48],score06);
//			_mm256_store_ps(&buffer[56],score07);
//
//			std::cout << std::endl << "<TRANSPOSE II> -------------------------" << std::endl;
//			for(uint32_t m = 0; m < 64; m+=8){
//				for(uint32_t mm = m; mm < m+8; mm++ ){
//					std::cout << buffer[mm] << " | ";
//				}
//				std::cout << std::endl;
//			}
//			std::cout << "-------------------------" << std::endl;

			tmp = _mm256_permute2f128_ps(score00, score04, 0x20);
			score04 = _mm256_permute2f128_ps(score00, score04, 0x31);
			score00 = tmp;

			tmp = _mm256_permute2f128_ps(score01, score05, 0x20);
			score05 = _mm256_permute2f128_ps(score01, score05, 0x31);
			score01 = tmp;

			tmp = _mm256_permute2f128_ps(score02, score06, 0x20);
			score06 = _mm256_permute2f128_ps(score02, score06, 0x31);
			score02 = tmp;

			tmp = _mm256_permute2f128_ps(score03, score07, 0x20);
			score07 = _mm256_permute2f128_ps(score03, score07, 0x31);
			score03 = tmp;

			//__m256i idx = _mm256_set_epi32(0x4,0x5,0x6,0x7,0x3,0x2,0x1,0x0);
	//		score00 = _mm256_permutevar8x32_ps(score00,_mm256_set_epi32(0x4,0x5,0x6,0x7,0x3,0x2,0x1,0x0));
	//		score04 = _mm256_permutevar8x32_ps(score04,_mm256_set_epi32(0x3,0x2,0x1,0x0,0x4,0x5,0x6,0x7));

			_mm256_store_ps(&buffer[0],score00);
			_mm256_store_ps(&buffer[8],score01);
			_mm256_store_ps(&buffer[16],score02);
			_mm256_store_ps(&buffer[24],score03);
			_mm256_store_ps(&buffer[32],score04);
			_mm256_store_ps(&buffer[40],score05);
			_mm256_store_ps(&buffer[48],score06);
			_mm256_store_ps(&buffer[56],score07);

			std::cout << std::endl << "<TRANSPOSE III> -------------------------" << std::endl;
			for(uint32_t m = 0; m < 64; m+=8){
				for(uint32_t mm = m; mm < m+8; mm++ ){
					std::cout << buffer[mm] << " | ";
				}
				std::cout << std::endl;
			}
			std::cout << "-------------------------" << std::endl;


			/////////////////////////
			//bitonic-sort in cache//
			/////////////////////////

			//Find-Max//
			break;
		}
		break;
	}
	free(max_scores);
}
