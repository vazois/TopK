#ifndef NRA_H
#define NRA_H


template<class T, class Z>
class NRA : public AA<T,Z>
{
	public:
		NRA(uint64_t n, uint64_t d) : AA<T,Z>(n,d)
		{
			this->algo = "LARA";
			this->lists = NULL;
		};

		~NRA()
		{
			if(this->lists!=NULL){ for(uint8_t m = 0;m < this->d;m++){ if(this->lists[m] != NULL){ free(this->lists[m]); } } }
			free(this->lists);
		};

		void init();
		void findTopK(uint64_t k, uint8_t qq, T *weights, uint8_t *attr);
		void findTopK2(uint64_t k, uint8_t qq, T *weights, uint8_t *attr);

	private:
		qpair<T,Z> ** lists;
};

template<class T, class Z>
void NRA<T,Z>::init()
{
	normalize<T,Z>(this->cdata, this->n, this->d);
	this->lists = (qpair<T,Z> **) malloc(sizeof(qpair<T,Z>*)*this->d);
	for(uint8_t m = 0; m < this->d; m++){ this->lists[m] = (qpair<T,Z>*) malloc(sizeof(qpair<T,Z>)*this->n); }

	this->t.start();
	for(uint64_t i = 0; i < this->n; i++)
	{
		for(uint8_t m = 0; m < this->d; m++)
		{
			this->lists[m][i].id = i;
			this->lists[m][i].score = this->cdata[i*this->d + m];
		}
	}
	for(uint8_t m = 0; m < this->d; m++) __gnu_parallel::sort(this->lists[m],lists[m]+this->n,lara_pair_descending<T,Z>);
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void NRA<T,Z>::findTopK(uint64_t k,uint8_t qq, T *weights, uint8_t *attr)
{
	std::cout << this->algo << " find top-" << k << " (" << (int)qq << "D) ...";
	if(this->res.size() > 0) this->res.clear();
	if(STATS_EFF) this->pred_count=0;
	if(STATS_EFF) this->tuple_count = 0;
	if(STATS_EFF) this->pop_count=0;
	if(STATS_EFF) this->candidate_count=0;


}

#endif
