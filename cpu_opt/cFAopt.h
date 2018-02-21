#ifndef C_FA_OPT_H
#define C_FA_OPT_H

#include "../cpu/AA.h"

#include <bmi2intrin.h>

#define INTERLEAVED false //position index

template<class T, class Z>
struct cfa_pair{
	Z id;
	T attr;
};

template<class Z>
struct cfa_obj{
	uint8_t  attr_ii;
	Z attr_pos;
};

template<class T,class Z>
inline static bool cfa_pair_max(const cfa_pair<T,Z> &a, const cfa_pair<T,Z> &b){ return a.attr > b.attr; };

template<class Z>
static inline bool cfa_attr_pos(const cfa_obj<Z> &a, const cfa_obj<Z> &b){ return a.attr_pos < b.attr_pos; };

template<class T,class Z>
class cFAopt : public AA<T,Z>{
	public:
		cFAopt(uint64_t n,uint64_t d) : AA<T,Z>(n,d){
			this->algo = "cFAopt";
			this->lists = (cfa_pair<T,Z>**)malloc(sizeof(cfa_pair<T,Z>*)*this->d);
			for(uint64_t i = 0;i < this->d;i++) this->lists[i] = (cfa_pair<T,Z>*)malloc(sizeof(cfa_pair<T,Z>) * this->n);
			this->position_table = (Z *)malloc(sizeof(Z) * this->d * this->n);

			this->qsize = (uint8_t)this->d;
			this->q = (uint8_t *)malloc(sizeof(uint8_t) * this->qsize);
			for(uint8_t m = 0; m < this->qsize; m++) this->q[m]=m;
			this->II = NULL;
		};

		cFAopt(uint64_t n, uint64_t d, uint8_t *q, uint8_t qsize) : AA<T,Z>(n,d){
			this->algo = "cFAopt";
			this->lists = (cfa_pair<T,Z>**)malloc(sizeof(cfa_pair<T,Z>*)*this->d);
			for(uint64_t i = 0;i < this->d;i++) this->lists[i] = (cfa_pair<T,Z>*)malloc(sizeof(cfa_pair<T,Z>) * this->n);
			this->position_table = (Z *)malloc(sizeof(Z) * this->d * this->n);

			this->qsize = qsize;
			this->q = (uint8_t *)malloc(sizeof(uint8_t) * this->qsize);
			memcpy(this->q,q,sizeof(uint8_t) * this->qsize);
			this->II = NULL;
		};

		~cFAopt(){
			if(this->q != NULL) free(this->q);
			if(this->position_table != NULL) free(this->position_table);
			if(this->II != NULL) free(this->II);
		};

		void init();
		void findTopK(uint64_t k);

	private:
		std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> qq;
		cfa_pair<T,Z> **lists;
		uint8_t *q;
		uint8_t qsize;
		bool interleaved;

		Z *position_table;
		uint64_t *II;

};


template<class T,class Z>
void cFAopt<T,Z>::init(){
	this->t.start();
	//create sorted lists
	for(uint8_t m =0;m<this->d;m++){
		for(uint64_t i=0;i<this->n;i++){
			this->lists[m][i].id = i;
			this->lists[m][i].attr = this->cdata[m*this->n + i];
		}
	}
	for(uint8_t m =0;m<this->d;m++){ __gnu_parallel::sort(this->lists[m],this->lists[m] + this->n,cfa_pair_max<T,Z>); }
	//

	//Create position table

	for(uint64_t i=0;i<this->n;i++){
		for(uint8_t m =0;m<this->d;m++){
			cfa_pair<T,Z> cfap = this->lists[m][i];
			this->position_table[m * this->n + cfap.id] = i;
			//std::cout << std::setfill('0') << std::setw(3) << cfap.id << " ";
		}
		//std::cout <<std::endl;
	}
	for(uint64_t m = 0;m < this->d;m++) free(this->lists[m]);//Lists not longer necessary
	free(this->lists);
	//

	//Create index from position table
	this->II = (uint64_t *) malloc(sizeof(uint64_t) * this->n);
	cfa_obj<Z> *oo = (cfa_obj<Z>*)malloc(sizeof(cfa_obj<Z>)*this->d);
	for(uint64_t i = 0;i < this->n;i++){
		II[i] = 0;
		for(uint8_t m = 0; m < this->d;m++){
			oo[m].attr_ii = m;
			oo[m].attr_pos = this->position_table[m * this->n + i];

			//if( i < 25 ){ std::cout << std::dec << "[" << (int)oo[m].attr_ii << "," << std::setfill('0') << std::setw(3) <<oo[m].attr_pos << "]";}
		}

		//if(i < 25) std::cout << std::endl;
		std::sort(oo,oo + this->d,cfa_attr_pos<Z>);
		for(uint8_t m = 0; m < this->d;m++){
			II[i] = II[i] | ( m << (oo[m].attr_ii << 2));
			//if( i < 25 ){ std::cout << std::dec << "[" << (int)oo[m].attr_ii << "," <<(int)m << "]"; }
		}
		//if(i < 25) std::cout << "<0x" <<std::hex <<std::setfill('0') << std::setw(this->d) <<II[i] << ">" << std::endl << " --------------------------- " << std::endl;
	}
	free(oo);

	uint64_t src = 0xFEDCBA9876543210;
	uint64_t flg = 0xF00F0000FF0000F0;
	uint64_t dst = _pext_u64(src,flg);
//	std::cout << "src: 0x" <<  std::hex << std::setfill('0') << std::setw(16) << src << std::endl;
//	std::cout << "flg: 0x" <<  std::hex << std::setfill('0') << std::setw(16) << flg << std::endl;
//	std::cout << "dst: 0x" <<  std::hex << std::setfill('0') << std::setw(16) << dst << std::endl;

	this->tt_init = this->t.lap();
}


template<class T, class Z>
void cFAopt<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	Z *Tm = (Z *)malloc(sizeof(Z)*this->n);
	Z *Tw = (Z *)malloc(sizeof(Z)*this->n);
	Z *R = (Z *)malloc(sizeof(Z)*this->n);
	for(uint64_t i = 0;i <this->n;i++) R[i] = 0;

	this->t.start();
	uint64_t q_mask = 0;
	uint64_t i_attr = 0;
	//Identify which attributes need to be considered for the query
	for(uint8_t m = 0; m < this->qsize; m++){
		q_mask = q_mask | (0xF << (this->q[m] << 2));
		i_attr = i_attr | ((uint64_t) (this->q[m]) << (m << 2));
	}
	std::cout << std::endl << "q_mask: 0x" << std::hex << std::setfill('0') << std::setw(16) << q_mask << std::endl;
	std::cout << "-------------------------------------------" << std::endl;
	//std::cout << "i_attr: 0x" << std::hex << std::setfill('0') << std::setw(16) << i_attr << std::endl;

	Z seen_last_pos = 0;
	Z seen_first_pos = 0;
	uint64_t i_extr = i_attr;
	uint8_t tm_shf = (this->qsize-1) << 2;
	uint64_t tm_mask = (0xF << (tm_shf));
	uint64_t tw_mask = 0xF;
	for(uint64_t i = 0; i < this->n; i++){
		uint64_t p_attr = this->II[i];
		//uint64_t i_extr = _pext_u64(i_attr,q_mask);
		uint64_t p_extr = _pext_u64(p_attr,q_mask);

		uint64_t nible = 0xF;
		uint64_t shf = 0;
		uint64_t o_attr = 0;
		uint64_t m_attr = 0;
		for(uint8_t m = 0; m < qsize; m++){
			uint64_t ii = (i_extr & nible) >> shf;
			uint64_t pp = (p_extr & nible) >> shf;
			o_attr = o_attr | (ii << (pp << 2));
			m_attr = m_attr | ((uint64_t)(0xF) << (pp << 2));
			nible = (nible << 4);
			shf+=4;
		}
		uint64_t o_extr = _pext_u64(o_attr,m_attr);
//		std::cout << "i_attr: 0x" << std::hex << std::setfill('0') << std::setw(16) << i_attr << std::endl;
//		std::cout << "p_attr: 0x" << std::hex << std::setfill('0') << std::setw(16) << p_attr << std::endl;

//		std::cout << "i_extr: 0x" << std::hex << std::setfill('0') << std::setw(16) << i_extr << std::endl;
//		std::cout << "p_extr: 0x" << std::hex << std::setfill('0') << std::setw(16) << p_extr << std::endl;
//
//		std::cout << "o_attr: 0x" << std::hex << std::setfill('0') << std::setw(16) << o_attr << std::endl;
//		std::cout << "m_attr: 0x" << std::hex << std::setfill('0') << std::setw(16) << m_attr << std::endl;

//		std::cout << "o_extr: 0x" << std::hex << std::setfill('0') << std::setw(16) << o_extr << std::endl;
//
//		std::cout << "-------------------------------------------" << std::endl;
		seen_last_pos = this->position_table[((o_extr & tm_mask) >> tm_shf) * this->n + i];
		seen_first_pos = this->position_table[(o_extr & tw_mask) * this->n + i];
		R[seen_last_pos]++;
		Tw[i] = seen_first_pos == 0 ? 0 : seen_first_pos-1;

		if(i < 25){
		std::cout << std::dec << std::setfill('0') << std::setw(2) << i << ": ";
		for(uint8_t m = 0;m<this->d;m++){
			std::cout << std::dec << std::setfill('0') << std::setw(3) << this->position_table[m * this->n + i]<< " ";
		}
		std::cout << "| 0x" << std::hex << std::setfill('0') << std::setw(this->d) << II[i];
		std::cout << " | " << std::dec << std::setfill('0') << std::setw(3) << seen_last_pos;
		std::cout << " | " << std::dec << std::setfill('0') << std::setw(3) << seen_first_pos;
		//std::cout << " | " << std::dec << std::setfill('0') << std::setw(3) << R[i];
		std::cout << std::endl;
		}
	}

	//Calculate prefix sum
	uint64_t upper = 0;
	R[0] = 1;
	std::cout <<std::endl << "<RS>" << std::endl;
	for(uint64_t i = 1; i < this->n; i++){
		R[i] = R[i-1] + R[i];
		upper = i;// Find offset of prefix that is less than k

		std::cout << i<<  ": " << R[i] <<std::endl;
		if(R[i] >= k) break;// >= or > ?
	}

//	//calculate scores
	for(uint64_t i = 0; i <this->n; i++){
		if(Tw[i]-1 <= upper){
			Z id = i;
			T score = 0;
			for(uint8_t m = 0; m < this->d; m++){
				score+=this->cdata[m * this->n + id];
			}
			if(STATS_EFF) this->tuple_count++;
			if(STATS_EFF) this->pred_count+=this->d;

			if(this->qq.size() < k){//insert if empty space in queue
				this->qq.push(tuple<T,Z>(id,score));
			}else if(this->qq.top().score<score){//delete smallest element if current score is bigger
				this->qq.pop();
				this->qq.push(tuple<T,Z>(id,score));
			}
		}
	}
	this->tt_processing = this->t.lap();

	free(Tm);
	free(Tw);
	free(R);

	T threshold = this->qq.top().score;

	std::cout << " threshold=[" << threshold <<"] (" << this->res.size() << ")" << std::endl;
	this->threshold = threshold;

}


#endif
