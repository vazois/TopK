#ifndef QLA_H
#define QLA_H

#include "../cpu/AA.h"

#include <stdlib.h>

template<class T, class Z>
struct qla_pair{
	Z id;
	T score;
};

template<class T, class Z>
struct qla_block_info{
	qla_block_info(){}
	qla_block_info(Z s, T ub){ size = s; upper_bound = ub;}
	Z size;
	T upper_bound;
};

template<class T,class Z>
static bool cmp_qla_pair(const qla_pair<T,Z> &a, const qla_pair<T,Z> &b){ return a.score > b.score; };

template<class T, class Z>
class QLA : public AA<T,Z>{
	public:
		QLA(uint64_t n, uint64_t d) : AA<T,Z>(n,d){
			this->algo = "QLA";
			this->sbins = NULL;
		}

		~QLA(){
			if(this->sbins != NULL) free(this->sbins);
			for(uint32_t m = 0; m < this->blocks.size(); m++){
				if(this->blocks[m] != NULL) free(this->blocks[m]);
			}
		}

		void init();
		void findTopK(uint64_t k);
	private:
		uint64_t block_num = 0;
		uint64_t partitions = 8;
		T *sbins;
		std::vector<qla_block_info<T,Z>> parts;
		std::vector<T*> blocks;
};

template<class T, class Z>
void QLA<T,Z>::init(){
//	qla_pair<T,Z> *lists = (qla_pair<T,Z>*)malloc(sizeof(qla_pair<T,Z>)*this->n*this->d);
	this->sbins = (T*)malloc(sizeof(T)*this->d*this->partitions);
	this->t.start();

//	for(uint8_t m = 0; m < this->d; m++){
//		for(uint64_t i = 0; i < this->n; i++){
//			lists[m*this->n + i].id = i;
//			lists[m*this->n + i].score = this->cdata[m*this->n + i];
//		}
//	}
//	for(uint8_t m = 0;m<this->d;m++){ __gnu_parallel::sort(&lists[m*this->n],(&lists[m*this->n]) + this->n,cmp_qla_pair<T,Z>); }

	//Data driven partitions//
//	for(uint8_t m = 0; m < this->d; m++){
//		for(uint64_t j = 0; j < this->partitions; j++){
//			// std::cout << (uint64_t)((((float)j)/this->partitions)*this->n) << std::endl;
//			this->sbins[m*this->partitions+ j] = lists[m*this->n + j*(this->n / this->partitions)].score;
//			//this->sbins[m*this->partitions + j] = j*(this->n / this->partitions);
//		}
//	}

	//Fixed partitions//
	for(uint8_t m = 0; m < this->d; m++){
		for(uint64_t j = 0; j < this->partitions; j++){
			this->sbins[m*this->partitions+ j] = ((float)j+1)/this->partitions;
		}
	}

//	for(uint8_t m = 0; m < this->d; m++){
//		std::cout << "a" << std::dec << std::setfill('0') << std::setw(2) << (int)m <<"[ ";
//		std::cout << std::fixed << std::setprecision(4);
//		std::cout << this->sbins[m*this->partitions];
//		for(uint8_t j = 1; j < this->partitions; j++){
//			std::cout << "," << this->sbins[m*this->partitions + j];
//		}
//		std::cout << "]" << std::endl;
//	}

	//Assign Upper Bound to Tuple
	qla_pair<T,Z> *tuples = (qla_pair<T,Z>*)malloc(sizeof(qla_pair<T,Z>)*this->n);
	for(uint64_t i = 0; i < this->n; i++){
		T uscore = 0;
		T tscore = 0;
		for(uint8_t m = 0; m < this->d; m++){
//			if( i < 10 ){
//				std::cout << std::fixed << std::setprecision(4);
//				std::cout << this->cdata[m * this->n + i] << " ";
//			}

			T attr = this->cdata[m * this->n + i];
			tscore+=attr;
			for(uint64_t j = 0; j < this->partitions; j++){
				if ( attr <= this->sbins[m*this->partitions+ j] ){
					uscore+= this->sbins[m*this->partitions+ j];
					break;
				}
			}
		}
		tuples[i].id = i;
		tuples[i].score = uscore;
//		if( i < 10 ){
//			std::cout << "| ";
//			std::cout << tscore << " < "<< uscore << std::endl;
//		}
	}
	__gnu_parallel::sort(tuples,tuples + this->n,cmp_qla_pair<T,Z>);
	////////////////////////////////////////////////////////////////////

	//Calculate partitions//
	this->block_num = 0;
	Z size = 0;
	Z total = 0;
	for(uint64_t i = 0; i < this->n;i++){
		if(tuples[i].score<tuples[i-1].score){
			this->block_num++;
			this->parts.push_back(qla_block_info<T,Z>(size,tuples[i-1].score));
			total+=size;
			size=0;
		}
		size++;
	}
	total+=size;
	std::cout << "grids: " << this->block_num <<std::endl;
	//std::cout << "total: " << total <<std::endl;

	this->blocks.resize(this->parts.size());
	Z start=0;
	Z end=0;
	for(uint32_t m = 0; m < this->parts.size(); m++){
		Z size = (((this->parts[m].size-1)/16) + 1)*16;
//		std::cout << "[" << std::dec << std::setfill('0') << std::setw(3) << (int)(m) << "]: ";
//		std::cout << std::dec << std::setfill('0') << std::setw(8);
//		std::cout << this->parts[m].size << ",";
//		std::cout << std::dec << std::setfill('0') << std::setw(8);
//		std::cout << size;
//		std::cout << std::fixed << std::setprecision(4);
//		std:: cout << " = " << this->parts[m].upper_bound;
//		std::cout << std::endl;

		this->blocks[m] = static_cast<T*>(aligned_alloc(32, sizeof(T) * size * (this->d)));
		end = start + this->parts[m].size;
		T *block = this->blocks[m];
		Z ii = 0;
		for(uint64_t i = 0; i < size * (this->d); i++) block[i] = 0;
		for(uint64_t i = start; i < end; i++){
			qla_pair<T,Z> p = tuples[i];
			//if (i < 272){std::cout << "----> ";}
			for(uint8_t j = 0; j < this->d; j++){
				//if (i < 272){ std::cout << this->cdata[j * this->n + p.id] << " "; }
				block[j * size + ii] = this->cdata[j * this->n + p.id];
			}
			//if (i < 272){std::cout << std::endl;}
			ii++;
		}
		start += this->parts[m].size;
		//this->parts[m].size = size;
	}

	for(uint32_t m = 0; m < this->parts.size(); m++){
		Z size = (((this->parts[m].size-1)/16) + 1)*16;
		this->parts[m].size = size;
	}
	free(this->cdata); this->cdata = NULL;
	free(tuples);
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void QLA<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	this->t.start();
	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
	Z offset = 0;
	for(uint32_t j = 0; j < this->parts.size(); j++){
		T *block = this->blocks[j];
		Z size = this->parts[j].size;

//		if( j  < 0 ){
//		std::cout << "[" << std::dec << std::setfill('0') << std::setw(3) << (int)(j) << "]: ";
//		std::cout << std::dec << std::setfill('0') << std::setw(8);
//		std::cout << this->parts[j].size;
//		std::cout << std::fixed << std::setprecision(4);
//		std:: cout << " = " << this->parts[j].upper_bound;
//		std::cout << std::endl;
//		}

		for(uint64_t i = 0; i < size; i+=4){
			T score00=0;
			T score01=0;
			T score02=0;
			T score03=0;
			for(uint8_t m = 0; m < this->d; m++){
				uint64_t offset0 = m * size + i;
				score00+=block[offset0];
				score01+=block[offset0+1];
				score02+=block[offset0+2];
				score03+=block[offset0+3];
			}
			if(q.size() < k){//insert if empty space in queue
				q.push(tuple<T,Z>(offset+i,score00));
				q.push(tuple<T,Z>(offset+i+1,score01));
				q.push(tuple<T,Z>(offset+i+2,score02));
				q.push(tuple<T,Z>(offset+i+3,score03));
			}else{//delete smallest element if current score is bigger
				if(q.top().score < score00){ q.pop(); q.push(tuple<T,Z>(offset+i,score00)); }
				if(q.top().score < score01){ q.pop(); q.push(tuple<T,Z>(offset+i+1,score01)); }
				if(q.top().score < score02){ q.pop(); q.push(tuple<T,Z>(offset+i+2,score02)); }
				if(q.top().score < score03){ q.pop(); q.push(tuple<T,Z>(offset+i+3,score03)); }
			}
		}

		if(STATS_EFF) this->tuple_count = size;
		if( j < this->parts.size()-1 ){
			if(q.top().score >= this->parts[j+1].upper_bound ){
				break;
			}
		}
		offset+=size;
	}

	this->tt_processing = this->t.lap();
	T threshold = q.top().score;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}




#endif
