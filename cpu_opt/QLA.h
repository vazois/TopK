#ifndef QLA_H
#define QLA_H

#include "../cpu/AA.h"

template<class T, class Z>
struct qla_pair{
	Z id;
	T score;
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
		}

		void init();
		void findTopK(uint64_t k);
	private:
		uint64_t partitions = 4;
		T *sbins;
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

	qla_pair<T,Z> *tuples = (qla_pair<T,Z>*)malloc(sizeof(qla_pair<T,Z>)*this->n);
	for(uint64_t i = 0; i < this->n; i++){
		T uscore = 0;
		T tscore = 0;
		for(uint8_t m = 0; m < this->d; m++){
			if( i < 10 ){
				std::cout << std::fixed << std::setprecision(4);
				std::cout << this->cdata[m * this->n + i] << " ";
			}

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
		if( i < 10 ){
			std::cout << "| ";
			std::cout << tscore << " < "<< uscore << std::endl;
		}
	}
	__gnu_parallel::sort(tuples,tuples + this->n,cmp_qla_pair<T,Z>);

	Z *bins = (Z*)malloc(sizeof(Z)*this->partitions * this->partitions);
	uint64_t jj = 0;
	for(uint8_t m = 0; m < this->partitions * this->partitions; m++){bins[m]=0;}
	bins[0]++;
	for(uint64_t i = 1; i < this->n;i++){
		if(tuples[i].score<tuples[i-1].score){
			jj++;
		}
		bins[jj]++;
	}

	for(uint8_t m = 0; m < this->partitions; m++){
		for(uint8_t l = 0; l < this->partitions; l++){
			std::cout << "[" << std::dec << std::setfill('0') << std::setw(3) << (int)(m*this->partitions + l) << "]: ";
			std::cout << std::dec << std::setfill('0') << std::setw(8);
			std::cout << bins[m*this->partitions + l] << " ";
		}
		std::cout << std::endl;
	}

	free(tuples);
//	free(lists);
	this->tt_init = this->t.lap();
}

template<class T, class Z>
void QLA<T,Z>::findTopK(uint64_t k){
	std::cout << this->algo << " find topK ...";
	std::priority_queue<T, std::vector<tuple<T,Z>>, PQComparison<T,Z>> q;
	this->t.start();

	this->tt_processing = this->t.lap();
	T threshold = 1313;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << " threshold=[" << threshold <<"] (" << q.size() << ")" << std::endl;
	this->threshold = threshold;
}




#endif
