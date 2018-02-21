#include<limits>

#include "validation/bench_cpu.h"
#include "validation/debug_cpu.h"
#include <stdio.h>
#include <cstdint>

#include <stdio.h>
#include <tmmintrin.h>

#define RUN_PAR false
#define K 5

void test(){
	uint8_t qsize = 8;
	uint64_t i_attr = 0xFDECBA9876543210;
	uint64_t p_attr = 0x05FDA17CE2489B36;
	uint64_t q_mask = 0xF000FF00F000FFFF;
	uint64_t i_extr = _pext_u64(i_attr,q_mask);
	uint64_t p_extr = _pext_u64(p_attr,q_mask);

	std::cout << "i_attr: 0x" << std::hex << std::setfill('0') << std::setw(16) << i_attr << std::endl;
	std::cout << "p_attr: 0x" << std::hex << std::setfill('0') << std::setw(16) << p_attr << std::endl;

	std::cout << "q_mask: 0x"<< std::hex << std::setfill('0') << std::setw(16) << q_mask<< std::endl;
	std::cout << "i_extr: 0x"<< std::hex << std::setfill('0') << std::setw(16) << i_extr<< std::endl;
	std::cout << "p_extr: 0x"<< std::hex << std::setfill('0') << std::setw(16) << p_extr<< std::endl;

	uint64_t nible = 0xF;
	uint64_t shf = 0;
	uint64_t o_attr = 0;
	uint64_t m_attr = 0;
	for(uint8_t m = 0; m < qsize; m++){
		uint64_t ii = (i_extr & nible) >> shf;
		uint64_t pp = (p_extr & nible) >> shf;
		//std::cout << std::hex << (int)ii << "," << (int)pp <<" <";
		o_attr = o_attr | (ii << (pp << 2));
		m_attr = m_attr | ((uint64_t)(0xF) << (pp << 2));
		nible = (nible << 4);
		shf+=4;
		//std::cout << (int)m<<"> o_attr: 0x "<< std::hex << std::setfill('0') << std::setw(16) << o_attr<< std::endl;
	}


	std::cout << "o_attr: 0x"<< std::hex << std::setfill('0') << std::setw(16) << o_attr<< std::endl;
	std::cout << "m_attr: 0x"<< std::hex << std::setfill('0') << std::setw(16) << m_attr<< std::endl;
	uint64_t o_extr = _pext_u64(o_attr,m_attr);
	std::cout << "o_extr: 0x"<< std::hex << std::setfill('0') << std::setw(16) << o_extr<< std::endl;
}

void test2(){
	uint64_t II[2];
	II[0] = II[1] = 0;
	uint64_t e=0x0F0F0F0F0F0F0F0F;
	uint64_t o=0xF0F0F0F0F0F0F0F0;

	uint64_t i_attr = 0xFEDCBA9876543210;
	uint64_t p_attr = 0x54F876ABC13D20E9;
	//				  0x0f0d0b0907050301
	//				  0x050f070a0c03020e
	//				  0xd1070900b0f05300
	//				  0xD147890CBAFE5362
	uint64_t e_perm = (p_attr & e);
	uint64_t o_perm = (p_attr & o) >> 4;
	//uint64_t p_attr = 0x7777777777777777;

	__m128i *pII =(__m128i*)II;
	__m128i v_in = _mm_set_epi64x(i_attr & e,((i_attr & o)>>4));

	_mm_store_si128(pII,v_in);
	std::cout << "v_in: 0x" << std::hex << std::setfill('0') << std::setw(16) << II[0] << " | 0x" << std::setfill('0') << std::setw(16) << II[1] << std::endl;

	__m128i perm = _mm_set_epi64x(e_perm,o_perm);
	_mm_store_si128(pII,perm);
	std::cout << "perm: 0x" << std::hex << std::setfill('0') << std::setw(16) << II[0] << " | 0x" << std::setfill('0') << std::setw(16) << II[1] << std::endl;

	__m128i half = _mm_set_epi16(0x22,0x22,0x22,0x22,0x22,0x22,0x22,0x22);
	//perm = _mm_div_epi8(perm,half);
	_mm_store_si128(pII,perm);
	std::cout << "perm: 0x" << std::hex << std::setfill('0') << std::setw(16) << II[0] << " | 0x" << std::setfill('0') << std::setw(16) << II[1] << std::endl;
}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	if(!ap.exists("-f")){
		std::cout << "Missing file input!!!" << std::endl;
		exit(1);
	}

	if(!ap.exists("-d")){
		std::cout << "Missing d!!!" << std::endl;
		exit(1);
	}

	if(!ap.exists("-n")){
		std::cout << "Missing n!!!" << std::endl;
		exit(1);
	}

	uint64_t n = ap.getInt("-n");
	uint64_t d = ap.getInt("-d");
	uint64_t nu;
	if(!ap.exists("-nu")){
		nu = n;
	}else{
		nu = ap.getInt("-nu");
	}

	uint64_t nl;
	if(!ap.exists("-nl")){
		nl = n;
	}else{
		nl = ap.getInt("-nl");
	}

	//debug(ap.getString("-f"),n,d,K);
	//bench_fa(ap.getString("-f"),n,d,K);
	bench_ta(ap.getString("-f"),n,d,K);
	//bench_cfa(ap.getString("-f"),n,d,K);
	//bench_maxscore(ap.getString("-f"),n,d,K);
	//bench_cba(ap.getString("-f"),n,d,K);
	bench_cba_opt(ap.getString("-f"),n,d,K);
	//bench_cfa_opt(ap.getString("-f"),n,d,K);

	bench_msa(ap.getString("-f"),n,d,K);
	bench_gsa(ap.getString("-f"),n,d,K);

	//test2();

	return 0;
}
