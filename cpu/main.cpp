#include<limits>

#include "validation/bench_cpu.h"
#include <stdio.h>
#include <cstdint>

#include <stdio.h>
#include <tmmintrin.h>

#define RUN_PAR false
#define K 100

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

const __m128i pass1_add4 = _mm_setr_epi32(1, 1, 3, 3);
const __m128i pass2_add4 = _mm_setr_epi32(2, 3, 2, 3);
const __m128i pass3_add4 = _mm_setr_epi32(0, 2, 2, 3);
void simdsort4(int* __restrict v) {
	__m128i b, a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(v));

	b = _mm_shuffle_epi32(a, 177);
	b = _mm_cmpgt_epi32(b, a);
	b = _mm_add_epi32(b, pass1_add4);
	a = _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(a), b));

	b = _mm_shuffle_epi32(a, 78);
	b = _mm_cmpgt_epi32(b, a);
	b = _mm_add_epi32(b, b);
	b = _mm_add_epi32(b, pass2_add4);
	a = _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(a), b));

	b = _mm_shuffle_epi32(a, 216);
	b = _mm_cmpgt_epi32(b, a);
	b = _mm_add_epi32(b, pass3_add4);
	__m128 ret = _mm_permutevar_ps(_mm_castsi128_ps(a), b);

	_mm_storeu_ps(reinterpret_cast<float*>(v), ret);
}

void simd4sort(float *v){
	__m128 in = _mm_set_ps(v[0],v[1],v[2],v[3]);
	__m128 pair;
	__m128i cmp;

	pair =_mm_permute_ps(in,177);
	_mm_store_ps(v,pair);
	std::cout << "1a: "<< v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;
	cmp = _mm_castps_si128(_mm_cmpgt_ps(in,pair));
	cmp = _mm_add_epi32(cmp, _mm_setr_epi32(1, 1, 3, 3));
	in = _mm_permutevar_ps(in,cmp);
	_mm_store_ps(v,in);
	std::cout << "1b: "<< v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;

	pair =_mm_permute_ps(in,78);
	_mm_store_ps(v,pair);
	std::cout << "2a: "<< v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;
	cmp = _mm_castps_si128(_mm_cmpgt_ps(in,pair));
	int vv[4];
	_mm_storeu_si128((__m128i*)vv,cmp);
	std::cout << "<<a "<< vv[0] << "," << vv[1] << "," << vv[2] << "," << vv[3] << std::endl;
	cmp = _mm_add_epi32(cmp, cmp);
	_mm_storeu_si128((__m128i*)vv,cmp);
	std::cout << "<<b "<< vv[0] << "," << vv[1] << "," << vv[2] << "," << vv[3] << std::endl;
	cmp = _mm_add_epi32(cmp, _mm_setr_epi32(2, 3, 2, 3));
	_mm_storeu_si128((__m128i*)vv,cmp);
	std::cout << "<<c "<< vv[0] << "," << vv[1] << "," << vv[2] << "," << vv[3] << std::endl;
	in = _mm_permutevar_ps(in,cmp);
	_mm_store_ps(v,in);
	std::cout << "2b: "<< v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;

	pair =_mm_permute_ps(in,216);
	_mm_store_ps(v,pair);
	std::cout << "3a: "<< v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;
	cmp = _mm_castps_si128(_mm_cmpgt_ps(in,pair));
	cmp = _mm_add_epi32(cmp, _mm_setr_epi32(0, 2, 2, 3));
	in = _mm_permutevar_ps(in,cmp);
	_mm_store_ps(v,in);
	std::cout << "3b: "<< v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;

//	pair =_mm_permute_ps(in,177);
////	_mm_store_ps(v,pair);
////	std::cout << "1: "<< v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;
//	cmp = _mm_castps_si128(_mm_cmpgt_ps(in,pair));
//	cmp = _mm_add_epi32(cmp, _mm_setr_epi32(1, 1, 3, 3));
//	in = _mm_permutevar_ps(in,cmp);
//	_mm_store_ps(v,in);
//	std::cout << "1: "<< v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;
//
//	pair =_mm_permute_ps(in,78);
//	cmp = _mm_castps_si128(_mm_cmpgt_ps(in,pair));
//	cmp = _mm_add_epi32(cmp, _mm_setr_epi32(1, 3, 1, 3));
//	in = _mm_permutevar_ps(in,cmp);
//	_mm_store_ps(v,in);
//	std::cout << "2: "<< v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;
//
//	pair =_mm_permute_ps(in,216);
//	cmp = _mm_castps_si128(_mm_cmpgt_ps(in,pair));
//	cmp = _mm_add_epi32(cmp, _mm_setr_epi32(0, 2, 2, 3));
//	in = _mm_permutevar_ps(in,cmp);
//	_mm_store_ps(v,in);
//	std::cout << "3: "<< v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;
//
//	_mm_store_ps(v,in);
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
	//bench_na(ap.getString("-f"),n,d,K);
//	bench_ta(ap.getString("-f"),n,d,K);
//	bench_lsa(ap.getString("-f"),n,d,K);
//	bench_msa(ap.getString("-f"),n,d,K);
//	bench_gsa(ap.getString("-f"),n,d,K);
//	bench_tpac(ap.getString("-f"),n,d,K);
//	bench_tpar(ap.getString("-f"),n,d,K);
//	bench_ta_simd(ap.getString("-f"),n,d,K);

//	bench_qla(ap.getString("-f"),n,d,K);

//	float v[4] = {0.3,0.2,0.7,0.5};
//	std::cout << "00: "<<v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;
//	simd4sort(v);
//	std::cout << "ee: " <<v[0] << "," << v[1] << "," << v[2] << "," << v[3] << std::endl;

	bench_tpac(ap.getString("-f"),n,d,K);
	//bench_ta_simd(ap.getString("-f"),n,d,K);
	bench_qla(ap.getString("-f"),n,d,K);

	return 0;
}
