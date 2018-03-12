#ifndef TESTS_H
#define TESTS_H

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

bool DT(float *p, float *q, uint8_t d){
	uint32_t dt = 0xFF;
	uint32_t mask = 0xFF;

	__m128 p4_00,q4_00;
	__m128 p4_01,q4_01;
	__m128 gt4_00, gt4_01;
	__m256 p8_00,q8_00;
	__m256 p8_01,q8_01;
	__m256 gt8_00,gt8_01;

	switch(d){
		case 4:
			mask =0xF;
			p4_00 = _mm_load_ps(p);
			q4_00 = _mm_load_ps(q);
			gt4_00 = _mm_cmp_ps(p4_00,q4_00,14);
			dt = dt & _mm_movemask_ps(gt4_00);
			break;
		case 8:
			p8_00 = _mm256_load_ps(p);
			q8_00 = _mm256_load_ps(q);
			gt8_00 = _mm256_cmp_ps(p8_00,q8_00,14);
			dt = dt & _mm256_movemask_ps(gt8_00);
			break;
		case 12:
			p8_00 = _mm256_load_ps(p);
			q8_00 = _mm256_load_ps(q);
			p4_00 = _mm_load_ps(&p[8]);
			q4_00 = _mm_load_ps(&q[8]);

			gt4_00 = _mm_cmp_ps(p4_00,q4_00,14);
			gt8_00 = _mm256_set_m128(gt4_00,gt4_00);
			gt8_01 = _mm256_cmp_ps(p8_00,q8_00,14);

			gt8_00 = _mm256_and_ps(gt8_00,gt8_01);
			dt = dt & _mm256_movemask_ps(gt8_00);
			break;
		case 16:
			p8_00 = _mm256_load_ps(p);
			q8_00 = _mm256_load_ps(q);
			p8_01 = _mm256_load_ps(&p[8]);
			q8_01 = _mm256_load_ps(&q[8]);

			gt8_00 = _mm256_cmp_ps(p8_00,q8_00,14);
			gt8_01 = _mm256_cmp_ps(p8_01,q8_01,14);
			gt8_00 = _mm256_and_ps(gt8_00,gt8_01);

			dt = dt & _mm256_movemask_ps(gt8_00);
			break;
		default:
			break;
	};

	return (mask == dt);
}

void test_dt(){
	float cdata4[8] __attribute__((aligned(32))) = {
		0.730,0.620,0.811,0.919,
		0.630,0.520,0.411,0.319
	};
	float cdata8[16] __attribute__((aligned(32))) = {
		0.730,0.620,0.811,0.919,0.542,0.410,0.901,0.722,
		0.630,0.520,0.411,0.319,0.242,0.110,0.001,0.222
	};
	float cdata12[24] __attribute__((aligned(32))) = {
		0.730,0.620,0.811,0.919,0.542,0.410,0.901,0.722,0.730,0.620,0.811,0.919,
		0.630,0.520,0.411,0.319,0.242,0.110,0.001,0.222,0.630,0.520,0.411,0.319
	};

	float cdata16[32] __attribute__((aligned(32))) = {
		0.730,0.620,0.811,0.919,0.542,0.410,0.901,0.722,0.730,0.620,0.811,0.919,0.542,0.410,0.901,0.722,
		0.630,0.520,0.411,0.319,0.242,0.110,0.001,0.222,0.630,0.520,0.411,0.319,0.242,0.110,0.001,0.222
	};
	bool dom;

	for(uint64_t i = 0; i < 100;i++){
	dom = DT(&cdata4[0],&cdata4[4],4);
	std::cout << "(4) p dominates q ? " << "(" << dom << ")" << std::endl;
	dom = DT(&cdata4[4],&cdata4[0],4);
	std::cout << "(4) q dominates p ? " << "(" << dom << ")" << std::endl;
	}

	for(uint64_t i = 0; i < 100;i++){
	dom = DT(&cdata8[0],&cdata8[8],8);
	std::cout << "(8) p dominates q ? " << "(" << dom << ")" << std::endl;
	dom = DT(&cdata8[8],&cdata8[0],8);
	std::cout << "(8) q dominates p ? " << "(" << dom << ")" << std::endl;
	}

	for(uint64_t i = 0; i < 100;i++){
	dom = DT(&cdata12[0],&cdata12[12],12);
	std::cout << "(12) p dominates q ? " << "(" << dom << ")" << std::endl;
	dom = DT(&cdata12[12],&cdata12[0],12);
	std::cout << "(12) q dominates p ? " << "(" << dom << ")" << std::endl;
	}

	for(uint64_t i = 0; i < 100;i++){
	dom = DT(&cdata16[0],&cdata16[16],16);
	std::cout << "(16) p dominates q ? " << "(" << dom << ")" << std::endl;
	dom = DT(&cdata16[16],&cdata16[0],16);
	std::cout << "(16) q dominates p ? " << "(" << dom << ")" << std::endl;
	}
}

#endif
