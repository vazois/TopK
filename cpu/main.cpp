#include<limits>
#include <stdio.h>
#include <cstdint>
#include <stdio.h>
#include <tmmintrin.h>
#include <immintrin.h>

#include "../validation/bench_cpu.h"

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
//	uint64_t nu;
//	if(!ap.exists("-nu")){
//		nu = n;
//	}else{
//		nu = ap.getInt("-nu");
//	}
//
//	uint64_t nl;
//	if(!ap.exists("-nl")){
//		nl = n;
//	}else{
//		nl = ap.getInt("-nl");
//	}

	if (TA_B == 1){ bench_ta(ap.getString("-f"),n,d,KKS,KKE); }
	if (NRA_B == 1){ bench_nra(ap.getString("-f"),n,d,KKS,KKE); }
	if (LARA_B == 1){ bench_lara(ap.getString("-f"),n,d,KKS,KKE); }
	if (BPA_B == 1){ bench_bpa(ap.getString("-f"),n,d,KKS,KKE); }
	if (HLi_B == 1){ bench_hli(ap.getString("-f"),n,d,KKS,KKE); }
	if (T2S_B == 1){ bench_t2s(ap.getString("-f"),n,d,KKS,KKE); }
	if (TPAc_B == 1){ bench_tpac(ap.getString("-f"),n,d,KKS,KKE); }
	if (VTA_B == 1){ bench_vta(ap.getString("-f"),n,d,KKS,KKE); }
	if (SLA_B == 1){ bench_sla(ap.getString("-f"),n,d,KKS,KKE); }
	if (PTA_B == 1){ bench_pta(ap.getString("-f"),n,d,KKS,KKE); }

	if (Onion_B == 1){ bench_onion(ap.getString("-f"),n,d,KKS,KKE); }
	if (DL_B == 1){ bench_dl(ap.getString("-f"),n,d,KKS,KKE); }
	if (TPAr_B == 1){ bench_tpar(ap.getString("-f"),n,d,KKS,KKE); }

	return 0;
}
