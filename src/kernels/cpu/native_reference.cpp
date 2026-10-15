#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
namespace strata::kernels::cpu {
void iq512_reference_gu_rows(int,const uint8_t*,size_t,size_t,int,const void*const*,int,float*const*,int,int);
void iq4nl_reference_rows(const uint8_t*,size_t,int,const void*const*,int,float*const*,int,int);
void native_gu_rows_invariant(const NativeFmt& f,const uint8_t* blob,const void*const* act,int nt,float*const* ff,int r0,int r1) {
 // An explicitly selected multi-token-from-one policy already defines an invariant solo target.
 const int mt=native_gu_mt_min(f.gu_type);
 if(mt<=1){native_gu_rows(f,blob,act,nt,ff,r0,r1);return;}
 static const bool avx512_enabled=std::getenv("STRATA_NO_IQ512")==nullptr;
 if(nt>=mt && avx512_enabled && cpu_avx512_ok() && ggml_cpu_has_avx2() && ggml_cpu_has_fma() && (f.gu_type==18 || f.gu_type==21 || f.gu_type==22)) {
  iq512_reference_gu_rows(f.gu_type,blob,f.gu_row,f.up_off,(int)f.n_embd,act,nt,ff,r0,r1);return;
 }
 for(int t=0;t<nt;++t) native_gu_rows(f,blob,act+t,1,ff+t,r0,r1);
}
void native_down_rows_invariant(const NativeFmt& f,const uint8_t* blob,const void*const* hq,int nt,float*const* out,int r0,int r1) {
 static const int mt=[] {const char* e=std::getenv("STRATA_IQ_MT_MIN");return e?std::atoi(e):2;}();
 if(mt<=1){native_down_rows(f,blob,hq,nt,out,r0,r1);return;}
 static const bool iq4nl_enabled=std::getenv("STRATA_NO_IQ4NL")==nullptr;
 if(nt>=mt && iq4nl_enabled && cpu_avx2_ok() && ggml_cpu_has_avx2() && ggml_cpu_has_fma() && f.d_type==20){iq4nl_reference_rows(blob+f.down_off,f.d_row,(int)f.n_ff,hq,nt,out,r0,r1);return;}
 for(int t=0;t<nt;++t) native_down_rows(f,blob,hq+t,1,out+t,r0,r1);
}
}
