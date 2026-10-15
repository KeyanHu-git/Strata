#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
namespace strata::kernels::cpu {
namespace {
float h2f(uint16_t x){return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)));}
float hsum(__m256 x){__m128 s=_mm256_extractf128_ps(x,1);s=_mm_add_ps(s,_mm256_castps256_ps128(x));s=_mm_add_ps(s,_mm_movehl_ps(s,s));s=_mm_add_ss(s,_mm_movehdup_ps(s));return _mm_cvtss_f32(s);}
template<int NT> void rows(const uint8_t* w,size_t row_bytes,int n,const void*const* hq,float*const* out,int r0,int r1){
 const __m128i values=_mm_loadu_si128((const __m128i*)kvalues_iq4nl),mask=_mm_set1_epi8(15);const __m256i ones=_mm256_set1_epi16(1);
 const int nb=n/QK4_NL;
 for(int r=r0;r<r1;++r){
  auto* x=(const block_iq4_nl*)(w+r*row_bytes);__m256 even[NT],odd[NT];
  for(int t=0;t<NT;++t){even[t]=_mm256_setzero_ps();odd[t]=_mm256_setzero_ps();}
  int ib=0;
  for(;ib+1<nb;ib+=2){
   const __m128i b0=_mm_loadu_si128((const __m128i*)x[ib].qs),b1=_mm_loadu_si128((const __m128i*)x[ib+1].qs);
   const __m256i w0=_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values,_mm_and_si128(b0,mask))),_mm_shuffle_epi8(values,_mm_and_si128(_mm_srli_epi16(b0,4),mask)),1);
   const __m256i w1=_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values,_mm_and_si128(b1,mask))),_mm_shuffle_epi8(values,_mm_and_si128(_mm_srli_epi16(b1,4),mask)),1);
   const __m256i a0=_mm256_sign_epi8(w0,w0),a1=_mm256_sign_epi8(w1,w1);
   const float d0=h2f(x[ib].d),d1=h2f(x[ib+1].d);
   for(int t=0;t<NT;++t){auto* y=(const block_q8_0*)hq[t];
    auto p0=_mm256_madd_epi16(_mm256_maddubs_epi16(a0,_mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)y[ib].qs),w0)),ones);
    auto p1=_mm256_madd_epi16(_mm256_maddubs_epi16(a1,_mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)y[ib+1].qs),w1)),ones);
    even[t]=_mm256_fmadd_ps(_mm256_set1_ps(h2f(y[ib].d)*d0),_mm256_cvtepi32_ps(p0),even[t]);
    odd[t]=_mm256_fmadd_ps(_mm256_set1_ps(h2f(y[ib+1].d)*d1),_mm256_cvtepi32_ps(p1),odd[t]);
   }
  }
  for(int t=0;t<NT;++t){float s=hsum(_mm256_add_ps(even[t],odd[t]));
   if(ib<nb){auto* y=(const block_q8_0*)hq[t];int a=0,b=0;for(int j=0;j<QK4_NL/2;++j){a+=kvalues_iq4nl[x[ib].qs[j]&15]*y[ib].qs[j];b+=kvalues_iq4nl[x[ib].qs[j]>>4]*y[ib].qs[j+QK4_NL/2];}const float d=h2f(y[ib].d)*h2f(x[ib].d);s+=d*(a+b);}
   out[t][r]=s;
  }
 }
}
}
void iq4nl_reference_rows(const uint8_t*w,size_t row_bytes,int n,const void*const*hq,int nt,float*const*out,int r0,int r1){
 for(int at=0;at<nt;at+=4){int k=nt-at<4?nt-at:4;switch(k){
 case 1:rows<1>(w,row_bytes,n,hq+at,out+at,r0,r1);break;
 case 2:rows<2>(w,row_bytes,n,hq+at,out+at,r0,r1);break;
 case 3:rows<3>(w,row_bytes,n,hq+at,out+at,r0,r1);break;
 case 4:rows<4>(w,row_bytes,n,hq+at,out+at,r0,r1);break;}}
}
}
