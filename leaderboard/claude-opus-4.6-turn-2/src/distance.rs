#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute L2 squared distance for 128-dim f32 vectors.
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    l2_distance_f32(a, b) as f64
}

#[inline(always)]
pub fn l2_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe { l2_distance_avx512_impl(a, b) }
}

/// Compute dot product for 128-dim f32 vectors using AVX-512.
#[inline(always)]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe { dot_product_avx512_impl(a, b) }
}

/// Compute squared norm of a 128-dim f32 vector.
#[inline(always)]
pub fn squared_norm_f32(a: &[f32]) -> f32 {
    unsafe { dot_product_avx512_impl(a, a) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn l2_distance_avx512_impl(a: &[f32], b: &[f32]) -> f32 {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    
    let d0 = _mm512_sub_ps(_mm512_loadu_ps(ap), _mm512_loadu_ps(bp));
    let mut sum = _mm512_mul_ps(d0, d0);
    
    let d1 = _mm512_sub_ps(_mm512_loadu_ps(ap.add(16)), _mm512_loadu_ps(bp.add(16)));
    sum = _mm512_fmadd_ps(d1, d1, sum);
    
    let d2 = _mm512_sub_ps(_mm512_loadu_ps(ap.add(32)), _mm512_loadu_ps(bp.add(32)));
    sum = _mm512_fmadd_ps(d2, d2, sum);
    
    let d3 = _mm512_sub_ps(_mm512_loadu_ps(ap.add(48)), _mm512_loadu_ps(bp.add(48)));
    sum = _mm512_fmadd_ps(d3, d3, sum);
    
    let d4 = _mm512_sub_ps(_mm512_loadu_ps(ap.add(64)), _mm512_loadu_ps(bp.add(64)));
    sum = _mm512_fmadd_ps(d4, d4, sum);
    
    let d5 = _mm512_sub_ps(_mm512_loadu_ps(ap.add(80)), _mm512_loadu_ps(bp.add(80)));
    sum = _mm512_fmadd_ps(d5, d5, sum);
    
    let d6 = _mm512_sub_ps(_mm512_loadu_ps(ap.add(96)), _mm512_loadu_ps(bp.add(96)));
    sum = _mm512_fmadd_ps(d6, d6, sum);
    
    let d7 = _mm512_sub_ps(_mm512_loadu_ps(ap.add(112)), _mm512_loadu_ps(bp.add(112)));
    sum = _mm512_fmadd_ps(d7, d7, sum);
    
    _mm512_reduce_add_ps(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn dot_product_avx512_impl(a: &[f32], b: &[f32]) -> f32 {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    
    let mut sum = _mm512_mul_ps(_mm512_loadu_ps(ap), _mm512_loadu_ps(bp));
    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(16)), _mm512_loadu_ps(bp.add(16)), sum);
    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(32)), _mm512_loadu_ps(bp.add(32)), sum);
    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(48)), _mm512_loadu_ps(bp.add(48)), sum);
    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(64)), _mm512_loadu_ps(bp.add(64)), sum);
    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(80)), _mm512_loadu_ps(bp.add(80)), sum);
    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(96)), _mm512_loadu_ps(bp.add(96)), sum);
    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(112)), _mm512_loadu_ps(bp.add(112)), sum);
    
    _mm512_reduce_add_ps(sum)
}
