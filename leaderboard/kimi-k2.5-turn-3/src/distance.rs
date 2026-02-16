use std::arch::x86_64::*;

#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    
    unsafe {
        l2_distance_avx512(a, b)
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len();
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    
    // Process 32 elements at a time (2 AVX-512 registers)
    let mut i = 0;
    while i + 32 <= len {
        let va0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm512_sub_ps(va0, vb0);
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);
        
        let va1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let vb1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        let diff1 = _mm512_sub_ps(va1, vb1);
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
        
        i += 32;
    }
    
    // Process remaining 16 elements
    if i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        sum0 = _mm512_fmadd_ps(diff, diff, sum0);
        i += 16;
    }
    
    // Sum the accumulators
    sum0 = _mm512_add_ps(sum0, sum1);
    let mut result = _mm512_reduce_add_ps(sum0) as f64;
    
    // Handle remaining elements
    while i < len {
        let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
        result += (diff * diff) as f64;
        i += 1;
    }
    
    result
}
