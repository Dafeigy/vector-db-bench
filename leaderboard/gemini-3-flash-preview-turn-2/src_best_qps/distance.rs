pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { l2_distance_avx512(a, b) };
        }
    }
    l2_distance_fallback(a, b)
}

#[inline(always)]
fn l2_distance_fallback(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0;
    for (i, &val) in a.iter().enumerate() {
        let diff = val - b[i];
        sum += diff * diff;
    }
    (sum as f64).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx512dq")]
unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    
    // Unroll by 2 (32 floats = 128 bytes)
    for i in (0..128).step_by(32) {
        let va0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm512_loadu_ps(b.as_ptr().add(i));
        let va1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let vb1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        
        let d0 = _mm512_sub_ps(va0, vb0);
        let d1 = _mm512_sub_ps(va1, vb1);
        
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
    }
    
    let sum = _mm512_add_ps(sum0, sum1);
    let res = _mm512_reduce_add_ps(sum);
    (res as f64).sqrt()
}
