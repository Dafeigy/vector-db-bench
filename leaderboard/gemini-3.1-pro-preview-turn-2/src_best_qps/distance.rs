use std::arch::x86_64::*;

#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            unsafe {
                return l2_avx512(a, b);
            }
        }
    }
    
    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum as f64
}

#[target_feature(enable = "avx512f")]
unsafe fn l2_avx512(a: &[f32], b: &[f32]) -> f64 {
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();
    
    let mut i = 0;
    let n = a.len();
    while i + 63 < n {
        let a0 = _mm512_loadu_ps(a.as_ptr().add(i) as *const f32);
        let b0 = _mm512_loadu_ps(b.as_ptr().add(i) as *const f32);
        let d0 = _mm512_sub_ps(a0, b0);
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);

        let a1 = _mm512_loadu_ps(a.as_ptr().add(i + 16) as *const f32);
        let b1 = _mm512_loadu_ps(b.as_ptr().add(i + 16) as *const f32);
        let d1 = _mm512_sub_ps(a1, b1);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);

        let a2 = _mm512_loadu_ps(a.as_ptr().add(i + 32) as *const f32);
        let b2 = _mm512_loadu_ps(b.as_ptr().add(i + 32) as *const f32);
        let d2 = _mm512_sub_ps(a2, b2);
        sum2 = _mm512_fmadd_ps(d2, d2, sum2);

        let a3 = _mm512_loadu_ps(a.as_ptr().add(i + 48) as *const f32);
        let b3 = _mm512_loadu_ps(b.as_ptr().add(i + 48) as *const f32);
        let d3 = _mm512_sub_ps(a3, b3);
        sum3 = _mm512_fmadd_ps(d3, d3, sum3);

        i += 64;
    }
    
    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);
    
    let mut res = _mm512_reduce_add_ps(sum0);
    
    while i < n {
        let diff = a[i] - b[i];
        res += diff * diff;
        i += 1;
    }
    
    res as f64
}
