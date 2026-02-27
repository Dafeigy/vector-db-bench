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
    sum as f64
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vl")]
#[target_feature(enable = "avx512dq")]
unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    
    // Unroll by 4 (64 floats = 256 bytes)
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();
    
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    // Loop twice for 128 floats
    for i in (0..128).step_by(64) {
        let va0 = _mm512_loadu_ps(pa.add(i));
        let vb0 = _mm512_loadu_ps(pb.add(i));
        let va1 = _mm512_loadu_ps(pa.add(i + 16));
        let vb1 = _mm512_loadu_ps(pb.add(i + 16));
        let va2 = _mm512_loadu_ps(pa.add(i + 32));
        let vb2 = _mm512_loadu_ps(pb.add(i + 32));
        let va3 = _mm512_loadu_ps(pa.add(i + 48));
        let vb3 = _mm512_loadu_ps(pb.add(i + 48));
        
        let d0 = _mm512_sub_ps(va0, vb0);
        let d1 = _mm512_sub_ps(va1, vb1);
        let d2 = _mm512_sub_ps(va2, vb2);
        let d3 = _mm512_sub_ps(va3, vb3);
        
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
        sum2 = _mm512_fmadd_ps(d2, d2, sum2);
        sum3 = _mm512_fmadd_ps(d3, d3, sum3);
    }
    
    let sum01 = _mm512_add_ps(sum0, sum1);
    let sum23 = _mm512_add_ps(sum2, sum3);
    let sum = _mm512_add_ps(sum01, sum23);
    let res = _mm512_reduce_add_ps(sum);
    res as f64
}
