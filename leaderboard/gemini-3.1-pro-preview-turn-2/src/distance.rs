use std::arch::x86_64::*;

#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            unsafe {
                return l2_avx512(a, b);
            }
        } else if std::is_x86_feature_detected!("avx2") {
            unsafe {
                return l2_avx2(a, b);
            }
        }
    }
    
    let mut sum: f32 = 0.0;
    for i in 0..128 {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum as f64
}

#[target_feature(enable = "avx512f")]
unsafe fn l2_avx512(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = _mm512_setzero_ps();
    let n = a.len();
    let mut i = 0;
    
    while i + 15 < n {
        let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
        let d0 = _mm512_sub_ps(a0, b0);
        sum = _mm512_fmadd_ps(d0, d0, sum);
        i += 16;
    }
    
    let mut res = _mm512_reduce_add_ps(sum) as f64;
    
    while i < n {
        let diff = a[i] - b[i];
        res += (diff * diff) as f64;
        i += 1;
    }
    
    res
}

#[target_feature(enable = "avx2")]
unsafe fn l2_avx2(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = _mm256_setzero_ps();
    let n = a.len();
    let mut i = 0;
    
    while i + 7 < n {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let d0 = _mm256_sub_ps(a0, b0);
        sum = _mm256_fmadd_ps(d0, d0, sum);
        i += 8;
    }
    
    let mut x = [0.0; 8];
    _mm256_storeu_ps(x.as_mut_ptr(), sum);
    let mut res = x.iter().fold(0.0, |acc, &v| acc + v as f64);
    
    while i < n {
        let diff = a[i] - b[i];
        res += (diff * diff) as f64;
        i += 1;
    }
    
    res
}
