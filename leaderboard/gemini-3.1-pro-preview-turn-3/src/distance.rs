pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { l2_distance_avx512(a, b) };
        } else if is_x86_feature_detected!("avx2") {
            return unsafe { l2_distance_avx2(a, b) };
        }
    }
    l2_distance_fallback(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let va0 = _mm512_loadu_ps(a_ptr);
    let vb0 = _mm512_loadu_ps(b_ptr);
    let d0 = _mm512_sub_ps(va0, vb0);
    let mut sum0 = _mm512_mul_ps(d0, d0);

    let va1 = _mm512_loadu_ps(a_ptr.add(16));
    let vb1 = _mm512_loadu_ps(b_ptr.add(16));
    let d1 = _mm512_sub_ps(va1, vb1);
    let mut sum1 = _mm512_mul_ps(d1, d1);

    for i in (32..128).step_by(32) {
        let va0 = _mm512_loadu_ps(a_ptr.add(i));
        let vb0 = _mm512_loadu_ps(b_ptr.add(i));
        let d0 = _mm512_sub_ps(va0, vb0);
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);

        let va1 = _mm512_loadu_ps(a_ptr.add(i + 16));
        let vb1 = _mm512_loadu_ps(b_ptr.add(i + 16));
        let d1 = _mm512_sub_ps(va1, vb1);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
    }
    
    let sum = _mm512_add_ps(sum0, sum1);
    
    // faster reduce add
    let h1 = _mm512_castps512_ps256(sum);
    let h2 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(sum), 1));
    let s256 = _mm256_add_ps(h1, h2);
    
    let h3 = _mm256_castps256_ps128(s256);
    let h4 = _mm256_extractf128_ps(s256, 1);
    let s128 = _mm_add_ps(h3, h4);
    
    let h5 = _mm_movehl_ps(s128, s128);
    let s64 = _mm_add_ps(s128, h5);
    
    let h6 = _mm_shuffle_ps(s64, s64, 1);
    let s32 = _mm_add_ss(s64, h6);
    
    _mm_cvtss_f32(s32) as f64
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    
    let mut sum = _mm256_setzero_ps();
    for i in (0..128).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    // Horizontal add
    let h1 = _mm256_hadd_ps(sum, sum);
    let h2 = _mm256_hadd_ps(h1, h1);
    let mut arr = [0.0f32; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), h2);
    (arr[0] + arr[4]) as f64
}

fn l2_distance_fallback(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = (*x - *y) as f64;
            diff * diff
        })
        .sum()
}