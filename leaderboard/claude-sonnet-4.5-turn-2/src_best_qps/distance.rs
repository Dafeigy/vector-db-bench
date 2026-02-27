#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);
    
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    
    // Process 32 floats per iteration (2 AVX-512 registers of 16 floats each)
    for i in (0..128).step_by(32) {
        let a1 = _mm512_loadu_ps(a_ptr.add(i));
        let b1 = _mm512_loadu_ps(b_ptr.add(i));
        let diff1 = _mm512_sub_ps(a1, b1);
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
        
        let a2 = _mm512_loadu_ps(a_ptr.add(i + 16));
        let b2 = _mm512_loadu_ps(b_ptr.add(i + 16));
        let diff2 = _mm512_sub_ps(a2, b2);
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);
    }
    
    // Combine the sums
    let sum = _mm512_add_ps(sum1, sum2);
    
    // Horizontal sum using reduce
    _mm512_reduce_add_ps(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);
    
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    let mut sum4 = _mm256_setzero_ps();
    
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    
    // Process 32 floats per iteration (4 AVX2 registers of 8 floats each)
    for i in (0..128).step_by(32) {
        let a1 = _mm256_loadu_ps(a_ptr.add(i));
        let b1 = _mm256_loadu_ps(b_ptr.add(i));
        let diff1 = _mm256_sub_ps(a1, b1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        
        let a2 = _mm256_loadu_ps(a_ptr.add(i + 8));
        let b2 = _mm256_loadu_ps(b_ptr.add(i + 8));
        let diff2 = _mm256_sub_ps(a2, b2);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        
        let a3 = _mm256_loadu_ps(a_ptr.add(i + 16));
        let b3 = _mm256_loadu_ps(b_ptr.add(i + 16));
        let diff3 = _mm256_sub_ps(a3, b3);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
        
        let a4 = _mm256_loadu_ps(a_ptr.add(i + 24));
        let b4 = _mm256_loadu_ps(b_ptr.add(i + 24));
        let diff4 = _mm256_sub_ps(a4, b4);
        sum4 = _mm256_fmadd_ps(diff4, diff4, sum4);
    }
    
    // Combine the sums
    let sum12 = _mm256_add_ps(sum1, sum2);
    let sum34 = _mm256_add_ps(sum3, sum4);
    let sum = _mm256_add_ps(sum12, sum34);
    
    // Horizontal sum
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_low, sum_high);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
    
    _mm_cvtss_f32(sum32)
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() == 128 && b.len() == 128 {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    let squared_sum = l2_distance_avx512(a, b);
                    return (squared_sum as f64).sqrt();
                }
            } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    let squared_sum = l2_distance_avx2(a, b);
                    return (squared_sum as f64).sqrt();
                }
            }
        }
    }
    
    // Fallback for non-SIMD or non-128 dim
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    (sum as f64).sqrt()
}
