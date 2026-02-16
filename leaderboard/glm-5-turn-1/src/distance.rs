#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute L2 (Euclidean) distance between two vectors.
/// Uses AVX-512 SIMD when available for optimal performance.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() >= 128);
    
    let mut sum = _mm512_setzero_ps();
    
    // Process 16 floats at a time
    for i in (0..128).step_by(16) {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }
    
    // Horizontal sum of 16 floats
    _mm512_reduce_add_ps(sum) as f64
}

/// Compute L2 distance using AVX2 (fallback if AVX-512 not available)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    
    // Process 32 floats at a time (4 x 8-wide AVX registers)
    for i in (0..128).step_by(32) {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
        
        let diff0 = _mm256_sub_ps(va0, vb0);
        let diff1 = _mm256_sub_ps(va1, vb1);
        let diff2 = _mm256_sub_ps(va2, vb2);
        let diff3 = _mm256_sub_ps(va3, vb3);
        
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }
    
    // Combine sums
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    
    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum0, 1);
    let lo = _mm256_castps256_ps128(sum0);
    let sum128 = _mm_add_ps(hi, lo);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    
    _mm_cvtss_f32(sum128) as f64
}

/// Scalar fallback for non-x86_64 platforms
pub fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = *x - *y;
            diff * diff
        })
        .sum::<f32>() as f64
}

/// Main entry point - dispatches to the best available implementation
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx512_supported() {
            unsafe { l2_distance_avx512(a, b) }
        } else if is_avx2_supported() {
            unsafe { l2_distance_avx2(a, b) }
        } else {
            l2_distance_scalar(a, b)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        l2_distance_scalar(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
fn is_avx512_supported() -> bool {
    use std::arch::x86_64::__cpuid;
    unsafe {
        let eax7 = __cpuid(7);
        // Check AVX512F (bit 16 of ebx)
        (eax7.ebx & (1 << 16)) != 0
    }
}

#[cfg(target_arch = "x86_64")]
fn is_avx2_supported() -> bool {
    use std::arch::x86_64::__cpuid;
    unsafe {
        let eax7 = __cpuid(7);
        // Check AVX2 (bit 5 of ebx)
        (eax7.ebx & (1 << 5)) != 0
    }
}
