/// Computes L2 (Euclidean) distance between two vectors using AVX2 SIMD.
/// Returns distance as f64 for better precision with large sums.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_distance_avx2(a, b) };
        }
    }
    
    // Fallback to scalar
    let mut sum: f64 = 0.0;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]) as f64;
        sum += diff * diff;
    }
    sum.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    
    let len = a.len();
    let mut sum256: __m256 = _mm256_setzero_ps();
    
    let mut i = 0;
    
    // Process 8 floats at a time with AVX2 (256-bit)
    while i + 7 < len {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        let sq = _mm256_mul_ps(diff, diff);
        sum256 = _mm256_add_ps(sum256, sq);
        i += 8;
    }
    
    // Horizontal add on 256-bit register to get 128-bit
    let sum128_from_256 = _mm256_castps256_ps128(sum256);
    let sum_high = _mm256_extractf128_ps(sum256, 1);
    let sum128_combined = _mm_add_ps(sum128_from_256, sum_high);
    
    // Horizontal add on 128-bit to get scalar
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum128_combined);
    
    let mut sum = (result[0] + result[1] + result[2] + result[3]) as f64;
    
    // Process remaining scalars
    while i < len {
        let diff = (a[i] - b[i]) as f64;
        sum += diff * diff;
        i += 1;
    }
    
    sum.sqrt()
}
