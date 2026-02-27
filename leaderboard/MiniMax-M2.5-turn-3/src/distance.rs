use std::arch::x86_64::*;

/// Computes L2 (Euclidean) distance between two 128-dimensional vectors.
/// Uses AVX-512 SIMD instructions for maximum throughput.
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), 128);

    // Load all 128 floats of query vector once
    let a0 = _mm512_loadu_ps(a.as_ptr());
    let a1 = _mm512_loadu_ps(a.as_ptr().add(16));
    let a2 = _mm512_loadu_ps(a.as_ptr().add(32));
    let a3 = _mm512_loadu_ps(a.as_ptr().add(48));
    let a4 = _mm512_loadu_ps(a.as_ptr().add(64));
    let a5 = _mm512_loadu_ps(a.as_ptr().add(80));
    let a6 = _mm512_loadu_ps(a.as_ptr().add(96));
    let a7 = _mm512_loadu_ps(a.as_ptr().add(112));
    
    // First half
    let b0 = _mm512_loadu_ps(b.as_ptr());
    let diff0 = _mm512_sub_ps(a0, b0);
    let sum0 = _mm512_mul_ps(diff0, diff0);
    
    let b1 = _mm512_loadu_ps(b.as_ptr().add(16));
    let diff1 = _mm512_sub_ps(a1, b1);
    let sum1 = _mm512_mul_ps(diff1, diff1);
    
    let b2 = _mm512_loadu_ps(b.as_ptr().add(32));
    let diff2 = _mm512_sub_ps(a2, b2);
    let sum2 = _mm512_mul_ps(diff2, diff2);
    
    let b3 = _mm512_loadu_ps(b.as_ptr().add(48));
    let diff3 = _mm512_sub_ps(a3, b3);
    let sum3 = _mm512_mul_ps(diff3, diff3);
    
    // Second half
    let b4 = _mm512_loadu_ps(b.as_ptr().add(64));
    let diff4 = _mm512_sub_ps(a4, b4);
    let mut sum4 = _mm512_mul_ps(diff4, diff4);
    
    let b5 = _mm512_loadu_ps(b.as_ptr().add(80));
    let diff5 = _mm512_sub_ps(a5, b5);
    let sum5 = _mm512_mul_ps(diff5, diff5);
    
    let b6 = _mm512_loadu_ps(b.as_ptr().add(96));
    let diff6 = _mm512_sub_ps(a6, b6);
    let mut sum6 = _mm512_mul_ps(diff6, diff6);
    
    let b7 = _mm512_loadu_ps(b.as_ptr().add(112));
    let diff7 = _mm512_sub_ps(a7, b7);
    let sum7 = _mm512_mul_ps(diff7, diff7);
    
    // Sum all accumulators
    let mut sum01 = _mm512_add_ps(sum0, sum1);
    let mut sum23 = _mm512_add_ps(sum2, sum3);
    sum4 = _mm512_add_ps(sum4, sum5);
    sum6 = _mm512_add_ps(sum6, sum7);
    
    sum01 = _mm512_add_ps(sum01, sum23);
    let mut sum47 = _mm512_add_ps(sum4, sum6);
    
    let sum = _mm512_add_ps(sum01, sum47);

    // Sum all lanes of the AVX-512 register
    let mut sum_array = [0.0f32; 16];
    _mm512_storeu_ps(sum_array.as_mut_ptr(), sum);
    
    let mut total = 0.0f32;
    for v in &sum_array {
        total += v;
    }
    
    (total as f64).sqrt()
}

/// Computes L2 distance using AVX2 (fallback if AVX-512 not available)
#[target_feature(enable = "avx2")]
pub unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), 128);

    let mut sum = _mm256_setzero_ps();
    
    let mut i = 0;
    // Process 8 floats (256 bits) at a time
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        let sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
        i += 8;
    }
    
    // Handle remaining elements
    while i < a.len() {
        let diff = a[i] - b[i];
        sum = _mm256_add_ps(sum, _mm256_set1_ps(diff * diff));
        i += 1;
    }

    // Sum all lanes of the AVX-256 register
    // Extract high and low 128 bits and add
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum = _mm_add_ps(sum_low, sum_high);
    
    // Sum the 4 floats in the 128-bit register
    let mut sum_array = [0.0f32; 4];
    _mm_storeu_ps(sum_array.as_mut_ptr(), sum);
    
    let total = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    (total as f64).sqrt()
}

/// Scalar fallback for L2 distance
pub fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]) as f64;
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Main L2 distance function that uses the best available SIMD
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        panic!("Vector dimensions must match");
    }
    
    // For 128-dim vectors, use SIMD
    if a.len() == 128 {
        // Try AVX-512 first
        if is_x86_feature_detected!("avx512f") {
            return unsafe { l2_distance_avx512(a, b) };
        }
        // Fall back to AVX2
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_distance_avx2(a, b) };
        }
    }
    
    // Fallback to scalar
    l2_distance_scalar(a, b)
}
