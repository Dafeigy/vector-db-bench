use std::arch::x86_64::*;

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    squared_distance(a, b).sqrt()
}

pub fn squared_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    // Use SIMD if available
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { squared_distance_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { squared_distance_avx2(a, b) };
        }
    }
    squared_distance_scalar(a, b)
}

fn squared_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let diff = a[i] as f64 - b[i] as f64;
        sum += diff * diff;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn squared_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut acc = _mm512_setzero_ps();
    // Process 16 floats per iteration
    for i in (0..128).step_by(16) {
        let va = _mm512_loadu_ps(a_ptr.add(i));
        let vb = _mm512_loadu_ps(b_ptr.add(i));
        let diff = _mm512_sub_ps(va, vb);
        let diff_sq = _mm512_mul_ps(diff, diff);
        acc = _mm512_add_ps(acc, diff_sq);
    }
    // Horizontal sum across 16 lanes
    let sum = _mm512_reduce_add_ps(acc); // Requires AVX512F
    sum as f64
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn squared_distance_avx2(a: &[f32], b: &[f32]) -> f64 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    // Process 8 floats per iteration
    for i in (0..128).step_by(8) {
        let va = _mm256_loadu_ps(a_ptr.add(i));
        let vb = _mm256_loadu_ps(b_ptr.add(i));
        let diff = _mm256_sub_ps(va, vb);
        let diff_sq = _mm256_mul_ps(diff, diff);
        if i % 16 == 0 {
            acc0 = _mm256_add_ps(acc0, diff_sq);
        } else {
            acc1 = _mm256_add_ps(acc1, diff_sq);
        }
    }
    // Sum the two accumulators
    let acc = _mm256_add_ps(acc0, acc1);
    // Horizontal sum
    let shuf = _mm256_shuffle_ps(acc, acc, 0b01001110);
    let sum256 = _mm256_add_ps(acc, shuf);
    let sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
    let sum = _mm_cvtss_f32(sum128);
    sum as f64
}