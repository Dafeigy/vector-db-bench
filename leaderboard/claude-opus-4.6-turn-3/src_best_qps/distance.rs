#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute squared L2 distance between two 128-dimensional f32 vectors.
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    unsafe { l2_distance_avx512_inner(a, b) as f64 }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn l2_distance_avx512_inner(a: &[f32], b: &[f32]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // 128 floats = 8 AVX-512 registers, unrolled as 4+4
    let va0 = _mm512_loadu_ps(a_ptr);
    let vb0 = _mm512_loadu_ps(b_ptr);
    let d0 = _mm512_sub_ps(va0, vb0);
    let mut sum0 = _mm512_mul_ps(d0, d0);

    let va1 = _mm512_loadu_ps(a_ptr.add(16));
    let vb1 = _mm512_loadu_ps(b_ptr.add(16));
    let d1 = _mm512_sub_ps(va1, vb1);
    let mut sum1 = _mm512_mul_ps(d1, d1);

    let va2 = _mm512_loadu_ps(a_ptr.add(32));
    let vb2 = _mm512_loadu_ps(b_ptr.add(32));
    let d2 = _mm512_sub_ps(va2, vb2);
    sum0 = _mm512_fmadd_ps(d2, d2, sum0);

    let va3 = _mm512_loadu_ps(a_ptr.add(48));
    let vb3 = _mm512_loadu_ps(b_ptr.add(48));
    let d3 = _mm512_sub_ps(va3, vb3);
    sum1 = _mm512_fmadd_ps(d3, d3, sum1);

    let va4 = _mm512_loadu_ps(a_ptr.add(64));
    let vb4 = _mm512_loadu_ps(b_ptr.add(64));
    let d4 = _mm512_sub_ps(va4, vb4);
    sum0 = _mm512_fmadd_ps(d4, d4, sum0);

    let va5 = _mm512_loadu_ps(a_ptr.add(80));
    let vb5 = _mm512_loadu_ps(b_ptr.add(80));
    let d5 = _mm512_sub_ps(va5, vb5);
    sum1 = _mm512_fmadd_ps(d5, d5, sum1);

    let va6 = _mm512_loadu_ps(a_ptr.add(96));
    let vb6 = _mm512_loadu_ps(b_ptr.add(96));
    let d6 = _mm512_sub_ps(va6, vb6);
    sum0 = _mm512_fmadd_ps(d6, d6, sum0);

    let va7 = _mm512_loadu_ps(a_ptr.add(112));
    let vb7 = _mm512_loadu_ps(b_ptr.add(112));
    let d7 = _mm512_sub_ps(va7, vb7);
    sum1 = _mm512_fmadd_ps(d7, d7, sum1);

    let sum = _mm512_add_ps(sum0, sum1);
    _mm512_reduce_add_ps(sum)
}

/// Fast L2 distance, always uses AVX-512
#[inline(always)]
pub unsafe fn l2_distance_fast(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_avx512_inner(a, b)
}

/// Compute L2 distances from query to a contiguous block of vectors
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_distances_batch(
    query: &[f32],
    vectors: &[f32],
    n: usize,
    results: &mut [f32],
) {
    for i in 0..n {
        let b = std::slice::from_raw_parts(vectors.as_ptr().add(i * 128), 128);
        *results.get_unchecked_mut(i) = l2_distance_avx512_inner(query, b);
    }
}
