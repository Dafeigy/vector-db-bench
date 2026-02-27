use std::arch::x86_64::*;

// Requires AVX512F
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn avx512_l2_sq(a: *const f32, b: *const f32) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let lanes = 128 / 16;
    for _ in 0..lanes {
        let qa = _mm512_loadu_ps(a);
        let qb = _mm512_loadu_ps(b);
        let diff = _mm512_sub_ps(qa, qb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
        a = a.add(16);
        b = b.add(16);
    }
    _mm512_reduce_add_ps(sum)
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), 128);
    unsafe {
        let sq_dist_f32 = avx512_l2_sq(a.as_ptr(), b.as_ptr());
        (sq_dist_f32 as f64).sqrt()
    }
}