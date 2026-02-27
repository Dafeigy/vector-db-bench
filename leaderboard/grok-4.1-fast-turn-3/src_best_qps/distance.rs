use std::arch::x86_64::{
    _mm512_add_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_reduce_add_ps,
    _mm512_setzero_ps, _mm512_sub_ps,
};

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn avx512_l2_squared(a: *const f32, b: *const f32) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let lanes = 128 / 16;
    for i in 0..lanes {
        let qa = _mm512_loadu_ps(a.add(i * 16));
        let qb = _mm512_loadu_ps(b.add(i * 16));
        let diff = _mm512_sub_ps(qa, qb);
        let sq = _mm512_mul_ps(diff, diff);
        sum = _mm512_add_ps(sum, sq);
    }
    _mm512_reduce_add_ps(sum)
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), 128);
    unsafe {
        let sq_f32: f32 = avx512_l2_squared(a.as_ptr(), b.as_ptr());
        (sq_f32 as f64).sqrt()
    }
}