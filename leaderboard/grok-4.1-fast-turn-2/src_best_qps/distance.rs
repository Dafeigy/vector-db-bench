use std::arch::x86_64::*;
use std::sync::RwLock;

#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    unsafe { l2_distance_simd(a, b) }
}

#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_simd(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);

    let mut sum_sq = _mm512_setzero_ps();
    let ap = a.as_ptr() as *const f32;
    let bp = b.as_ptr() as *const f32;

    for chunk in 0..8usize {
        let offset = chunk * 16;
        let va = _mm512_loadu_ps(ap.add(offset));
        let vb = _mm512_loadu_ps(bp.add(offset));
        let diff = _mm512_sub_ps(va, vb);
        let sq = _mm512_mul_ps(diff, diff);
        sum_sq = _mm512_add_ps(sum_sq, sq);
    }

    let sum_f32 = _mm512_reduce_add_ps(sum_sq);
    (sum_f32 as f64).sqrt()
}
