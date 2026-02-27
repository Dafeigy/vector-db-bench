use std::arch::x86_64::*;

#[inline(always)]
pub fn sq_l2_distance(a: &[f32], b: &[f32]) -> f64 {
    if std::arch::is_x86_feature_detected!("avx512f") {
        unsafe { sq_l2_avx512(a.as_ptr(), b.as_ptr()) }
    } else {
        scalar_sq_l2(a, b)
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn sq_l2_avx512(aptr: *const f32, bptr: *const f32) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..8 {
        let va = _mm512_loadu_ps(aptr.add(i * 16));
        let vb = _mm512_loadu_ps(bptr.add(i * 16));
        let diff = _mm512_sub_ps(va, vb);
        let sq = _mm512_mul_ps(diff, diff);
        sum += horizontal_sum(sq) as f64;
    }
    sum
}

unsafe fn horizontal_sum(v: __m512) -> f32 {
    let sum0 = _mm512_extractf32x4_ps::<0>(v);
    let sum1 = _mm512_extractf32x4_ps::<1>(v);
    let sum2 = _mm512_extractf32x4_ps::<2>(v);
    let sum3 = _mm512_extractf32x4_ps::<3>(v);
    let sum01 = _mm_add_ps(sum0, sum1);
    let sum23 = _mm_add_ps(sum2, sum3);
    let sum_ab = _mm_add_ps(sum01, sum23);
    let sum_high = _mm_movehl_ps(sum_ab, sum_ab);
    let sum_a = _mm_add_ps(sum_ab, sum_high);
    let sum_b = _mm_hadd_ps(sum_a, sum_a);
    let sum_c = _mm_hadd_ps(sum_b, sum_b);
    _mm_cvtss_f32(sum_c)
}

fn scalar_sq_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..128 {
        let d = (a[i] - b[i]) as f64;
        sum += d * d;
    }
    sum
}
