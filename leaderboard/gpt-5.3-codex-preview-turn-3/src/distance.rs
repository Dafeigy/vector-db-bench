#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    l2_distance_squared(a, b).sqrt()
}

#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if a.len() == 128 && std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: guarded by runtime feature detection and fixed length check.
            return unsafe { l2_distance_sq_128_avx512(a, b) };
        }
    }

    l2_distance_sq_scalar(a, b)
}

#[inline]
fn l2_distance_sq_scalar(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]) as f64;
        sum += d * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_sq_128_avx512(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc = _mm512_setzero_ps();

    macro_rules! step {
        ($offset:expr) => {{
            let va = _mm512_loadu_ps(pa.add($offset));
            let vb = _mm512_loadu_ps(pb.add($offset));
            let diff = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }};
    }

    step!(0);
    step!(16);
    step!(32);
    step!(48);
    step!(64);
    step!(80);
    step!(96);
    step!(112);

    _mm512_reduce_add_ps(acc) as f64
}
