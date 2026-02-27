use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_l2_dist_sq(query_ptr: *const f32, vec_ptr: *const f32) -> f64 {
    let mut sum = 0.0f64;
    let mut i = 0isize;
    while i < 128 {
        let q = _mm512_loadu_ps(query_ptr.offset(i));
        let v = _mm512_loadu_ps(vec_ptr.offset(i));
        let diff = _mm512_sub_ps(q, v);
        let sq = _mm512_mul_ps(diff, diff);
        sum += _mm512_reduce_add_ps(sq) as f64;
        i += 16;
    }
    sum
}

pub fn l2_dist_sq(a: &[f32], b: &[f32]) -> f64 {
    unsafe {
        #[cfg(all(target_arch = "x86_64"))]
        {
            avx512_l2_dist_sq(a.as_ptr(), b.as_ptr())
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut sum = 0.0f64;
            for i in 0..128 {
                let d = (a[i] as f64) - (b[i] as f64);
                sum += d * d;
            }
            sum
        }
    }
}
