#[inline(always)]
fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f64;

    let mut i = 0usize;
    while i + 7 < len {
        let d0 = (a[i] - b[i]) as f64;
        let d1 = (a[i + 1] - b[i + 1]) as f64;
        let d2 = (a[i + 2] - b[i + 2]) as f64;
        let d3 = (a[i + 3] - b[i + 3]) as f64;
        let d4 = (a[i + 4] - b[i + 4]) as f64;
        let d5 = (a[i + 5] - b[i + 5]) as f64;
        let d6 = (a[i + 6] - b[i + 6]) as f64;
        let d7 = (a[i + 7] - b[i + 7]) as f64;
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
        i += 8;
    }

    while i < len {
        let d = (a[i] - b[i]) as f64;
        sum += d * d;
        i += 1;
    }

    sum.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_avx512_128(a: *const f32, b: *const f32) -> f64 {
    use std::arch::x86_64::*;

    let mut acc = _mm512_setzero_ps();
    let mut off = 0isize;
    while off < 128 {
        let va = _mm512_loadu_ps(a.offset(off));
        let vb = _mm512_loadu_ps(b.offset(off));
        let d = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(d, d, acc);
        off += 16;
    }

    (_mm512_reduce_add_ps(acc) as f64).sqrt()
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    if a.len() == 128
        && b.len() == 128
        && {
            #[cfg(target_arch = "x86_64")]
            {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        }
    {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return l2_distance_avx512_128(a.as_ptr(), b.as_ptr());
        }
    }

    l2_distance_scalar(a, b)
}
