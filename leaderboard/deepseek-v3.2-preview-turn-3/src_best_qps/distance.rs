#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { l2_distance_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_distance_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.2") {
            return unsafe { l2_distance_sse(a, b) };
        }
    }
    l2_distance_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i < a.len() {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(diff, diff, acc);
        i += 16;
    }
    // reduce acc to scalar
    let sum = reduce512_ps(acc);
    (sum as f64).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn reduce512_ps(v: __m512) -> f32 {
    // store to array and sum (simple, possibly slower but correct)
    let mut buf = [0.0f32; 16];
    _mm512_storeu_ps(buf.as_mut_ptr(), v);
    buf.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f64 {
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut i = 0;
    while i < a.len() {
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
        let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let diff2 = _mm256_sub_ps(va2, vb2);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);
        i += 16;
    }
    // horizontal sum
    let sum = reduce256_ps(acc1) + reduce256_ps(acc2);
    (sum as f64).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn reduce256_ps(v: __m256) -> f32 {
    // extract high/low 128 and add
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_extractf128_ps(v, 0);
    let sum128 = _mm_add_ps(hi, lo);
    reduce128_ps(sum128)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn l2_distance_sse(a: &[f32], b: &[f32]) -> f64 {
    let mut acc1 = _mm_setzero_ps();
    let mut acc2 = _mm_setzero_ps();
    let mut acc3 = _mm_setzero_ps();
    let mut acc4 = _mm_setzero_ps();
    let mut i = 0;
    while i < a.len() {
        let va1 = _mm_loadu_ps(a.as_ptr().add(i));
        let vb1 = _mm_loadu_ps(b.as_ptr().add(i));
        let diff1 = _mm_sub_ps(va1, vb1);
        acc1 = _mm_fmadd_ps(diff1, diff1, acc1);
        let va2 = _mm_loadu_ps(a.as_ptr().add(i + 4));
        let vb2 = _mm_loadu_ps(b.as_ptr().add(i + 4));
        let diff2 = _mm_sub_ps(va2, vb2);
        acc2 = _mm_fmadd_ps(diff2, diff2, acc2);
        let va3 = _mm_loadu_ps(a.as_ptr().add(i + 8));
        let vb3 = _mm_loadu_ps(b.as_ptr().add(i + 8));
        let diff3 = _mm_sub_ps(va3, vb3);
        acc3 = _mm_fmadd_ps(diff3, diff3, acc3);
        let va4 = _mm_loadu_ps(a.as_ptr().add(i + 12));
        let vb4 = _mm_loadu_ps(b.as_ptr().add(i + 12));
        let diff4 = _mm_sub_ps(va4, vb4);
        acc4 = _mm_fmadd_ps(diff4, diff4, acc4);
        i += 16;
    }
    let sum = reduce128_ps(acc1) + reduce128_ps(acc2) +
               reduce128_ps(acc3) + reduce128_ps(acc4);
    (sum as f64).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn reduce128_ps(v: __m128) -> f32 {
    let shuf = _mm_shuffle_ps(v, v, 0b10110001);
    let sum1 = _mm_add_ps(v, shuf);
    let shuf2 = _mm_shuffle_ps(sum1, sum1, 0b01001110);
    let sum2 = _mm_add_ps(sum1, shuf2);
    _mm_cvtss_f32(sum2)
}

fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let diff = *x as f64 - *y as f64;
        sum += diff * diff;
    }
    sum.sqrt()
}