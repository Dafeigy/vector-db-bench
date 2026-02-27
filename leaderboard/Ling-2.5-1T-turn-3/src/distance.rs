use std::arch::x86_64::*;

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { l2_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_avx2(a, b) };
        }
    }

    // Scalar fallback with fused multiply-add via manual loop
    let mut sum = 0.0f64;
    for i in 0..n {
        let diff = (a[i] as f64) - (b[i] as f64);
        sum += diff * diff;
    }
    sum.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_avx512(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let mut i = 0;
    // Use 4 accumulators to hide latency
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();

    while i + 64 <= n {
        let va0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm512_sub_ps(va0, vb0);
        acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

        let va1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let vb1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        let diff1 = _mm512_sub_ps(va1, vb1);
        acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

        let va2 = _mm512_loadu_ps(a.as_ptr().add(i + 32));
        let vb2 = _mm512_loadu_ps(b.as_ptr().add(i + 32));
        let diff2 = _mm512_sub_ps(va2, vb2);
        acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

        let va3 = _mm512_loadu_ps(a.as_ptr().add(i + 48));
        let vb3 = _mm512_loadu_ps(b.as_ptr().add(i + 48));
        let diff3 = _mm512_sub_ps(va3, vb3);
        acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);

        i += 64;
    }

    // Handle remaining 32 elements
    if i + 32 <= n {
        let va0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm512_sub_ps(va0, vb0);
        acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);
        i += 32;
    }

    // Combine accumulators
    let sum_ps = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut sum = _mm512_reduce_add_ps(sum_ps) as f64;

    // Tail
    while i < n {
        let diff = (a[i] as f64) - (b[i] as f64);
        sum += diff * diff;
        i += 1;
    }

    sum.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_avx2(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let mut i = 0;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 32 <= n {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let diff1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        let diff2 = _mm256_sub_ps(va2, vb2);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

        let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
        let diff3 = _mm256_sub_ps(va3, vb3);
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);

        i += 32;
    }

    // Handle remaining 16 elements
    if i + 16 <= n {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
        i += 16;
    }

    // Combine
    let sum_ps = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut sum = horizontal_sum_ps(sum_ps) as f64;

    while i < n {
        let diff = (a[i] as f64) - (b[i] as f64);
        sum += diff * diff;
        i += 1;
    }

    sum.sqrt()
}

#[cfg(target_arch = "x86_64")]
unsafe fn horizontal_sum_ps(v: __m256) -> f32 {
    let v64 = _mm256_castps256_ps128(v);
    let v64_high = _mm256_extractf128_ps(v, 1);
    let v128 = _mm_add_ps(v64, v64_high);
    let v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
    let v64 = _mm_add_ss(v64, _mm_shuffle_ps(v64, v64, 0x1));
    _mm_cvtss_f32(v64)
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn l2_avx512(_a: &[f32], _b: &[f32]) -> f64 {
    unreachable!()
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn l2_avx2(_a: &[f32], _b: &[f32]) -> f64 {
    unreachable!()
}
