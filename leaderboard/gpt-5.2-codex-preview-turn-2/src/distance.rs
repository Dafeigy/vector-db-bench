#[derive(Clone, Copy)]
pub enum DistanceImpl {
    Avx512,
    Avx2,
    Scalar,
}

pub fn select_distance_impl() -> DistanceImpl {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            DistanceImpl::Avx512
        } else if std::is_x86_feature_detected!("avx2") {
            DistanceImpl::Avx2
        } else {
            DistanceImpl::Scalar
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        DistanceImpl::Scalar
    }
}

#[inline]
pub unsafe fn l2_distance_ptr(a: *const f32, b: *const f32, kind: DistanceImpl) -> f32 {
    match kind {
        DistanceImpl::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            {
                l2_distance_avx512_ptr(a, b)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                l2_distance_scalar_ptr(a, b)
            }
        }
        DistanceImpl::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            {
                l2_distance_avx2_ptr(a, b)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                l2_distance_scalar_ptr(a, b)
            }
        }
        DistanceImpl::Scalar => l2_distance_scalar_ptr(a, b),
    }
}

#[inline]
unsafe fn l2_distance_scalar_ptr(a: *const f32, b: *const f32) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 8 <= 128 {
        let d0 = *a.add(i) - *b.add(i);
        let d1 = *a.add(i + 1) - *b.add(i + 1);
        let d2 = *a.add(i + 2) - *b.add(i + 2);
        let d3 = *a.add(i + 3) - *b.add(i + 3);
        let d4 = *a.add(i + 4) - *b.add(i + 4);
        let d5 = *a.add(i + 5) - *b.add(i + 5);
        let d6 = *a.add(i + 6) - *b.add(i + 6);
        let d7 = *a.add(i + 7) - *b.add(i + 7);
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
        i += 8;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_avx512_ptr(a: *const f32, b: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i < 128 {
        let va = _mm512_loadu_ps(a.add(i));
        let vb = _mm512_loadu_ps(b.add(i));
        let diff = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(diff, diff, acc);
        i += 16;
    }
    _mm512_reduce_add_ps(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2_ptr(a: *const f32, b: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < 128 {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
        i += 8;
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums)
}
