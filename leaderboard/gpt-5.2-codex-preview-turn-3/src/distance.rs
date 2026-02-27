#[derive(Copy, Clone)]
pub enum DistanceKind {
    Avx512,
    Avx2,
    Scalar,
}

pub fn select_distance_kind() -> DistanceKind {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return DistanceKind::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return DistanceKind::Avx2;
        }
    }
    DistanceKind::Scalar
}

#[inline(always)]
pub fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= a.len() {
        let av = _mm512_loadu_ps(a.as_ptr().add(i));
        let bv = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(av, bv);
        let prod = _mm512_mul_ps(diff, diff);
        sum = _mm512_add_ps(sum, prod);
        i += 16;
    }
    let mut total = _mm512_reduce_add_ps(sum);
    while i < a.len() {
        let diff = a.get_unchecked(i) - b.get_unchecked(i);
        total += diff * diff;
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= a.len() {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(av, bv);
        let prod = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, prod);
        i += 8;
    }
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut total: f32 = tmp.iter().sum();
    while i < a.len() {
        let diff = a.get_unchecked(i) - b.get_unchecked(i);
        total += diff * diff;
        i += 1;
    }
    total
}
