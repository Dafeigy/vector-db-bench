use std::arch::x86_64::*;

#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    l2_distance_f32(a, b) as f64
}

#[inline]
pub fn l2_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 128 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    return l2_distance_128_avx512(a, b);
                }
            }
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return l2_distance_128_avx2(a, b);
                }
            }
        }
    }
    l2_distance_scalar(a, b, len)
}

#[inline]
fn l2_distance_scalar(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    while i < len {
        let diff = a[i] - b[i];
        sum += diff * diff;
        i += 1;
    }
    sum
}

#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_128_avx512(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let mut i = 0;
    while i < 128 {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
        i += 16;
    }
    _mm512_reduce_add_ps(sum)
}

#[target_feature(enable = "avx2")]
unsafe fn l2_distance_128_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0;
    while i < 128 {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let diff0 = _mm256_sub_ps(va0, vb0);
        let diff1 = _mm256_sub_ps(va1, vb1);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        i += 16;
    }
    let sum = _mm256_add_ps(sum0, sum1);
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    tmp.iter().sum()
}
