#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512dq")]
unsafe fn l2_squared_distance_avx512_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;
    const LANES: usize = 16;
    let mut sums = _mm512_setzero_ps();
    while i + LANES <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        sums = _mm512_fmadd_ps(diff, diff, sums);
        i += LANES;
    }
    // horizontal sum
    let mut sum_arr = [0.0f32; LANES];
    _mm512_storeu_ps(sum_arr.as_mut_ptr(), sums);
    let mut sum = 0.0f32;
    for &s in &sum_arr {
        sum += s;
    }
    // remaining
    while i < len {
        let diff = a[i] - b[i];
        sum += diff * diff;
        i += 1;
    }
    sum
}

fn l2_squared_distance_fallback_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f32;
    for (&ai, &bi) in a.iter().zip(b) {
        let diff = ai - bi;
        sum += diff * diff;
    }
    sum
}

pub fn l2_squared_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
            return unsafe { l2_squared_distance_avx512_f32(a, b) };
        }
    }
    l2_squared_distance_fallback_f32(a, b)
}

pub fn l2_squared_distance(a: &[f32], b: &[f32]) -> f64 {
    l2_squared_distance_f32(a, b) as f64
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    l2_squared_distance_f32(a, b).sqrt() as f64
}