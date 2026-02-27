#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn l2_distance_squared_avx512(a: &[f32], b: &[f32]) -> f32 {
    // For 128 dimensions, we use exactly 8 AVX-512 registers (16 floats each)
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    
    let a0 = _mm512_loadu_ps(a_ptr);
    let b0 = _mm512_loadu_ps(b_ptr);
    let diff0 = _mm512_sub_ps(a0, b0);
    let mut sum = _mm512_mul_ps(diff0, diff0);
    
    let a1 = _mm512_loadu_ps(a_ptr.add(16));
    let b1 = _mm512_loadu_ps(b_ptr.add(16));
    let diff1 = _mm512_sub_ps(a1, b1);
    sum = _mm512_fmadd_ps(diff1, diff1, sum);
    
    let a2 = _mm512_loadu_ps(a_ptr.add(32));
    let b2 = _mm512_loadu_ps(b_ptr.add(32));
    let diff2 = _mm512_sub_ps(a2, b2);
    sum = _mm512_fmadd_ps(diff2, diff2, sum);
    
    let a3 = _mm512_loadu_ps(a_ptr.add(48));
    let b3 = _mm512_loadu_ps(b_ptr.add(48));
    let diff3 = _mm512_sub_ps(a3, b3);
    sum = _mm512_fmadd_ps(diff3, diff3, sum);
    
    let a4 = _mm512_loadu_ps(a_ptr.add(64));
    let b4 = _mm512_loadu_ps(b_ptr.add(64));
    let diff4 = _mm512_sub_ps(a4, b4);
    sum = _mm512_fmadd_ps(diff4, diff4, sum);
    
    let a5 = _mm512_loadu_ps(a_ptr.add(80));
    let b5 = _mm512_loadu_ps(b_ptr.add(80));
    let diff5 = _mm512_sub_ps(a5, b5);
    sum = _mm512_fmadd_ps(diff5, diff5, sum);
    
    let a6 = _mm512_loadu_ps(a_ptr.add(96));
    let b6 = _mm512_loadu_ps(b_ptr.add(96));
    let diff6 = _mm512_sub_ps(a6, b6);
    sum = _mm512_fmadd_ps(diff6, diff6, sum);
    
    let a7 = _mm512_loadu_ps(a_ptr.add(112));
    let b7 = _mm512_loadu_ps(b_ptr.add(112));
    let diff7 = _mm512_sub_ps(a7, b7);
    sum = _mm512_fmadd_ps(diff7, diff7, sum);
    
    _mm512_reduce_add_ps(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn l2_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    
    let mut sum = _mm256_setzero_ps();
    
    for i in 0..16 {
        let idx = i * 8;
        let a_vec = _mm256_loadu_ps(a_ptr.add(idx));
        let b_vec = _mm256_loadu_ps(b_ptr.add(idx));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    // Horizontal sum
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    result.iter().sum::<f32>()
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            let squared = unsafe { l2_distance_squared_avx512(a, b) };
            return (squared as f64).sqrt();
        } else if is_x86_feature_detected!("avx2") {
            let squared = unsafe { l2_distance_squared_avx2(a, b) };
            return (squared as f64).sqrt();
        }
    }
    
    l2_distance_scalar(a, b)
}

#[inline]
fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            (diff * diff) as f64
        })
        .sum::<f64>()
        .sqrt()
}
