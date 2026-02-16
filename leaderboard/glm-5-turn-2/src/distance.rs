#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_avx512_impl(a: *const f32, b: *const f32) -> f64 {
    let mut sum = _mm512_setzero_ps();
    
    // Process 16 floats at a time (128 / 16 = 8 iterations)
    // Unroll the loop for better performance
    let va0 = _mm512_loadu_ps(a);
    let vb0 = _mm512_loadu_ps(b);
    let diff0 = _mm512_sub_ps(va0, vb0);
    sum = _mm512_fmadd_ps(diff0, diff0, sum);
    
    let va1 = _mm512_loadu_ps(a.add(16));
    let vb1 = _mm512_loadu_ps(b.add(16));
    let diff1 = _mm512_sub_ps(va1, vb1);
    sum = _mm512_fmadd_ps(diff1, diff1, sum);
    
    let va2 = _mm512_loadu_ps(a.add(32));
    let vb2 = _mm512_loadu_ps(b.add(32));
    let diff2 = _mm512_sub_ps(va2, vb2);
    sum = _mm512_fmadd_ps(diff2, diff2, sum);
    
    let va3 = _mm512_loadu_ps(a.add(48));
    let vb3 = _mm512_loadu_ps(b.add(48));
    let diff3 = _mm512_sub_ps(va3, vb3);
    sum = _mm512_fmadd_ps(diff3, diff3, sum);
    
    let va4 = _mm512_loadu_ps(a.add(64));
    let vb4 = _mm512_loadu_ps(b.add(64));
    let diff4 = _mm512_sub_ps(va4, vb4);
    sum = _mm512_fmadd_ps(diff4, diff4, sum);
    
    let va5 = _mm512_loadu_ps(a.add(80));
    let vb5 = _mm512_loadu_ps(b.add(80));
    let diff5 = _mm512_sub_ps(va5, vb5);
    sum = _mm512_fmadd_ps(diff5, diff5, sum);
    
    let va6 = _mm512_loadu_ps(a.add(96));
    let vb6 = _mm512_loadu_ps(b.add(96));
    let diff6 = _mm512_sub_ps(va6, vb6);
    sum = _mm512_fmadd_ps(diff6, diff6, sum);
    
    let va7 = _mm512_loadu_ps(a.add(112));
    let vb7 = _mm512_loadu_ps(b.add(112));
    let diff7 = _mm512_sub_ps(va7, vb7);
    sum = _mm512_fmadd_ps(diff7, diff7, sum);
    
    let result = _mm512_reduce_add_ps(sum);
    (result as f64).sqrt()
}

#[cfg(target_arch = "x86_64")]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    unsafe {
        if is_x86_feature_detected!("avx512f") {
            l2_distance_avx512_impl(a.as_ptr(), b.as_ptr())
        } else {
            l2_distance_fallback(a, b)
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    l2_distance_fallback(a, b)
}

fn l2_distance_fallback(a: &[f32], b: &[f32]) -> f64 {
    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt() as f64
}
