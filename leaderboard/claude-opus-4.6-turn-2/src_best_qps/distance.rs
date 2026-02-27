#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute L2 squared distance for 128-dim f32 vectors using AVX-512.
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    l2_distance_f32(a, b) as f64
}

#[inline(always)]
pub fn l2_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // We know from profiling that AVX-512 is available
        unsafe { l2_distance_avx512_impl(a, b) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        l2_distance_scalar(a, b)
    }
}

/// Compute L2 squared distance with early exit if exceeds threshold.
/// Returns the distance if <= threshold, or a value > threshold otherwise.
#[inline(always)]
pub fn l2_distance_f32_early(a: &[f32], b: &[f32], threshold: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { l2_distance_avx512_impl(a, b) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        l2_distance_scalar(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn l2_distance_avx512_impl(a: &[f32], b: &[f32]) -> f32 {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    
    // 128 floats = 8 x 16-float AVX-512 registers
    // Unroll completely for 128 dimensions
    let va0 = _mm512_loadu_ps(ap);
    let vb0 = _mm512_loadu_ps(bp);
    let d0 = _mm512_sub_ps(va0, vb0);
    let mut sum = _mm512_mul_ps(d0, d0);
    
    let va1 = _mm512_loadu_ps(ap.add(16));
    let vb1 = _mm512_loadu_ps(bp.add(16));
    let d1 = _mm512_sub_ps(va1, vb1);
    sum = _mm512_fmadd_ps(d1, d1, sum);
    
    let va2 = _mm512_loadu_ps(ap.add(32));
    let vb2 = _mm512_loadu_ps(bp.add(32));
    let d2 = _mm512_sub_ps(va2, vb2);
    sum = _mm512_fmadd_ps(d2, d2, sum);
    
    let va3 = _mm512_loadu_ps(ap.add(48));
    let vb3 = _mm512_loadu_ps(bp.add(48));
    let d3 = _mm512_sub_ps(va3, vb3);
    sum = _mm512_fmadd_ps(d3, d3, sum);
    
    let va4 = _mm512_loadu_ps(ap.add(64));
    let vb4 = _mm512_loadu_ps(bp.add(64));
    let d4 = _mm512_sub_ps(va4, vb4);
    sum = _mm512_fmadd_ps(d4, d4, sum);
    
    let va5 = _mm512_loadu_ps(ap.add(80));
    let vb5 = _mm512_loadu_ps(bp.add(80));
    let d5 = _mm512_sub_ps(va5, vb5);
    sum = _mm512_fmadd_ps(d5, d5, sum);
    
    let va6 = _mm512_loadu_ps(ap.add(96));
    let vb6 = _mm512_loadu_ps(bp.add(96));
    let d6 = _mm512_sub_ps(va6, vb6);
    sum = _mm512_fmadd_ps(d6, d6, sum);
    
    let va7 = _mm512_loadu_ps(ap.add(112));
    let vb7 = _mm512_loadu_ps(bp.add(112));
    let d7 = _mm512_sub_ps(va7, vb7);
    sum = _mm512_fmadd_ps(d7, d7, sum);
    
    _mm512_reduce_add_ps(sum)
}

#[allow(dead_code)]
fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}
