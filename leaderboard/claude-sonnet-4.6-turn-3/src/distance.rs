/// Compute squared L2 distance between two 128-dim f32 vectors.
/// Returns the squared distance as f64 (avoids sqrt for comparisons).
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    l2_squared_f64(a, b)
}

/// Squared L2 distance as f64.
#[inline(always)]
pub fn l2_squared_f64(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_feature = "avx512f")]
    {
        unsafe { l2_squared_avx512(a, b) }
    }
    #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
    {
        unsafe { l2_squared_avx2(a, b) }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
    {
        l2_squared_scalar(a, b)
    }
}

/// Squared L2 distance returned as f32 (faster for ranking).
#[inline(always)]
pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_feature = "avx512f")]
    {
        unsafe { l2_squared_avx512_f32(a, b) }
    }
    #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
    {
        unsafe { l2_squared_avx2_f32(a, b) }
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
    {
        l2_squared_scalar_f32(a, b)
    }
}

#[allow(dead_code)]
#[inline(always)]
fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| {
        let d = (*x - *y) as f64;
        d * d
    }).sum()
}

#[allow(dead_code)]
#[inline(always)]
fn l2_squared_scalar_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| {
        let d = *x - *y;
        d * d
    }).sum()
}

#[cfg(target_feature = "avx512f")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_squared_avx512(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    
    // Process 16 floats at a time with AVX-512
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();
    
    let chunks = n / 64;
    let mut i = 0usize;
    
    for _ in 0..chunks {
        let a0 = _mm512_loadu_ps(ap.add(i));
        let b0 = _mm512_loadu_ps(bp.add(i));
        let d0 = _mm512_sub_ps(a0, b0);
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);
        
        let a1 = _mm512_loadu_ps(ap.add(i + 16));
        let b1 = _mm512_loadu_ps(bp.add(i + 16));
        let d1 = _mm512_sub_ps(a1, b1);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
        
        let a2 = _mm512_loadu_ps(ap.add(i + 32));
        let b2 = _mm512_loadu_ps(bp.add(i + 32));
        let d2 = _mm512_sub_ps(a2, b2);
        sum2 = _mm512_fmadd_ps(d2, d2, sum2);
        
        let a3 = _mm512_loadu_ps(ap.add(i + 48));
        let b3 = _mm512_loadu_ps(bp.add(i + 48));
        let d3 = _mm512_sub_ps(a3, b3);
        sum3 = _mm512_fmadd_ps(d3, d3, sum3);
        
        i += 64;
    }
    
    // 128-dim => 2 chunks of 64, so handle remainder (128 % 64 = 0, perfect)
    // Handle remaining in groups of 16
    let chunks16 = (n - i) / 16;
    for _ in 0..chunks16 {
        let av = _mm512_loadu_ps(ap.add(i));
        let bv = _mm512_loadu_ps(bp.add(i));
        let d = _mm512_sub_ps(av, bv);
        sum0 = _mm512_fmadd_ps(d, d, sum0);
        i += 16;
    }
    
    // Combine
    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);
    
    _mm512_reduce_add_ps(sum0) as f64
}

#[cfg(target_feature = "avx512f")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_squared_avx512_f32(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();
    
    let chunks = n / 64;
    let mut i = 0usize;
    
    for _ in 0..chunks {
        let a0 = _mm512_loadu_ps(ap.add(i));
        let b0 = _mm512_loadu_ps(bp.add(i));
        let d0 = _mm512_sub_ps(a0, b0);
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);
        
        let a1 = _mm512_loadu_ps(ap.add(i + 16));
        let b1 = _mm512_loadu_ps(bp.add(i + 16));
        let d1 = _mm512_sub_ps(a1, b1);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
        
        let a2 = _mm512_loadu_ps(ap.add(i + 32));
        let b2 = _mm512_loadu_ps(bp.add(i + 32));
        let d2 = _mm512_sub_ps(a2, b2);
        sum2 = _mm512_fmadd_ps(d2, d2, sum2);
        
        let a3 = _mm512_loadu_ps(ap.add(i + 48));
        let b3 = _mm512_loadu_ps(bp.add(i + 48));
        let d3 = _mm512_sub_ps(a3, b3);
        sum3 = _mm512_fmadd_ps(d3, d3, sum3);
        
        i += 64;
    }
    
    let chunks16 = (n - i) / 16;
    for _ in 0..chunks16 {
        let av = _mm512_loadu_ps(ap.add(i));
        let bv = _mm512_loadu_ps(bp.add(i));
        let d = _mm512_sub_ps(av, bv);
        sum0 = _mm512_fmadd_ps(d, d, sum0);
        i += 16;
    }
    
    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);
    
    _mm512_reduce_add_ps(sum0)
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    
    let chunks = n / 32;
    let mut i = 0usize;
    
    for _ in 0..chunks {
        let a0 = _mm256_loadu_ps(ap.add(i));
        let b0 = _mm256_loadu_ps(bp.add(i));
        let d0 = _mm256_sub_ps(a0, b0);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        
        let a1 = _mm256_loadu_ps(ap.add(i + 8));
        let b1 = _mm256_loadu_ps(bp.add(i + 8));
        let d1 = _mm256_sub_ps(a1, b1);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
        
        let a2 = _mm256_loadu_ps(ap.add(i + 16));
        let b2 = _mm256_loadu_ps(bp.add(i + 16));
        let d2 = _mm256_sub_ps(a2, b2);
        sum2 = _mm256_fmadd_ps(d2, d2, sum2);
        
        let a3 = _mm256_loadu_ps(ap.add(i + 24));
        let b3 = _mm256_loadu_ps(bp.add(i + 24));
        let d3 = _mm256_sub_ps(a3, b3);
        sum3 = _mm256_fmadd_ps(d3, d3, sum3);
        
        i += 32;
    }
    
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    
    // Horizontal sum
    let low = _mm256_castps256_ps128(sum0);
    let high = _mm256_extractf128_ps(sum0, 1);
    let sum128 = _mm_add_ps(low, high);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x4e);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0xb1);
    let result = _mm_add_ps(sums, shuf2);
    _mm_cvtss_f32(result) as f64
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_squared_avx2_f32(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    
    let chunks = n / 32;
    let mut i = 0usize;
    
    for _ in 0..chunks {
        let a0 = _mm256_loadu_ps(ap.add(i));
        let b0 = _mm256_loadu_ps(bp.add(i));
        let d0 = _mm256_sub_ps(a0, b0);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        
        let a1 = _mm256_loadu_ps(ap.add(i + 8));
        let b1 = _mm256_loadu_ps(bp.add(i + 8));
        let d1 = _mm256_sub_ps(a1, b1);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
        
        let a2 = _mm256_loadu_ps(ap.add(i + 16));
        let b2 = _mm256_loadu_ps(bp.add(i + 16));
        let d2 = _mm256_sub_ps(a2, b2);
        sum2 = _mm256_fmadd_ps(d2, d2, sum2);
        
        let a3 = _mm256_loadu_ps(ap.add(i + 24));
        let b3 = _mm256_loadu_ps(bp.add(i + 24));
        let d3 = _mm256_sub_ps(a3, b3);
        sum3 = _mm256_fmadd_ps(d3, d3, sum3);
        
        i += 32;
    }
    
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    
    let low = _mm256_castps256_ps128(sum0);
    let high = _mm256_extractf128_ps(sum0, 1);
    let sum128 = _mm_add_ps(low, high);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x4e);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0xb1);
    let result = _mm_add_ps(sums, shuf2);
    _mm_cvtss_f32(result)
}
