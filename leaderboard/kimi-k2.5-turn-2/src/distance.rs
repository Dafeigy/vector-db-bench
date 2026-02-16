use std::arch::x86_64::*;

#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx512f") {
            return l2_distance_avx512(a, b);
        }
        if is_x86_feature_detected!("avx2") {
            return l2_distance_avx2(a, b);
        }
    }
    
    l2_distance_scalar(a, b)
}

#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len();
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();
    
    // Process 64 floats at a time (4 AVX-512 registers)
    let mut i = 0;
    while i + 64 <= len {
        let va0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm512_sub_ps(va0, vb0);
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);
        
        let va1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let vb1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        let diff1 = _mm512_sub_ps(va1, vb1);
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
        
        let va2 = _mm512_loadu_ps(a.as_ptr().add(i + 32));
        let vb2 = _mm512_loadu_ps(b.as_ptr().add(i + 32));
        let diff2 = _mm512_sub_ps(va2, vb2);
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);
        
        let va3 = _mm512_loadu_ps(a.as_ptr().add(i + 48));
        let vb3 = _mm512_loadu_ps(b.as_ptr().add(i + 48));
        let diff3 = _mm512_sub_ps(va3, vb3);
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
        
        i += 64;
    }
    
    // Handle remaining 32 floats
    while i + 32 <= len {
        let va0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm512_sub_ps(va0, vb0);
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);
        
        let va1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let vb1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        let diff1 = _mm512_sub_ps(va1, vb1);
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
        
        i += 32;
    }
    
    // Handle remaining 16 floats
    if i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        sum0 = _mm512_fmadd_ps(diff, diff, sum0);
        i += 16;
    }
    
    // Combine all accumulators
    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);
    
    let mut result = _mm512_reduce_add_ps(sum0) as f64;
    
    // Scalar cleanup
    while i < len {
        let diff = a[i] - b[i];
        result += (diff * diff) as f64;
        i += 1;
    }
    
    result
}

#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len();
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    
    // Process 32 floats at a time (4 AVX2 registers)
    let mut i = 0;
    while i + 32 <= len {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm256_sub_ps(va0, vb0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let diff1 = _mm256_sub_ps(va1, vb1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        
        let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        let diff2 = _mm256_sub_ps(va2, vb2);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        
        let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
        let diff3 = _mm256_sub_ps(va3, vb3);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
        
        i += 32;
    }
    
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    
    // Handle remaining 16 floats
    while i + 16 <= len {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm256_sub_ps(va0, vb0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let diff1 = _mm256_sub_ps(va1, vb1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        
        i += 16;
    }
    
    sum0 = _mm256_add_ps(sum0, sum1);
    
    // Handle remaining 8 floats
    if i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum0 = _mm256_fmadd_ps(diff, diff, sum0);
        i += 8;
    }
    
    let mut result = hsum256_ps_avx(sum0) as f64;
    
    // Scalar cleanup
    while i < len {
        let diff = a[i] - b[i];
        result += (diff * diff) as f64;
        i += 1;
    }
    
    result
}

#[target_feature(enable = "avx2")]
unsafe fn hsum256_ps_avx(x: __m256) -> f32 {
    let x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    let x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = *x - *y;
            (diff * diff) as f64
        })
        .sum()
}
