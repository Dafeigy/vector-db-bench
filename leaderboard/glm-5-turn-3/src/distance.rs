#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicUsize, Ordering};

// Cache for detected best implementation
static DISTANCE_IMPL: AtomicUsize = AtomicUsize::new(0);
const IMPL_SCALAR: usize = 1;
const IMPL_AVX2: usize = 2;
const IMPL_AVX512: usize = 3;

/// Compute L2 (Euclidean) distance between two 128-dimensional vectors
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let impl_type = DISTANCE_IMPL.load(Ordering::Relaxed);
        if impl_type == 0 {
            detect_and_cache_impl()
        }
        
        match DISTANCE_IMPL.load(Ordering::Relaxed) {
            IMPL_AVX512 => l2_distance_avx512(a.as_ptr(), b.as_ptr()),
            IMPL_AVX2 => l2_distance_avx2(a.as_ptr(), b.as_ptr()),
            _ => l2_distance_scalar(a, b),
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        l2_distance_scalar(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn detect_and_cache_impl() {
    let impl_type = if is_x86_feature_detected!("avx512f") {
        IMPL_AVX512
    } else if is_x86_feature_detected!("avx2") {
        IMPL_AVX2
    } else {
        IMPL_SCALAR
    };
    DISTANCE_IMPL.store(impl_type, Ordering::Relaxed);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn l2_distance_avx512(a: *const f32, b: *const f32) -> f64 {
    // Process 16 floats at a time (512 bits = 16 x 32-bit floats)
    // 128 floats = 8 iterations
    let mut sum = _mm512_setzero_ps();
    
    for i in 0..8 {
        let va = _mm512_loadu_ps(a.add(i * 16));
        let vb = _mm512_loadu_ps(b.add(i * 16));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }
    
    // Horizontal sum of the 512-bit register
    _mm512_reduce_add_ps(sum) as f64
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn l2_distance_avx2(a: *const f32, b: *const f32) -> f64 {
    // Process 8 floats at a time (256 bits = 8 x 32-bit floats)
    // 128 floats = 16 iterations
    let mut sum = _mm256_setzero_ps();
    
    for i in 0..16 {
        let va = _mm256_loadu_ps(a.add(i * 8));
        let vb = _mm256_loadu_ps(b.add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    // Horizontal sum of the 256-bit register
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let mut sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    _mm_cvtss_f32(sum128) as f64
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>() as f64
}

#[cfg(not(target_arch = "x86_64"))]
fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>() as f64
}
