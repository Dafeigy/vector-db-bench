#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    // Use AVX-512 intrinsics for 128-dim vectors (128 floats = 512 bytes = 8 AVX-512 zmm registers)
    // Fallback to scalar if not available at runtime; compile with target-cpu=native to enable.
    #[cfg(target_feature = "avx512f")]
    unsafe {
        use std::arch::x86_64::*;
        let mut sum = _mm512_setzero_ps();
        let mut i = 0;
        while i + 16 <= a.len() {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let diff = _mm512_sub_ps(va, vb);
            sum = _mm512_fmadd_ps(diff, diff, sum);
            i += 16;
        }
        let mut acc = _mm512_reduce_add_ps(sum);
        // handle remainder
        while i < a.len() {
            let diff = a[i] - b[i];
            acc += diff * diff;
            i += 1;
        }
        (acc as f64).sqrt()
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        let mut sum = 0.0f64;
        for i in 0..a.len() {
            let diff = a[i] as f64 - b[i] as f64;
            sum += diff * diff;
        }
        sum.sqrt()
    }
}
