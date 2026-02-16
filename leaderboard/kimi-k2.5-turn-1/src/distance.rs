use std::arch::x86_64::*;

const DIMENSION: usize = 128;

/// Compute L2 distance squared between two 128-dim vectors using AVX-512
/// Assumes both slices are exactly 128 elements
#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), DIMENSION);
    debug_assert_eq!(b.len(), DIMENSION);
    
    unsafe {
        // Process all 128 elements in fully unrolled AVX-512 operations
        // 128 floats = 8 AVX-512 registers worth (8 * 16 = 128)
        
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut acc4 = _mm512_setzero_ps();
        let mut acc5 = _mm512_setzero_ps();
        let mut acc6 = _mm512_setzero_ps();
        let mut acc7 = _mm512_setzero_ps();
        
        // Load and process 8 chunks of 16 floats each
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        
        let va0 = _mm512_loadu_ps(a_ptr.add(0));
        let vb0 = _mm512_loadu_ps(b_ptr.add(0));
        let diff0 = _mm512_sub_ps(va0, vb0);
        acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);
        
        let va1 = _mm512_loadu_ps(a_ptr.add(16));
        let vb1 = _mm512_loadu_ps(b_ptr.add(16));
        let diff1 = _mm512_sub_ps(va1, vb1);
        acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);
        
        let va2 = _mm512_loadu_ps(a_ptr.add(32));
        let vb2 = _mm512_loadu_ps(b_ptr.add(32));
        let diff2 = _mm512_sub_ps(va2, vb2);
        acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);
        
        let va3 = _mm512_loadu_ps(a_ptr.add(48));
        let vb3 = _mm512_loadu_ps(b_ptr.add(48));
        let diff3 = _mm512_sub_ps(va3, vb3);
        acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);
        
        let va4 = _mm512_loadu_ps(a_ptr.add(64));
        let vb4 = _mm512_loadu_ps(b_ptr.add(64));
        let diff4 = _mm512_sub_ps(va4, vb4);
        acc4 = _mm512_fmadd_ps(diff4, diff4, acc4);
        
        let va5 = _mm512_loadu_ps(a_ptr.add(80));
        let vb5 = _mm512_loadu_ps(b_ptr.add(80));
        let diff5 = _mm512_sub_ps(va5, vb5);
        acc5 = _mm512_fmadd_ps(diff5, diff5, acc5);
        
        let va6 = _mm512_loadu_ps(a_ptr.add(96));
        let vb6 = _mm512_loadu_ps(b_ptr.add(96));
        let diff6 = _mm512_sub_ps(va6, vb6);
        acc6 = _mm512_fmadd_ps(diff6, diff6, acc6);
        
        let va7 = _mm512_loadu_ps(a_ptr.add(112));
        let vb7 = _mm512_loadu_ps(b_ptr.add(112));
        let diff7 = _mm512_sub_ps(va7, vb7);
        acc7 = _mm512_fmadd_ps(diff7, diff7, acc7);
        
        // Combine all accumulators
        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc4 = _mm512_add_ps(acc4, acc5);
        acc6 = _mm512_add_ps(acc6, acc7);
        
        acc0 = _mm512_add_ps(acc0, acc2);
        acc4 = _mm512_add_ps(acc4, acc6);
        acc0 = _mm512_add_ps(acc0, acc4);
        
        _mm512_reduce_add_ps(acc0) as f64
    }
}

/// Compute L2 distance between two vectors using AVX-512
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    
    let len = a.len();
    let mut i = 0;
    let mut sum = 0.0f64;
    
    unsafe {
        // Process 16 elements at a time using AVX-512
        if len >= 16 {
            let mut acc = _mm512_setzero_ps();
            
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                let vb = _mm512_loadu_ps(b.as_ptr().add(i));
                let diff = _mm512_sub_ps(va, vb);
                acc = _mm512_fmadd_ps(diff, diff, acc);
                i += 16;
            }
            
            sum = _mm512_reduce_add_ps(acc) as f64;
        }
        
        // Process remaining elements
        while i < len {
            let diff = a[i] - b[i];
            sum += (diff * diff) as f64;
            i += 1;
        }
    }
    
    sum.sqrt()
}
