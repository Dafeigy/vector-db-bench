/// Compute L2 squared distance between two 128-dim f32 vectors, returned as f64.
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { l2_distance_avx512_single(a, b) };
        }
    }
    l2_distance_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn l2_distance_avx512_single(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let d0 = _mm512_sub_ps(_mm512_loadu_ps(a_ptr), _mm512_loadu_ps(b_ptr));
    let mut sum = _mm512_mul_ps(d0, d0);
    let d1 = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(16)), _mm512_loadu_ps(b_ptr.add(16)));
    sum = _mm512_fmadd_ps(d1, d1, sum);
    let d2 = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(32)), _mm512_loadu_ps(b_ptr.add(32)));
    sum = _mm512_fmadd_ps(d2, d2, sum);
    let d3 = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(48)), _mm512_loadu_ps(b_ptr.add(48)));
    sum = _mm512_fmadd_ps(d3, d3, sum);
    let d4 = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(64)), _mm512_loadu_ps(b_ptr.add(64)));
    sum = _mm512_fmadd_ps(d4, d4, sum);
    let d5 = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(80)), _mm512_loadu_ps(b_ptr.add(80)));
    sum = _mm512_fmadd_ps(d5, d5, sum);
    let d6 = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(96)), _mm512_loadu_ps(b_ptr.add(96)));
    sum = _mm512_fmadd_ps(d6, d6, sum);
    let d7 = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(112)), _mm512_loadu_ps(b_ptr.add(112)));
    sum = _mm512_fmadd_ps(d7, d7, sum);

    _mm512_reduce_add_ps(sum) as f64
}

#[inline]
fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum as f64
}

/// Batch L2 distance with software prefetching for better memory access patterns.
#[inline]
pub fn l2_distance_batch(query: &[f32], vectors_flat: &[f32], n_vectors: usize, distances: &mut [f32]) {
    debug_assert_eq!(query.len(), 128);
    debug_assert!(vectors_flat.len() >= n_vectors * 128);
    debug_assert!(distances.len() >= n_vectors);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { l2_distance_batch_avx512_prefetch(query, vectors_flat, n_vectors, distances) };
            return;
        }
    }

    for i in 0..n_vectors {
        let offset = i * 128;
        let v = &vectors_flat[offset..offset + 128];
        distances[i] = l2_distance_scalar(query, v) as f32;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_batch_avx512_prefetch(query: &[f32], vectors_flat: &[f32], n_vectors: usize, distances: &mut [f32]) {
    use std::arch::x86_64::*;

    let q_ptr = query.as_ptr();
    let q0 = _mm512_loadu_ps(q_ptr);
    let q1 = _mm512_loadu_ps(q_ptr.add(16));
    let q2 = _mm512_loadu_ps(q_ptr.add(32));
    let q3 = _mm512_loadu_ps(q_ptr.add(48));
    let q4 = _mm512_loadu_ps(q_ptr.add(64));
    let q5 = _mm512_loadu_ps(q_ptr.add(80));
    let q6 = _mm512_loadu_ps(q_ptr.add(96));
    let q7 = _mm512_loadu_ps(q_ptr.add(112));

    let v_ptr = vectors_flat.as_ptr();
    let d_ptr = distances.as_mut_ptr();

    // Prefetch distance: 2 vectors ahead (2 * 512 bytes = 1024 bytes)
    const PREFETCH_AHEAD: usize = 4;

    let mut i = 0;

    while i < n_vectors {
        let base = v_ptr.add(i * 128);

        // Prefetch next vector
        if i + PREFETCH_AHEAD < n_vectors {
            let pf_base = v_ptr.add((i + PREFETCH_AHEAD) * 128);
            _mm_prefetch(pf_base as *const i8, _MM_HINT_T0);
            _mm_prefetch(pf_base.add(16) as *const i8, _MM_HINT_T0);
            _mm_prefetch(pf_base.add(32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(pf_base.add(48) as *const i8, _MM_HINT_T0);
            _mm_prefetch(pf_base.add(64) as *const i8, _MM_HINT_T0);
            _mm_prefetch(pf_base.add(80) as *const i8, _MM_HINT_T0);
            _mm_prefetch(pf_base.add(96) as *const i8, _MM_HINT_T0);
            _mm_prefetch(pf_base.add(112) as *const i8, _MM_HINT_T0);
        }

        let d0 = _mm512_sub_ps(q0, _mm512_loadu_ps(base));
        let mut sum = _mm512_mul_ps(d0, d0);
        let d1 = _mm512_sub_ps(q1, _mm512_loadu_ps(base.add(16)));
        sum = _mm512_fmadd_ps(d1, d1, sum);
        let d2 = _mm512_sub_ps(q2, _mm512_loadu_ps(base.add(32)));
        sum = _mm512_fmadd_ps(d2, d2, sum);
        let d3 = _mm512_sub_ps(q3, _mm512_loadu_ps(base.add(48)));
        sum = _mm512_fmadd_ps(d3, d3, sum);
        let d4 = _mm512_sub_ps(q4, _mm512_loadu_ps(base.add(64)));
        sum = _mm512_fmadd_ps(d4, d4, sum);
        let d5 = _mm512_sub_ps(q5, _mm512_loadu_ps(base.add(80)));
        sum = _mm512_fmadd_ps(d5, d5, sum);
        let d6 = _mm512_sub_ps(q6, _mm512_loadu_ps(base.add(96)));
        sum = _mm512_fmadd_ps(d6, d6, sum);
        let d7 = _mm512_sub_ps(q7, _mm512_loadu_ps(base.add(112)));
        sum = _mm512_fmadd_ps(d7, d7, sum);

        *d_ptr.add(i) = _mm512_reduce_add_ps(sum);
        i += 1;
    }
}
