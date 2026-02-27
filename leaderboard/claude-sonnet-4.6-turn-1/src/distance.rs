/// Compute squared L2 distance between two 128-dim f32 vectors.
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    l2_distance_f32(a, b) as f64
}

/// Squared L2 distance as f32 (inner use).
#[inline(always)]
pub fn l2_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    // Always use scalar — the compiler will auto-vectorize with target-cpu=native
    let mut s = 0.0f32;
    // Manual unroll for 128 dims
    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        let d0 = a[i] - b[i];
        let d1 = a[i+1] - b[i+1];
        let d2 = a[i+2] - b[i+2];
        let d3 = a[i+3] - b[i+3];
        let d4 = a[i+4] - b[i+4];
        let d5 = a[i+5] - b[i+5];
        let d6 = a[i+6] - b[i+6];
        let d7 = a[i+7] - b[i+7];
        s += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7;
        i += 8;
    }
    while i < n {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    s
}
