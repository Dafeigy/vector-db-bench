pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    // Optimized safe version with loop unrolling and f64 accumulator.
    // Use slices for bounds-check elimination and better auto-vectorization.
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut sum = 0.0f64;
    let mut i = 0;
    // Prefetch friendly unrolled loop; let the compiler auto-vectorize with -C target-cpu=native
    while i + 8 <= n {
        let d0 = (a[i] - b[i]) as f64;
        let d1 = (a[i + 1] - b[i + 1]) as f64;
        let d2 = (a[i + 2] - b[i + 2]) as f64;
        let d3 = (a[i + 3] - b[i + 3]) as f64;
        let d4 = (a[i + 4] - b[i + 4]) as f64;
        let d5 = (a[i + 5] - b[i + 5]) as f64;
        let d6 = (a[i + 6] - b[i + 6]) as f64;
        let d7 = (a[i + 7] - b[i + 7]) as f64;
        sum += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7;
        i += 8;
    }
    while i < n {
        let d = (a[i] - b[i]) as f64;
        sum += d * d;
        i += 1;
    }
    sum
}
