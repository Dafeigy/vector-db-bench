pub fn l2_dist_sq(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);
    let mut sum = 0.0f64;
    for i in 0..128 {
        let d = f64::from(a[i]) - f64::from(b[i]);
        sum += d * d;
    }
    sum
}