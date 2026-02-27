#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let diff = a[i] as f64 - b[i] as f64;
        sum += diff * diff;
    }
    sum.sqrt()
}
