pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    (sum as f64).sqrt()
}
