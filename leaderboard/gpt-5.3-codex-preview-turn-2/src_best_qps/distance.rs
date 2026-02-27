pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }

    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]) as f64;
        sum += d * d;
    }
    sum.sqrt()
}
