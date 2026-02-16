pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (*x - *y) as f64;
        sum += diff * diff;
    }
    sum.sqrt()
}
