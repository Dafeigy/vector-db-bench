pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    // Ensure both slices have the same length (should be 128)
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f64;
    for (&ai, &bi) in a.iter().zip(b) {
        let diff = ai as f64 - bi as f64;
        sum += diff * diff;
    }
    sum
}