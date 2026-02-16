#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);
    
    let mut sum0: f32 = 0.0;
    let mut sum1: f32 = 0.0;
    let mut sum2: f32 = 0.0;
    let mut sum3: f32 = 0.0;
    let mut sum4: f32 = 0.0;
    let mut sum5: f32 = 0.0;
    let mut sum6: f32 = 0.0;
    let mut sum7: f32 = 0.0;
    
    // Unroll loop 8x for 128 elements (128 / 8 = 16 iterations)
    for i in (0..128).step_by(8) {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        let d4 = a[i + 4] - b[i + 4];
        let d5 = a[i + 5] - b[i + 5];
        let d6 = a[i + 6] - b[i + 6];
        let d7 = a[i + 7] - b[i + 7];
        
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
        sum4 += d4 * d4;
        sum5 += d5 * d5;
        sum6 += d6 * d6;
        sum7 += d7 * d7;
    }
    
    sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7
}

#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    (l2_distance_squared(a, b) as f64).sqrt()
}
