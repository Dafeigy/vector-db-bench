use std::arch::x86_64::*;

#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq")]
unsafe fn l2_avx512(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= a.len() {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
        i += 16;
    }
    
    // Reduce over AVX512
    let sum_256 = _mm256_add_ps(
        _mm512_castps512_ps256(sum),
        _mm512_extractf32x8_ps(sum, 1)
    );
    let sum_128 = _mm_add_ps(
        _mm256_castps256_ps128(sum_256),
        _mm256_extractf128_ps(sum_256, 1)
    );
    let mut arr = [0.0; 4];
    _mm_storeu_ps(arr.as_mut_ptr(), sum_128);
    let mut res = arr[0] + arr[1] + arr[2] + arr[3];

    for j in i..a.len() {
        let d = a[j] - b[j];
        res += d * d;
    }
    res as f64
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { l2_avx512(a, b) }
    } else {
        a.iter().zip(b.iter()).map(|(x, y)| {
            let d = x - y;
            (d * d) as f64
        }).sum()
    }
}
