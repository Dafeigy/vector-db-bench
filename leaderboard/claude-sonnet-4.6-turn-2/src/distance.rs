/// Compute squared L2 distance between two f32 vectors.
/// With target-cpu=native (Cascade Lake) the compiler will emit AVX-512 code.
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    l2_distance_f32(a, b) as f64
}

#[inline(always)]
pub fn l2_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    // Manual 16-wide unroll matches one AVX-512 register width.
    // The Rust compiler with opt-level=3 and target-cpu=native will auto-vectorize
    // this into vmovups/vsubps/vfmadd231ps patterns on Cascade Lake.
    let len = a.len().min(b.len());
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = len / 16;
    let mut i = 0usize;

    for _ in 0..chunks {
        let d0  = a[i]    - b[i];
        let d1  = a[i+1]  - b[i+1];
        let d2  = a[i+2]  - b[i+2];
        let d3  = a[i+3]  - b[i+3];
        let d4  = a[i+4]  - b[i+4];
        let d5  = a[i+5]  - b[i+5];
        let d6  = a[i+6]  - b[i+6];
        let d7  = a[i+7]  - b[i+7];
        let d8  = a[i+8]  - b[i+8];
        let d9  = a[i+9]  - b[i+9];
        let d10 = a[i+10] - b[i+10];
        let d11 = a[i+11] - b[i+11];
        let d12 = a[i+12] - b[i+12];
        let d13 = a[i+13] - b[i+13];
        let d14 = a[i+14] - b[i+14];
        let d15 = a[i+15] - b[i+15];
        sum0 += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        sum1 += d4*d4 + d5*d5 + d6*d6 + d7*d7;
        sum2 += d8*d8 + d9*d9 + d10*d10 + d11*d11;
        sum3 += d12*d12 + d13*d13 + d14*d14 + d15*d15;
        i += 16;
    }

    let mut sum = sum0 + sum1 + sum2 + sum3;
    for j in i..len {
        let d = a[j] - b[j];
        sum += d * d;
    }
    sum
}
