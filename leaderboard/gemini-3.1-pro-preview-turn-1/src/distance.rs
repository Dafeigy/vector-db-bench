use std::arch::x86_64::*;

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    // Both are 128 long
    if a.len() == 128 && b.len() == 128 {
        unsafe { l2_distance_avx512(a, b) }
    } else {
        l2_distance_fallback(a, b)
    }
}

// We will assume the CPU has AVX-512 as stated.
#[target_feature(enable = "avx512f", enable = "avx512vl", enable = "avx512dq")]
unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    let mut sum_vec = _mm512_setzero_ps();
    let mut sum_vec2 = _mm512_setzero_ps();
    let mut sum_vec3 = _mm512_setzero_ps();
    let mut sum_vec4 = _mm512_setzero_ps();

    let ap = a.as_ptr();
    let bp = b.as_ptr();

    // 128 floats = 8 zmm registers
    // Unroll manually
    let va1 = _mm512_loadu_ps(ap);
    let vb1 = _mm512_loadu_ps(bp);
    let diff1 = _mm512_sub_ps(va1, vb1);
    sum_vec = _mm512_fmadd_ps(diff1, diff1, sum_vec);

    let va2 = _mm512_loadu_ps(ap.add(16));
    let vb2 = _mm512_loadu_ps(bp.add(16));
    let diff2 = _mm512_sub_ps(va2, vb2);
    sum_vec2 = _mm512_fmadd_ps(diff2, diff2, sum_vec2);

    let va3 = _mm512_loadu_ps(ap.add(32));
    let vb3 = _mm512_loadu_ps(bp.add(32));
    let diff3 = _mm512_sub_ps(va3, vb3);
    sum_vec3 = _mm512_fmadd_ps(diff3, diff3, sum_vec3);

    let va4 = _mm512_loadu_ps(ap.add(48));
    let vb4 = _mm512_loadu_ps(bp.add(48));
    let diff4 = _mm512_sub_ps(va4, vb4);
    sum_vec4 = _mm512_fmadd_ps(diff4, diff4, sum_vec4);

    let va5 = _mm512_loadu_ps(ap.add(64));
    let vb5 = _mm512_loadu_ps(bp.add(64));
    let diff5 = _mm512_sub_ps(va5, vb5);
    sum_vec = _mm512_fmadd_ps(diff5, diff5, sum_vec);

    let va6 = _mm512_loadu_ps(ap.add(80));
    let vb6 = _mm512_loadu_ps(bp.add(80));
    let diff6 = _mm512_sub_ps(va6, vb6);
    sum_vec2 = _mm512_fmadd_ps(diff6, diff6, sum_vec2);

    let va7 = _mm512_loadu_ps(ap.add(96));
    let vb7 = _mm512_loadu_ps(bp.add(96));
    let diff7 = _mm512_sub_ps(va7, vb7);
    sum_vec3 = _mm512_fmadd_ps(diff7, diff7, sum_vec3);

    let va8 = _mm512_loadu_ps(ap.add(112));
    let vb8 = _mm512_loadu_ps(bp.add(112));
    let diff8 = _mm512_sub_ps(va8, vb8);
    sum_vec4 = _mm512_fmadd_ps(diff8, diff8, sum_vec4);
    
    let sum12 = _mm512_add_ps(sum_vec, sum_vec2);
    let sum34 = _mm512_add_ps(sum_vec3, sum_vec4);
    let sum = _mm512_add_ps(sum12, sum34);
    
    _mm512_reduce_add_ps(sum) as f64
}

fn l2_distance_fallback(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += (diff * diff) as f64;
    }
    sum
}
