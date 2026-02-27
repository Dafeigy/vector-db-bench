#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    // Specialized fast path for 128 dimensions
    if a.len() == 128 {
        return l2_distance_128(a, b);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { l2_distance_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_distance_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.2") {
            return unsafe { l2_distance_sse(a, b) };
        }
    }
    l2_distance_scalar(a, b)
}

// Specialized for 128 dimensions
fn l2_distance_128(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { l2_distance_128_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_distance_128_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.2") {
            return unsafe { l2_distance_128_sse(a, b) };
        }
    }
    l2_distance_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_128_avx512(a: &[f32], b: &[f32]) -> f64 {
    // 128 floats = 8 AVX-512 registers
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();

    // unrolled loop: each iteration processes 16 floats
    let va0 = _mm512_loadu_ps(a_ptr);
    let vb0 = _mm512_loadu_ps(b_ptr);
    let diff0 = _mm512_sub_ps(va0, vb0);
    acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

    let va1 = _mm512_loadu_ps(a_ptr.add(16));
    let vb1 = _mm512_loadu_ps(b_ptr.add(16));
    let diff1 = _mm512_sub_ps(va1, vb1);
    acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

    let va2 = _mm512_loadu_ps(a_ptr.add(32));
    let vb2 = _mm512_loadu_ps(b_ptr.add(32));
    let diff2 = _mm512_sub_ps(va2, vb2);
    acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

    let va3 = _mm512_loadu_ps(a_ptr.add(48));
    let vb3 = _mm512_loadu_ps(b_ptr.add(48));
    let diff3 = _mm512_sub_ps(va3, vb3);
    acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);

    let va4 = _mm512_loadu_ps(a_ptr.add(64));
    let vb4 = _mm512_loadu_ps(b_ptr.add(64));
    let diff4 = _mm512_sub_ps(va4, vb4);
    acc4 = _mm512_fmadd_ps(diff4, diff4, acc4);

    let va5 = _mm512_loadu_ps(a_ptr.add(80));
    let vb5 = _mm512_loadu_ps(b_ptr.add(80));
    let diff5 = _mm512_sub_ps(va5, vb5);
    acc5 = _mm512_fmadd_ps(diff5, diff5, acc5);

    let va6 = _mm512_loadu_ps(a_ptr.add(96));
    let vb6 = _mm512_loadu_ps(b_ptr.add(96));
    let diff6 = _mm512_sub_ps(va6, vb6);
    acc6 = _mm512_fmadd_ps(diff6, diff6, acc6);

    let va7 = _mm512_loadu_ps(a_ptr.add(112));
    let vb7 = _mm512_loadu_ps(b_ptr.add(112));
    let diff7 = _mm512_sub_ps(va7, vb7);
    acc7 = _mm512_fmadd_ps(diff7, diff7, acc7);

    // sum all accumulators
    let sum = reduce_add512_ps(acc0) + reduce_add512_ps(acc1) +
              reduce_add512_ps(acc2) + reduce_add512_ps(acc3) +
              reduce_add512_ps(acc4) + reduce_add512_ps(acc5) +
              reduce_add512_ps(acc6) + reduce_add512_ps(acc7);
    (sum as f64).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn reduce_add512_ps(v: __m512) -> f32 {
    // Use _mm512_reduce_add_ps if available (requires AVX512DQ)
    // Fallback to store and sum
    #[cfg(target_feature = "avx512dq")]
    {
        return _mm512_reduce_add_ps(v);
    }
    #[cfg(not(target_feature = "avx512dq"))]
    {
        let mut buf = [0.0f32; 16];
        _mm512_storeu_ps(buf.as_mut_ptr(), v);
        buf.iter().sum()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_128_avx2(a: &[f32], b: &[f32]) -> f64 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();

    // each iteration processes 8 floats
    for i in 0..16 {
        let va = _mm256_loadu_ps(a_ptr.add(i * 8));
        let vb = _mm256_loadu_ps(b_ptr.add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        match i {
            0 => acc0 = _mm256_fmadd_ps(diff, diff, acc0),
            1 => acc1 = _mm256_fmadd_ps(diff, diff, acc1),
            2 => acc2 = _mm256_fmadd_ps(diff, diff, acc2),
            3 => acc3 = _mm256_fmadd_ps(diff, diff, acc3),
            4 => acc4 = _mm256_fmadd_ps(diff, diff, acc4),
            5 => acc5 = _mm256_fmadd_ps(diff, diff, acc5),
            6 => acc6 = _mm256_fmadd_ps(diff, diff, acc6),
            7 => acc7 = _mm256_fmadd_ps(diff, diff, acc7),
            8 => acc0 = _mm256_fmadd_ps(diff, diff, acc0),
            9 => acc1 = _mm256_fmadd_ps(diff, diff, acc1),
            10 => acc2 = _mm256_fmadd_ps(diff, diff, acc2),
            11 => acc3 = _mm256_fmadd_ps(diff, diff, acc3),
            12 => acc4 = _mm256_fmadd_ps(diff, diff, acc4),
            13 => acc5 = _mm256_fmadd_ps(diff, diff, acc5),
            14 => acc6 = _mm256_fmadd_ps(diff, diff, acc6),
            15 => acc7 = _mm256_fmadd_ps(diff, diff, acc7),
            _ => unreachable!(),
        }
    }
    let sum = reduce_add256_ps(acc0) + reduce_add256_ps(acc1) +
              reduce_add256_ps(acc2) + reduce_add256_ps(acc3) +
              reduce_add256_ps(acc4) + reduce_add256_ps(acc5) +
              reduce_add256_ps(acc6) + reduce_add256_ps(acc7);
    (sum as f64).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn l2_distance_128_sse(a: &[f32], b: &[f32]) -> f64 {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut acc0 = _mm_setzero_ps();
    let mut acc1 = _mm_setzero_ps();
    let mut acc2 = _mm_setzero_ps();
    let mut acc3 = _mm_setzero_ps();
    let mut acc4 = _mm_setzero_ps();
    let mut acc5 = _mm_setzero_ps();
    let mut acc6 = _mm_setzero_ps();
    let mut acc7 = _mm_setzero_ps();
    let mut acc8 = _mm_setzero_ps();
    let mut acc9 = _mm_setzero_ps();
    let mut acc10 = _mm_setzero_ps();
    let mut acc11 = _mm_setzero_ps();
    let mut acc12 = _mm_setzero_ps();
    let mut acc13 = _mm_setzero_ps();
    let mut acc14 = _mm_setzero_ps();
    let mut acc15 = _mm_setzero_ps();

    // each iteration processes 4 floats
    for i in 0..32 {
        let va = _mm_loadu_ps(a_ptr.add(i * 4));
        let vb = _mm_loadu_ps(b_ptr.add(i * 4));
        let diff = _mm_sub_ps(va, vb);
        match i {
            0 => acc0 = _mm_fmadd_ps(diff, diff, acc0),
            1 => acc1 = _mm_fmadd_ps(diff, diff, acc1),
            2 => acc2 = _mm_fmadd_ps(diff, diff, acc2),
            3 => acc3 = _mm_fmadd_ps(diff, diff, acc3),
            4 => acc4 = _mm_fmadd_ps(diff, diff, acc4),
            5 => acc5 = _mm_fmadd_ps(diff, diff, acc5),
            6 => acc6 = _mm_fmadd_ps(diff, diff, acc6),
            7 => acc7 = _mm_fmadd_ps(diff, diff, acc7),
            8 => acc8 = _mm_fmadd_ps(diff, diff, acc8),
            9 => acc9 = _mm_fmadd_ps(diff, diff, acc9),
            10 => acc10 = _mm_fmadd_ps(diff, diff, acc10),
            11 => acc11 = _mm_fmadd_ps(diff, diff, acc11),
            12 => acc12 = _mm_fmadd_ps(diff, diff, acc12),
            13 => acc13 = _mm_fmadd_ps(diff, diff, acc13),
            14 => acc14 = _mm_fmadd_ps(diff, diff, acc14),
            15 => acc15 = _mm_fmadd_ps(diff, diff, acc15),
            16 => acc0 = _mm_fmadd_ps(diff, diff, acc0),
            17 => acc1 = _mm_fmadd_ps(diff, diff, acc1),
            18 => acc2 = _mm_fmadd_ps(diff, diff, acc2),
            19 => acc3 = _mm_fmadd_ps(diff, diff, acc3),
            20 => acc4 = _mm_fmadd_ps(diff, diff, acc4),
            21 => acc5 = _mm_fmadd_ps(diff, diff, acc5),
            22 => acc6 = _mm_fmadd_ps(diff, diff, acc6),
            23 => acc7 = _mm_fmadd_ps(diff, diff, acc7),
            24 => acc8 = _mm_fmadd_ps(diff, diff, acc8),
            25 => acc9 = _mm_fmadd_ps(diff, diff, acc9),
            26 => acc10 = _mm_fmadd_ps(diff, diff, acc10),
            27 => acc11 = _mm_fmadd_ps(diff, diff, acc11),
            28 => acc12 = _mm_fmadd_ps(diff, diff, acc12),
            29 => acc13 = _mm_fmadd_ps(diff, diff, acc13),
            30 => acc14 = _mm_fmadd_ps(diff, diff, acc14),
            31 => acc15 = _mm_fmadd_ps(diff, diff, acc15),
            _ => unreachable!(),
        }
    }
    let sum = reduce_add128_ps(acc0) + reduce_add128_ps(acc1) +
              reduce_add128_ps(acc2) + reduce_add128_ps(acc3) +
              reduce_add128_ps(acc4) + reduce_add128_ps(acc5) +
              reduce_add128_ps(acc6) + reduce_add128_ps(acc7) +
              reduce_add128_ps(acc8) + reduce_add128_ps(acc9) +
              reduce_add128_ps(acc10) + reduce_add128_ps(acc11) +
              reduce_add128_ps(acc12) + reduce_add128_ps(acc13) +
              reduce_add128_ps(acc14) + reduce_add128_ps(acc15);
    (sum as f64).sqrt()
}

// Generic SIMD versions (fallback for non-128 lengths)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i < a.len() {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(diff, diff, acc);
        i += 16;
    }
    let sum = reduce_add512_ps(acc);
    (sum as f64).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f64 {
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut i = 0;
    while i < a.len() {
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
        let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let diff2 = _mm256_sub_ps(va2, vb2);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);
        i += 16;
    }
    let sum = reduce_add256_ps(acc1) + reduce_add256_ps(acc2);
    (sum as f64).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn l2_distance_sse(a: &[f32], b: &[f32]) -> f64 {
    let mut acc1 = _mm_setzero_ps();
    let mut acc2 = _mm_setzero_ps();
    let mut acc3 = _mm_setzero_ps();
    let mut acc4 = _mm_setzero_ps();
    let mut i = 0;
    while i < a.len() {
        let va1 = _mm_loadu_ps(a.as_ptr().add(i));
        let vb1 = _mm_loadu_ps(b.as_ptr().add(i));
        let diff1 = _mm_sub_ps(va1, vb1);
        acc1 = _mm_fmadd_ps(diff1, diff1, acc1);
        let va2 = _mm_loadu_ps(a.as_ptr().add(i + 4));
        let vb2 = _mm_loadu_ps(b.as_ptr().add(i + 4));
        let diff2 = _mm_sub_ps(va2, vb2);
        acc2 = _mm_fmadd_ps(diff2, diff2, acc2);
        let va3 = _mm_loadu_ps(a.as_ptr().add(i + 8));
        let vb3 = _mm_loadu_ps(b.as_ptr().add(i + 8));
        let diff3 = _mm_sub_ps(va3, vb3);
        acc3 = _mm_fmadd_ps(diff3, diff3, acc3);
        let va4 = _mm_loadu_ps(a.as_ptr().add(i + 12));
        let vb4 = _mm_loadu_ps(b.as_ptr().add(i + 12));
        let diff4 = _mm_sub_ps(va4, vb4);
        acc4 = _mm_fmadd_ps(diff4, diff4, acc4);
        i += 16;
    }
    let sum = reduce_add128_ps(acc1) + reduce_add128_ps(acc2) +
               reduce_add128_ps(acc3) + reduce_add128_ps(acc4);
    (sum as f64).sqrt()
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn reduce_add256_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_extractf128_ps(v, 0);
    let sum128 = _mm_add_ps(hi, lo);
    reduce_add128_ps(sum128)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn reduce_add128_ps(v: __m128) -> f32 {
    let shuf = _mm_shuffle_ps(v, v, 0b10110001);
    let sum1 = _mm_add_ps(v, shuf);
    let shuf2 = _mm_shuffle_ps(sum1, sum1, 0b01001110);
    let sum2 = _mm_add_ps(sum1, shuf2);
    _mm_cvtss_f32(sum2)
}

fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let diff = *x as f64 - *y as f64;
        sum += diff * diff;
    }
    sum.sqrt()
}