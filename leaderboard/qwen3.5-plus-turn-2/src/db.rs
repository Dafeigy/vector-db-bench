use crate::api::SearchResult;
use std::sync::RwLock;
use rayon::prelude::*;

const DIM: usize = 128;

pub struct VectorDB {
    inner: RwLock<Inner>,
}

struct Inner {
    ids: Vec<u64>,
    data: Vec<f32>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            inner: RwLock::new(Inner {
                ids: Vec::new(),
                data: Vec::new(),
            }),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut inner = self.inner.write().unwrap();
        inner.ids.push(id);
        inner.data.extend_from_slice(&vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();
        let mut inner = self.inner.write().unwrap();
        
        inner.ids.reserve(count);
        inner.data.reserve(count * DIM);
        
        for (id, vector) in vectors {
            inner.ids.push(id);
            inner.data.extend_from_slice(&vector);
        }
        
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let inner = self.inner.read().unwrap();
        let n = inner.ids.len();
        
        if n == 0 {
            return Vec::new();
        }

        let ids = &inner.ids;
        let data = &inner.data;

        // Simple parallel iteration
        let mut results: Vec<(u64, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let start = i * DIM;
                let vec = &data[start..start + DIM];
                let dist = l2_distance_squared(query, vec);
                (ids[i], dist)
            })
            .collect();

        let k = (top_k as usize).min(results.len());
        if k > 0 {
            results.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap()
            });
            results.truncate(k);
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }

        results
            .into_iter()
            .map(|(id, dist)| SearchResult { id, distance: dist as f64 })
            .collect()
    }
}

#[inline]
fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { l2_distance_squared_avx512_impl(a, b) }
    } else {
        l2_distance_squared_fallback(a, b)
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn l2_distance_squared_avx512_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();
    let mut sum4 = _mm512_setzero_ps();
    let mut sum5 = _mm512_setzero_ps();
    let mut sum6 = _mm512_setzero_ps();
    let mut sum7 = _mm512_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let va0 = _mm512_loadu_ps(a_ptr.add(0));
    let vb0 = _mm512_loadu_ps(b_ptr.add(0));
    let diff0 = _mm512_sub_ps(va0, vb0);
    sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);

    let va1 = _mm512_loadu_ps(a_ptr.add(16));
    let vb1 = _mm512_loadu_ps(b_ptr.add(16));
    let diff1 = _mm512_sub_ps(va1, vb1);
    sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);

    let va2 = _mm512_loadu_ps(a_ptr.add(32));
    let vb2 = _mm512_loadu_ps(b_ptr.add(32));
    let diff2 = _mm512_sub_ps(va2, vb2);
    sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);

    let va3 = _mm512_loadu_ps(a_ptr.add(48));
    let vb3 = _mm512_loadu_ps(b_ptr.add(48));
    let diff3 = _mm512_sub_ps(va3, vb3);
    sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);

    let va4 = _mm512_loadu_ps(a_ptr.add(64));
    let vb4 = _mm512_loadu_ps(b_ptr.add(64));
    let diff4 = _mm512_sub_ps(va4, vb4);
    sum4 = _mm512_fmadd_ps(diff4, diff4, sum4);

    let va5 = _mm512_loadu_ps(a_ptr.add(80));
    let vb5 = _mm512_loadu_ps(b_ptr.add(80));
    let diff5 = _mm512_sub_ps(va5, vb5);
    sum5 = _mm512_fmadd_ps(diff5, diff5, sum5);

    let va6 = _mm512_loadu_ps(a_ptr.add(96));
    let vb6 = _mm512_loadu_ps(b_ptr.add(96));
    let diff6 = _mm512_sub_ps(va6, vb6);
    sum6 = _mm512_fmadd_ps(diff6, diff6, sum6);

    let va7 = _mm512_loadu_ps(a_ptr.add(112));
    let vb7 = _mm512_loadu_ps(b_ptr.add(112));
    let diff7 = _mm512_sub_ps(va7, vb7);
    sum7 = _mm512_fmadd_ps(diff7, diff7, sum7);

    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum4 = _mm512_add_ps(sum4, sum5);
    sum6 = _mm512_add_ps(sum6, sum7);
    
    sum0 = _mm512_add_ps(sum0, sum2);
    sum4 = _mm512_add_ps(sum4, sum6);
    sum0 = _mm512_add_ps(sum0, sum4);

    let mut result = [0.0f32; 16];
    _mm512_storeu_ps(result.as_mut_ptr(), sum0);
    
    result[0] + result[1] + result[2] + result[3] + 
    result[4] + result[5] + result[6] + result[7] +
    result[8] + result[9] + result[10] + result[11] +
    result[12] + result[13] + result[14] + result[15]
}

#[inline]
fn l2_distance_squared_fallback(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..DIM {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}
