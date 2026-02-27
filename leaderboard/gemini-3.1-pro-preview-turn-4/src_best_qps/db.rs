use std::arch::x86_64::*;
use crate::api::*;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct SearchResultInternal {
    pub id: u64,
    pub distance: f32,
}

pub struct VectorDB {
    ids: parking_lot::RwLock<Vec<u64>>,
    vecs: parking_lot::RwLock<Vec<f32>>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            ids: parking_lot::RwLock::new(Vec::with_capacity(1_000_000)),
            vecs: parking_lot::RwLock::new(Vec::with_capacity(1_000_000 * 128)),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        self.ids.write().push(id);
        self.vecs.write().extend_from_slice(&vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut ids = self.ids.write();
        let mut vecs = self.vecs.write();
        let len = vectors.len();
        for (id, vec) in vectors {
            ids.push(id);
            vecs.extend_from_slice(&vec);
        }
        len
    }

    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq")]
    unsafe fn search_avx512(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let ids = self.ids.read();
        let vecs = self.vecs.read();
        
        let chunk_size = 8192;
        let num_vecs = ids.len();
        let k = top_k as usize;
        let num_chunks = (num_vecs + chunk_size - 1) / chunk_size;

        let q0 = _mm512_loadu_ps(vector.as_ptr());
        let q1 = _mm512_loadu_ps(vector.as_ptr().add(16));
        let q2 = _mm512_loadu_ps(vector.as_ptr().add(32));
        let q3 = _mm512_loadu_ps(vector.as_ptr().add(48));
        let q4 = _mm512_loadu_ps(vector.as_ptr().add(64));
        let q5 = _mm512_loadu_ps(vector.as_ptr().add(80));
        let q6 = _mm512_loadu_ps(vector.as_ptr().add(96));
        let q7 = _mm512_loadu_ps(vector.as_ptr().add(112));

        let local_best: Vec<Vec<SearchResultInternal>> = (0..num_chunks).into_par_iter().map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = std::cmp::min(start + chunk_size, num_vecs);
            
            let mut best_k = vec![SearchResultInternal { id: 0, distance: std::f32::INFINITY }; k];
            let mut max_dist = std::f32::INFINITY;
            
            let id_ptr = ids.as_ptr();
            let vec_ptr = vecs.as_ptr();

            for i in start..end {
                let offset = i * 128;
                let ptr = vec_ptr.add(offset);
                
                let v0 = _mm512_loadu_ps(ptr);
                let d0 = _mm512_sub_ps(q0, v0);
                let mut sum = _mm512_mul_ps(d0, d0);
                
                let v1 = _mm512_loadu_ps(ptr.add(16));
                let d1 = _mm512_sub_ps(q1, v1);
                sum = _mm512_fmadd_ps(d1, d1, sum);
                
                let v2 = _mm512_loadu_ps(ptr.add(32));
                let d2 = _mm512_sub_ps(q2, v2);
                sum = _mm512_fmadd_ps(d2, d2, sum);
                
                let v3 = _mm512_loadu_ps(ptr.add(48));
                let d3 = _mm512_sub_ps(q3, v3);
                sum = _mm512_fmadd_ps(d3, d3, sum);
                
                let v4 = _mm512_loadu_ps(ptr.add(64));
                let d4 = _mm512_sub_ps(q4, v4);
                sum = _mm512_fmadd_ps(d4, d4, sum);
                
                let v5 = _mm512_loadu_ps(ptr.add(80));
                let d5 = _mm512_sub_ps(q5, v5);
                sum = _mm512_fmadd_ps(d5, d5, sum);
                
                let v6 = _mm512_loadu_ps(ptr.add(96));
                let d6 = _mm512_sub_ps(q6, v6);
                sum = _mm512_fmadd_ps(d6, d6, sum);
                
                let v7 = _mm512_loadu_ps(ptr.add(112));
                let d7 = _mm512_sub_ps(q7, v7);
                sum = _mm512_fmadd_ps(d7, d7, sum);
                
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
                let dist = arr[0] + arr[1] + arr[2] + arr[3];

                if dist < max_dist {
                    let mut pos = k - 1;
                    while pos > 0 && best_k[pos - 1].distance > dist {
                        best_k[pos] = best_k[pos - 1];
                        pos -= 1;
                    }
                    best_k[pos] = SearchResultInternal { id: *id_ptr.add(i), distance: dist };
                    max_dist = best_k[k - 1].distance;
                }
            }
            best_k
        }).collect();

        // Merge results
        let mut final_best = vec![SearchResultInternal { id: 0, distance: std::f32::INFINITY }; k];
        for local_k in local_best {
            for item in local_k {
                if item.distance < final_best[k - 1].distance {
                    let mut pos = k - 1;
                    while pos > 0 && final_best[pos - 1].distance > item.distance {
                        final_best[pos] = final_best[pos - 1];
                        pos -= 1;
                    }
                    final_best[pos] = item;
                }
            }
        }
        
        let mut results = Vec::with_capacity(k);
        for item in final_best {
            if item.distance < std::f32::INFINITY {
                results.push(SearchResult { id: item.id, distance: item.distance as f64 });
            }
        }
        results
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        if is_x86_feature_detected!("avx512f") {
            unsafe { self.search_avx512(vector, top_k) }
        } else {
            todo!("fallback")
        }
    }
}
