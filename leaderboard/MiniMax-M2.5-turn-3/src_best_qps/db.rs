use crate::api::*;
use std::sync::RwLock;
use rayon::prelude::*;

/// VectorDB stores 128-dimensional vectors with unique IDs.
/// Uses SIMD-accelerated distance computation with parallel search.
/// Memory layout: flat contiguous storage for better cache performance.
pub struct VectorDB {
    // Vector IDs
    ids: RwLock<Vec<u64>>,
    // Flat vector data: [v1_dim0, v1_dim1, ..., v1_dim127, v2_dim0, ...]
    vectors: RwLock<Vec<f32>>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            ids: RwLock::new(Vec::new()),
            vectors: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut ids = self.ids.write().unwrap();
        let mut vectors = self.vectors.write().unwrap();
        ids.push(id);
        vectors.extend(vector);
    }

    pub fn bulk_insert(&self, new_vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut ids = self.ids.write().unwrap();
        let mut vectors = self.vectors.write().unwrap();
        let count = new_vectors.len();
        for (id, vec) in new_vectors {
            ids.push(id);
            vectors.extend(vec);
        }
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let ids = self.ids.read().unwrap();
        let vectors = self.vectors.read().unwrap();
        
        if vectors.is_empty() {
            return Vec::new();
        }

        let k = top_k as usize;
        let dim = 128;
        let count = ids.len();
        let query_vec = query.to_vec();
        
        // Compute distances in parallel
        let mut results: Vec<(u64, f64)> = (0..count)
            .into_par_iter()
            .map(|i| {
                let dist = unsafe {
                    let vec_ptr = vectors.as_ptr().add(i * dim);
                    let vec_slice = std::slice::from_raw_parts(vec_ptr, dim);
                    crate::distance::l2_distance(&query_vec, vec_slice)
                };
                (ids[i], dist)
            })
            .collect();
        
        // Use partial sort: get top-k without sorting all 1M elements
        // select_nth_unstable_by partitions so elements [0..k) are the k smallest
        if results.len() > k {
            results.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Sort just the top k elements
            results[..k].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        results
            .into_iter()
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }
}
